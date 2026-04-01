/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/common/global_flags.h"
#include "framework/model/model_input_params.h"
#include "framework/sampling/rec_sampler.h"
#include "rec_worker_impl.h"
#if defined(USE_NPU)
#include "platform/npu/device_capture_lock.h"
#endif
#include "util/timer.h"

namespace xllm {

void RecWorkerImpl::RecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
#if defined(USE_NPU)
  std::optional<std::unique_lock<std::mutex>> lock_guard;
  if (FLAGS_enable_graph) {
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(
            runtime_.worker.device().index());
    lock_guard.emplace(capture_lock);
  }
#endif
  processed_inputs =
      inputs.to(runtime_.worker.device(), runtime_.worker.dtype());
  auto& input_params = processed_inputs.input_params;
#if defined(USE_NPU)
  if (input_params.swap_blocks.size() > 0 && !FLAGS_enable_block_copy_kernel) {
    auto& swap_blocks = input_params.swap_blocks;

    std::vector<int64_t> src_indices;
    std::vector<int64_t> dst_indices;
    src_indices.reserve(swap_blocks.size());
    dst_indices.reserve(swap_blocks.size());

    for (const auto& block : swap_blocks) {
      src_indices.push_back(block.src_block_id);
      dst_indices.push_back(block.dst_block_id);
    }

    torch::Tensor src_tensor = torch::tensor(
        src_indices,
        torch::dtype(torch::kLong).device(runtime_.worker.device_));
    torch::Tensor dst_tensor = torch::tensor(
        dst_indices,
        torch::dtype(torch::kLong).device(runtime_.worker.device_));
    const int64_t num_layers = runtime_.context->get_model_args().n_layers();
    for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      runtime_.worker.kv_caches_[layer_id].swap_blocks(src_tensor, dst_tensor);
    }
  }
  if (runtime_.context->get_model_args().enable_mla() &&
      input_params.batch_forward_type.is_chunked_prefill()) {
    runtime_.worker.prepare_mla_prefixcache_inputs(input_params);
  }

  if (!runtime_.context->get_parallel_args().mapping_data().empty() &&
      (runtime_.context->get_parallel_args().dp_size() > 1 ||
       runtime_.context->get_parallel_args().ep_size() > 1)) {
    torch::Tensor token_size_per_dp_group =
        torch::tensor(processed_inputs.input_params.dp_global_token_nums,
                      torch::TensorOptions()
                          .device(torch::kCPU)
                          .dtype(torch::kInt32)
                          .pinned_memory(true));
    bool is_prefill =
        processed_inputs.input_params.batch_forward_type.is_prefill();
    DpEpPadding dp_ep_padding(
        token_size_per_dp_group,
        runtime_.context->get_model_args().num_experts_per_tok(),
        runtime_.context->get_parallel_args().mapping_data(),
        runtime_.worker.device(),
        runtime_.worker.dtype(),
        is_prefill);
    processed_inputs.input_params.dp_ep_padding_data = dp_ep_padding.build();
  }
#endif
}

ForwardInput RecWorkerImpl::RecWorkPipeline::prepare_inputs(Batch& batch) {
  return runtime_.worker.WorkerImpl::prepare_inputs(batch);
}

std::optional<ForwardOutput> RecWorkerImpl::RecWorkPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  const auto& sampling_params = input.sampling_params;

  std::vector<folly::SemiFuture<bool>> futures;

  if (runtime_.worker.options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            runtime_.context->get_model_args().n_layers());
    const_cast<ModelInputParams*>(&(input.input_params))->layer_synchronizer =
        layer_synchronizer;

    futures.emplace_back(
        runtime_.worker.kv_cache_transfer_->push_kv_blocks_async(
            input.transfer_kv_infos,
            runtime_.context->get_parallel_args(),
            layer_synchronizer,
            runtime_.worker.is_spec_draft_));
#endif
  }

  if (FLAGS_enable_eplb) {
    runtime_.eplb_executor->eplb_execute(input.eplb_info);
  }

  auto model_output = runtime_.executor->forward(input.token_ids,
                                                 input.positions,
                                                 runtime_.worker.kv_caches_,
                                                 input.input_params);
  if (!model_output.hidden_states.defined()) {
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = runtime_.model->logits(model_output.hidden_states,
                                    sampling_params.selected_token_idxes);
  }

  ForwardOutput output;
  if (FLAGS_enable_eplb) {
    output.expert_load_data = runtime_.expert_load_data;
    output.prepared_layer_id = runtime_.eplb_executor->get_ready_layer_id();
    if (output.prepared_layer_id != -1) {
      runtime_.eplb_executor->reset_ready_layer_id();
    }
  }

  if (!runtime_.worker.driver_ && !runtime_.worker.dp_driver_ &&
      !runtime_.worker.options_.enable_speculative_decode()) {
    runtime_.stream->synchronize();
    if (runtime_.worker.options_.kv_cache_transfer_mode() == "PUSH" &&
        !input.transfer_kv_infos.empty()) {
      auto results =
          folly::collectAll(futures).within(std::chrono::seconds(60)).get();
      for (const auto& result : results) {
        if (!result.value()) {
          LOG(ERROR) << "kv_cache_transfer_ failed";
          break;
        }
      }
    }
    if (FLAGS_enable_eplb) {
      return output;
    }
    return std::nullopt;
  }

  SampleOutput sample_output;
  if (sampling_params.selected_token_idxes.defined()) {
    sample_output = runtime_.worker.sampler_->forward(logits, sampling_params);
    output.logits = logits;

    BeamSearchOutput beam_search_output;
    if (sampling_params.use_beam_search && input.acc_logprob.defined() &&
        input.acc_logprob.numel() > 0) {
      beam_search_output =
          runtime_.worker.beam_searcher_->forward(input.acc_logprob,
                                                  sample_output.top_tokens,
                                                  sample_output.top_logprobs);
    }

    output.sample_output = sample_output;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
    output.beam_search_output = beam_search_output;
  }

  if (runtime_.worker.options_.enable_speculative_decode()) {
    if (!input.input_params.batch_forward_type.is_decode() &&
        !runtime_.worker.is_spec_draft_) {
      output.sample_output.embeddings = model_output.hidden_states;
    } else if (sampling_params.selected_token_idxes.defined()) {
      torch::Tensor embeddings = model_output.hidden_states.index_select(
          /*dim=*/0, sampling_params.selected_token_idxes);
      output.sample_output.embeddings = embeddings;
    }
  }

  runtime_.stream->synchronize();

  if (runtime_.worker.options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
    auto results =
        folly::collectAll(futures).within(std::chrono::seconds(60)).get();
    for (const auto& result : results) {
      if (!result.value()) {
        LOG(ERROR) << "kv_cache_transfer_ failed";
        break;
      }
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      runtime_.worker.device_.index());

  return output;
}

void RecWorkerImpl::LlmRecWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);
  runtime_.worker.prepare_multi_modal_data(processed_inputs);
}

ForwardInput RecWorkerImpl::OneRecWorkPipeline::prepare_inputs(Batch& batch) {
  ThreadPool* thread_pool =
      runtime_.worker.input_builder_thread_pool_
          ? runtime_.worker.input_builder_thread_pool_.get()
          : nullptr;

  return batch.prepare_rec_forward_input(
      runtime_.worker.options_.num_decoding_tokens(),
      /*min_decoding_batch_size=*/0,
      runtime_.context->get_model_args(),
      thread_pool);
}

std::optional<ForwardOutput> RecWorkerImpl::OneRecWorkPipeline::step(
    const ForwardInput& input) {
  Timer timer;
  runtime_.worker.device_.set_device();

  const auto& sampling_params = input.sampling_params;
  const auto& input_params = input.input_params;

  const auto* onerec_params = input_params.onerec_params();
  CHECK(onerec_params != nullptr) << "OneRec requires rec_params.";

  const OneRecModelInputParams& rec_params = *onerec_params;

  torch::Tensor hidden_states;
  if (rec_params.rec_stage == OneRecModelInputParams::RecStage::PREFILL) {
    if (!rec_params.is_first_prefill) {
      ModelInputParams decoder_params = input_params;
      decoder_params.mutable_onerec_params().is_encoder_forward = false;
      decoder_params.mutable_onerec_params().has_encoder_output =
          rec_params.has_encoder_output;
      auto model_output = runtime_.executor->forward(input.token_ids,
                                                     input.positions,
                                                     runtime_.worker.kv_caches_,
                                                     decoder_params);
      hidden_states = model_output.hidden_states;
    } else {
      const bool has_sparse_embedding =
          rec_params.encoder_sparse_embedding.defined();
      const bool has_encoder_tokens = rec_params.encoder_token_ids.defined() &&
                                      rec_params.encoder_positions.defined();

      if (!has_sparse_embedding && !has_encoder_tokens) {
        LOG(ERROR) << "OneRec first prefill requires encoder inputs.";
        return std::nullopt;
      }

      ModelInputParams encoder_params = input_params;
      auto& mutable_onerec_params = encoder_params.mutable_onerec_params();
      mutable_onerec_params.is_encoder_forward = true;
      mutable_onerec_params.is_hybrid_mode = has_sparse_embedding;

      torch::Tensor encoder_tokens;
      if (has_sparse_embedding) {
        encoder_tokens = rec_params.encoder_sparse_embedding;
      } else {
        mutable_onerec_params.is_hybrid_mode = false;
        encoder_tokens = rec_params.encoder_token_ids;
      }

      auto encoder_output =
          runtime_.executor->forward(encoder_tokens,
                                     rec_params.encoder_positions,
                                     runtime_.worker.kv_caches_,
                                     encoder_params);

      ModelInputParams decoder_params = input_params;
      auto& decoder_onerec_params = decoder_params.mutable_onerec_params();
      decoder_onerec_params.is_encoder_forward = false;
      decoder_onerec_params.has_encoder_output =
          encoder_output.hidden_states.defined();
      if (encoder_output.hidden_states.defined() &&
          !decoder_onerec_params.decoder_context_embedding.defined()) {
        decoder_onerec_params.decoder_context_embedding =
            encoder_output.hidden_states;
      }
      auto model_output = runtime_.executor->forward(input.token_ids,
                                                     input.positions,
                                                     runtime_.worker.kv_caches_,
                                                     decoder_params);
      hidden_states = model_output.hidden_states;
    }
  } else {
    ModelInputParams decoder_params = input_params;
    decoder_params.mutable_onerec_params().is_encoder_forward = false;
    decoder_params.mutable_onerec_params().has_encoder_output =
        rec_params.has_encoder_output;
    auto model_output = runtime_.executor->forward(input.token_ids,
                                                   input.positions,
                                                   runtime_.worker.kv_caches_,
                                                   decoder_params);
    hidden_states = model_output.hidden_states;
  }

  if (!hidden_states.defined()) {
    return std::nullopt;
  }

  if (!runtime_.worker.driver_ && !runtime_.worker.dp_driver_ &&
      !runtime_.worker.options_.enable_speculative_decode()) {
    runtime_.stream->synchronize();
    COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
    DeviceMonitor::get_instance().update_active_activation_memory(
        runtime_.worker.device_.index());
    return std::nullopt;
  }

  torch::Tensor logits;
  if (sampling_params.selected_token_idxes.defined()) {
    logits = runtime_.model->logits(hidden_states,
                                    sampling_params.selected_token_idxes);
  }

  ForwardOutput output;

  if (sampling_params.selected_token_idxes.defined()) {
    auto sample_output =
        runtime_.worker.sampler_->forward(logits, sampling_params);
    output.logits = logits;
    output.sample_output = sample_output;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }

  runtime_.stream->synchronize();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  DeviceMonitor::get_instance().update_active_activation_memory(
      runtime_.worker.device_.index());

  return output;
}

void RecWorkerImpl::LlmRecWithMmDataWorkPipeline::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  RecWorkPipeline::prepare_work_before_execute(inputs, processed_inputs);

  if (!inputs.input_params.mm_data.valid()) {
    return;
  }

  torch::Tensor input_embedding;
  torch::Tensor input_tokens_tensor;
  torch::Tensor input_indices_tensor;

  const auto& mm_data = inputs.input_params.mm_data;
  const auto& processed_mm_data = processed_inputs.input_params.mm_data;

  if (auto res = processed_mm_data.get<torch::Tensor>(LLM_REC_INPUT_TOKENS)) {
    input_tokens_tensor = res.value();
  }

  if (auto res = mm_data.get<torch::Tensor>(LLM_REC_INPUT_INDICES)) {
    input_indices_tensor = res.value();
  }

  if (auto res =
          processed_mm_data.get<torch::Tensor>(LLM_REC_INPUT_EMBEDDING)) {
    input_embedding = res.value();
  }

  if (input_embedding.defined()) {
    input_embedding = input_embedding.to(runtime_.worker.dtype());
  }

  if (input_indices_tensor.defined()) {
    CHECK(input_tokens_tensor.defined())
        << "LLM_REC_INPUT_TOKENS is required when LLM_REC_INPUT_INDICES is "
           "set.";

#if defined(USE_NPU)
    layer::NpuWordEmbedding npu_word_embedding =
        runtime_.worker.get_npu_word_embedding();
    torch::Tensor input_tokens_embedding =
        npu_word_embedding(input_tokens_tensor, 0);
#else
    layer::WordEmbedding word_embedding = runtime_.worker.get_word_embedding();
    torch::Tensor input_tokens_embedding =
        word_embedding->forward(input_tokens_tensor);
#endif

    if (input_embedding.defined()) {
      torch::Tensor input_indices_cpu =
          input_indices_tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
      const auto* input_indices_ptr = input_indices_cpu.data_ptr<int64_t>();
      std::vector<int64_t> input_indices(
          input_indices_ptr, input_indices_ptr + input_indices_cpu.numel());

      processed_inputs.input_params.input_embedding =
          runtime_.worker.merge_embeddings_by_indices(
              input_tokens_embedding, input_embedding, input_indices);
    } else {
      processed_inputs.input_params.input_embedding = input_tokens_embedding;
    }
  } else if (input_embedding.defined()) {
    processed_inputs.input_params.input_embedding = input_embedding;
  }
}

}  // namespace xllm
