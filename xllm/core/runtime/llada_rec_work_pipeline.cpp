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

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "common/rec_model_utils.h"
#include "framework/model/model_input_params.h"
#include "rec_worker_impl.h"
#include "runtime/dlm_runtime.h"

namespace xllm {

namespace {

torch::ScalarType get_dlm_probability_dtype() {
#if defined(USE_NPU)
  return torch::kFloat32;
#else
  return torch::kFloat64;
#endif
}

torch::Tensor add_gumbel_noise(const torch::Tensor& logits, float temperature) {
  const auto prob_dtype = get_dlm_probability_dtype();
  auto uniform = torch::rand_like(logits, logits.options().dtype(prob_dtype))
                     .clamp_(1e-12, 1.0 - 1e-12);
  auto gumbel = -torch::log(-torch::log(uniform));
  return logits.to(prob_dtype) / static_cast<double>(temperature) + gumbel;
}

}  // namespace

ForwardInput RecWorkerImpl::DlmDecodePipeline::prepare_inputs(Batch& batch) {
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

torch::Tensor RecWorkerImpl::DlmDecodePipeline::initialize_tokens(
    const ForwardInput& input,
    int32_t prompt_length,
    int32_t total_length,
    int32_t mask_token_id) const {
  const torch::Device device = runtime_.worker.device();
  const auto token_options =
      torch::TensorOptions().device(device).dtype(torch::kInt32);
  torch::Tensor prompt = input.token_ids.view({1, -1}).to(token_options);
  torch::Tensor tokens =
      torch::full({1, total_length}, mask_token_id, token_options);
  tokens.slice(/*dim=*/1, 0, prompt_length).copy_(prompt);
  return tokens;
}

int64_t RecWorkerImpl::DlmDecodePipeline::finalize_answer_length(
    const torch::Tensor& generated_tokens,
    int32_t eos_token_id,
    int32_t mask_token_id) const {
  int64_t answer_length = generated_tokens.size(1);
  torch::Tensor eos_positions = generated_tokens.eq(eos_token_id).nonzero();
  if (eos_positions.numel() > 0) {
    answer_length = eos_positions[0][1].item<int64_t>() + 1;
  }

  torch::Tensor truncated_tokens =
      generated_tokens.slice(/*dim=*/1, 0, answer_length);
  torch::Tensor unresolved_positions =
      truncated_tokens.eq(mask_token_id).nonzero();
  if (unresolved_positions.numel() == 0) {
    return answer_length;
  }

  const int64_t first_mask_index = unresolved_positions[0][1].item<int64_t>();
  LOG(ERROR) << "LLaDA mask_finalize failed with unresolved mask tokens"
             << ", rank=" << runtime_.context->get_parallel_args().rank()
             << ", generated_tokens=" << generated_tokens.size(1)
             << ", first_unresolved_index=" << first_mask_index;
  return std::max<int64_t>(first_mask_index, 0);
}

RawForwardOutput RecWorkerImpl::DlmDecodePipeline::build_raw_output(
    const torch::Tensor& generated_tokens,
    int64_t answer_length) const {
  RawForwardOutput raw_output;
  raw_output.final_sequence_output = true;
  RawSampleOutput sample;
  sample.tokens.reserve(answer_length);

  const auto accessor = generated_tokens.accessor<int32_t, 2>();
  for (int64_t i = 0; i < answer_length; ++i) {
    RawToken token;
    token.id = accessor[0][i];
    sample.tokens.push_back(std::move(token));
  }

  raw_output.outputs.push_back(std::move(sample));
  return raw_output;
}

std::pair<torch::Tensor, torch::Tensor>
RecWorkerImpl::DlmDecodePipeline::sample_next_tokens(
    const torch::Tensor& logits,
    const SamplingParameters& sampling_params) {
  torch::Tensor filtered_logits = logits;
  float temperature = 0.0f;
  if (sampling_params.temperatures.defined() &&
      sampling_params.temperatures.numel() > 0) {
    temperature = sampling_params.temperatures[0].item<float>();
  }
  if (temperature <= 0.0f) {
    torch::Tensor next_tokens =
        std::get<1>(filtered_logits.max(-1, true)).squeeze(-1);
    const auto prob_dtype = get_dlm_probability_dtype();
    torch::Tensor next_probs =
        torch::softmax(filtered_logits.to(prob_dtype), -1)
            .gather(-1, next_tokens.to(torch::kInt64).unsqueeze(-1))
            .squeeze(-1);
    return {std::move(next_tokens), std::move(next_probs)};
  }

  int64_t top_k = -1;
  if (sampling_params.top_k.defined() && sampling_params.top_k.numel() > 0) {
    top_k = sampling_params.top_k[0].item<int64_t>();
  }
  if (top_k > 0 && top_k < filtered_logits.size(-1)) {
    auto topk = std::get<0>(filtered_logits.topk(top_k, -1));
    auto kth = topk.select(-1, top_k - 1).unsqueeze(-1);
    filtered_logits = filtered_logits.masked_fill(
        filtered_logits < kth, -std::numeric_limits<float>::infinity());
  }

  float top_p = 1.0f;
  if (sampling_params.top_p.defined() && sampling_params.top_p.numel() > 0) {
    top_p = sampling_params.top_p[0].item<float>();
  }
  if (top_p < 1.0f) {
    auto sorted = torch::sort(filtered_logits, -1, /*descending=*/true);
    auto sorted_logits = std::get<0>(sorted);
    auto sorted_indices = std::get<1>(sorted);
    auto probs = torch::softmax(sorted_logits, -1);
    auto cumulative_probs = torch::cumsum(probs, -1);
    auto sorted_mask = cumulative_probs > top_p;
    if (sorted_mask.size(-1) > 1) {
      auto shifted_mask = torch::zeros_like(sorted_mask);
      shifted_mask.slice(-1, 1, sorted_mask.size(-1))
          .copy_(sorted_mask.slice(-1, 0, sorted_mask.size(-1) - 1));
      sorted_mask = shifted_mask;
    }
    auto remove_mask = torch::zeros_like(sorted_mask);
    remove_mask.scatter_(-1, sorted_indices, sorted_mask);
    filtered_logits = filtered_logits.masked_fill(
        remove_mask, -std::numeric_limits<float>::infinity());
  }

  torch::Tensor sample_logits = add_gumbel_noise(filtered_logits, temperature);
  torch::Tensor next_tokens =
      std::get<1>(sample_logits.max(-1, true)).squeeze(-1);
  const auto prob_dtype = get_dlm_probability_dtype();
  torch::Tensor probs = torch::softmax(filtered_logits.to(prob_dtype), -1);
  torch::Tensor next_probs =
      probs.gather(-1, next_tokens.to(torch::kInt64).unsqueeze(-1)).squeeze(-1);
  return {std::move(next_tokens), std::move(next_probs)};
}

std::optional<ForwardOutput> RecWorkerImpl::DlmDecodePipeline::step(
    const ForwardInput& input) {
  runtime_.worker.device_.set_device();
  const auto* dlm_params = input.input_params.dlm_params();
  CHECK(dlm_params != nullptr) << "DLM worker requires dlm_params.";

  const int32_t prompt_length = dlm_params->prompt_length;
  const int32_t gen_length = dlm_params->max_generated_tokens;
  CHECK_GT(prompt_length, 0);
  CHECK_GT(gen_length, 0);
  LLaDARuntimeConfig llada_runtime_config;
  std::string config_error_message;
  CHECK(load_llada_runtime_config(
      runtime_.context->get_model_args().eos_token_id(),
      &llada_runtime_config,
      &config_error_message))
      << "LLaDA runtime_config_validation failed: " << config_error_message;
  DlmDecodeAlgorithm decode_algorithm(llada_runtime_config);

  auto device = runtime_.worker.device();
  auto token_options =
      torch::TensorOptions().device(device).dtype(torch::kInt32);
  auto bool_options = torch::TensorOptions().device(device).dtype(torch::kBool);

  const int32_t effective_steps =
      decode_algorithm.compute_effective_steps(gen_length);
  const int32_t block_length = decode_algorithm.config().block_length;
  const int32_t num_blocks =
      decode_algorithm.compute_total_blocks(prompt_length, gen_length);
  const int32_t total_length = num_blocks * block_length;
  const bool enable_prefix_cache =
      decode_algorithm.config().cache_mode == DlmCacheMode::kPrefix;
  const int32_t num_layers = runtime_.context->get_model_args().n_layers();

  torch::Tensor x = initialize_tokens(
      input, prompt_length, total_length, decode_algorithm.config().mask_id);
  torch::Tensor position_ids =
      torch::arange(total_length, token_options).view({1, total_length});

  const int32_t prefill_blocks = prompt_length / block_length;
  DlmPrefixCacheManager prefix_cache_manager(num_layers);
  std::vector<KVCache> no_cache_kv_caches(static_cast<size_t>(num_layers));
  bool eos_early_stopped = false;
  int32_t total_refine_steps = 0;
  int32_t total_cache_updates = 0;
  LOG(INFO) << "DLM worker starts generation"
            << ", model_type="
            << runtime_.context->get_model_args().model_type() << ", rec_type="
            << rec_model_kind_to_string(runtime_.worker.rec_model_kind_)
            << ", family="
            << rec_model_family_to_string(
                   get_rec_model_family(runtime_.worker.rec_model_kind_))
            << ", pipeline_type="
            << rec_pipeline_type_to_string(
                   get_rec_pipeline_type(runtime_.worker.rec_model_kind_))
            << ", rank=" << runtime_.context->get_parallel_args().rank()
            << ", device=" << static_cast<int>(runtime_.worker.device().index())
            << ", tp_size="
            << runtime_.context->get_parallel_args().world_size()
            << ", prompt_tokens=" << prompt_length
            << ", max_tokens=" << gen_length << ", total_blocks=" << num_blocks
            << ", "
            << dlm_runtime_config_to_string(
                   get_dlm_runtime_config(runtime_.worker.rec_model_kind_))
            << ", " << llada_runtime_config_to_string(decode_algorithm.config())
            << ", worker_cache_path="
            << (enable_prefix_cache ? "local_prefix_cache" : "disabled");

  auto run_cached_block_forward = [&](const torch::Tensor& block_tokens,
                                      const torch::Tensor& block_positions,
                                      const DlmBlockRange& cache_write_range,
                                      bool update_cache) -> torch::Tensor {
    const int32_t active_cache_length =
        prefix_cache_manager.active_length_for(cache_write_range);
    DlmForwardBatch forward_batch;
    forward_batch.forward_mode = DlmForwardMode::kDecode;
    forward_batch.tokens = block_tokens;
    forward_batch.positions = block_positions;
    forward_batch.attention_mask = build_dlm_visible_attention_mask(
        device, block_tokens.size(1), active_cache_length);
    forward_batch.cache_write_range = cache_write_range;
    forward_batch.committed_prefix_length =
        prefix_cache_manager.committed_prefix_length();
    forward_batch.active_cache_length = active_cache_length;
    forward_batch.use_cache =
        prefix_cache_manager.can_use_prefix_for(cache_write_range);
    forward_batch.update_cache = update_cache;
    ModelInputParams model_input_params =
        make_dlm_model_input_params(forward_batch);
    auto model_output =
        runtime_.model->forward(block_tokens.reshape({-1}),
                                block_positions.reshape({-1}),
                                prefix_cache_manager.mutable_caches(),
                                model_input_params);
    if (update_cache) {
      prefix_cache_manager.mark_prefix_committed(cache_write_range);
      ++total_cache_updates;
    }
    return runtime_.model->logits(model_output.hidden_states, torch::Tensor())
        .view({1, block_tokens.size(1), -1});
  };

  auto run_full_window_forward =
      [&](const torch::Tensor& window_tokens,
          const torch::Tensor& window_positions,
          const torch::Tensor& attention_mask) -> torch::Tensor {
    DlmForwardBatch forward_batch;
    forward_batch.forward_mode = DlmForwardMode::kPrefill;
    forward_batch.tokens = window_tokens;
    forward_batch.positions = window_positions;
    forward_batch.attention_mask = attention_mask;
    ModelInputParams model_input_params =
        make_dlm_model_input_params(forward_batch);
    auto model_output = runtime_.model->forward(window_tokens.reshape({-1}),
                                                window_positions.reshape({-1}),
                                                no_cache_kv_caches,
                                                model_input_params);
    return runtime_.model->logits(model_output.hidden_states, torch::Tensor())
        .view({1, window_tokens.size(1), -1});
  };

  if (enable_prefix_cache) {
    for (int32_t block_id = 0; block_id < prefill_blocks; ++block_id) {
      const DlmBlockRange prompt_block_range = {block_id * block_length,
                                                (block_id + 1) * block_length};
      torch::Tensor prompt_block =
          x.slice(1, prompt_block_range.start, prompt_block_range.end).clone();
      torch::Tensor prompt_positions = position_ids.slice(
          1, prompt_block_range.start, prompt_block_range.end);
      (void)run_cached_block_forward(prompt_block,
                                     prompt_positions,
                                     prompt_block_range,
                                     /*update_history_cache=*/true);
    }
  }

  for (int32_t num_block = prefill_blocks; num_block < num_blocks;
       ++num_block) {
    const DlmBlockRange block_range = {num_block * block_length,
                                       (num_block + 1) * block_length};
    torch::Tensor cur_tokens =
        (enable_prefix_cache ? x.slice(1, block_range.start, block_range.end)
                             : x.slice(1, 0, block_range.end))
            .clone();
    torch::Tensor cur_positions =
        (enable_prefix_cache
             ? position_ids.slice(1, block_range.start, block_range.end)
             : position_ids.slice(1, 0, block_range.end));
    torch::Tensor full_window_attn_mask =
        enable_prefix_cache ? torch::Tensor()
                            : build_dlm_block_attention_mask(
                                  device, cur_tokens.size(1), block_length);

    int32_t refine_steps = 0;
    int32_t post_edit_steps = 0;
    bool cache_matches_current_tokens = false;
    while (true) {
      ++refine_steps;
      ++total_refine_steps;
      torch::Tensor block_tokens_view =
          enable_prefix_cache
              ? cur_tokens
              : cur_tokens.slice(1, block_range.start, block_range.end);
      torch::Tensor old_block_tokens = block_tokens_view.clone();
      torch::Tensor active_block_mask =
          old_block_tokens.eq(decode_algorithm.config().mask_id);
      DlmDecodeStepState step_state;
      step_state.has_mask = active_block_mask.any().item<bool>();
      if (!step_state.has_mask) {
        ++post_edit_steps;
      }
      step_state.post_edit_step = post_edit_steps;
      if (!decode_algorithm.should_continue(step_state, refine_steps)) {
        LOG(WARNING) << "DLM block_refine_step_cap"
                     << ", rank="
                     << runtime_.context->get_parallel_args().rank()
                     << ", block=" << num_block
                     << ", refine_steps=" << refine_steps;
        break;
      }

      torch::Tensor prompt_mask_in_block =
          decode_algorithm.build_prompt_mask_in_block(
              bool_options, block_range, prompt_length);

      torch::Tensor active_logits =
          enable_prefix_cache
              ? run_cached_block_forward(cur_tokens,
                                         cur_positions,
                                         block_range,
                                         /*update_cache=*/false)
              : run_full_window_forward(
                    cur_tokens, cur_positions, full_window_attn_mask)
                    .slice(1, block_range.start, block_range.end);
      cache_matches_current_tokens = true;
      if (decode_algorithm.config().algorithm ==
          LLaDAAlgorithmType::kJointThreshold) {
        auto logits_view = active_logits.squeeze(0);
        decode_algorithm.apply_penalty(logits_view,
                                       old_block_tokens.squeeze(0));
        active_logits = logits_view.unsqueeze(0);
      }
      std::pair<torch::Tensor, torch::Tensor> sampling_result =
          sample_next_tokens(active_logits, input.sampling_params);
      torch::Tensor next_tokens = sampling_result.first.to(token_options);
      torch::Tensor next_probs = sampling_result.second;

      DlmTransferPlan transfer_plan =
          decode_algorithm.build_transfer_plan(next_probs,
                                               next_tokens,
                                               old_block_tokens,
                                               prompt_mask_in_block,
                                               step_state,
                                               bool_options);

      if (transfer_plan.transfer_index.any().item<bool>()) {
        block_tokens_view.masked_scatter_(
            transfer_plan.transfer_index,
            next_tokens.masked_select(transfer_plan.transfer_index));
        cache_matches_current_tokens = false;
      }

      if (transfer_plan.finished) {
        break;
      }
    }

    if (enable_prefix_cache) {
      x.slice(1, block_range.start, block_range.end).copy_(cur_tokens);
      if (!cache_matches_current_tokens) {
        (void)run_cached_block_forward(cur_tokens,
                                       cur_positions,
                                       block_range,
                                       /*update_cache=*/true);
      } else if (prefix_cache_manager.committed_prefix_length() <
                 block_range.end) {
        prefix_cache_manager.mark_prefix_committed(block_range);
        ++total_cache_updates;
      }
    } else {
      x.slice(1, 0, block_range.end).copy_(cur_tokens);
    }
    if (decode_algorithm.config().eos_early_stop) {
      torch::Tensor generated_part = x.slice(1, prompt_length, block_range.end);
      if (!generated_part.eq(decode_algorithm.config().mask_id)
               .any()
               .item<bool>()) {
        torch::Tensor eos_hits = generated_part.eq(
            runtime_.context->get_model_args().eos_token_id());
        if (eos_hits.any().item<bool>()) {
          eos_early_stopped = true;
          break;
        }
      }
    }
  }

  torch::Tensor generated_answer = x.slice(1, 0, prompt_length + gen_length);
  torch::Tensor generated_tokens =
      generated_answer.slice(1, prompt_length, prompt_length + gen_length);
  generated_tokens = generated_tokens.to(torch::kCPU);
  const int64_t answer_length =
      finalize_answer_length(generated_tokens,
                             runtime_.context->get_model_args().eos_token_id(),
                             decode_algorithm.config().mask_id);
  generated_tokens = generated_tokens.slice(1, 0, answer_length);
  RawForwardOutput raw_output =
      build_raw_output(generated_tokens, answer_length);

  LOG(INFO) << "DLM worker finished generation"
            << ", rank=" << runtime_.context->get_parallel_args().rank()
            << ", device=" << static_cast<int>(runtime_.worker.device().index())
            << ", prompt_tokens=" << prompt_length
            << ", max_tokens=" << gen_length
            << ", block_length=" << block_length
            << ", steps=" << effective_steps << ", total_blocks=" << num_blocks
            << ", total_refine_steps=" << total_refine_steps
            << ", cache_updates=" << total_cache_updates
            << ", generated_tokens=" << answer_length
            << ", eos_early_stop=" << eos_early_stopped;

  ForwardOutput output;
  output.raw_output = std::move(raw_output);
  return output;
}

}  // namespace xllm
