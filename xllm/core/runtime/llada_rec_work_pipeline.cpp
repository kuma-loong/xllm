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
#include <limits>
#include <string>

#include "common/rec_model_utils.h"
#include "framework/model/model_input_params.h"
#include "rec_worker_impl.h"

namespace xllm {

namespace {

int32_t compute_llada_effective_steps(int32_t gen_length,
                                      const LLaDARuntimeConfig& config) {
  return std::min(config.steps,
                  std::max(gen_length / std::max(config.minimal_topk, 1), 1));
}

int32_t compute_llada_total_blocks(int32_t prompt_length,
                                   int32_t gen_length,
                                   int32_t block_length) {
  return (prompt_length + gen_length + block_length - 1) / block_length;
}

torch::Tensor build_prompt_mask_in_block(const torch::TensorOptions& options,
                                         int32_t block_start_pos,
                                         int32_t prompt_length,
                                         int32_t block_length) {
  torch::Tensor prompt_mask_in_block = torch::zeros({block_length}, options);
  if (block_start_pos < prompt_length) {
    const int32_t prompt_end_in_block =
        std::min(prompt_length - block_start_pos, block_length);
    prompt_mask_in_block.slice(0, 0, prompt_end_in_block).fill_(true);
  }
  return prompt_mask_in_block;
}

torch::Tensor compute_llada_mask_transfer_index(
    const torch::Tensor& next_probs,
    const torch::Tensor& active_block_mask,
    const LLaDARuntimeConfig& config,
    const torch::TensorOptions& bool_options) {
  torch::Tensor mask_transfer_index =
      torch::zeros_like(next_probs, bool_options);
  if (!active_block_mask.any().item<bool>()) {
    return mask_transfer_index;
  }

  torch::Tensor mask_confidence = torch::where(
      active_block_mask,
      next_probs,
      torch::full_like(next_probs, -std::numeric_limits<float>::infinity()));
  torch::Tensor high_conf_mask =
      mask_confidence.gt(config.threshold).logical_and(active_block_mask);
  const int64_t num_high_confidence = high_conf_mask.sum().item<int64_t>();
  if (num_high_confidence >= config.num_to_transfer) {
    return high_conf_mask;
  }

  const int64_t num_available = active_block_mask.sum().item<int64_t>();
  if (num_available <= 0) {
    return mask_transfer_index;
  }

  torch::Tensor topk_indices = std::get<1>(mask_confidence.topk(
      std::min<int64_t>(config.num_to_transfer, num_available), -1));
  mask_transfer_index.scatter_(1, topk_indices, true);
  return mask_transfer_index.logical_and(active_block_mask);
}

torch::Tensor compute_llada_editing_transfer_index(
    const torch::Tensor& next_probs,
    const torch::Tensor& next_tokens,
    const torch::Tensor& old_block_tokens,
    const torch::Tensor& active_block_mask,
    const torch::Tensor& prompt_mask_in_block,
    const LLaDARuntimeConfig& config) {
  torch::Tensor editable_positions =
      active_block_mask.logical_not().logical_and(
          prompt_mask_in_block.unsqueeze(0).logical_not());
  torch::Tensor editing_confidence = torch::where(
      editable_positions,
      next_probs,
      torch::full_like(next_probs, -std::numeric_limits<float>::infinity()));
  return editing_confidence.gt(config.editing_threshold)
      .logical_and(editable_positions)
      .logical_and(next_tokens.ne(old_block_tokens));
}

}  // namespace

ForwardInput RecWorkerImpl::LLaDARecWorkPipeline::prepare_inputs(Batch& batch) {
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

torch::Tensor RecWorkerImpl::LLaDARecWorkPipeline::build_attention_mask(
    int32_t active_length,
    int32_t block_length) const {
  const int32_t num_active_blocks =
      (active_length + block_length - 1) / block_length;
  const torch::Device device = runtime_.worker.device();
  torch::Tensor block_mask = torch::tril(torch::ones(
      {num_active_blocks, num_active_blocks},
      torch::TensorOptions().device(device).dtype(torch::kFloat32)));
  torch::Tensor attention_mask = block_mask.repeat_interleave(block_length, 0)
                                     .repeat_interleave(block_length, 1)
                                     .slice(0, 0, active_length)
                                     .slice(1, 0, active_length)
                                     .unsqueeze(0)
                                     .unsqueeze(0);
  return (torch::ones_like(attention_mask) - attention_mask) * -1e9;
}

torch::Tensor RecWorkerImpl::LLaDARecWorkPipeline::initialize_tokens(
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

int64_t RecWorkerImpl::LLaDARecWorkPipeline::finalize_answer_length(
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

RawForwardOutput RecWorkerImpl::LLaDARecWorkPipeline::build_raw_output(
    const torch::Tensor& generated_tokens,
    int64_t answer_length) const {
  RawForwardOutput raw_output;
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

torch::Tensor RecWorkerImpl::LLaDARecWorkPipeline::sample_next_tokens(
    const torch::Tensor& logits,
    const SamplingParameters& sampling_params) {
  torch::Tensor filtered_logits = logits;
  float temperature = 0.0f;
  if (sampling_params.temperatures.defined() &&
      sampling_params.temperatures.numel() > 0) {
    temperature = sampling_params.temperatures[0].item<float>();
  }
  if (temperature > 0.0f) {
    filtered_logits = filtered_logits / temperature;
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

  bool do_sample = sampling_params.do_sample.defined() &&
                   sampling_params.do_sample.numel() > 0 &&
                   sampling_params.do_sample[0].item<bool>();
  if (!do_sample && temperature <= 0.0f && top_k <= 0 && top_p >= 1.0f) {
    return std::get<1>(filtered_logits.max(-1, true));
  }

  auto probs = torch::softmax(filtered_logits, -1);
  return torch::multinomial(probs.view({-1, probs.size(-1)}), 1)
      .view({logits.size(0), logits.size(1)});
}

std::optional<ForwardOutput> RecWorkerImpl::LLaDARecWorkPipeline::step(
    const ForwardInput& input) {
  runtime_.worker.device_.set_device();
  const auto* llada_params = input.input_params.llada_params();
  CHECK(llada_params != nullptr) << "LLaDA worker requires llada_params.";

  const int32_t prompt_length = llada_params->prompt_length;
  const int32_t gen_length = llada_params->max_generated_tokens;
  CHECK_GT(prompt_length, 0);
  CHECK_GT(gen_length, 0);
  LLaDARuntimeConfig llada_runtime_config;
  std::string config_error_message;
  CHECK(load_llada_runtime_config(
      runtime_.context->get_model_args().eos_token_id(),
      &llada_runtime_config,
      &config_error_message))
      << "LLaDA runtime_config_validation failed: " << config_error_message;

  auto device = runtime_.worker.device();
  auto token_options =
      torch::TensorOptions().device(device).dtype(torch::kInt32);
  auto bool_options = torch::TensorOptions().device(device).dtype(torch::kBool);

  const int32_t effective_steps =
      compute_llada_effective_steps(gen_length, llada_runtime_config);
  const int32_t block_length = llada_runtime_config.block_length;
  const int32_t num_blocks =
      compute_llada_total_blocks(prompt_length, gen_length, block_length);
  const int32_t total_length = num_blocks * block_length;

  torch::Tensor x = initialize_tokens(
      input, prompt_length, total_length, llada_runtime_config.mask_id);
  torch::Tensor position_ids =
      torch::arange(total_length, token_options).view({1, total_length});

  const int32_t prefill_blocks = prompt_length / block_length;
  bool eos_early_stopped = false;
  LOG(INFO) << "LLaDA worker starts generation"
            << ", model_type="
            << runtime_.context->get_model_args().model_type() << ", rec_type="
            << rec_model_kind_to_string(runtime_.worker.rec_model_kind_)
            << ", pipeline_type="
            << rec_pipeline_type_to_string(
                   get_rec_pipeline_type(runtime_.worker.rec_model_kind_))
            << ", rank=" << runtime_.context->get_parallel_args().rank()
            << ", device=" << runtime_.worker.device().index() << ", tp_size="
            << runtime_.context->get_parallel_args().world_size()
            << ", prompt_tokens=" << prompt_length
            << ", max_tokens=" << gen_length << ", total_blocks=" << num_blocks
            << ", " << llada_runtime_config_to_string(llada_runtime_config);

  for (int32_t num_block = prefill_blocks; num_block < num_blocks;
       ++num_block) {
    const int32_t current_window_end = (num_block + 1) * block_length;
    torch::Tensor cur_x = x.slice(1, 0, current_window_end).clone();
    torch::Tensor cur_positions =
        position_ids.slice(1, 0, current_window_end).reshape({-1});
    torch::Tensor cur_attn_mask =
        build_attention_mask(current_window_end, block_length);
    const int32_t block_start_pos = num_block * block_length;

    int32_t post_steps = 0;
    int32_t refine_steps = 0;
    while (true) {
      ++refine_steps;
      torch::Tensor old_block_tokens =
          cur_x.slice(1, current_window_end - block_length, current_window_end)
              .clone();
      torch::Tensor active_block_mask =
          old_block_tokens.eq(llada_runtime_config.mask_id);
      if (!active_block_mask.any().item<bool>()) {
        ++post_steps;
      }
      if (post_steps > llada_runtime_config.max_post_steps) {
        break;
      }
      if (refine_steps >
          effective_steps + llada_runtime_config.max_post_steps) {
        LOG(WARNING) << "LLaDA block_refine_step_cap"
                     << ", rank="
                     << runtime_.context->get_parallel_args().rank()
                     << ", block=" << num_block
                     << ", refine_steps=" << refine_steps;
        break;
      }

      torch::Tensor prompt_mask_in_block = build_prompt_mask_in_block(
          bool_options, block_start_pos, prompt_length, block_length);

      ModelInputParams model_input_params;
      model_input_params.graph_buffer.attn_mask = cur_attn_mask;
      auto model_output = runtime_.model->forward(cur_x.reshape({-1}),
                                                  cur_positions,
                                                  runtime_.worker.kv_caches_,
                                                  model_input_params);
      torch::Tensor logits =
          runtime_.model->logits(model_output.hidden_states, torch::Tensor());
      torch::Tensor active_logits =
          logits.view({1, current_window_end, -1})
              .slice(1, current_window_end - block_length, current_window_end);
      torch::Tensor next_tokens =
          sample_next_tokens(active_logits, input.sampling_params)
              .to(token_options);
      torch::Tensor next_probs =
          torch::softmax(active_logits, -1)
              .gather(-1, next_tokens.to(torch::kInt64).unsqueeze(-1))
              .squeeze(-1);

      torch::Tensor mask_transfer_index = compute_llada_mask_transfer_index(
          next_probs, active_block_mask, llada_runtime_config, bool_options);
      torch::Tensor editing_transfer_index =
          compute_llada_editing_transfer_index(next_probs,
                                               next_tokens,
                                               old_block_tokens,
                                               active_block_mask,
                                               prompt_mask_in_block,
                                               llada_runtime_config);
      torch::Tensor final_transfer_index =
          mask_transfer_index.logical_or(editing_transfer_index);

      if (final_transfer_index.any().item<bool>()) {
        auto block_slice = cur_x.slice(
            1, current_window_end - block_length, current_window_end);
        block_slice.masked_scatter_(
            final_transfer_index,
            next_tokens.masked_select(final_transfer_index));
      }

      if (!active_block_mask.any().item<bool>() &&
          !editing_transfer_index.any().item<bool>()) {
        break;
      }
    }

    x.slice(1, 0, current_window_end).copy_(cur_x);
    if (llada_runtime_config.eos_early_stop) {
      torch::Tensor generated_part =
          x.slice(1, prompt_length, current_window_end);
      if (!generated_part.eq(llada_runtime_config.mask_id).any().item<bool>()) {
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
                             llada_runtime_config.mask_id);
  generated_tokens = generated_tokens.slice(1, 0, answer_length);
  RawForwardOutput raw_output =
      build_raw_output(generated_tokens, answer_length);

  LOG(INFO) << "LLaDA worker finished generation"
            << ", rank=" << runtime_.context->get_parallel_args().rank()
            << ", device=" << runtime_.worker.device().index()
            << ", prompt_tokens=" << prompt_length
            << ", max_tokens=" << gen_length
            << ", block_length=" << block_length
            << ", steps=" << effective_steps << ", total_blocks=" << num_blocks
            << ", generated_tokens=" << answer_length
            << ", eos_early_stop=" << eos_early_stopped;

  ForwardOutput output;
  output.raw_output = std::move(raw_output);
  return output;
}

}  // namespace xllm
