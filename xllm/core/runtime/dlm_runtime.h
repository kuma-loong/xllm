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

#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/rec_model_utils.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"

namespace xllm {

struct DlmBlockRange {
  int32_t start = 0;
  int32_t end = 0;

  int32_t length() const { return std::max(end - start, 0); }
};

enum class DlmForwardMode : uint8_t {
  kPrefill = 0,
  kDecode = 1,
};

struct DlmForwardBatch {
  DlmForwardMode forward_mode = DlmForwardMode::kPrefill;
  DlmModelInputParams::ReqPhase req_phase =
      DlmModelInputParams::ReqPhase::kIncomingPrefill;
  torch::Tensor tokens;
  torch::Tensor positions;
  torch::Tensor attention_mask;
  int32_t block_offset = 0;
  int32_t block_length = 0;
  DlmBlockRange cache_write_range;
  int32_t committed_prefix_length = 0;
  int32_t active_cache_length = 0;
  bool use_cache = false;
  bool update_cache = false;
};

inline ModelInputParams make_dlm_model_input_params(
    const DlmForwardBatch& forward_batch) {
  ModelInputParams input_params;
  input_params.batch_forward_type =
      forward_batch.forward_mode == DlmForwardMode::kPrefill
          ? BatchForwardType::PREFILL
          : BatchForwardType::DECODE;
  input_params.graph_buffer.attn_mask = forward_batch.attention_mask;
  auto& dlm_params = input_params.mutable_dlm_params();
  dlm_params.committed_prefix_length = forward_batch.committed_prefix_length;
  dlm_params.active_cache_length = forward_batch.active_cache_length;
  dlm_params.block_offset = forward_batch.block_offset;
  dlm_params.block_length = forward_batch.block_length;
  dlm_params.req_phase = forward_batch.req_phase;
  dlm_params.cache_write_start = forward_batch.cache_write_range.start;
  dlm_params.cache_write_end = forward_batch.cache_write_range.end;
  dlm_params.use_history_cache = forward_batch.use_cache;
  dlm_params.update_history_cache = forward_batch.update_cache;
  return input_params;
}

inline torch::Tensor build_dlm_block_attention_mask(const torch::Device& device,
                                                    int32_t active_length,
                                                    int32_t block_length) {
  const int32_t num_active_blocks =
      (active_length + block_length - 1) / block_length;
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

inline torch::Tensor build_dlm_visible_attention_mask(
    const torch::Device& device,
    int32_t query_length,
    int32_t key_length) {
  return torch::zeros(
      {1, 1, query_length, key_length},
      torch::TensorOptions().device(device).dtype(torch::kFloat32));
}

class DlmPrefixCacheManager {
 public:
  DlmPrefixCacheManager() = default;

  explicit DlmPrefixCacheManager(int32_t num_layers) { reset(num_layers); }

  void reset(int32_t num_layers) {
    kv_caches_.assign(static_cast<size_t>(std::max(num_layers, 0)), KVCache());
    committed_prefix_length_ = 0;
  }

  int32_t committed_prefix_length() const { return committed_prefix_length_; }

  bool has_committed_prefix() const { return committed_prefix_length_ > 0; }

  std::vector<KVCache>& mutable_caches() { return kv_caches_; }

  const std::vector<KVCache>& caches() const { return kv_caches_; }

  int32_t active_length_for(const DlmBlockRange& cache_write_range) const {
    return std::max(committed_prefix_length_, cache_write_range.end);
  }

  bool can_use_prefix_for(const DlmBlockRange& cache_write_range) const {
    return cache_write_range.start <= committed_prefix_length_;
  }

  void mark_prefix_committed(const DlmBlockRange& cache_write_range) {
    CHECK_LE(cache_write_range.start, committed_prefix_length_)
        << "DLM prefix cache requires monotonic committed ranges";
    committed_prefix_length_ =
        std::max(committed_prefix_length_, cache_write_range.end);
  }

 private:
  std::vector<KVCache> kv_caches_;
  int32_t committed_prefix_length_ = 0;
};

struct DlmDecodeStepState {
  bool has_mask = false;
  int32_t post_edit_step = 0;
};

struct DlmTransferPlan {
  torch::Tensor transfer_index;
  bool finished = false;
};

class DlmDecodeAlgorithm {
 public:
  explicit DlmDecodeAlgorithm(const LLaDARuntimeConfig& config)
      : config_(config) {}

  const LLaDARuntimeConfig& config() const { return config_; }

  int32_t compute_effective_steps(int32_t gen_length) const {
    if (config_.algorithm == LLaDAAlgorithmType::kLowConfidence) {
      return std::min(
          config_.steps,
          std::max(gen_length / std::max(config_.minimal_topk, 1), 1));
    }
    return config_.block_length + config_.max_post_steps;
  }

  int32_t compute_total_blocks(int32_t prompt_length,
                               int32_t gen_length) const {
    return (prompt_length + gen_length + config_.block_length - 1) /
           config_.block_length;
  }

  torch::Tensor build_prompt_mask_in_block(
      const torch::Tensor& initial_block_tokens) const {
    return initial_block_tokens.eq(config_.mask_id).logical_not();
  }

  void apply_penalty(torch::Tensor& logits,
                     const torch::Tensor& current_tokens) const {
    if (config_.algorithm != LLaDAAlgorithmType::kJointThreshold ||
        config_.penalty_lambda <= 0.0 || current_tokens.size(0) <= 1) {
      return;
    }
    auto prev_ids =
        current_tokens.slice(/*dim=*/0, 0, current_tokens.size(0) - 1)
            .to(torch::kLong)
            .unsqueeze(-1);
    logits.slice(/*dim=*/0, 1, logits.size(0))
        .scatter_add_(
            /*dim=*/1,
            prev_ids,
            torch::full({prev_ids.size(0), 1},
                        -config_.penalty_lambda,
                        logits.options()));
  }

  torch::Tensor select_mask_transfer_index(
      const torch::Tensor& next_probs,
      const torch::Tensor& next_tokens,
      const torch::Tensor& active_block_mask,
      const torch::TensorOptions& bool_options) const {
    torch::Tensor transfer_index = torch::zeros_like(next_probs, bool_options);
    if (!active_block_mask.any().item<bool>()) {
      return transfer_index;
    }

    torch::Tensor valid_mask =
        active_block_mask.logical_and(next_tokens.ne(config_.mask_id));
    torch::Tensor confidence = torch::where(
        valid_mask,
        next_probs,
        torch::full_like(next_probs, -std::numeric_limits<float>::infinity()));
    if (!valid_mask.any().item<bool>()) {
      return transfer_index;
    }

    torch::Tensor high_conf_mask;
    if (config_.algorithm == LLaDAAlgorithmType::kJointThreshold) {
      high_conf_mask = confidence.gt(config_.threshold).logical_and(valid_mask);
    } else {
      torch::Tensor actual_threshold =
          std::get<0>(confidence.max(/*dim=*/1, /*keepdim=*/true))
              .sub(1e-5)
              .clamp(-1000.0, config_.threshold);
      high_conf_mask = confidence.ge(actual_threshold).logical_and(valid_mask);
    }
    if (high_conf_mask.any().item<bool>()) {
      return high_conf_mask;
    }

    const int64_t num_available = valid_mask.sum().item<int64_t>();
    if (num_available <= 0) {
      return transfer_index;
    }

    const int64_t fallback_topk = 1;
    auto topk = std::get<1>(
        confidence.topk(std::min<int64_t>(fallback_topk, num_available), -1));
    transfer_index.scatter_(1, topk, true);
    return transfer_index.logical_and(valid_mask);
  }

  torch::Tensor select_edit_transfer_index(
      const torch::Tensor& next_probs,
      const torch::Tensor& next_tokens,
      const torch::Tensor& old_block_tokens,
      const torch::Tensor& active_block_mask,
      const torch::Tensor& prompt_mask_in_block,
      const torch::TensorOptions& bool_options) const {
    torch::Tensor editable_positions =
        active_block_mask.logical_not().logical_and(
            prompt_mask_in_block.unsqueeze(0).logical_not());
    if (!editable_positions.any().item<bool>()) {
      return torch::zeros_like(next_probs, bool_options);
    }
    torch::Tensor editing_confidence = torch::where(
        editable_positions,
        next_probs,
        torch::full_like(next_probs, -std::numeric_limits<float>::infinity()));
    return editing_confidence.gt(config_.editing_threshold)
        .logical_and(editable_positions)
        .logical_and(next_tokens.ne(config_.mask_id))
        .logical_and(next_tokens.ne(old_block_tokens));
  }

  DlmTransferPlan build_transfer_plan(
      const torch::Tensor& next_probs,
      const torch::Tensor& next_tokens,
      const torch::Tensor& old_block_tokens,
      const torch::Tensor& prompt_mask_in_block,
      const DlmDecodeStepState& state,
      const torch::TensorOptions& bool_options) const {
    DlmTransferPlan plan;
    torch::Tensor active_block_mask = old_block_tokens.eq(config_.mask_id);
    torch::Tensor mask_transfer_index = select_mask_transfer_index(
        next_probs, next_tokens, active_block_mask, bool_options);
    torch::Tensor edit_transfer_index;
    if (config_.algorithm == LLaDAAlgorithmType::kJointThreshold &&
        !state.has_mask) {
      edit_transfer_index = select_edit_transfer_index(next_probs,
                                                       next_tokens,
                                                       old_block_tokens,
                                                       active_block_mask,
                                                       prompt_mask_in_block,
                                                       bool_options);
    } else {
      edit_transfer_index = torch::zeros_like(next_probs, bool_options);
    }
    plan.transfer_index = mask_transfer_index.logical_or(edit_transfer_index);
    const bool changed = plan.transfer_index.any().item<bool>();
    plan.finished =
        (!state.has_mask &&
         config_.algorithm == LLaDAAlgorithmType::kLowConfidence) ||
        (!changed && config_.algorithm == LLaDAAlgorithmType::kJointThreshold);
    return plan;
  }

  bool should_continue(const DlmDecodeStepState& state,
                       int32_t refine_step) const {
    if (config_.algorithm == LLaDAAlgorithmType::kLowConfidence) {
      return refine_step <= config_.steps;
    }
    return state.post_edit_step <= config_.max_post_steps &&
           refine_step <= config_.block_length + config_.max_post_steps;
  }

 private:
  LLaDARuntimeConfig config_;
};

}  // namespace xllm
