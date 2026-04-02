/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

#include "core/common/global_flags.h"

namespace xllm {

enum class RecModelKind : int8_t {
  kNone = 0,
  kOneRec = 1,
  kLlmRec = 2,
  kLLaDARec = 3,
};

// Pipeline strategy types (extensible for future strategies)
enum class RecPipelineType : uint8_t {
  kLlmRecDefault = 0,             // LlmRec without mm_data (pure qwen)
  kLlmRecWithMmData = 1,          // LlmRec with mm_data (qwen + embedding)
  kOneRecDefault = 2,             // OneRec
  kLlmRecMultiRoundPipeline = 3,  // LlmRec multi-round pipeline (device loop)
  kLLaDARecWorkerLoop = 4,        // LLaDA worker-driven generation loop
};

// Check if Rec multi-round mode is enabled.
// Rec multi-round mode: multi-round decode loop runs on device (worker layer),
// while the engine issues a single step.
inline bool is_rec_multi_round_mode() { return FLAGS_max_decode_rounds > 0; }

// Get the number of decode rounds for Rec multi-round mode.
// Returns 0 if Rec multi-round mode is disabled.
inline int32_t get_rec_multi_round_decode_rounds() {
  return is_rec_multi_round_mode() ? FLAGS_max_decode_rounds : 0;
}

// Pipeline strategy selector: choose strategy based on RecModelKind
inline RecPipelineType get_rec_pipeline_type(RecModelKind kind) {
  switch (kind) {
    case RecModelKind::kLlmRec:
      if (is_rec_multi_round_mode()) {
        return RecPipelineType::kLlmRecMultiRoundPipeline;
      } else {
        return RecPipelineType::kLlmRecDefault;
      }
    case RecModelKind::kOneRec:
      return RecPipelineType::kOneRecDefault;
    case RecModelKind::kLLaDARec:
      return RecPipelineType::kLLaDARecWorkerLoop;
    default:
      return RecPipelineType::kLlmRecDefault;
  }
}

inline constexpr bool is_onerec_model_type(std::string_view model_type) {
  return model_type == "onerec";
}

inline constexpr bool is_llmrec_model_type(std::string_view model_type) {
  return model_type == "qwen2" || model_type == "qwen3" ||
         model_type == "qwen3_moe";
}

inline constexpr bool is_llada_rec_model_type(std::string_view model_type) {
  return model_type == "llada2_moe";
}

inline constexpr RecModelKind get_rec_model_kind(std::string_view model_type) {
  if (is_onerec_model_type(model_type)) {
    return RecModelKind::kOneRec;
  }
  if (is_llmrec_model_type(model_type)) {
    return RecModelKind::kLlmRec;
  }
  if (is_llada_rec_model_type(model_type)) {
    return RecModelKind::kLLaDARec;
  }
  return RecModelKind::kNone;
}

struct LLaDARuntimeConfig {
  int32_t block_length = 0;
  int32_t steps = 0;
  double threshold = 0.0;
  double editing_threshold = 0.0;
  int32_t max_post_steps = 0;
  int32_t minimal_topk = 0;
  int32_t num_to_transfer = 0;
  bool eos_early_stop = true;
  int32_t mask_id = 0;
};

inline constexpr const char* rec_model_kind_to_string(RecModelKind kind) {
  switch (kind) {
    case RecModelKind::kOneRec:
      return "OneRec";
    case RecModelKind::kLlmRec:
      return "LlmRec";
    case RecModelKind::kLLaDARec:
      return "LLaDARec";
    case RecModelKind::kNone:
      return "None";
  }
  return "Unknown";
}

inline constexpr const char* rec_pipeline_type_to_string(RecPipelineType type) {
  switch (type) {
    case RecPipelineType::kLlmRecDefault:
      return "LlmRecDefault";
    case RecPipelineType::kLlmRecWithMmData:
      return "LlmRecWithMmData";
    case RecPipelineType::kOneRecDefault:
      return "OneRecDefault";
    case RecPipelineType::kLlmRecMultiRoundPipeline:
      return "LlmRecMultiRoundPipeline";
    case RecPipelineType::kLLaDARecWorkerLoop:
      return "LLaDARecWorkerLoop";
  }
  return "Unknown";
}

inline std::string llada_runtime_config_to_string(
    const LLaDARuntimeConfig& config) {
  std::ostringstream oss;
  oss << "block_length=" << config.block_length << ", steps=" << config.steps
      << ", threshold=" << config.threshold
      << ", editing_threshold=" << config.editing_threshold
      << ", max_post_steps=" << config.max_post_steps
      << ", minimal_topk=" << config.minimal_topk
      << ", num_to_transfer=" << config.num_to_transfer
      << ", eos_early_stop=" << config.eos_early_stop
      << ", mask_id=" << config.mask_id;
  return oss.str();
}

inline bool validate_llada_runtime_config(const LLaDARuntimeConfig& config,
                                          int32_t eos_token_id,
                                          std::string* error_message) {
  if (config.block_length <= 0) {
    *error_message = "LLaDA requires llada_block_length > 0";
    return false;
  }
  if (config.steps <= 0) {
    *error_message = "LLaDA requires llada_steps > 0";
    return false;
  }
  if (!std::isfinite(config.threshold) || config.threshold < 0.0 ||
      config.threshold > 1.0) {
    *error_message = "LLaDA requires llada_threshold in [0, 1]";
    return false;
  }
  if (!std::isfinite(config.editing_threshold) ||
      config.editing_threshold < 0.0 || config.editing_threshold > 1.0) {
    *error_message = "LLaDA requires llada_editing_threshold in [0, 1]";
    return false;
  }
  if (config.max_post_steps < 0) {
    *error_message = "LLaDA requires llada_max_post_steps >= 0";
    return false;
  }
  if (config.minimal_topk <= 0) {
    *error_message = "LLaDA requires llada_minimal_topk > 0";
    return false;
  }
  if (config.num_to_transfer <= 0) {
    *error_message = "LLaDA requires llada_num_to_transfer > 0";
    return false;
  }
  if (config.mask_id < 0) {
    *error_message = "LLaDA requires llada_mask_id >= 0";
    return false;
  }
  if (eos_token_id >= 0 && config.mask_id == eos_token_id) {
    *error_message = "LLaDA requires llada_mask_id to differ from eos_token_id";
    return false;
  }
  return true;
}

inline LLaDARuntimeConfig get_llada_runtime_config() {
  LLaDARuntimeConfig config;
  config.block_length = FLAGS_llada_block_length;
  config.steps = FLAGS_llada_steps;
  config.threshold = FLAGS_llada_threshold;
  config.editing_threshold = FLAGS_llada_editing_threshold;
  config.max_post_steps = FLAGS_llada_max_post_steps;
  config.minimal_topk = FLAGS_llada_minimal_topk;
  config.num_to_transfer = FLAGS_llada_num_to_transfer;
  config.eos_early_stop = FLAGS_llada_eos_early_stop;
  config.mask_id = FLAGS_llada_mask_id;
  return config;
}

inline bool load_llada_runtime_config(int32_t eos_token_id,
                                      LLaDARuntimeConfig* config,
                                      std::string* error_message) {
  *config = get_llada_runtime_config();
  return validate_llada_runtime_config(*config, eos_token_id, error_message);
}

template <typename OptionsT>
bool validate_llada_runtime_options(const OptionsT& options,
                                    std::string* error_message) {
  if (options.enable_chunked_prefill()) {
    *error_message = "LLaDA does not support chunked prefill";
    return false;
  }
  if (options.enable_prefix_cache()) {
    *error_message = "LLaDA does not support prefix cache";
    return false;
  }
  if (options.enable_schedule_overlap()) {
    *error_message = "LLaDA does not support schedule overlap";
    return false;
  }
  if (options.enable_disagg_pd() || options.enable_service_routing()) {
    *error_message = "LLaDA does not support PD/service routing modes";
    return false;
  }
  if (options.num_speculative_tokens() > 0) {
    *error_message = "LLaDA does not support speculative decode";
    return false;
  }
  if (options.max_seqs_per_batch() != 1) {
    *error_message = "LLaDA requires max_seqs_per_batch=1";
    return false;
  }
  if (options.rec_worker_max_concurrency() != 1) {
    *error_message = "LLaDA requires rec_worker_max_concurrency=1";
    return false;
  }
  if (options.dp_size() != 1 || options.cp_size() != 1 ||
      options.ep_size() != 1) {
    *error_message = "LLaDA v1 only supports TP-only parallelism";
    return false;
  }
  return true;
}

}  // namespace xllm
