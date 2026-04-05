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

enum class RecModelFamily : uint8_t {
  kUnknown = 0,
  kOneRec = 1,
  kAutoregressive = 2,
  kDLM = 3,
};

enum class DlmCacheMode : uint8_t {
  kNone = 0,
  kPrefix = 1,
};

enum class LLaDAAlgorithmType : uint8_t {
  kJointThreshold = 0,
  kLowConfidence = 1,
};

// Pipeline strategy types (extensible for future strategies)
enum class RecPipelineType : uint8_t {
  kLlmRecDefault = 0,             // LlmRec without mm_data (pure qwen)
  kLlmRecWithMmData = 1,          // LlmRec with mm_data (qwen + embedding)
  kOneRecDefault = 2,             // OneRec
  kLlmRecMultiRoundPipeline = 3,  // LlmRec multi-round pipeline (device loop)
  kDlmWorkerLoop = 4,             // DLM worker-driven generation loop
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
      return RecPipelineType::kDlmWorkerLoop;
    default:
      return RecPipelineType::kLlmRecDefault;
  }
}

inline constexpr RecModelFamily get_rec_model_family(RecModelKind kind) {
  switch (kind) {
    case RecModelKind::kOneRec:
      return RecModelFamily::kOneRec;
    case RecModelKind::kLlmRec:
      return RecModelFamily::kAutoregressive;
    case RecModelKind::kLLaDARec:
      return RecModelFamily::kDLM;
    case RecModelKind::kNone:
      return RecModelFamily::kUnknown;
  }
  return RecModelFamily::kUnknown;
}

inline constexpr bool is_dlm_model_kind(RecModelKind kind) {
  return get_rec_model_family(kind) == RecModelFamily::kDLM;
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
  std::string preset_mode = "quality";
  LLaDAAlgorithmType algorithm = LLaDAAlgorithmType::kJointThreshold;
  DlmCacheMode cache_mode = DlmCacheMode::kPrefix;
  int32_t block_length = 0;
  int32_t steps = 0;
  double threshold = 0.0;
  double editing_threshold = 0.0;
  double penalty_lambda = 0.0;
  int32_t max_post_steps = 0;
  int32_t minimal_topk = 0;
  int32_t num_to_transfer = 0;
  bool eos_early_stop = true;
  int32_t mask_id = 0;
};

struct DlmRuntimeConfig {
  RecModelFamily family = RecModelFamily::kUnknown;
  std::string preset_mode = "quality";
  LLaDAAlgorithmType algorithm = LLaDAAlgorithmType::kJointThreshold;
  DlmCacheMode cache_mode = DlmCacheMode::kPrefix;
  int32_t block_length = 0;
  int32_t steps = 0;
  int32_t default_max_tokens = 16384;
  bool eos_early_stop = true;
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

inline constexpr const char* rec_model_family_to_string(RecModelFamily family) {
  switch (family) {
    case RecModelFamily::kUnknown:
      return "Unknown";
    case RecModelFamily::kOneRec:
      return "OneRec";
    case RecModelFamily::kAutoregressive:
      return "Autoregressive";
    case RecModelFamily::kDLM:
      return "DLM";
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
    case RecPipelineType::kDlmWorkerLoop:
      return "DlmWorkerLoop";
  }
  return "Unknown";
}

inline constexpr const char* dlm_cache_mode_to_string(DlmCacheMode mode) {
  switch (mode) {
    case DlmCacheMode::kNone:
      return "none";
    case DlmCacheMode::kPrefix:
      return "prefix";
  }
  return "unknown";
}

inline constexpr const char* llada_algorithm_type_to_string(
    LLaDAAlgorithmType algorithm) {
  switch (algorithm) {
    case LLaDAAlgorithmType::kJointThreshold:
      return "joint_threshold";
    case LLaDAAlgorithmType::kLowConfidence:
      return "low_confidence";
  }
  return "unknown";
}

inline bool is_valid_dlm_cache_mode(std::string_view mode) {
  return mode == "none" || mode == "prefix";
}

inline bool is_valid_llada_algorithm(std::string_view algorithm) {
  return algorithm == "joint_threshold" || algorithm == "low_confidence";
}

inline LLaDAAlgorithmType parse_llada_algorithm(std::string_view algorithm) {
  if (algorithm == "low_confidence") {
    return LLaDAAlgorithmType::kLowConfidence;
  }
  return LLaDAAlgorithmType::kJointThreshold;
}

inline DlmCacheMode parse_dlm_cache_mode(std::string_view mode) {
  if (mode == "none") {
    return DlmCacheMode::kNone;
  }
  return DlmCacheMode::kPrefix;
}

inline std::string llada_runtime_config_to_string(
    const LLaDARuntimeConfig& config) {
  std::ostringstream oss;
  oss << "preset_mode=" << config.preset_mode
      << ", algorithm=" << llada_algorithm_type_to_string(config.algorithm)
      << ", cache_mode=" << dlm_cache_mode_to_string(config.cache_mode)
      << ", block_length=" << config.block_length << ", steps=" << config.steps
      << ", threshold=" << config.threshold
      << ", editing_threshold=" << config.editing_threshold
      << ", penalty_lambda=" << config.penalty_lambda
      << ", max_post_steps=" << config.max_post_steps
      << ", minimal_topk=" << config.minimal_topk
      << ", num_to_transfer=" << config.num_to_transfer
      << ", eos_early_stop=" << config.eos_early_stop
      << ", mask_id=" << config.mask_id;
  return oss.str();
}

inline std::string dlm_runtime_config_to_string(
    const DlmRuntimeConfig& config) {
  std::ostringstream oss;
  oss << "family=" << rec_model_family_to_string(config.family)
      << ", preset_mode=" << config.preset_mode
      << ", algorithm=" << llada_algorithm_type_to_string(config.algorithm)
      << ", cache_mode=" << dlm_cache_mode_to_string(config.cache_mode)
      << ", block_length=" << config.block_length << ", steps=" << config.steps
      << ", default_max_tokens=" << config.default_max_tokens
      << ", eos_early_stop=" << config.eos_early_stop;
  return oss.str();
}

inline bool is_valid_llada_mode(std::string_view mode) {
  return mode == "quality" || mode == "speed" || mode == "custom";
}

inline bool validate_llada_runtime_config(const LLaDARuntimeConfig& config,
                                          int32_t eos_token_id,
                                          std::string* error_message) {
  if (!is_valid_llada_mode(config.preset_mode)) {
    *error_message =
        "LLaDA requires llada_mode to be one of quality, speed, custom";
    return false;
  }
  if (!is_valid_llada_algorithm(FLAGS_llada_algorithm)) {
    *error_message =
        "LLaDA requires llada_algorithm to be one of joint_threshold, "
        "low_confidence";
    return false;
  }
  if (!is_valid_dlm_cache_mode(FLAGS_llada_cache_mode)) {
    *error_message =
        "LLaDA requires llada_cache_mode to be one of none, prefix";
    return false;
  }
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
  if (!std::isfinite(config.penalty_lambda) || config.penalty_lambda < 0.0) {
    *error_message = "LLaDA requires llada_penalty_lambda >= 0";
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
  config.preset_mode = FLAGS_llada_mode;
  config.algorithm = parse_llada_algorithm(FLAGS_llada_algorithm);
  config.cache_mode = parse_dlm_cache_mode(FLAGS_llada_cache_mode);
  config.block_length = FLAGS_llada_block_length;
  config.steps = FLAGS_llada_steps;
  if (FLAGS_llada_mode == "speed") {
    config.threshold = 0.5;
    config.editing_threshold = 0.0;
    config.max_post_steps = std::max(FLAGS_llada_max_post_steps, 16);
  } else if (FLAGS_llada_mode == "quality") {
    config.threshold = 0.7;
    config.editing_threshold = 0.5;
    config.max_post_steps = std::max(FLAGS_llada_max_post_steps, 16);
  } else {
    config.threshold = FLAGS_llada_threshold;
    config.editing_threshold = FLAGS_llada_editing_threshold;
    config.max_post_steps = FLAGS_llada_max_post_steps;
  }
  config.penalty_lambda = FLAGS_llada_penalty_lambda;
  config.minimal_topk = FLAGS_llada_minimal_topk;
  config.num_to_transfer = FLAGS_llada_num_to_transfer;
  config.eos_early_stop = FLAGS_llada_eos_early_stop;
  config.mask_id = FLAGS_llada_mask_id;
  return config;
}

inline DlmRuntimeConfig get_dlm_runtime_config(RecModelKind kind) {
  DlmRuntimeConfig config;
  config.family = get_rec_model_family(kind);
  config.preset_mode = FLAGS_llada_mode;
  config.algorithm = parse_llada_algorithm(FLAGS_llada_algorithm);
  config.cache_mode = parse_dlm_cache_mode(FLAGS_llada_cache_mode);
  config.block_length = FLAGS_llada_block_length;
  config.steps = FLAGS_llada_steps;
  config.default_max_tokens = 16384;
  config.eos_early_stop = FLAGS_llada_eos_early_stop;
  return config;
}

inline bool load_llada_runtime_config(int32_t eos_token_id,
                                      LLaDARuntimeConfig* config,
                                      std::string* error_message) {
  *config = get_llada_runtime_config();
  return validate_llada_runtime_config(*config, eos_token_id, error_message);
}

template <typename OptionsT>
bool validate_dlm_runtime_options(const OptionsT& options,
                                  std::string* error_message) {
  if (options.enable_chunked_prefill()) {
    *error_message = "DLM worker pipeline does not support chunked prefill";
    return false;
  }
  if (options.enable_schedule_overlap()) {
    *error_message = "DLM worker pipeline does not support schedule overlap";
    return false;
  }
  if (options.enable_disagg_pd() || options.enable_service_routing()) {
    *error_message =
        "DLM worker pipeline does not support PD/service routing modes";
    return false;
  }
  if (options.num_speculative_tokens() > 0) {
    *error_message = "DLM worker pipeline does not support speculative decode";
    return false;
  }
  if (options.max_seqs_per_batch() != 1) {
    *error_message = "DLM worker pipeline requires max_seqs_per_batch=1";
    return false;
  }
  if (options.rec_worker_max_concurrency() != 1) {
    *error_message =
        "DLM worker pipeline requires rec_worker_max_concurrency=1";
    return false;
  }
  if (options.dp_size() != 1 || options.cp_size() != 1 ||
      options.ep_size() != 1) {
    *error_message = "DLM worker pipeline currently supports TP-only serving";
    return false;
  }
  return true;
}

template <typename OptionsT>
bool validate_llada_runtime_options(const OptionsT& options,
                                    std::string* error_message) {
  return validate_dlm_runtime_options(options, error_message);
}

}  // namespace xllm
