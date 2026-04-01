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

#include <cstdint>
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
