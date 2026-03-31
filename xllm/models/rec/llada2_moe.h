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

#include <unordered_set>

#include "models/model_registry.h"
#include "models/rec/onerec.h"

namespace xllm {

// Module A only wires model registration and metadata loading. The dedicated
// runtime/model implementation will replace this placeholder in later modules.
using LLaDA2MoeForConditionalGeneration = OneRecForConditionalGeneration;

REGISTER_REC_MODEL(llada2_moe, LLaDA2MoeForConditionalGeneration);

REGISTER_MODEL_ARGS(llada2_moe, [&] {
  LOAD_ARG_OR(model_type, "model_type", "llada2_moe");
  LOAD_ARG_OR_FUNC(dtype, "dtype", [&] {
    return json.value_or<std::string>("torch_dtype", "bfloat16");
  });

  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 5120);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 20);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", 4);
  LOAD_ARG_OR(head_dim, "head_dim", 128);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6f);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rope_theta, "rope_theta", 600000.0f);
  LOAD_ARG_OR(partial_rotary_factor, "partial_rotary_factor", 0.5f);
  LOAD_ARG_OR(use_qk_norm, "norm_head", false);
  LOAD_ARG_OR(first_k_dense_replace, "first_k_dense_replace", 1);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(num_experts, "num_experts", 256);
  LOAD_ARG_OR(n_routed_experts, "num_experts", 256);
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 8);
  LOAD_ARG_OR(n_shared_experts, "num_shared_experts", 1);
  LOAD_ARG_OR(moe_intermediate_size, "moe_intermediate_size", 512);
  LOAD_ARG_OR(n_group, "n_group", 8);
  LOAD_ARG_OR(topk_group, "topk_group", 4);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(scoring_func, "score_function", "sigmoid");

  LOAD_ARG_OR(vocab_size, "vocab_size", 157184);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", -1);
  LOAD_ARG_OR(pad_token_id, "pad_token_id", 0);

  args->stop_token_ids().clear();
  if (args->eos_token_id() >= 0) {
    args->stop_token_ids().insert(args->eos_token_id());
  }
});

REGISTER_TOKENIZER_ARGS(llada2_moe, [&] { SET_ARG(tokenizer_type, "fast"); });

}  // namespace xllm
