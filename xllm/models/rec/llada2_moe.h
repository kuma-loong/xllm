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

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "core/layers/common/rms_norm.h"
#include "core/layers/common/word_embedding.h"
#include "core/layers/llada2_moe_decoder_layer.h"
#include "models/model_registry.h"
#include "models/rec/rec_model_base.h"

namespace xllm {
class LLaDA2MoeModelImpl final : public torch::nn::Module {
 public:
  explicit LLaDA2MoeModelImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    embed_tokens_ =
        register_module("word_embeddings", layer::WordEmbedding(context));
    norm_ = register_module(
        "norm",
        layer::RMSNorm(
            model_args_.hidden_size(), model_args_.rms_norm_eps(), options_));
    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args_.n_layers());
    for (int32_t i = 0; i < model_args_.n_layers(); ++i) {
      auto block = layer::LLaDA2MoeDecoderLayer(context, i);
      blocks_->push_back(block);
      layers_.push_back(block);
    }
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    auto flat_tokens = tokens;
    auto flat_positions = positions;
    if (flat_tokens.dim() > 1) {
      flat_tokens = flat_tokens.reshape({-1});
    }
    if (flat_positions.dim() > 1) {
      flat_positions = flat_positions.reshape({-1});
    }
    if (flat_tokens.numel() == 0) {
      flat_tokens = torch::zeros({1}, options_.dtype(torch::kInt32));
      flat_positions = torch::zeros({1}, options_.dtype(torch::kInt32));
    }

    auto inputs_embeds = input_params.input_embedding;
    torch::Tensor hidden_states;
    if (inputs_embeds.defined()) {
      hidden_states = inputs_embeds.dim() > 2
                          ? inputs_embeds.reshape({-1, inputs_embeds.size(-1)})
                          : inputs_embeds;
    } else {
      hidden_states = embed_tokens_->forward(flat_tokens);
    }

    torch::Tensor attention_mask;
    if (input_params.attn_metadata &&
        input_params.attn_metadata->attn_mask.defined()) {
      attention_mask = input_params.attn_metadata->attn_mask;
    } else if (input_params.graph_buffer.attn_mask.defined()) {
      attention_mask = input_params.graph_buffer.attn_mask;
    }
    CHECK(attention_mask.defined())
        << "LLaDA forward requires explicit 4D block attention mask";

    const auto* dlm_runtime_params = input_params.dlm_params();
    const bool use_history_cache =
        dlm_runtime_params != nullptr && dlm_runtime_params->use_history_cache;
    const bool update_history_cache = dlm_runtime_params != nullptr &&
                                      dlm_runtime_params->update_history_cache;
    const int32_t active_cache_length =
        dlm_runtime_params != nullptr ? dlm_runtime_params->active_cache_length
                                      : 0;
    const int32_t cache_write_start =
        dlm_runtime_params != nullptr ? dlm_runtime_params->cache_write_start
                                      : 0;
    const int32_t cache_write_end =
        dlm_runtime_params != nullptr ? dlm_runtime_params->cache_write_end : 0;
    const int32_t block_offset =
        dlm_runtime_params != nullptr ? dlm_runtime_params->block_offset : 0;
    const int32_t block_length =
        dlm_runtime_params != nullptr ? dlm_runtime_params->block_length : 0;
    const auto req_phase =
        dlm_runtime_params != nullptr
            ? dlm_runtime_params->req_phase
            : DlmModelInputParams::ReqPhase::kIncomingPrefill;
    if (kv_caches.size() < layers_.size()) {
      kv_caches.resize(layers_.size());
    }

    for (size_t layer_id = 0; layer_id < layers_.size(); ++layer_id) {
      hidden_states = layers_[layer_id]->forward(hidden_states,
                                                 flat_positions,
                                                 attention_mask,
                                                 kv_caches[layer_id],
                                                 active_cache_length,
                                                 use_history_cache,
                                                 update_history_cache,
                                                 block_offset,
                                                 block_length,
                                                 req_phase,
                                                 cache_write_start,
                                                 cache_write_end);
    }
    auto norm_input = hidden_states;
    auto [final_hidden_states, residual] = norm_->forward(norm_input);
    return ModelOutput(final_hidden_states, residual);
  }

  void load_state_dict(const StateDict& state_dict) {
    auto embed_state_dict = state_dict.get_dict_with_prefix("word_embeddings.");
    if (embed_state_dict.size() > 0) {
      embed_tokens_->load_state_dict(embed_state_dict);
      embed_tokens_is_loaded_ = embed_tokens_is_loaded_ ||
                                embed_state_dict.get_tensor("weight").defined();
    }

    for (size_t i = 0; i < layers_.size(); ++i) {
      auto layer_state_dict =
          state_dict.get_dict_with_prefix("layers." + std::to_string(i) + ".");
      if (layer_state_dict.size() > 0) {
        layers_[i]->load_state_dict(layer_state_dict);
      }
    }

    auto norm_state_dict = state_dict.get_dict_with_prefix("norm.");
    if (norm_state_dict.size() == 0) {
      norm_state_dict = state_dict.get_dict_with_prefix("final_layernorm.");
    }
    if (norm_state_dict.size() > 0) {
      norm_->load_state_dict(norm_state_dict);
      norm_weight_is_loaded_ = norm_weight_is_loaded_ ||
                               norm_state_dict.get_tensor("weight").defined();
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(embed_tokens_is_loaded_)
        << "weight is not loaded for " << prefix + "word_embeddings.weight";
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->verify_loaded_weights(prefix + "layers." + std::to_string(i) +
                                        ".");
    }
    CHECK(norm_weight_is_loaded_)
        << "weight is not loaded for " << prefix + "norm.weight";
  }

  layer::WordEmbedding get_word_embedding() { return embed_tokens_; }

  void set_word_embedding(layer::WordEmbedding& embedding) {
    embed_tokens_ = embedding;
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  layer::WordEmbedding embed_tokens_{nullptr};
  layer::RMSNorm norm_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<layer::LLaDA2MoeDecoderLayer> layers_;

  bool embed_tokens_is_loaded_ = false;
  bool norm_weight_is_loaded_ = false;
};
TORCH_MODULE(LLaDA2MoeModel);

class LLaDA2MoeForConditionalGenerationImpl final
    : public RecForCausalLMImplBase<LLaDA2MoeModel> {
 public:
  explicit LLaDA2MoeForConditionalGenerationImpl(const ModelContext& context)
      : RecForCausalLMImplBase<LLaDA2MoeModel>(context) {}

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const auto& state_dict : loader->get_state_dicts()) {
      auto model_state_dict = state_dict->get_dict_with_prefix(prefix);
      model_->load_state_dict(model_state_dict);

      StateDict lm_head_state_dict(
          std::unordered_map<std::string, torch::Tensor>{});
      if (tie_word_embeddings_) {
        lm_head_state_dict =
            model_state_dict.get_dict_with_prefix("word_embeddings.");
      } else {
        lm_head_state_dict = state_dict->get_dict_with_prefix("lm_head.");
      }
      if (lm_head_state_dict.size() > 0) {
        lm_head_->load_state_dict(lm_head_state_dict);
        lm_head_is_loaded_ = lm_head_is_loaded_ ||
                             has_parallel_linear_weight(lm_head_state_dict);
      }
    }

    model_->verify_loaded_weights(prefix);
    CHECK(lm_head_is_loaded_)
        << "weight is not loaded for "
        << (tie_word_embeddings_ ? prefix + "word_embeddings.weight"
                                 : "lm_head.weight");
  }

 private:
  static bool has_parallel_linear_weight(const StateDict& state_dict) {
    return state_dict.get_tensor("weight").defined() ||
           state_dict.get_tensor("qweight").defined();
  }

  bool lm_head_is_loaded_ = false;
};
TORCH_MODULE(LLaDA2MoeForConditionalGeneration);

using LLaDA2MoeCausalLM = CausalLMImpl<LLaDA2MoeForConditionalGeneration>;
static_assert(std::is_base_of_v<CausalLM, LLaDA2MoeCausalLM>,
              "LLaDA2Moe must satisfy CausalLM contract.");

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
  LOAD_ARG_OR(rotary_dim, "rotary_dim", 64);
  LOAD_ARG_OR(use_qk_norm, "use_qk_norm", true);
  LOAD_ARG_OR(qkv_bias, "use_qkv_bias", false);
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
  LOAD_ARG_OR(
      moe_router_enable_expert_bias, "moe_router_enable_expert_bias", true);
  LOAD_ARG_OR(n_group, "n_group", 8);
  LOAD_ARG_OR(topk_group, "topk_group", 4);
  LOAD_ARG_OR(routed_scaling_factor, "routed_scaling_factor", 2.5f);
  LOAD_ARG_OR(scoring_func, "score_function", "sigmoid");

  LOAD_ARG_OR(vocab_size, "vocab_size", 157184);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 156892);
  LOAD_ARG_OR(pad_token_id, "pad_token_id", 156892);

  args->stop_token_ids().clear();
  if (args->eos_token_id() >= 0) {
    args->stop_token_ids().insert(args->eos_token_id());
  }
});

}  // namespace xllm
