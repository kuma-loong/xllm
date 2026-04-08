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

#include <string>
#include <tuple>
#include <vector>

#include "common/dense_mlp.h"
#include "common/linear.h"
#include "common/partial_rotary_embedding.h"
#include "common/qwen3_next_rms_norm.h"
#include "common/rms_norm.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {

class LLaDA2MoeGateImpl : public torch::nn::Module {
 public:
  explicit LLaDA2MoeGateImpl(const ModelContext& context);

  std::tuple<torch::Tensor, torch::Tensor> forward(
      const torch::Tensor& hidden_states);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix) const;

 private:
  int64_t num_experts_ = 0;
  int64_t top_k_ = 0;
  int64_t n_group_ = 0;
  int64_t topk_group_ = 0;
  float routed_scaling_factor_ = 1.0f;
  bool norm_topk_prob_ = false;
  bool use_expert_bias_ = false;

  ReplicatedLinear gate_{nullptr};
  torch::Tensor expert_bias_;

  bool gate_weight_is_loaded_ = false;
  bool expert_bias_is_loaded_ = false;
};
TORCH_MODULE(LLaDA2MoeGate);

class LLaDA2SparseMoeBlockImpl : public torch::nn::Module {
 public:
  explicit LLaDA2SparseMoeBlockImpl(const ModelContext& context);

  torch::Tensor forward(const torch::Tensor& hidden_states);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix) const;

 private:
#if defined(USE_NPU)
  torch::Tensor forward_npu_llada_grouped_moe(
      const torch::Tensor& hidden_states);
#endif

  int64_t hidden_size_ = 0;
  int64_t moe_intermediate_size_ = 0;
  int64_t num_experts_ = 0;
  int64_t local_intermediate_size_ = 0;
  std::string hidden_act_;

  ProcessGroup* tp_group_ = nullptr;
  int64_t tp_rank_ = 0;
  int64_t tp_world_size_ = 1;

  LLaDA2MoeGate gate_{nullptr};
  DenseMLP shared_experts_{nullptr};
  torch::Tensor w13_;
  torch::Tensor w2_;

  std::vector<bool> expert_gate_is_loaded_;
  std::vector<bool> expert_up_is_loaded_;
  std::vector<bool> expert_down_is_loaded_;
  bool shared_gate_is_loaded_ = false;
  bool shared_up_is_loaded_ = false;
  bool shared_down_is_loaded_ = false;
};
TORCH_MODULE(LLaDA2SparseMoeBlock);

class LLaDA2AttentionImpl : public torch::nn::Module {
 public:
  explicit LLaDA2AttentionImpl(const ModelContext& context);

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& positions,
                        const torch::Tensor& attention_mask,
                        KVCache& kv_cache,
                        int32_t active_cache_length,
                        bool use_history_cache,
                        bool update_history_cache,
                        int32_t block_offset,
                        int32_t block_length,
                        DlmModelInputParams::ReqPhase req_phase,
                        int32_t cache_write_start,
                        int32_t cache_write_end);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix) const;

 private:
  int64_t tp_rank_ = 0;
  int64_t tp_world_size_ = 1;
  int64_t total_num_heads_ = 0;
  int64_t total_num_kv_heads_ = 0;
  int64_t num_heads_ = 0;
  int64_t num_kv_heads_ = 0;
  int64_t num_kv_head_replicas_ = 1;
  int64_t attn_num_kv_repeats_ = 1;
  int64_t head_dim_ = 0;
  int64_t total_q_size_ = 0;
  int64_t total_kv_size_ = 0;
  int64_t q_size_ = 0;
  int64_t kv_size_ = 0;
  int64_t max_position_embeddings_ = 0;
  float scaling_ = 1.0f;
  bool use_qk_norm_ = false;
  bool qkv_bias_ = false;

  QKVParallelLinear qkv_proj_{nullptr};
  RowParallelLinear o_proj_{nullptr};
  Qwen3NextRMSNorm q_norm_{nullptr};
  Qwen3NextRMSNorm k_norm_{nullptr};
  PartialRotaryEmbedding rotary_emb_{nullptr};

  bool qkv_weight_is_loaded_ = false;
  bool qkv_bias_is_loaded_ = false;
  bool o_proj_weight_is_loaded_ = false;
  bool q_norm_weight_is_loaded_ = false;
  bool k_norm_weight_is_loaded_ = false;
};
TORCH_MODULE(LLaDA2Attention);

class LLaDA2MoeDecoderLayerImpl : public torch::nn::Module {
 public:
  explicit LLaDA2MoeDecoderLayerImpl(const ModelContext& context,
                                     int32_t layer_id);

  torch::Tensor forward(const torch::Tensor& hidden_states,
                        const torch::Tensor& positions,
                        const torch::Tensor& attention_mask,
                        KVCache& kv_cache,
                        int32_t active_cache_length,
                        bool use_history_cache,
                        bool update_history_cache,
                        int32_t block_offset,
                        int32_t block_length,
                        DlmModelInputParams::ReqPhase req_phase,
                        int32_t cache_write_start,
                        int32_t cache_write_end);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights(const std::string& prefix) const;

 private:
  LLaDA2Attention attention_{nullptr};
  RMSNorm input_norm_{nullptr};
  RMSNorm post_norm_{nullptr};
  DenseMLP dense_mlp_{nullptr};
  LLaDA2SparseMoeBlock moe_{nullptr};

  bool input_norm_weight_is_loaded_ = false;
  bool post_norm_weight_is_loaded_ = false;
  bool dense_gate_is_loaded_ = false;
  bool dense_up_is_loaded_ = false;
  bool dense_down_is_loaded_ = false;
};
TORCH_MODULE(LLaDA2MoeDecoderLayer);

}  // namespace layer
}  // namespace xllm
