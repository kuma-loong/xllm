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

#include "llada2_moe_decoder_layer.h"

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"
#if defined(USE_NPU)
#include "kernels/npu/npu_ops_api.h"
#endif
#include "models/rec/llada_dlm_cache.h"

namespace xllm {
namespace layer {

namespace {

bool has_parallel_linear_weight(const StateDict& state_dict) {
  return state_dict.get_tensor("weight").defined() ||
         state_dict.get_tensor("qweight").defined();
}

torch::Tensor normalize_attention_mask(torch::Tensor attn_mask) {
  CHECK_EQ(attn_mask.dim(), 4)
      << "Expected 4D attention mask, got " << attn_mask.sizes();
  return attn_mask;
}

torch::Tensor apply_hidden_activation(const std::string& hidden_act,
                                      const torch::Tensor& x) {
  if (hidden_act == "silu" || hidden_act == "swish") {
    return x * torch::sigmoid(x);
  }
  if (hidden_act == "gelu") {
    return torch::nn::functional::gelu(x);
  }
  if (hidden_act == "relu") {
    return torch::relu(x);
  }
  LOG(FATAL) << "Unsupported LLaDA hidden_act: " << hidden_act;
  return torch::Tensor();
}

torch::Tensor repeat_kv(const torch::Tensor& hidden_states,
                        int64_t num_kv_head_replicas) {
  if (num_kv_head_replicas == 1) {
    return hidden_states;
  }
  const auto batch = hidden_states.size(0);
  const auto num_kv_heads = hidden_states.size(1);
  const auto seq_len = hidden_states.size(2);
  const auto head_dim = hidden_states.size(3);
  return hidden_states.unsqueeze(2)
      .expand({batch, num_kv_heads, num_kv_head_replicas, seq_len, head_dim})
      .reshape({batch, num_kv_heads * num_kv_head_replicas, seq_len, head_dim});
}

void split_qkv_tensor(const torch::Tensor& fused_tensor,
                      int64_t total_q_size,
                      int64_t total_kv_size,
                      int64_t dim,
                      const std::string& name,
                      std::unordered_map<std::string, torch::Tensor>* out) {
  if (!fused_tensor.defined()) {
    return;
  }

  if (fused_tensor.numel() == 1) {
    (*out)["q_proj." + name] = fused_tensor;
    (*out)["k_proj." + name] = fused_tensor;
    (*out)["v_proj." + name] = fused_tensor;
    return;
  }

  CHECK_LT(dim, fused_tensor.dim())
      << "Invalid split dim " << dim << " for " << name;
  CHECK_EQ(fused_tensor.size(dim), total_q_size + 2 * total_kv_size)
      << "Invalid fused QKV size for " << name << ", expected "
      << total_q_size + 2 * total_kv_size << ", got " << fused_tensor.size(dim);

  (*out)["q_proj." + name] =
      fused_tensor.narrow(dim, 0, total_q_size).contiguous();
  (*out)["k_proj." + name] =
      fused_tensor.narrow(dim, total_q_size, total_kv_size).contiguous();
  (*out)["v_proj." + name] =
      fused_tensor.narrow(dim, total_q_size + total_kv_size, total_kv_size)
          .contiguous();
}

}  // namespace

LLaDA2MoeGateImpl::LLaDA2MoeGateImpl(const ModelContext& context)
    : num_experts_(context.get_model_args().num_experts()),
      top_k_(context.get_model_args().num_experts_per_tok()),
      n_group_(context.get_model_args().n_group()),
      topk_group_(context.get_model_args().topk_group()),
      routed_scaling_factor_(context.get_model_args().routed_scaling_factor()),
      norm_topk_prob_(context.get_model_args().norm_topk_prob()),
      use_expert_bias_(
          context.get_model_args().moe_router_enable_expert_bias()) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& options = context.get_tensor_options();

  gate_ = register_module("gate",
                          ReplicatedLinear(args.hidden_size(),
                                           num_experts_,
                                           /*bias=*/false,
                                           quant_args,
                                           options));
  if (use_expert_bias_) {
    expert_bias_ = register_parameter(
        "expert_bias",
        torch::zeros({num_experts_}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
  }
}

std::tuple<torch::Tensor, torch::Tensor> LLaDA2MoeGateImpl::forward(
    const torch::Tensor& hidden_states) {
  auto logits = gate_->forward(hidden_states).to(torch::kFloat32);
  auto scores = torch::sigmoid(logits);
  auto scores_for_routing = scores;
  if (expert_bias_.defined()) {
    scores_for_routing = scores_for_routing + expert_bias_;
  }

  const auto num_tokens = scores.size(0);
  CHECK_GT(n_group_, 0) << "n_group must be positive for LLaDA MoE.";
  CHECK_GT(topk_group_, 0) << "topk_group must be positive for LLaDA MoE.";
  CHECK_EQ(num_experts_ % n_group_, 0)
      << "num_experts must be divisible by n_group.";
  const int64_t experts_per_group = num_experts_ / n_group_;
  const int64_t top2 = std::min<int64_t>(2, experts_per_group);

  auto grouped_scores =
      scores_for_routing.view({num_tokens, n_group_, experts_per_group});
  auto group_scores =
      std::get<0>(grouped_scores.topk(top2, /*dim=*/-1)).sum(/*dim=*/-1);

  auto group_idx =
      std::get<1>(group_scores.topk(topk_group_, /*dim=*/-1, true, false));
  auto group_mask = torch::zeros_like(group_scores);
  group_mask.scatter_(1, group_idx, 1);

  auto score_mask = group_mask.unsqueeze(-1)
                        .expand({num_tokens, n_group_, experts_per_group})
                        .reshape({num_tokens, num_experts_});
  auto masked_scores =
      scores_for_routing.masked_fill(score_mask.to(torch::kBool).logical_not(),
                                     -std::numeric_limits<float>::infinity());

  auto topk_result = masked_scores.topk(top_k_, /*dim=*/-1);
  auto topk_idx = std::get<1>(topk_result);
  auto topk_weight = torch::gather(scores, /*dim=*/1, topk_idx);
  if (norm_topk_prob_ && top_k_ > 1) {
    topk_weight =
        topk_weight / (topk_weight.sum(/*dim=*/-1, /*keepdim=*/true) + 1e-20);
  }
  topk_weight = topk_weight * routed_scaling_factor_;
  return {topk_idx, topk_weight};
}

void LLaDA2MoeGateImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() > 0) {
    gate_->load_state_dict(state_dict);
    gate_weight_is_loaded_ =
        gate_weight_is_loaded_ || has_parallel_linear_weight(state_dict);
  }

  auto tensor = state_dict.get_tensor("expert_bias");
  if (tensor.defined()) {
    CHECK(expert_bias_.defined())
        << "Unexpected expert_bias in checkpoint without router bias support";
    CHECK_EQ(expert_bias_.sizes(), tensor.sizes())
        << "expert_bias size mismatch";
    torch::NoGradGuard no_grad;
    expert_bias_.copy_(
        tensor.to(expert_bias_.device()).to(expert_bias_.dtype()));
    expert_bias_is_loaded_ = true;
  }
}

void LLaDA2MoeGateImpl::verify_loaded_weights(const std::string& prefix) const {
  CHECK(gate_weight_is_loaded_)
      << "weight is not loaded for " << prefix + "gate.weight";
  if (use_expert_bias_) {
    CHECK(expert_bias_is_loaded_)
        << "weight is not loaded for " << prefix + "expert_bias";
  }
}

LLaDA2SparseMoeBlockImpl::LLaDA2SparseMoeBlockImpl(const ModelContext& context)
    : hidden_size_(context.get_model_args().hidden_size()),
      moe_intermediate_size_(context.get_model_args().moe_intermediate_size()),
      num_experts_(context.get_model_args().num_experts()),
      hidden_act_(context.get_model_args().hidden_act()),
      tp_group_(context.get_parallel_args().tp_group_),
      tp_rank_(tp_group_->rank()),
      tp_world_size_(tp_group_->world_size()),
      expert_gate_is_loaded_(num_experts_, false),
      expert_up_is_loaded_(num_experts_, false),
      expert_down_is_loaded_(num_experts_, false) {
  CHECK_EQ(moe_intermediate_size_ % tp_world_size_, 0)
      << "moe_intermediate_size must be divisible by TP world size.";
  local_intermediate_size_ = moe_intermediate_size_ / tp_world_size_;

  const auto& args = context.get_model_args();
  const auto& options = context.get_tensor_options();
  gate_ = register_module("gate", LLaDA2MoeGate(context));
  if (args.n_shared_experts() > 0) {
    shared_experts_ = register_module(
        "shared_experts",
        DenseMLP(hidden_size_,
                 moe_intermediate_size_ * args.n_shared_experts(),
                 /*is_gated=*/true,
                 /*has_bias=*/false,
                 hidden_act_,
                 /*enable_result_reduction=*/false,
                 context.get_quant_args(),
                 tp_group_,
                 options));
  }

  auto expert_options = options;
  w13_ = register_parameter(
      "w13",
      torch::empty({num_experts_, local_intermediate_size_ * 2, hidden_size_},
                   expert_options),
      /*requires_grad=*/false);
  w2_ = register_parameter(
      "w2",
      torch::empty({num_experts_, hidden_size_, local_intermediate_size_},
                   expert_options),
      /*requires_grad=*/false);
}

torch::Tensor LLaDA2SparseMoeBlockImpl::forward(
    const torch::Tensor& hidden_states) {
#if defined(USE_NPU)
  return forward_npu_fused(hidden_states);
#else
  auto [topk_idx, topk_weight] = gate_->forward(hidden_states);

  auto local_output = torch::zeros({hidden_states.size(0), hidden_size_},
                                   hidden_states.options());

  for (int64_t expert_id = 0; expert_id < num_experts_; ++expert_id) {
    auto expert_matches = torch::nonzero(topk_idx == expert_id);
    if (expert_matches.numel() == 0) {
      continue;
    }

    auto token_indices =
        expert_matches.select(/*dim=*/1, /*index=*/0).to(torch::kLong);
    auto topk_slots =
        expert_matches.select(/*dim=*/1, /*index=*/1).to(torch::kLong);

    auto expert_input = hidden_states.index_select(/*dim=*/0, token_indices);
    auto gate_up = torch::matmul(expert_input, w13_[expert_id].transpose(0, 1));
    auto gate_part = gate_up.slice(/*dim=*/-1, 0, local_intermediate_size_);
    auto up_part = gate_up.slice(
        /*dim=*/-1, local_intermediate_size_, local_intermediate_size_ * 2);
    auto activated = apply_hidden_activation(hidden_act_, gate_part) * up_part;
    auto expert_output =
        torch::matmul(activated, w2_[expert_id].transpose(0, 1));

    auto expert_weight = topk_weight.index({token_indices, topk_slots})
                             .to(expert_output.dtype())
                             .unsqueeze(-1);
    local_output.index_add_(
        /*dim=*/0, token_indices, expert_output * expert_weight);
  }

  if (shared_experts_) {
    local_output = local_output + shared_experts_->forward(hidden_states);
  }
  if (tp_world_size_ > 1) {
    local_output = parallel_state::reduce(local_output, tp_group_);
  }
  return local_output;
#endif
}

torch::Tensor LLaDA2SparseMoeBlockImpl::forward_npu_fused(
    const torch::Tensor& hidden_states) {
  auto [topk_idx, topk_weight] = gate_->forward(hidden_states);

  std::optional<torch::Tensor> shared_output = std::nullopt;
  if (shared_experts_) {
    shared_output = shared_experts_->forward(hidden_states);
  }

  auto hidden_states_2d = hidden_states.reshape({-1, hidden_size_});
  auto expert_idx = topk_idx.to(torch::kInt32).contiguous();
  auto reduce_weight = topk_weight.to(torch::kFloat32).contiguous();
  const std::array<int64_t, 2> active_expert_range = {0, num_experts_};

  xllm::kernel::MoeInitRoutingV2Params moe_init_routing_params;
  moe_init_routing_params.x = hidden_states_2d;
  moe_init_routing_params.expert_idx = expert_idx;
  moe_init_routing_params.scale = std::nullopt;
  moe_init_routing_params.offset = std::nullopt;
  moe_init_routing_params.active_num =
      hidden_states_2d.size(0) * expert_idx.size(1);
  moe_init_routing_params.expert_capacity = 0;
  moe_init_routing_params.expert_num = num_experts_;
  moe_init_routing_params.drop_pad_mode = 0;
  moe_init_routing_params.expert_tokens_num_type = 1;
  moe_init_routing_params.expert_tokens_num_flag = true;
  moe_init_routing_params.row_idx_type = 0;
  moe_init_routing_params.quant_mode = -1;
  moe_init_routing_params.active_expert_range = active_expert_range;

  auto [expand_hidden_states, expand_row_ids, group_list, dynamic_scale] =
      xllm::kernel::moe_init_routing_v2(moe_init_routing_params);
  (void)dynamic_scale;

  xllm::kernel::GroupGemmParams gemm1_params;
  gemm1_params.a = expand_hidden_states;
  gemm1_params.b = w13_;
  if (gemm1_params.b.size(1) != expand_hidden_states.size(1)) {
    gemm1_params.b = gemm1_params.b.transpose(1, 2);
  }
  gemm1_params.group_list = group_list;
  gemm1_params.split_item = 2;
  gemm1_params.group_type = 0;
  gemm1_params.group_list_type = 1;
  auto gemm1_out = xllm::kernel::group_gemm(gemm1_params);

  xllm::kernel::ActivationParams activation_params;
  activation_params.input = gemm1_out;
  activation_params.act_mode = hidden_act_;
  activation_params.is_gated = true;
  xllm::kernel::active(activation_params);
  auto act_out = activation_params.output;

  xllm::kernel::GroupGemmParams gemm2_params;
  gemm2_params.a = act_out;
  gemm2_params.b = w2_;
  if (gemm2_params.b.size(1) != act_out.size(1)) {
    gemm2_params.b = gemm2_params.b.transpose(1, 2);
  }
  gemm2_params.group_list = group_list;
  gemm2_params.split_item = 2;
  gemm2_params.group_type = 0;
  gemm2_params.group_list_type = 1;
  auto gemm2_out = xllm::kernel::group_gemm(gemm2_params);

  xllm::kernel::MoeCombineResultParams moe_combine_params;
  moe_combine_params.input = gemm2_out;
  moe_combine_params.reduce_weight = reduce_weight;
  moe_combine_params.gather_ids = expand_row_ids;
  auto output = xllm::kernel::moe_combine_result(moe_combine_params)
                    .reshape(hidden_states.sizes());
  if (shared_output.has_value()) {
    output = output + shared_output.value();
  }
  if (tp_world_size_ > 1) {
    output = parallel_state::reduce(output, tp_group_);
  }
  return output;
}

void LLaDA2SparseMoeBlockImpl::load_state_dict(const StateDict& state_dict) {
  auto gate_state_dict = state_dict.get_dict_with_prefix("gate.");
  if (gate_state_dict.size() > 0) {
    gate_->load_state_dict(gate_state_dict);
  }

  if (shared_experts_) {
    auto shared_state_dict = state_dict.get_dict_with_prefix("shared_experts.");
    if (shared_state_dict.size() > 0) {
      shared_experts_->load_state_dict(shared_state_dict);
      shared_gate_is_loaded_ =
          shared_gate_is_loaded_ ||
          has_parallel_linear_weight(
              shared_state_dict.get_dict_with_prefix("gate_proj."));
      shared_up_is_loaded_ =
          shared_up_is_loaded_ ||
          has_parallel_linear_weight(
              shared_state_dict.get_dict_with_prefix("up_proj."));
      shared_down_is_loaded_ =
          shared_down_is_loaded_ ||
          has_parallel_linear_weight(
              shared_state_dict.get_dict_with_prefix("down_proj."));
    }
  }

  torch::NoGradGuard no_grad;
  for (int64_t expert_id = 0; expert_id < num_experts_; ++expert_id) {
    const auto prefix = "experts." + std::to_string(expert_id) + ".";
    auto gate_weight = state_dict.get_sharded_tensor(
        prefix + "gate_proj.weight", /*dim=*/0, tp_rank_, tp_world_size_);
    auto up_weight = state_dict.get_sharded_tensor(
        prefix + "up_proj.weight", /*dim=*/0, tp_rank_, tp_world_size_);
    auto down_weight = state_dict.get_sharded_tensor(
        prefix + "down_proj.weight", /*dim=*/1, tp_rank_, tp_world_size_);

    if (gate_weight.defined()) {
      auto target = w13_[expert_id].slice(
          /*dim=*/0, 0, local_intermediate_size_);
      CHECK_EQ(target.sizes(), gate_weight.sizes())
          << "LLaDA expert gate shape mismatch for expert " << expert_id;
      target.copy_(
          gate_weight.to(target.device()).to(target.dtype()).contiguous());
      expert_gate_is_loaded_[expert_id] = true;
    }
    if (up_weight.defined()) {
      auto target = w13_[expert_id].slice(
          /*dim=*/0, local_intermediate_size_, local_intermediate_size_ * 2);
      CHECK_EQ(target.sizes(), up_weight.sizes())
          << "LLaDA expert up shape mismatch for expert " << expert_id;
      target.copy_(
          up_weight.to(target.device()).to(target.dtype()).contiguous());
      expert_up_is_loaded_[expert_id] = true;
    }
    if (down_weight.defined()) {
      CHECK_EQ(w2_[expert_id].sizes(), down_weight.sizes())
          << "LLaDA expert down shape mismatch for expert " << expert_id;
      w2_[expert_id].copy_(
          down_weight.to(w2_.device()).to(w2_.dtype()).contiguous());
      expert_down_is_loaded_[expert_id] = true;
    }
  }
}

void LLaDA2SparseMoeBlockImpl::verify_loaded_weights(
    const std::string& prefix) const {
  gate_->verify_loaded_weights(prefix + "gate.");
  if (shared_experts_) {
    CHECK(shared_gate_is_loaded_) << "weight is not loaded for "
                                  << prefix + "shared_experts.gate_proj.weight";
    CHECK(shared_up_is_loaded_) << "weight is not loaded for "
                                << prefix + "shared_experts.up_proj.weight";
    CHECK(shared_down_is_loaded_) << "weight is not loaded for "
                                  << prefix + "shared_experts.down_proj.weight";
  }

  for (int64_t expert_id = 0; expert_id < num_experts_; ++expert_id) {
    const auto expert_prefix =
        prefix + "experts." + std::to_string(expert_id) + ".";
    CHECK(expert_gate_is_loaded_[expert_id])
        << "weight is not loaded for " << expert_prefix + "gate_proj.weight";
    CHECK(expert_up_is_loaded_[expert_id])
        << "weight is not loaded for " << expert_prefix + "up_proj.weight";
    CHECK(expert_down_is_loaded_[expert_id])
        << "weight is not loaded for " << expert_prefix + "down_proj.weight";
  }
}

LLaDA2AttentionImpl::LLaDA2AttentionImpl(const ModelContext& context)
    : tp_rank_(context.get_parallel_args().tp_group_->rank()),
      tp_world_size_(context.get_parallel_args().tp_group_->world_size()) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  total_num_heads_ = args.n_heads();
  total_num_kv_heads_ = args.n_kv_heads().value_or(args.n_heads());
  CHECK_EQ(total_num_heads_ % tp_world_size_, 0)
      << "num_attention_heads must be divisible by TP world size.";

  num_heads_ = total_num_heads_ / tp_world_size_;
  if (total_num_kv_heads_ >= tp_world_size_) {
    CHECK_EQ(total_num_kv_heads_ % tp_world_size_, 0)
        << "num_key_value_heads must be divisible by TP world size.";
    num_kv_heads_ = total_num_kv_heads_ / tp_world_size_;
    num_kv_head_replicas_ = 1;
  } else {
    CHECK_EQ(tp_world_size_ % total_num_kv_heads_, 0)
        << "TP world size must be divisible by num_key_value_heads.";
    num_kv_heads_ = 1;
    num_kv_head_replicas_ = tp_world_size_ / total_num_kv_heads_;
  }
  CHECK_EQ(num_heads_ % num_kv_heads_, 0)
      << "Local attention heads must be divisible by local kv heads.";
  attn_num_kv_repeats_ = num_heads_ / num_kv_heads_;

  head_dim_ = args.head_dim();
  total_q_size_ = total_num_heads_ * head_dim_;
  total_kv_size_ = total_num_kv_heads_ * head_dim_;
  q_size_ = num_heads_ * head_dim_;
  kv_size_ = num_kv_heads_ * head_dim_;
  max_position_embeddings_ = args.max_position_embeddings();
  scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  use_qk_norm_ = args.use_qk_norm();
  qkv_bias_ = args.qkv_bias();

  qkv_proj_ = register_module("query_key_value",
                              QKVParallelLinear(args.hidden_size(),
                                                num_heads_,
                                                num_kv_heads_,
                                                head_dim_,
                                                num_kv_head_replicas_,
                                                qkv_bias_,
                                                /*gather_output=*/false,
                                                parallel_args,
                                                options,
                                                quant_args));
  o_proj_ = register_module("dense",
                            RowParallelLinear(total_num_heads_ * head_dim_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*enable_result_reduction=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));
  if (use_qk_norm_) {
    q_norm_ = register_module(
        "query_layernorm",
        Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));
    k_norm_ = register_module(
        "key_layernorm",
        Qwen3NextRMSNorm(head_dim_, args.rms_norm_eps(), options));
  }
  rotary_emb_ = register_module(
      "rotary_emb",
      PartialRotaryEmbedding(
          static_cast<int64_t>(head_dim_ * args.partial_rotary_factor()),
          args.max_position_embeddings(),
          static_cast<int64_t>(args.rope_theta()),
          head_dim_,
          /*is_neox_style=*/true,
          /*interleaved=*/false,
          options));
}

torch::Tensor LLaDA2AttentionImpl::forward(const torch::Tensor& hidden_states,
                                           const torch::Tensor& positions,
                                           const torch::Tensor& attention_mask,
                                           KVCache& kv_cache,
                                           int32_t active_cache_length,
                                           bool use_history_cache,
                                           bool update_history_cache,
                                           int32_t cache_write_start,
                                           int32_t cache_write_end) {
  auto qkv = qkv_proj_->forward(hidden_states);
  auto q = qkv.slice(/*dim=*/-1, 0, q_size_);
  auto k = qkv.slice(/*dim=*/-1, q_size_, q_size_ + kv_size_);
  auto v = qkv.slice(/*dim=*/-1, q_size_ + kv_size_, q_size_ + kv_size_ * 2);

  const auto seq_len = hidden_states.size(0);
  if (use_qk_norm_) {
    auto q_reshaped = q.reshape({seq_len, num_heads_, head_dim_});
    auto k_reshaped = k.reshape({seq_len, num_kv_heads_, head_dim_});
    q = q_norm_->forward(q_reshaped).reshape({seq_len, q_size_});
    k = k_norm_->forward(k_reshaped).reshape({seq_len, kv_size_});
  }

  auto rope_positions = positions.reshape({-1});
  rotary_emb_->forward(rope_positions, q, k);

  auto query_states =
      q.view({1, seq_len, num_heads_, head_dim_}).transpose(1, 2);
  auto current_key_states =
      k.view({1, seq_len, num_kv_heads_, head_dim_}).transpose(1, 2);
  auto current_value_states =
      v.view({1, seq_len, num_kv_heads_, head_dim_}).transpose(1, 2);
  auto key_states = current_key_states;
  auto value_states = current_value_states;
  if ((use_history_cache || update_history_cache) &&
      cache_write_end > cache_write_start) {
    CHECK_GE(cache_write_start, 0)
        << "LLaDA cache_write_start must be non-negative";
    CHECK_GT(cache_write_end, cache_write_start)
        << "LLaDA cache_write_end must be greater than cache_write_start";
    CHECK_LE(cache_write_end, max_position_embeddings_)
        << "LLaDA cache_write_end exceeds max_position_embeddings";
    LLaDAHistoryCacheRange cache_range;
    cache_range.start = cache_write_start;
    cache_range.end = cache_write_end;
    cache_range.active_length = std::max(active_cache_length, cache_write_end);
    std::tie(key_states, value_states) =
        LLaDAHistoryCache::materialize_attention_kv(kv_cache,
                                                    current_key_states,
                                                    current_value_states,
                                                    cache_range,
                                                    update_history_cache);
  } else if (update_history_cache || use_history_cache) {
    kv_cache = KVCache(current_key_states.contiguous(),
                       current_value_states.contiguous());
  }

  auto attn_mask = normalize_attention_mask(attention_mask)
                       .to(query_states.device())
                       .to(query_states.dtype());
  if (attn_mask.size(-1) != key_states.size(-2)) {
    attn_mask = attn_mask.slice(/*dim=*/-1, 0, key_states.size(-2));
  }
#if defined(USE_NPU)
  const bool can_use_npu_batch_prefill = !use_history_cache &&
                                         !update_history_cache &&
                                         cache_write_end <= cache_write_start;
  if (can_use_npu_batch_prefill) {
    auto query_3d = q.view({seq_len, num_heads_, head_dim_});
    auto key_3d = key_states.squeeze(0).transpose(0, 1).contiguous();
    auto value_3d = value_states.squeeze(0).transpose(0, 1).contiguous();
    auto output_3d = torch::empty_like(query_3d);
    auto kv_seq_lens_host =
        torch::tensor({static_cast<int32_t>(key_3d.size(0))},
                      torch::TensorOptions().dtype(torch::kInt));
    xllm::kernel::npu::batch_prefill(query_3d,
                                     key_3d,
                                     value_3d,
                                     attn_mask,
                                     kv_seq_lens_host,
                                     scaling_,
                                     output_3d);
    auto attn_output = output_3d.reshape({seq_len, q_size_});
    return o_proj_->forward(attn_output);
  }
#endif
  key_states = repeat_kv(key_states, attn_num_kv_repeats_);
  value_states = repeat_kv(value_states, attn_num_kv_repeats_);
  auto attn_output = at::scaled_dot_product_attention(
      query_states, key_states, value_states, attn_mask, 0.0, false);
  attn_output =
      attn_output.transpose(1, 2).contiguous().reshape({seq_len, q_size_});
  return o_proj_->forward(attn_output);
}

void LLaDA2AttentionImpl::load_state_dict(const StateDict& state_dict) {
  std::unordered_map<std::string, torch::Tensor> split_qkv_tensors;
  split_qkv_tensor(state_dict.get_tensor("query_key_value.weight"),
                   total_q_size_,
                   total_kv_size_,
                   /*dim=*/0,
                   "weight",
                   &split_qkv_tensors);
  split_qkv_tensor(state_dict.get_tensor("query_key_value.bias"),
                   total_q_size_,
                   total_kv_size_,
                   /*dim=*/0,
                   "bias",
                   &split_qkv_tensors);
  split_qkv_tensor(state_dict.get_tensor("query_key_value.weight_scale"),
                   total_q_size_,
                   total_kv_size_,
                   /*dim=*/0,
                   "weight_scale",
                   &split_qkv_tensors);
  split_qkv_tensor(state_dict.get_tensor("query_key_value.input_scale"),
                   total_q_size_,
                   total_kv_size_,
                   /*dim=*/0,
                   "input_scale",
                   &split_qkv_tensors);
  if (!split_qkv_tensors.empty()) {
    qkv_proj_->load_state_dict(StateDict(std::move(split_qkv_tensors)),
                               {"q_proj.", "k_proj.", "v_proj."});
  }
  qkv_weight_is_loaded_ =
      qkv_weight_is_loaded_ ||
      state_dict.get_tensor("query_key_value.weight").defined();
  if (qkv_bias_) {
    qkv_bias_is_loaded_ =
        qkv_bias_is_loaded_ ||
        state_dict.get_tensor("query_key_value.bias").defined();
  }

  auto dense_state_dict = state_dict.get_dict_with_prefix("dense.");
  if (dense_state_dict.size() > 0) {
    o_proj_->load_state_dict(dense_state_dict);
    o_proj_weight_is_loaded_ = o_proj_weight_is_loaded_ ||
                               has_parallel_linear_weight(dense_state_dict);
  }

  auto q_weight = state_dict.get_tensor("query_layernorm.weight");
  if (q_weight.defined()) {
    CHECK(q_norm_)
        << "Unexpected query_layernorm weights when use_qk_norm=false";
    q_norm_->load_state_dict(StateDict({{"weight", q_weight}}));
    q_norm_weight_is_loaded_ = true;
  }

  auto k_weight = state_dict.get_tensor("key_layernorm.weight");
  if (k_weight.defined()) {
    CHECK(k_norm_) << "Unexpected key_layernorm weights when use_qk_norm=false";
    k_norm_->load_state_dict(StateDict({{"weight", k_weight}}));
    k_norm_weight_is_loaded_ = true;
  }
}

void LLaDA2AttentionImpl::verify_loaded_weights(
    const std::string& prefix) const {
  CHECK(qkv_weight_is_loaded_)
      << "weight is not loaded for " << prefix + "query_key_value.weight";
  if (qkv_bias_) {
    CHECK(qkv_bias_is_loaded_)
        << "weight is not loaded for " << prefix + "query_key_value.bias";
  }
  CHECK(o_proj_weight_is_loaded_)
      << "weight is not loaded for " << prefix + "dense.weight";
  if (use_qk_norm_) {
    CHECK(q_norm_weight_is_loaded_)
        << "weight is not loaded for " << prefix + "query_layernorm.weight";
    CHECK(k_norm_weight_is_loaded_)
        << "weight is not loaded for " << prefix + "key_layernorm.weight";
  }
}

LLaDA2MoeDecoderLayerImpl::LLaDA2MoeDecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const auto& args = context.get_model_args();
  const auto& options = context.get_tensor_options();

  attention_ = register_module("attention", LLaDA2Attention(context));
  input_norm_ = register_module(
      "input_layernorm",
      RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));
  post_norm_ = register_module(
      "post_attention_layernorm",
      RMSNorm(args.hidden_size(), args.rms_norm_eps(), options));

  if (layer_id >= args.first_k_dense_replace()) {
    moe_ = register_module("mlp", LLaDA2SparseMoeBlock(context));
  } else {
    dense_mlp_ = register_module("mlp",
                                 DenseMLP(args.hidden_size(),
                                          args.intermediate_size(),
                                          /*is_gated=*/true,
                                          /*has_bias=*/false,
                                          args.hidden_act(),
                                          /*enable_result_reduction=*/true,
                                          context.get_quant_args(),
                                          context.get_parallel_args().tp_group_,
                                          options));
  }
}

torch::Tensor LLaDA2MoeDecoderLayerImpl::forward(
    const torch::Tensor& hidden_states,
    const torch::Tensor& positions,
    const torch::Tensor& attention_mask,
    KVCache& kv_cache,
    int32_t active_cache_length,
    bool use_history_cache,
    bool update_history_cache,
    int32_t cache_write_start,
    int32_t cache_write_end) {
  auto residual = hidden_states;
  auto norm_input = hidden_states;
  auto x = std::get<0>(input_norm_->forward(norm_input));
  x = attention_->forward(x,
                          positions,
                          attention_mask,
                          kv_cache,
                          active_cache_length,
                          use_history_cache,
                          update_history_cache,
                          cache_write_start,
                          cache_write_end);
  x = residual + x;

  residual = x;
  auto post_input = x;
  x = std::get<0>(post_norm_->forward(post_input));
  if (moe_) {
    x = moe_->forward(x);
  } else {
    x = dense_mlp_->forward(x);
  }
  return residual + x;
}

void LLaDA2MoeDecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  auto attention_state_dict = state_dict.get_dict_with_prefix("attention.");
  if (attention_state_dict.size() > 0) {
    attention_->load_state_dict(attention_state_dict);
  }

  auto input_norm_state_dict =
      state_dict.get_dict_with_prefix("input_layernorm.");
  if (input_norm_state_dict.size() > 0) {
    input_norm_->load_state_dict(input_norm_state_dict);
    input_norm_weight_is_loaded_ =
        input_norm_weight_is_loaded_ ||
        input_norm_state_dict.get_tensor("weight").defined();
  }

  auto post_norm_state_dict =
      state_dict.get_dict_with_prefix("post_attention_layernorm.");
  if (post_norm_state_dict.size() > 0) {
    post_norm_->load_state_dict(post_norm_state_dict);
    post_norm_weight_is_loaded_ =
        post_norm_weight_is_loaded_ ||
        post_norm_state_dict.get_tensor("weight").defined();
  }

  auto mlp_state_dict = state_dict.get_dict_with_prefix("mlp.");
  if (mlp_state_dict.size() == 0) {
    return;
  }
  if (moe_) {
    moe_->load_state_dict(mlp_state_dict);
  } else {
    dense_mlp_->load_state_dict(mlp_state_dict);
    dense_gate_is_loaded_ =
        dense_gate_is_loaded_ ||
        has_parallel_linear_weight(
            mlp_state_dict.get_dict_with_prefix("gate_proj."));
    dense_up_is_loaded_ = dense_up_is_loaded_ ||
                          has_parallel_linear_weight(
                              mlp_state_dict.get_dict_with_prefix("up_proj."));
    dense_down_is_loaded_ =
        dense_down_is_loaded_ ||
        has_parallel_linear_weight(
            mlp_state_dict.get_dict_with_prefix("down_proj."));
  }
}

void LLaDA2MoeDecoderLayerImpl::verify_loaded_weights(
    const std::string& prefix) const {
  attention_->verify_loaded_weights(prefix + "attention.");
  CHECK(input_norm_weight_is_loaded_)
      << "weight is not loaded for " << prefix + "input_layernorm.weight";
  CHECK(post_norm_weight_is_loaded_)
      << "weight is not loaded for "
      << prefix + "post_attention_layernorm.weight";
  if (moe_) {
    moe_->verify_loaded_weights(prefix + "mlp.");
  } else {
    CHECK(dense_gate_is_loaded_)
        << "weight is not loaded for " << prefix + "mlp.gate_proj.weight";
    CHECK(dense_up_is_loaded_)
        << "weight is not loaded for " << prefix + "mlp.up_proj.weight";
    CHECK(dense_down_is_loaded_)
        << "weight is not loaded for " << prefix + "mlp.down_proj.weight";
  }
}

}  // namespace layer
}  // namespace xllm
