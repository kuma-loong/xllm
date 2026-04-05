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

#include "runtime/dlm_runtime.h"

#include <gtest/gtest.h>

#include "models/rec/llada_dlm_cache.h"

namespace xllm {

TEST(DlmRuntimeTest, PrefixCacheManagerTracksCommittedLength) {
  DlmPrefixCacheManager manager(/*num_layers=*/3);
  EXPECT_FALSE(manager.has_committed_prefix());
  EXPECT_EQ(manager.committed_prefix_length(), 0);
  EXPECT_EQ(manager.mutable_caches().size(), 3);

  manager.mark_prefix_committed(DlmBlockRange{0, 32});
  EXPECT_TRUE(manager.has_committed_prefix());
  EXPECT_EQ(manager.committed_prefix_length(), 32);
  EXPECT_EQ(manager.active_length_for(DlmBlockRange{32, 64}), 64);
  EXPECT_TRUE(manager.can_use_prefix_for(DlmBlockRange{32, 64}));

  manager.mark_prefix_committed(DlmBlockRange{0, 16});
  EXPECT_EQ(manager.committed_prefix_length(), 32);
}

TEST(DlmRuntimeTest, PrefixCacheManagerRejectsNonPrefixCommit) {
  DlmPrefixCacheManager manager(/*num_layers=*/1);
  EXPECT_DEATH(manager.mark_prefix_committed(DlmBlockRange{32, 64}),
               "prefix cache requires monotonic committed ranges");
}

TEST(DlmRuntimeTest, ThresholdHelperKeepsMinimalTopKFallback) {
  LLaDARuntimeConfig config;
  config.algorithm = LLaDAAlgorithmType::kLowConfidence;
  config.threshold = 0.9;
  config.minimal_topk = 1;
  config.num_to_transfer = 1;
  config.mask_id = 42;
  DlmDecodeAlgorithm helper(config);

  auto probs = torch::tensor({{0.4f, 0.3f, 0.2f}});
  auto next_tokens = torch::tensor({{10, 11, 42}}, torch::kInt64);
  auto active_mask = torch::tensor({{true, true, false}}, torch::kBool);
  auto transfer = helper.select_mask_transfer_index(
      probs,
      next_tokens,
      active_mask,
      torch::TensorOptions().dtype(torch::kBool));

  EXPECT_TRUE(transfer[0][0].item<bool>());
  EXPECT_FALSE(transfer[0][1].item<bool>());
  EXPECT_FALSE(transfer[0][2].item<bool>());
}

TEST(DlmRuntimeTest, ThresholdHelperAcceptsAllHighConfidenceMasks) {
  LLaDARuntimeConfig config;
  config.algorithm = LLaDAAlgorithmType::kLowConfidence;
  config.threshold = 0.7;
  config.minimal_topk = 1;
  config.num_to_transfer = 1;
  config.mask_id = 42;
  DlmDecodeAlgorithm helper(config);

  auto probs = torch::tensor({{0.91f, 0.85f, 0.1f}});
  auto next_tokens = torch::tensor({{10, 11, 42}}, torch::kInt64);
  auto active_mask = torch::tensor({{true, true, false}}, torch::kBool);
  auto transfer = helper.select_mask_transfer_index(
      probs,
      next_tokens,
      active_mask,
      torch::TensorOptions().dtype(torch::kBool));

  EXPECT_TRUE(transfer[0][0].item<bool>());
  EXPECT_TRUE(transfer[0][1].item<bool>());
  EXPECT_FALSE(transfer[0][2].item<bool>());
}

TEST(DlmRuntimeTest, JointThresholdUsesAbsoluteThresholdForMasks) {
  LLaDARuntimeConfig config;
  config.algorithm = LLaDAAlgorithmType::kJointThreshold;
  config.threshold = 0.5;
  config.minimal_topk = 1;
  config.num_to_transfer = 1;
  config.mask_id = 42;
  DlmDecodeAlgorithm helper(config);

  auto probs = torch::tensor({{0.9f, 0.4f, 0.1f}});
  auto next_tokens = torch::tensor({{10, 11, 12}}, torch::kInt64);
  auto active_mask = torch::tensor({{true, true, false}}, torch::kBool);
  auto transfer = helper.select_mask_transfer_index(
      probs,
      next_tokens,
      active_mask,
      torch::TensorOptions().dtype(torch::kBool));

  EXPECT_TRUE(transfer[0][0].item<bool>());
  EXPECT_FALSE(transfer[0][1].item<bool>());
}

TEST(DlmRuntimeTest, JointThresholdEditsOnlyNonPromptChangedTokens) {
  LLaDARuntimeConfig config;
  config.algorithm = LLaDAAlgorithmType::kJointThreshold;
  config.editing_threshold = 0.5;
  config.mask_id = 42;
  DlmDecodeAlgorithm helper(config);

  auto probs = torch::tensor({{0.8f, 0.9f, 0.7f}});
  auto next_tokens = torch::tensor({{42, 13, 15}}, torch::kInt64);
  auto old_tokens = torch::tensor({{7, 14, 15}}, torch::kInt64);
  auto active_mask = torch::tensor({{false, false, false}}, torch::kBool);
  auto prompt_mask = torch::tensor({true, false, false}, torch::kBool);
  auto transfer = helper.select_edit_transfer_index(
      probs,
      next_tokens,
      old_tokens,
      active_mask,
      prompt_mask,
      torch::TensorOptions().dtype(torch::kBool));

  EXPECT_FALSE(transfer[0][0].item<bool>());
  EXPECT_TRUE(transfer[0][1].item<bool>());
  EXPECT_FALSE(transfer[0][2].item<bool>());
}

TEST(DlmRuntimeTest, JointThresholdPenaltyLowersRepeatedTokenLogit) {
  LLaDARuntimeConfig config;
  config.algorithm = LLaDAAlgorithmType::kJointThreshold;
  config.penalty_lambda = 0.5;
  DlmDecodeAlgorithm helper(config);

  auto logits = torch::zeros({3, 5}, torch::kFloat32);
  auto tokens = torch::tensor({1, 2, 3}, torch::kInt64);
  helper.apply_penalty(logits, tokens);

  EXPECT_FLOAT_EQ(logits[1][1].item<float>(), -0.5f);
  EXPECT_FLOAT_EQ(logits[2][2].item<float>(), -0.5f);
  EXPECT_FLOAT_EQ(logits[0][1].item<float>(), 0.0f);
}

TEST(DlmRuntimeTest, JointThresholdTransferPlanStopsWhenNoEditApplies) {
  LLaDARuntimeConfig config;
  config.algorithm = LLaDAAlgorithmType::kJointThreshold;
  config.threshold = 0.9;
  config.editing_threshold = 0.9;
  config.max_post_steps = 4;
  config.mask_id = 42;
  DlmDecodeAlgorithm helper(config);

  DlmDecodeStepState state;
  state.has_mask = false;
  state.post_edit_step = 1;
  auto probs = torch::tensor({{0.2f, 0.3f, 0.4f}});
  auto next_tokens = torch::tensor({{8, 9, 10}}, torch::kInt64);
  auto old_tokens = torch::tensor({{8, 9, 10}}, torch::kInt64);
  auto prompt_mask = torch::tensor({false, false, false}, torch::kBool);

  auto plan =
      helper.build_transfer_plan(probs,
                                 next_tokens,
                                 old_tokens,
                                 prompt_mask,
                                 state,
                                 torch::TensorOptions().dtype(torch::kBool));
  EXPECT_TRUE(plan.finished);
  EXPECT_FALSE(plan.transfer_index.any().item<bool>());
}

TEST(DlmRuntimeTest, LowConfidenceTransferPlanStopsWhenNoMaskRemains) {
  LLaDARuntimeConfig config;
  config.algorithm = LLaDAAlgorithmType::kLowConfidence;
  config.mask_id = 42;
  DlmDecodeAlgorithm helper(config);

  DlmDecodeStepState state;
  state.has_mask = false;
  auto probs = torch::tensor({{0.2f, 0.3f, 0.4f}});
  auto next_tokens = torch::tensor({{8, 9, 10}}, torch::kInt64);
  auto old_tokens = torch::tensor({{8, 9, 10}}, torch::kInt64);
  auto prompt_mask = torch::tensor({false, false, false}, torch::kBool);

  auto plan =
      helper.build_transfer_plan(probs,
                                 next_tokens,
                                 old_tokens,
                                 prompt_mask,
                                 state,
                                 torch::TensorOptions().dtype(torch::kBool));
  EXPECT_TRUE(plan.finished);
  EXPECT_FALSE(plan.transfer_index.any().item<bool>());
}

TEST(DlmRuntimeTest, LLaDAHistoryCacheRangeUpdateReplacesBlockSlice) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  KVCache kv_cache(torch::zeros({1, 1, 4, 2}, options),
                   torch::zeros({1, 1, 4, 2}, options));
  auto current_k = torch::ones({1, 1, 2, 2}, options);
  auto current_v = torch::full({1, 1, 2, 2}, 2.0f, options);

  LLaDAHistoryCacheRange cache_range;
  cache_range.start = 2;
  cache_range.end = 4;
  cache_range.active_length = 4;
  auto [materialized_k, materialized_v] =
      LLaDAHistoryCache::materialize_attention_kv(
          kv_cache, current_k, current_v, cache_range, /*commit_cache=*/true);

  EXPECT_EQ(materialized_k.size(2), 4);
  EXPECT_EQ(materialized_v.size(2), 4);
  EXPECT_TRUE(torch::allclose(materialized_k.slice(2, 0, 2),
                              torch::zeros({1, 1, 2, 2}, options)));
  EXPECT_TRUE(torch::allclose(materialized_k.slice(2, 2, 4), current_k));
  EXPECT_TRUE(torch::allclose(materialized_v.slice(2, 2, 4), current_v));
  EXPECT_TRUE(
      torch::allclose(kv_cache.get_k_cache().slice(2, 2, 4), current_k));
  EXPECT_TRUE(
      torch::allclose(kv_cache.get_v_cache().slice(2, 2, 4), current_v));
}

TEST(DlmRuntimeTest,
     LLaDAHistoryCacheMaterializeWithoutCommitReusesWorkerLocalCache) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  KVCache kv_cache(torch::zeros({1, 1, 2, 2}, options),
                   torch::zeros({1, 1, 2, 2}, options));
  auto current_k = torch::ones({1, 1, 2, 2}, options);
  auto current_v = torch::full({1, 1, 2, 2}, 3.0f, options);

  LLaDAHistoryCacheRange cache_range;
  cache_range.start = 2;
  cache_range.end = 4;
  cache_range.active_length = 4;
  auto [materialized_k, materialized_v] =
      LLaDAHistoryCache::materialize_attention_kv(
          kv_cache, current_k, current_v, cache_range, /*commit_cache=*/false);

  EXPECT_EQ(materialized_k.size(2), 4);
  EXPECT_EQ(materialized_v.size(2), 4);
  EXPECT_TRUE(torch::allclose(materialized_k.slice(2, 2, 4), current_k));
  EXPECT_TRUE(torch::allclose(materialized_v.slice(2, 2, 4), current_v));
  EXPECT_EQ(kv_cache.get_k_cache().size(2), 4);
  EXPECT_EQ(kv_cache.get_v_cache().size(2), 4);
  EXPECT_TRUE(
      torch::allclose(kv_cache.get_k_cache().slice(2, 2, 4), current_k));
  EXPECT_TRUE(
      torch::allclose(kv_cache.get_v_cache().slice(2, 2, 4), current_v));
}

TEST(DlmRuntimeTest, LLaDAHistoryCacheOverwritesMiddleSliceWithLongerCache) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto base_k = torch::arange(12, options).view({1, 1, 6, 2});
  auto base_v = torch::arange(12, options).view({1, 1, 6, 2}) + 100.0f;
  KVCache kv_cache(base_k.clone(), base_v.clone());
  auto current_k = torch::full({1, 1, 2, 2}, 7.0f, options);
  auto current_v = torch::full({1, 1, 2, 2}, 9.0f, options);

  LLaDAHistoryCacheRange cache_range;
  cache_range.start = 2;
  cache_range.end = 4;
  cache_range.active_length = 6;
  auto [materialized_k, materialized_v] =
      LLaDAHistoryCache::materialize_attention_kv(
          kv_cache, current_k, current_v, cache_range, /*commit_cache=*/true);

  EXPECT_TRUE(
      torch::allclose(materialized_k.slice(2, 0, 2), base_k.slice(2, 0, 2)));
  EXPECT_TRUE(torch::allclose(materialized_k.slice(2, 2, 4), current_k));
  EXPECT_TRUE(
      torch::allclose(materialized_k.slice(2, 4, 6), base_k.slice(2, 4, 6)));
  EXPECT_TRUE(
      torch::allclose(materialized_v.slice(2, 0, 2), base_v.slice(2, 0, 2)));
  EXPECT_TRUE(torch::allclose(materialized_v.slice(2, 2, 4), current_v));
  EXPECT_TRUE(
      torch::allclose(materialized_v.slice(2, 4, 6), base_v.slice(2, 4, 6)));
}

}  // namespace xllm
