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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <utility>

#include "framework/kv_cache/kv_cache.h"

namespace xllm {

struct LLaDAHistoryCacheRange {
  int32_t start = 0;
  int32_t end = 0;
  int32_t active_length = 0;

  bool enabled() const { return end > start; }
};

class LLaDAHistoryCache {
 public:
  static std::pair<torch::Tensor, torch::Tensor> materialize_attention_kv(
      KVCache& kv_cache,
      const torch::Tensor& current_key_states,
      const torch::Tensor& current_value_states,
      const LLaDAHistoryCacheRange& cache_range,
      bool commit_cache) {
    CHECK(cache_range.enabled()) << "LLaDA history cache range must be enabled";
    const int64_t target_length =
        std::max<int64_t>(cache_range.active_length, cache_range.end);
    CHECK_GT(target_length, 0)
        << "LLaDA history cache target length must be positive";

    torch::Tensor key_cache = ensure_cache_length(
        kv_cache.get_k_cache(), current_key_states, target_length);
    torch::Tensor value_cache = ensure_cache_length(
        kv_cache.get_v_cache(), current_value_states, target_length);
    write_cache_slice(key_cache, current_key_states, cache_range);
    write_cache_slice(value_cache, current_value_states, cache_range);

    if (commit_cache || caches_changed(kv_cache, key_cache, value_cache)) {
      kv_cache = KVCache(key_cache.contiguous(), value_cache.contiguous());
    }

    return {key_cache.slice(/*dim=*/2, 0, target_length),
            value_cache.slice(/*dim=*/2, 0, target_length)};
  }

 private:
  static torch::Tensor ensure_cache_length(const torch::Tensor& existing_cache,
                                           const torch::Tensor& current_states,
                                           int64_t target_length) {
    CHECK_GT(target_length, 0)
        << "LLaDA history cache target length must be positive";
    if (!existing_cache.defined()) {
      return torch::zeros({current_states.size(0),
                           current_states.size(1),
                           target_length,
                           current_states.size(3)},
                          current_states.options());
    }

    if (existing_cache.size(2) >= target_length) {
      return existing_cache;
    }

    return torch::cat({existing_cache,
                       torch::zeros({existing_cache.size(0),
                                     existing_cache.size(1),
                                     target_length - existing_cache.size(2),
                                     existing_cache.size(3)},
                                    existing_cache.options())},
                      /*dim=*/2);
  }

  static bool caches_changed(const KVCache& kv_cache,
                             const torch::Tensor& key_cache,
                             const torch::Tensor& value_cache) {
    return !kv_cache.get_k_cache().defined() ||
           !kv_cache.get_v_cache().defined() ||
           !kv_cache.get_k_cache().is_same(key_cache) ||
           !kv_cache.get_v_cache().is_same(value_cache);
  }

  static void write_cache_slice(torch::Tensor& cache,
                                const torch::Tensor& current_states,
                                const LLaDAHistoryCacheRange& cache_range) {
    cache.slice(/*dim=*/2, cache_range.start, cache_range.end)
        .copy_(current_states);
  }
};

}  // namespace xllm
