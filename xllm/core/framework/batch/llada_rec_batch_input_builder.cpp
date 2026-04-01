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

#include "llada_rec_batch_input_builder.h"

#include <glog/logging.h>

#include <vector>

#include "framework/sampling/sampling_params.h"

namespace xllm {

namespace {

std::vector<Sequence*> collect_sequences(
    const std::vector<Sequence*>& sequences,
    const std::vector<SequencesGroup*>& sequence_groups) {
  if (!sequences.empty()) {
    return sequences;
  }

  std::vector<Sequence*> collected_sequences;
  for (SequencesGroup* sequence_group : sequence_groups) {
    if (sequence_group == nullptr) {
      continue;
    }
    for (const auto& sequence : sequence_group->sequences()) {
      collected_sequences.push_back(sequence.get());
    }
  }
  return collected_sequences;
}

std::vector<int32_t> build_positions(int32_t prompt_length) {
  std::vector<int32_t> positions;
  positions.reserve(prompt_length);
  for (int32_t index = 0; index < prompt_length; ++index) {
    positions.push_back(index);
  }
  return positions;
}

torch::Tensor make_seq_len_tensor(int32_t prompt_length) {
  return torch::tensor({prompt_length}, torch::dtype(torch::kInt32));
}

}  // namespace

LLaDARecBatchInputBuilder::LLaDARecBatchInputBuilder(
    const std::vector<Sequence*>& sequences,
    const std::vector<SequencesGroup*>& sequence_groups,
    const std::vector<uint32_t>& allowed_max_tokens,
    const std::vector<torch::Tensor>& input_embeddings_vec,
    const std::vector<MMData>& mm_data_vec,
    std::vector<BlockTransferInfo>* swap_block_transfer_infos,
    uint64_t batch_id,
    const ModelArgs* args,
    BatchForwardType batch_forward_type,
    ThreadPool* thread_pool)
    : sequences_(collect_sequences(sequences, sequence_groups)),
      allowed_max_tokens_(allowed_max_tokens),
      batch_id_(batch_id) {
  (void)input_embeddings_vec;
  (void)mm_data_vec;
  (void)swap_block_transfer_infos;
  (void)args;
  (void)batch_forward_type;
  (void)thread_pool;

  (void)sequence_groups;
}

ForwardInput LLaDARecBatchInputBuilder::build_rec_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size) {
  (void)num_decoding_tokens;
  (void)min_decoding_batch_size;

  CHECK_EQ(sequences_.size(), 1)
      << "LLaDA v1 only supports a single active sequence per batch.";
  CHECK_EQ(allowed_max_tokens_.size(), 1)
      << "LLaDA v1 expects exactly one token budget entry.";

  auto* sequence = sequences_.front();
  CHECK(sequence != nullptr);

  const auto token_ids = sequence->tokens();
  const int32_t prompt_length = static_cast<int32_t>(token_ids.size());
  CHECK_GT(prompt_length, 0) << "LLaDA prompt tokens must not be empty.";

  std::vector<int32_t> flat_tokens(token_ids.begin(), token_ids.end());
  std::vector<int32_t> positions = build_positions(prompt_length);

  SamplingParameters sampling_params;
  sampling_params.init({sequence->sampling_param()},
                       {std::max(prompt_length - 1, 0)},
                       {0},
                       {{}},
                       {{}},
                       {0});

  ForwardInput input;
  input.token_ids = torch::tensor(flat_tokens, torch::dtype(torch::kInt32));
  input.positions = torch::tensor(positions, torch::dtype(torch::kInt32));
  input.sampling_params = std::move(sampling_params);
  input.input_params.batch_forward_type = BatchForwardType::PREFILL;
  input.input_params.num_sequences = 1;
  input.input_params.kv_max_seq_len = prompt_length;
  input.input_params.q_max_seq_len = prompt_length;
  input.input_params.q_seq_lens = make_seq_len_tensor(prompt_length);
  input.input_params.kv_seq_lens = make_seq_len_tensor(prompt_length);
  input.input_params.q_cu_seq_lens =
      torch::tensor({0, prompt_length}, torch::dtype(torch::kInt32));
  input.input_params.q_seq_lens_vec = {prompt_length};
  input.input_params.kv_seq_lens_vec = {prompt_length};
  input.input_params.batch_id = batch_id_;
  auto& llada_params = input.input_params.mutable_llada_params();
  llada_params.prompt_length = prompt_length;
  llada_params.max_generated_tokens =
      static_cast<int32_t>(allowed_max_tokens_.front());
  return input;
}

}  // namespace xllm
