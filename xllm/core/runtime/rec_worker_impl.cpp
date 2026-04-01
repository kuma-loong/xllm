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

#include "rec_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/global_flags.h"
#include "core/common/global_flags.h"
#include "framework/model_loader.h"
#if defined(USE_CUDA)
#include "layers/cuda/flashinfer_workspace.h"
#include "layers/cuda/xattention_workspace.h"
#endif
#include "models/model_registry.h"
#include "util/env_var.h"

namespace xllm {

void RecWorkerImpl::initialize_xattention_workspace() {
#if defined(USE_CUDA)
  if (FLAGS_enable_xattention_one_stage) {
    return;
  }
  ::xllm::layer::xattention::XAttentionWorkspace::get_instance().initialize(
      device_);
#endif
}

RecWorkerImpl::RecWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : LLMWorkerImpl(parallel_args, device, options) {
  initialize_xattention_workspace();

  if (!is_driver()) {
    return;
  }

  step_threadpool_ = std::make_unique<ThreadPool>(
      options_.rec_worker_max_concurrency(), [this]() mutable {
        device_.set_device();
#if defined(USE_CUDA)
        ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance()
            .initialize(device_);
        initialize_xattention_workspace();
#endif
      });

  LOG(INFO) << "RecWorkerImpl constructor: "
            << options_.rec_worker_max_concurrency();
  const int64_t num_threads = std::max<int64_t>(
      1, util::get_int_env("XLLM_REC_INPUT_BUILDER_THREADS", 16));
  input_builder_thread_pool_ =
      std::make_shared<ThreadPool>(static_cast<size_t>(num_threads));
}

RecWorkerImpl::~RecWorkerImpl() {
  model_.release();
  model_executor_.release();

  if (FLAGS_enable_eplb) {
    eplb_executor_.release();
  }
}

bool RecWorkerImpl::init_model(const std::string& model_weights_path,
                               int32_t random_seed,
                               MasterStatus master_status) {
  if (!WorkerImpl::init_model(model_weights_path, random_seed, master_status)) {
    return false;
  }

  if (FLAGS_enable_eplb) {
    work_pipelines_[0]->runtime().expert_load_data = expert_load_data_;

    for (size_t i = 1; i < work_pipelines_.size(); ++i) {
      work_pipelines_[i]->runtime().expert_load_data =
          work_pipelines_[0]->runtime().expert_load_data.clone();
    }
  }

  return true;
}

bool RecWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  const auto& model_type = context.get_model_args().model_type();
  rec_model_kind_ = get_rec_model_kind(model_type);
  CHECK(rec_model_kind_ != RecModelKind::kNone)
      << "Unsupported rec model_type: " << model_type;

  RecPipelineType pipeline_type = get_rec_pipeline_type(rec_model_kind_);

  work_pipelines_.reserve(options_.rec_worker_max_concurrency());
  for (size_t i = 0; i < options_.rec_worker_max_concurrency(); ++i) {
    RecPipelineRuntime runtime(*this);
    auto stream = device_.get_stream_from_pool();
    runtime.stream = std::move(stream);
    auto stream_guard = runtime.stream->set_stream_guard();

    runtime.context =
        std::make_unique<ModelContext>(context.get_parallel_args(),
                                       context.get_model_args(),
                                       context.get_quant_args(),
                                       context.get_tensor_options());

    if (rec_model_kind_ == RecModelKind::kOneRec ||
        rec_model_kind_ == RecModelKind::kLLaDARec) {
      runtime.model = create_rec_model(*runtime.context.get());
    } else {
      runtime.model = create_llm_model(*runtime.context.get());
    }
    CHECK(runtime.model != nullptr) << "Failed to create model instance " << i;

    runtime.executor =
        std::make_unique<Executor>(runtime.model.get(),
                                   runtime.context->get_model_args(),
                                   runtime.worker.device(),
                                   runtime.worker.options_);

    if (FLAGS_enable_eplb) {
      runtime.eplb_executor = std::make_unique<EplbExecutor>(
          runtime.model.get(), runtime.worker.device());
    }

    work_pipelines_.emplace_back(create_pipeline(pipeline_type, runtime));
    index_queue_.enqueue(i);
  }

  model_.reset(work_pipelines_[0]->runtime().model.get());
  model_executor_.reset(work_pipelines_[0]->runtime().executor.get());

  if (FLAGS_enable_beam_search_kernel) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }

  if (FLAGS_enable_eplb) {
    eplb_executor_.reset(work_pipelines_[0]->runtime().eplb_executor.get());
  }

  LOG(INFO) << "Created " << work_pipelines_.size()
            << " pipelines for concurrent execution";
  return true;
}

void RecWorkerImpl::load_model(std::unique_ptr<ModelLoader> loader) {
  CHECK(!work_pipelines_.empty())
      << "Model instances are not initialized. Call init_model() first.";

  std::string model_weights_path = loader->model_weights_path();

  work_pipelines_[0]->runtime().model->load_model(std::move(loader));
  LOG(INFO) << "Loaded weights for model instance 0";

  for (size_t i = 1; i < work_pipelines_.size(); ++i) {
    auto model_loader = ModelLoader::create(model_weights_path);
    CHECK(model_loader != nullptr)
        << "Failed to create ModelLoader for model instance " << i;
    work_pipelines_[i]->runtime().model->load_model(std::move(model_loader));
    LOG(INFO) << "Loaded weights for model instance " << i;
  }

  LOG(INFO) << "Loaded weights for all " << work_pipelines_.size() << " models";
}

bool RecWorkerImpl::init_onerec_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";
  device_.set_device();

  model_ = create_rec_model(context);
  CHECK(model_ != nullptr) << "Failed to create rec model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  if (FLAGS_enable_eplb) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }
  return true;
}

ForwardInput RecWorkerImpl::prepare_inputs(Batch& batch) {
  CHECK(!work_pipelines_.empty()) << "RecWorkerImpl is not initialized.";
  return work_pipelines_[0]->prepare_inputs(batch);
}

void RecWorkerImpl::prepare_work_before_execute(
    const ForwardInput& inputs,
    ForwardInput& processed_inputs) {
  LOG(FATAL)
      << "RecWorkerImpl::prepare_work_before_execute should not be called.";
}

void RecWorkerImpl::prepare_multi_modal_data(ForwardInput& processed_inputs) {
  if (!processed_inputs.input_params.mm_data.valid()) {
    return;
  }

  torch::Tensor multi_modal_values;
  torch::Tensor multi_modal_indices;

  const auto& processed_mm_data = processed_inputs.input_params.mm_data;
  if (auto res = processed_mm_data.get<torch::Tensor>("MULTI_MODAL_VALUES")) {
    multi_modal_values = res.value();
  }

  if (auto res = processed_mm_data.get<torch::Tensor>("MULTI_MODAL_INDICES")) {
    multi_modal_indices = res.value();
  }

  if (!multi_modal_values.defined() || !multi_modal_indices.defined()) {
    return;
  }

#if defined(USE_NPU)
  layer::NpuWordEmbedding npu_word_embedding = get_npu_word_embedding();
  torch::Tensor input_tokens_embedding =
      npu_word_embedding(processed_inputs.token_ids, 0);
#else
  layer::WordEmbedding word_embedding = get_word_embedding();
  torch::Tensor input_tokens_embedding =
      word_embedding->forward(processed_inputs.token_ids);
#endif

  std::vector<torch::indexing::TensorIndex> indices = {
      torch::indexing::TensorIndex(multi_modal_indices),
      torch::indexing::Slice()};

  input_tokens_embedding.index_put_(indices, multi_modal_values);
  processed_inputs.input_params.input_embedding = input_tokens_embedding;
}

std::optional<ForwardOutput> RecWorkerImpl::step(const ForwardInput& input) {
  LOG(FATAL) << "RecWorkerImpl::step should not be called.";
  return std::nullopt;
}

folly::SemiFuture<std::optional<ForwardOutput>> RecWorkerImpl::step_async(
    const ForwardInput& input) {
  folly::Promise<std::optional<ForwardOutput>> promise;

  size_t index;
  index_queue_.wait_dequeue(index);
  auto future = promise.getSemiFuture();

  step_threadpool_->schedule_with_tid(
      [this, &input, index, promise = std::move(promise)]() mutable {
        auto stream_guard =
            work_pipelines_[index]->runtime().stream->set_stream_guard();

        ForwardInput input_on_device;
        work_pipelines_[index]->prepare_work_before_execute(input,
                                                            input_on_device);

        if (hierarchy_kv_cache_transfer_ != nullptr) {
          hierarchy_kv_cache_transfer_->set_layer_synchronizer(
              input_on_device.input_params);
        }

        const auto output = work_pipelines_[index]->step(input_on_device);
        promise.setValue(output);

        index_queue_.enqueue(index);
      },
      index);

  return future;
}

std::unique_ptr<RecWorkerImpl::RecWorkPipeline> RecWorkerImpl::create_pipeline(
    RecPipelineType type,
    RecPipelineRuntime& runtime) {
  switch (type) {
    case RecPipelineType::kLlmRecDefault:
      return std::make_unique<LlmRecWorkPipeline>(runtime);
    case RecPipelineType::kOneRecDefault:
      return std::make_unique<OneRecWorkPipeline>(runtime);
    case RecPipelineType::kLlmRecMultiRoundPipeline:
      return std::make_unique<LlmRecMultiRoundPipeline>(runtime);
    case RecPipelineType::kLLaDARecWorkerLoop:
      return std::make_unique<LLaDARecWorkPipeline>(runtime);
    default:
      LOG(FATAL) << "Unknown RecWorkerImpl pipeline type: "
                 << static_cast<int>(type);
      return nullptr;
  }
}

torch::Tensor RecWorkerImpl::merge_embeddings_by_indices(
    const torch::Tensor& input_tokens_embedding,
    const torch::Tensor& input_embedding,
    const std::vector<int64_t>& input_indices) {
  CHECK_EQ(input_embedding.dim(), 2);
  CHECK_EQ(input_tokens_embedding.dim(), 2);
  CHECK_EQ(input_tokens_embedding.size(1), input_embedding.size(1));
  CHECK_EQ(input_tokens_embedding.dtype(), input_embedding.dtype());
  CHECK_EQ(input_tokens_embedding.device(), input_embedding.device());

  const int64_t total_rows =
      input_tokens_embedding.size(0) + input_embedding.size(0);
  const int64_t cols = input_embedding.size(1);

  torch::Device device = input_embedding.device();
  torch::Tensor merged = torch::empty(
      {total_rows, cols}, torch::dtype(input_embedding.dtype()).device(device));

  std::vector<int64_t> input_embedding_indices;
  for (int64_t i = 0; i < total_rows; ++i) {
    if (std::find(input_indices.begin(), input_indices.end(), i) ==
        input_indices.end()) {
      input_embedding_indices.push_back(i);
    }
  }

  CHECK_EQ(input_embedding_indices.size(), input_embedding.size(0));

  torch::Tensor input_embedding_indices_tensor =
      torch::tensor(input_embedding_indices, torch::kInt64).to(device);
  merged.index_put_({input_embedding_indices_tensor, torch::indexing::Ellipsis},
                    input_embedding);

  torch::Tensor input_indices_tensor =
      torch::tensor(input_indices, torch::kInt64).to(device);
  merged.index_put_({input_indices_tensor, torch::indexing::Ellipsis},
                    input_tokens_embedding);

  return merged;
}

}  // namespace xllm
