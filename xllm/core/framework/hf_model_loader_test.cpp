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

#include "hf_model_loader.h"

#include <gtest/gtest.h>

#include <filesystem>

#include "core/common/rec_model_utils.h"
#include "core/platform/device.h"
#if defined(USE_NPU)
#include "models/model_registry.h"
#endif

namespace xllm {

namespace {

std::filesystem::path find_llada_model_dir() {
  auto current = std::filesystem::current_path();
  for (int i = 0; i < 8; ++i) {
    auto candidate = current / "models" / "LLaDA2.1-mini";
    if (std::filesystem::exists(candidate / "config.json") &&
        std::filesystem::exists(candidate / "tokenizer_config.json")) {
      return candidate;
    }
    if (current == current.root_path()) {
      break;
    }
    current = current.parent_path();
  }
  return {};
}

}  // namespace

TEST(HFModelLoaderTest, LoadCompressedTensorsFp8StaticConfig) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "quantization_config": {
        "config_groups": {
          "group_0": {
            "input_activations": {
              "dynamic": false,
              "num_bits": 8,
              "type": "float"
            },
            "weights": {
              "num_bits": 8,
              "type": "float"
            }
          }
        },
        "quant_method": "compressed-tensors"
      }
    }
  )json"));

  QuantArgs quant_args;
  if (Device::type_str() == "cuda") {
    ASSERT_TRUE(load_quant_cfg(reader, quant_args));
    EXPECT_EQ(quant_args.quant_method(), kQuantMethodFp8);
    EXPECT_EQ(quant_args.bits(), 8);
    EXPECT_EQ(quant_args.moe_weight_bits(), 8);
    EXPECT_FALSE(quant_args.activation_dynamic());
  }
}

TEST(HFModelLoaderTest, KeepLegacyFp8ConfigUnchanged) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "quantization_config": {
        "activation_scheme": "static",
        "quant_method": "fp8"
      }
    }
  )json"));

  QuantArgs quant_args;
  ASSERT_TRUE(load_quant_cfg(reader, quant_args));
  EXPECT_EQ(quant_args.quant_method(), kQuantMethodFp8);
  EXPECT_FALSE(quant_args.activation_dynamic());
}

#if defined(USE_NPU)
TEST(HFModelLoaderTest, Qwen35MtpModelArgsFromDenseConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_mtp");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5",
      "text_config": {
        "mtp_num_hidden_layers": 1,
        "layer_types": ["linear_attention"]
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_mtp");
  EXPECT_EQ(args.num_nextn_predict_layers(), 1);
  EXPECT_EQ(args.n_layers(), 1);
  ASSERT_EQ(args.layer_types().size(), 1);
  EXPECT_EQ(args.layer_types()[0], "full_attention");
}

TEST(HFModelLoaderTest, Qwen35MtpModelArgsFromMoeConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_moe_mtp");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5_moe",
      "text_config": {
        "mtp_num_hidden_layers": 2,
        "layer_types": ["linear_attention", "linear_attention"]
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_moe_mtp");
  EXPECT_EQ(args.num_nextn_predict_layers(), 2);
  EXPECT_EQ(args.n_layers(), 2);
  ASSERT_EQ(args.layer_types().size(), 2);
  EXPECT_EQ(args.layer_types()[0], "full_attention");
  EXPECT_EQ(args.layer_types()[1], "full_attention");
}

TEST(HFModelLoaderTest, LLaDAResolveModelRegistration) {
  std::string effective_backend;
  std::string resolved_name;
  std::string error_message;

  ASSERT_TRUE(resolve_model_registration("llada2_moe",
                                         "AUTO",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "llada2_moe");

  ASSERT_TRUE(resolve_model_registration("llada2_moe",
                                         "TORCH",
                                         &effective_backend,
                                         &resolved_name,
                                         &error_message));
  EXPECT_EQ(effective_backend, "TORCH");
  EXPECT_EQ(resolved_name, "llada2_moe");

  EXPECT_FALSE(resolve_model_registration(
      "llada2_moe", "ATB", &effective_backend, &resolved_name, &error_message));
  EXPECT_NE(error_message.find("only supports --npu_kernel_backend=TORCH"),
            std::string::npos);
}

TEST(HFModelLoaderTest, LLaDARecModelKindAndPipelineType) {
  EXPECT_EQ(get_rec_model_kind("llada2_moe"), RecModelKind::kLLaDARec);
  EXPECT_EQ(get_rec_pipeline_type(RecModelKind::kLLaDARec),
            RecPipelineType::kLLaDARecWorkerLoop);
  EXPECT_EQ(ModelRegistry::get_model_backend("llada2_moe"), "rec");
}

TEST(HFModelLoaderTest, LLaDARuntimeConfigDefaultsAreValid) {
  LLaDARuntimeConfig config = get_llada_runtime_config();
  std::string error_message;
  EXPECT_TRUE(validate_llada_runtime_config(config, -1, &error_message))
      << error_message;
}

TEST(HFModelLoaderTest, LLaDARuntimeConfigRejectsInvalidRanges) {
  LLaDARuntimeConfig config = get_llada_runtime_config();
  std::string error_message;

  config.block_length = 0;
  EXPECT_FALSE(validate_llada_runtime_config(config, -1, &error_message));

  config = get_llada_runtime_config();
  config.threshold = 1.5;
  EXPECT_FALSE(validate_llada_runtime_config(config, -1, &error_message));

  config = get_llada_runtime_config();
  config.mask_id = 42;
  EXPECT_FALSE(validate_llada_runtime_config(config, 42, &error_message));
}

TEST(HFModelLoaderTest, LLaDAModelArgsLoader) {
  auto loader = ModelRegistry::get_model_args_loader("llada2_moe");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "dtype": "bfloat16",
      "first_k_dense_replace": 1,
      "head_dim": 128,
      "hidden_act": "silu",
      "hidden_size": 2048,
      "initializer_range": 0.02,
      "intermediate_size": 5120,
      "max_position_embeddings": 32768,
      "model_type": "llada2_moe",
      "moe_intermediate_size": 512,
      "moe_router_enable_expert_bias": true,
      "n_group": 8,
      "use_qk_norm": true,
      "use_qkv_bias": false,
      "norm_topk_prob": true,
      "num_attention_heads": 16,
      "num_experts": 256,
      "num_experts_per_tok": 8,
      "num_hidden_layers": 20,
      "num_key_value_heads": 4,
      "num_shared_experts": 1,
      "output_router_logits": false,
      "pad_token_id": 156892,
      "partial_rotary_factor": 0.5,
      "rms_norm_eps": 1e-06,
      "rope_theta": 600000,
      "routed_scaling_factor": 2.5,
      "score_function": "sigmoid",
      "tie_word_embeddings": false,
      "topk_group": 4,
      "vocab_size": 157184
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "llada2_moe");
  EXPECT_EQ(args.dtype(), "bfloat16");
  EXPECT_EQ(args.hidden_size(), 2048);
  EXPECT_EQ(args.intermediate_size(), 5120);
  EXPECT_EQ(args.n_layers(), 20);
  EXPECT_EQ(args.n_heads(), 16);
  ASSERT_TRUE(args.n_kv_heads().has_value());
  EXPECT_EQ(args.n_kv_heads().value(), 4);
  EXPECT_EQ(args.head_dim(), 128);
  EXPECT_FLOAT_EQ(args.rms_norm_eps(), 1e-6f);
  EXPECT_EQ(args.max_position_embeddings(), 32768);
  EXPECT_FLOAT_EQ(args.rope_theta(), 600000.0f);
  EXPECT_FLOAT_EQ(args.partial_rotary_factor(), 0.5f);
  EXPECT_TRUE(args.use_qk_norm());
  EXPECT_TRUE(args.moe_router_enable_expert_bias());
  EXPECT_EQ(args.first_k_dense_replace(), 1);
  EXPECT_EQ(args.num_experts(), 256);
  EXPECT_EQ(args.n_routed_experts(), 256);
  EXPECT_EQ(args.num_experts_per_tok(), 8);
  EXPECT_EQ(args.n_shared_experts(), 1);
  EXPECT_EQ(args.moe_intermediate_size(), 512);
  EXPECT_EQ(args.n_group(), 8);
  EXPECT_EQ(args.topk_group(), 4);
  EXPECT_FLOAT_EQ(args.routed_scaling_factor(), 2.5f);
  EXPECT_EQ(args.scoring_func(), "sigmoid");
  EXPECT_EQ(args.pad_token_id(), 156892);
  EXPECT_EQ(args.vocab_size(), 157184);
  EXPECT_TRUE(args.stop_token_ids().empty());
}

TEST(HFModelLoaderTest, LLaDAModelArgsLoaderDefaultsUseQkNorm) {
  auto loader = ModelRegistry::get_model_args_loader("llada2_moe");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "llada2_moe",
      "hidden_size": 2048,
      "intermediate_size": 5120,
      "num_hidden_layers": 20,
      "num_attention_heads": 16,
      "num_key_value_heads": 4,
      "head_dim": 128,
      "moe_intermediate_size": 512,
      "num_experts": 256,
      "num_experts_per_tok": 8,
      "num_shared_experts": 1,
      "n_group": 8,
      "topk_group": 4,
      "norm_head": false
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_TRUE(args.use_qk_norm());
}

TEST(HFModelLoaderTest, LLaDAModelArgsLoaderUsesTorchDtypeFallback) {
  auto loader = ModelRegistry::get_model_args_loader("llada2_moe");
  ASSERT_TRUE(loader != nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "llada2_moe",
      "torch_dtype": "float16",
      "hidden_size": 2048,
      "intermediate_size": 5120,
      "num_hidden_layers": 20,
      "num_attention_heads": 16,
      "num_key_value_heads": 4,
      "head_dim": 128,
      "moe_intermediate_size": 512,
      "num_experts": 256,
      "num_experts_per_tok": 8,
      "num_shared_experts": 1,
      "n_group": 8,
      "topk_group": 4
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.dtype(), "float16");
}

TEST(HFModelLoaderTest, LLaDARealModelDirectoryLoadsArgsAndTokenizer) {
  const auto model_dir = find_llada_model_dir();
  if (model_dir.empty()) {
    GTEST_SKIP() << "LLaDA2.1-mini local model directory not found";
  }

  HFModelLoader loader(model_dir.string());

  const ModelArgs& model_args = loader.model_args();
  EXPECT_EQ(model_args.model_type(), "llada2_moe");
  EXPECT_EQ(model_args.dtype(), "bfloat16");
  EXPECT_EQ(model_args.hidden_size(), 2048);
  EXPECT_EQ(model_args.n_layers(), 20);
  EXPECT_EQ(model_args.n_heads(), 16);
  ASSERT_TRUE(model_args.n_kv_heads().has_value());
  EXPECT_EQ(model_args.n_kv_heads().value(), 4);
  EXPECT_EQ(model_args.num_experts(), 256);
  EXPECT_EQ(model_args.num_experts_per_tok(), 8);
  EXPECT_EQ(model_args.n_shared_experts(), 1);
  EXPECT_TRUE(model_args.use_qk_norm());
  EXPECT_TRUE(model_args.moe_router_enable_expert_bias());
  EXPECT_EQ(model_args.scoring_func(), "sigmoid");

  const TokenizerArgs& tokenizer_args = loader.tokenizer_args();
  EXPECT_EQ(tokenizer_args.tokenizer_type(), "fast");
  EXPECT_EQ(tokenizer_args.tokenizer_class(), "PreTrainedTokenizerFast");
  EXPECT_FALSE(tokenizer_args.chat_template().empty());
  EXPECT_NE(tokenizer_args.chat_template().find("<role>SYSTEM</role>"),
            std::string::npos);
  EXPECT_NE(tokenizer_args.chat_template().find("add_generation_prompt"),
            std::string::npos);
  EXPECT_EQ(tokenizer_args.bos_token(), "<|startoftext|>");
  EXPECT_EQ(tokenizer_args.eos_token(), "<|endoftext|>");
}
#endif

}  // namespace xllm
