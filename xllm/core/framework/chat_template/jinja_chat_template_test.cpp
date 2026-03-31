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

#include "jinja_chat_template.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>

#include "core/util/json_reader.h"

namespace xllm {

namespace {

std::filesystem::path find_llada_model_dir() {
  auto current = std::filesystem::current_path();
  for (int i = 0; i < 8; ++i) {
    auto candidate = current / "models" / "LLaDA2.1-mini";
    if (std::filesystem::exists(candidate / "tokenizer_config.json")) {
      return candidate;
    }
    if (current == current.root_path()) {
      break;
    }
    current = current.parent_path();
  }
  return {};
}

std::string normalize_ws(std::string text) {
  text.erase(std::remove_if(text.begin(),
                            text.end(),
                            [](unsigned char ch) {
                              return ch == ' ' || ch == '\n' || ch == '\r' ||
                                     ch == '\t';
                            }),
             text.end());
  return text;
}

TokenizerArgs make_llada_tokenizer_args() {
  const auto model_dir = find_llada_model_dir();
  if (model_dir.empty()) {
    return {};
  }

  JsonReader reader;
  const auto tokenizer_config_path = model_dir / "tokenizer_config.json";
  EXPECT_TRUE(reader.parse(tokenizer_config_path.string()));

  TokenizerArgs args;
  if (auto v = reader.value<std::string>("chat_template")) {
    args.chat_template() = v.value();
  }
  if (auto v = reader.value<std::string>("bos_token")) {
    args.bos_token() = v.value();
  }
  if (auto v = reader.value<std::string>("eos_token")) {
    args.eos_token() = v.value();
  }
  return args;
}

}  // namespace

class TestableJinjaChatTemplate : public JinjaChatTemplate {
 public:
  TestableJinjaChatTemplate(const TokenizerArgs& args)
      : JinjaChatTemplate(args) {}

  using JinjaChatTemplate::apply;
};

TEST(JinjaChatTemplate, OpenChatModel) {
  // clang-format off
  const std::string template_str =
      "<s>"
      "{% for message in messages %}"
        "{{ 'GPT4 Correct ' + message['role'] + ': ' + message['content'] + '<|end_of_turn|>'}}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "you are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "what i can do for you?"}},
      {{"role", "user"}, {"content", "how are you?"}}};
  const std::string expected =
    "<s>"
    "GPT4 Correct system: you are a helpful assistant.<|end_of_turn|>"
    "GPT4 Correct user: hi<|end_of_turn|>"
    "GPT4 Correct assistant: what i can do for you?<|end_of_turn|>"
    "GPT4 Correct user: how are you?<|end_of_turn|>"
    "GPT4 Correct Assistant:";
  // clang-format on

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("<|end_of_turn|>");
  TestableJinjaChatTemplate template_(args);
  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), expected);
}

TEST(JinjaChatTemplate, LLaDATemplateSystemAndUser) {
  const auto model_dir = find_llada_model_dir();
  if (model_dir.empty()) {
    GTEST_SKIP() << "LLaDA2.1-mini local model directory not found";
  }

  TokenizerArgs args = make_llada_tokenizer_args();
  TestableJinjaChatTemplate template_(args);
  const ChatMessages messages = {
      Message("system", "sys"),
      Message("user", "u1"),
  };

  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  const std::string expected =
      "<role>SYSTEM</role>\n"
      "sys\n"
      "detailed thinking off<|role_end|>"
      "<role>HUMAN</role>u1<|role_end|>"
      "<role>ASSISTANT</role>";
  EXPECT_EQ(normalize_ws(result.value()), normalize_ws(expected));
}

TEST(JinjaChatTemplate, LLaDATemplateWithAssistantHistory) {
  const auto model_dir = find_llada_model_dir();
  if (model_dir.empty()) {
    GTEST_SKIP() << "LLaDA2.1-mini local model directory not found";
  }

  TokenizerArgs args = make_llada_tokenizer_args();
  TestableJinjaChatTemplate template_(args);
  const ChatMessages messages = {
      Message("system", "sys"),
      Message("user", "u1"),
      Message("assistant", "a1"),
      Message("user", "u2"),
  };

  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  const std::string expected =
      "<role>SYSTEM</role>\n"
      "sys\n"
      "detailed thinking off<|role_end|>"
      "<role>HUMAN</role>u1<|role_end|>"
      "<role>ASSISTANT</role>a1<|role_end|>"
      "<role>HUMAN</role>u2<|role_end|>"
      "<role>ASSISTANT</role>";
  EXPECT_EQ(normalize_ws(result.value()), normalize_ws(expected));
}

}  // namespace xllm
