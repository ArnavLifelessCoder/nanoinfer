#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <random>
#include <vector>

#include "nanoinfer/cached.hpp"

namespace nanoinfer::test {
extern std::uint32_t random_seed;
}

namespace nanoinfer {
namespace {
const auto model_path = std::filesystem::path("fixtures") / "tiny.bin";

void check_row(std::span<const float> actual, std::span<const float> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    ASSERT_EQ(actual[i], expected[i]) << "index=" << i;
}

TEST(CachedForward, EveryPositionMatchesFullRecomputeIncludingAllHiddenStates) {
  const ReferenceTransformer reference(model_path);
  CachedTransformer cached(model_path);
  std::mt19937 generator(test::random_seed);
  std::uniform_int_distribution<std::uint32_t> distribution(0, reference.config().vocab - 1);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  std::vector<std::uint32_t> tokens(reference.config().ctx);
  for (auto& token : tokens) token = distribution(generator);
  const auto width = reference.config().d_model;
  const auto vocab = reference.config().vocab;
  for (const std::size_t prefix : {1, 7, 31, 63, 64}) {
    const auto n = std::min(prefix, tokens.size());
    SCOPED_TRACE("prefill=" + std::to_string(n));
    auto full = reference.forward(std::span(tokens).first(n));
    auto actual = cached.prefill(std::span(tokens).first(n), true);
    EXPECT_EQ(actual.logits, full.logits);
    EXPECT_EQ(actual.hidden_states, full.hidden_states);
    EXPECT_EQ(actual.final_ln, full.final_ln);
    EXPECT_EQ(cached.position(), n);
    for (std::size_t position = n; position < tokens.size(); ++position) {
      SCOPED_TRACE("absolute_position=" + std::to_string(position));
      actual = cached.decode(tokens[position], true);
      full = reference.forward(std::span(tokens).first(position + 1));
      check_row(actual.logits, std::span(full.logits).last(vocab));
      check_row(actual.final_ln, std::span(full.final_ln).last(width));
      for (std::size_t layer = 0; layer < full.hidden_states.size(); ++layer) {
        check_row(actual.hidden_states[layer], std::span(full.hidden_states[layer]).last(width));
      }
      EXPECT_EQ(cached.position(), position + 1);
    }
  }
}

TEST(CachedForward, ResetShorterPrefillIndependentInstancesAndInvalidCalls) {
  CachedTransformer first(model_path), second(model_path);
  const std::vector<std::uint32_t> prompt{1, 2, 3, 4, 5, 6, 7};
  EXPECT_THROW(static_cast<void>(first.decode(1)), std::invalid_argument);
  static_cast<void>(first.prefill(prompt));
  EXPECT_THROW(static_cast<void>(first.prefill({})), std::invalid_argument);
  EXPECT_EQ(first.position(), prompt.size());
  EXPECT_THROW(static_cast<void>(first.decode(first.config().vocab)), std::out_of_range);
  EXPECT_EQ(first.position(), prompt.size());
  const auto a = first.decode(8);
  static_cast<void>(second.prefill(prompt));
  EXPECT_EQ(a.logits, second.decode(8).logits);
  const auto short_prompt = std::span(prompt).first(1);
  EXPECT_EQ(first.prefill(short_prompt).logits, second.prefill(short_prompt).logits);
  first.reset();
  EXPECT_EQ(first.position(), 0U);
  EXPECT_THROW(static_cast<void>(first.decode(1)), std::invalid_argument);
  std::vector<std::uint32_t> full(first.config().ctx, 1);
  static_cast<void>(first.prefill(full));
  EXPECT_THROW(static_cast<void>(first.decode(1)), std::invalid_argument);
  full.push_back(1);
  EXPECT_THROW(static_cast<void>(first.prefill(full)), std::invalid_argument);
  EXPECT_EQ(first.position(), first.config().ctx);
}
}  // namespace
}  // namespace nanoinfer
