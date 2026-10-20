#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "nanoinfer/sampling.hpp"

namespace nanoinfer {
namespace {
TEST(Sampling, GreedyChoosesLowestIdOnTies) {
  Sampler sampler;
  const std::array<float, 4> logits{-10000, 3, 3, -5};
  EXPECT_EQ(sampler.sample(logits), 1U);
}

TEST(Sampling, SameSeedReproducesSequenceAndDifferentSeedChangesIt) {
  Sampler first({0.8, 0, 1}, 1234), second({0.8, 0, 1}, 1234), other({0.8, 0, 1}, 5678);
  const std::array<float, 7> logits{0, 1, 2, 0.5F, 1.5F, -1, 2};
  std::vector<std::uint32_t> a, b;
  for (int i = 0; i < 200; ++i) {
    const auto token = first.sample(logits);
    EXPECT_EQ(token, second.sample(logits));
    a.push_back(token);
    b.push_back(other.sample(logits));
  }
  EXPECT_NE(a, b);
}

TEST(Sampling, TopKAndNucleusKeepOnlyAllowedSupportIncludingCrossingToken) {
  const std::array<float, 4> logits{0, 0, 0, 0};
  Sampler top_k({1, 2, 1}, 1234), nucleus({1, 0, 0.3}, 1234);
  std::array<bool, 2> seen{};
  for (int i = 0; i < 200; ++i) {
    EXPECT_LT(top_k.sample(logits), 2U);
    const auto token = nucleus.sample(logits);
    ASSERT_LT(token, 2U);  // Uniform mass .25 needs two tokens to reach .3.
    seen[token] = true;
  }
  EXPECT_TRUE(seen[0] && seen[1]);
  Sampler combined({1, 2, 0.4}, 1234);  // Top-k renormalizes to .5 per token.
  for (int i = 0; i < 10; ++i) EXPECT_EQ(combined.sample(logits), 0U);
  Sampler clamped({1, 999, 1}, 1234), unfiltered({1, 0, 1}, 1234);
  for (int i = 0; i < 20; ++i) EXPECT_EQ(clamped.sample(logits), unfiltered.sample(logits));
}

TEST(Sampling, StableTemperatureScalingChangesDistribution) {
  const std::array<float, 2> logits{10000, 9998};
  Sampler cold({0.01, 0, 1}, 42), hot({1000, 0, 1}, 42);
  int second_count = 0;
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(cold.sample(logits), 0U);
    if (hot.sample(logits) == 1) ++second_count;
  }
  EXPECT_GT(second_count, 400);
  EXPECT_LT(second_count, 600);
  Sampler tiny({std::numeric_limits<double>::min(), 0, 1}, 42);
  const std::array<float, 2> extremes{std::numeric_limits<float>::max(),
                                      -std::numeric_limits<float>::max()};
  EXPECT_EQ(tiny.sample(extremes), 0U);
}

TEST(Sampling, RejectsInvalidOptionsAndLogitsWithoutAdvancingRandomState) {
  EXPECT_THROW(Sampler(SamplingOptions{-1, 0, 1}), std::invalid_argument);
  EXPECT_THROW(Sampler(SamplingOptions{1, 0, 0}), std::invalid_argument);
  EXPECT_THROW(Sampler(SamplingOptions{1, 0, 1.1}), std::invalid_argument);
  EXPECT_THROW(Sampler(SamplingOptions{std::numeric_limits<double>::infinity(), 0, 1}),
               std::invalid_argument);
  Sampler sampler({1, 0, 1}, 1234), control({1, 0, 1}, 1234);
  EXPECT_THROW(static_cast<void>(sampler.sample({})), std::invalid_argument);
  const std::array<float, 1> invalid{std::numeric_limits<float>::quiet_NaN()};
  EXPECT_THROW(static_cast<void>(sampler.sample(invalid)), std::invalid_argument);
  const std::array<float, 2> logits{1, 2};
  EXPECT_EQ(sampler.sample(logits), control.sample(logits));
}
}  // namespace
}  // namespace nanoinfer
