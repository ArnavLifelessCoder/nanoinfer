#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "nanoinfer/ref_ops.hpp"

namespace nanoinfer::test {
extern std::uint32_t random_seed;
}

namespace nanoinfer {
namespace {
namespace ref = ops::ref;

TEST(ReferenceOps, MatmulRandomAwkwardShapesAndTransposedStorage) {
  std::mt19937 generator(test::random_seed);
  std::uniform_real_distribution<float> distribution(-1, 1);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  const std::array<std::array<std::size_t, 3>, 10> shapes{{
      {1, 7, 63},
      {7, 1, 65},
      {3, 65, 7},
      {1, 767, 1},
      {1, 769, 7},
      {7, 3, 769},
      {63, 1, 3},
      {65, 1, 3},
      {767, 1, 1},
      {769, 1, 1},
  }};
  for (const auto& [m, n, k] : shapes) {
    SCOPED_TRACE(std::to_string(m) + "," + std::to_string(n) + "," + std::to_string(k));
    std::vector<float> a(m * k), b(k * n), transposed(n * k), actual(m * n), alternate(m * n);
    for (auto& value : a) value = distribution(generator);
    for (auto& value : b) value = distribution(generator);
    for (std::size_t i = 0; i < k; ++i) {
      for (std::size_t j = 0; j < n; ++j) transposed[j * k + i] = b[i * n + j];
    }
    ref::matmul(a, b, m, n, k, actual);
    ref::matmul(a, transposed, m, n, k, alternate, true);
    EXPECT_EQ(actual, alternate);
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        double expected = 0;
        for (std::size_t p = 0; p < k; ++p) {
          expected += static_cast<double>(a[i * k + p]) * b[p * n + j];
        }
        EXPECT_NEAR(actual[i * n + j], expected, 2e-5 * std::max(1.0, std::abs(expected)));
      }
    }
  }
}

TEST(ReferenceOps, EmbeddingsBiasAndResidual) {
  const std::vector<float> weights{1, 2, 3, 4, 5, 6};
  const std::array<std::uint32_t, 3> ids{2, 0, 2};
  std::vector<float> values(6);
  ref::embedding(ids, weights, 3, 2, values);
  EXPECT_EQ(values, (std::vector<float>{5, 6, 1, 2, 5, 6}));
  const std::vector<float> shifts{10, -1};
  ref::bias(values, shifts);
  ref::residual(values, weights);
  EXPECT_EQ(values, (std::vector<float>{16, 7, 14, 5, 20, 11}));
  const std::array<std::uint32_t, 1> bad{3};
  EXPECT_THROW(ref::embedding(bad, weights, 3, 2, std::span(values).first(2)), std::out_of_range);
}

TEST(ReferenceOps, LayerNormRandomWidthsPopulationVarianceAndEpsilon) {
  std::mt19937 generator(test::random_seed);
  std::uniform_real_distribution<float> distribution(-3, 3);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  for (const std::size_t width : {1, 7, 63, 65, 767, 769}) {
    SCOPED_TRACE(width);
    std::vector<float> input(width * 3), scale(width), shift(width), output(input.size());
    for (auto& value : input) value = distribution(generator);
    for (auto& value : scale) value = distribution(generator);
    for (auto& value : shift) value = distribution(generator);
    std::fill_n(input.begin(), width, 2.0F);
    ref::layer_norm(input, scale, shift, output);
    for (std::size_t row = 0; row < 3; ++row) {
      double mean = 0, variance = 0;
      for (std::size_t j = 0; j < width; ++j) mean += input[row * width + j];
      mean /= static_cast<double>(width);
      for (std::size_t j = 0; j < width; ++j) {
        variance += std::pow(input[row * width + j] - mean, 2);
      }
      variance /= static_cast<double>(width);
      for (std::size_t j = 0; j < width; ++j) {
        const double expected =
            (input[row * width + j] - mean) / std::sqrt(variance + 1e-5) * scale[j] + shift[j];
        EXPECT_NEAR(output[row * width + j], expected, 1e-5);
      }
    }
  }
  std::vector<float> input{0, 0.001F}, scale(2, 1), shift(2), output(2);
  ref::layer_norm(input, scale, shift, output);
  EXPECT_NEAR(output[1], 0.0005 / std::sqrt(0.00000025 + 1e-5), 1e-7);
}

TEST(ReferenceOps, GeluUsesTanhApproximation) {
  std::vector<float> values{-10, -3, -1, 0, 1, 3, 10};
  const auto input = values;
  ref::gelu(values);
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double x = input[i];
    const double expected =
        0.5 * x * (1 + std::tanh(std::sqrt(2 / std::acos(-1.0)) * (x + 0.044715 * x * x * x)));
    EXPECT_NEAR(values[i], expected, 1e-6);
  }
  const double exact = 0.5 * 3 * (1 + std::erf(3 / std::sqrt(2.0)));
  EXPECT_GT(std::abs(values[5] - exact), 1e-4);
}

TEST(ReferenceOps, StableSoftmaxWithLargeAndNegativeValues) {
  for (const float offset : {-10000.0F, 0.0F, 10000.0F}) {
    std::vector<float> values{offset - 2, offset, offset - 1};
    ref::softmax(values);
    const double denominator = 1 + std::exp(-1.0) + std::exp(-2.0);
    EXPECT_NEAR(values[0], std::exp(-2.0) / denominator, 1e-7);
    EXPECT_NEAR(values[1], 1 / denominator, 1e-7);
    EXPECT_NEAR(values[2], std::exp(-1.0) / denominator, 1e-7);
  }
}

TEST(ReferenceOps, GeluRoundsTheCubicTermBeforeScaling) {
  for (int index = -4000; index <= 4000; ++index) {
    const float x = static_cast<float>(index) / 1000.0F;
    const float cube = x * x * x;
    const float expected =
        0.5F * x * (1.0F + std::tanh(0.7978845608028654F * (x + 0.044715F * cube)));
    std::array<float, 1> actual{x};
    ref::gelu(actual);
    ASSERT_EQ(actual[0], expected) << "x=" << x;
  }
}

TEST(ReferenceOps, AttentionIncludesDiagonalAndKeepsHeadsSeparate) {
  // Zero Q/K gives a uniform average of V at positions [0, query], per head.
  std::vector<float> qkv(3 * 3 * 4), output(3 * 4);
  for (std::size_t t = 0; t < 3; ++t) {
    for (std::size_t d = 0; d < 4; ++d) {
      qkv[t * 12 + 8 + d] = static_cast<float>(t * 10 + d);
    }
  }
  ref::causal_attention(qkv, 3, 4, 2, output);
  for (std::size_t t = 0; t < 3; ++t) {
    for (std::size_t d = 0; d < 4; ++d) {
      EXPECT_NEAR(output[t * 4 + d], static_cast<float>(t * 5 + d), 1e-6);
    }
  }
  auto changed = qkv;
  std::fill(changed.begin() + 24, changed.end(), 1000.0F);
  std::vector<float> alternate(12);
  ref::causal_attention(changed, 3, 4, 2, alternate);
  EXPECT_TRUE(std::equal(output.begin(), output.begin() + 8, alternate.begin()));
}

TEST(ReferenceOps, InvalidShapesFailBeforeIndexing) {
  std::vector<float> values(7), weights(7);
  EXPECT_THROW(ref::matmul(values, weights, 1, 7, 7, values), std::invalid_argument);
  EXPECT_THROW(ref::layer_norm(values, weights, {}, values), std::invalid_argument);
  EXPECT_THROW(ref::bias(values, {}), std::invalid_argument);
  EXPECT_THROW(ref::residual(values, {}), std::invalid_argument);
  EXPECT_THROW(ref::softmax({}), std::invalid_argument);
  EXPECT_THROW(ref::causal_attention(values, 1, 7, 2, values), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(ref::elements(std::numeric_limits<std::size_t>::max(), 2)),
               std::overflow_error);
}
}  // namespace
}  // namespace nanoinfer
