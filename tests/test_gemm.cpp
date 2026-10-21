#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "nanoinfer/gemm.hpp"
#include "nanoinfer/ref_ops.hpp"

namespace nanoinfer::test {
extern std::uint32_t random_seed;
}

namespace nanoinfer {
namespace {
using kernels::GemmStep;

constexpr std::array<GemmStep, 5> kSteps{GemmStep::naive, GemmStep::reordered, GemmStep::blocked,
                                         GemmStep::packed, GemmStep::avx2};

struct Shape {
  std::size_t m, n, k;
};

std::string label(GemmStep step, const Shape& shape, bool transpose_b) {
  return std::string(kernels::name(step)) + " m=" + std::to_string(shape.m) +
         " n=" + std::to_string(shape.n) + " k=" + std::to_string(shape.k) +
         (transpose_b ? " transposed" : "");
}

// Compares one kernel with the scalar reference. The error of every output is
// measured relative to sum_p |a_ip| |b_pj|, the magnitude its rounding error
// scales with, so cancellation near zero cannot hide or fake an error. The
// whole matrix must also agree in relative Frobenius norm.
void check(GemmStep step, const Shape& shape, bool transpose_b, std::mt19937& generator) {
  SCOPED_TRACE(label(step, shape, transpose_b));
  std::uniform_real_distribution<float> distribution(-1, 1);
  const auto [m, n, k] = shape;
  std::vector<float> a(m * k), b(k * n);
  for (auto& value : a) value = distribution(generator);
  for (auto& value : b) value = distribution(generator);
  std::vector<float> expected(m * n), actual(m * n, std::numeric_limits<float>::quiet_NaN());
  ops::ref::matmul(a, b, m, n, k, expected, transpose_b);
  kernels::gemm(step, a, b, m, n, k, actual, transpose_b);
  double error_squared = 0, norm_squared = 0, worst = 0;
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double magnitude = 0;
      for (std::size_t p = 0; p < k; ++p) {
        const double right = transpose_b ? b[j * k + p] : b[p * n + j];
        magnitude += std::abs(static_cast<double>(a[i * k + p]) * right);
      }
      const double value = actual[i * n + j];
      ASSERT_TRUE(std::isfinite(value)) << "output not written at " << i << "," << j;
      const double difference = std::abs(value - expected[i * n + j]);
      worst = std::max(worst, magnitude > 0 ? difference / magnitude : difference);
      error_squared += difference * difference;
      norm_squared += static_cast<double>(expected[i * n + j]) * expected[i * n + j];
    }
  }
  EXPECT_LT(worst, 1e-5);
  EXPECT_LT(std::sqrt(error_squared / std::max(norm_squared, 1e-300)), 1e-5);
}

TEST(Gemm, EveryStepMatchesReferenceOnAwkwardAndBlockCrossingShapes) {
  std::mt19937 generator(test::random_seed);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  // Awkward sizes from the spec, register-tile edges (6 x 16) and shapes that
  // cross the KC = 256, MC = 144 and NC = 3072 cache blocks.
  const std::vector<Shape> shapes{
      {1, 1, 1},    {1, 7, 63},    {7, 65, 1},     {63, 767, 65},   {65, 769, 767},
      {769, 63, 7}, {6, 16, 256},  {12, 32, 512},  {5, 15, 257},    {145, 17, 513},
      {7, 3089, 9}, {1, 769, 768}, {2, 2304, 768}, {145, 3089, 33},
  };
  for (const auto step : kSteps) {
    if (!kernels::available(step)) {
      GTEST_LOG_(INFO) << "skipping unavailable step " << kernels::name(step);
      continue;
    }
    for (const auto& shape : shapes) {
      for (const bool transpose_b : {false, true}) check(step, shape, transpose_b, generator);
    }
  }
}

TEST(Gemm, RandomShapesForEveryStep) {
  std::mt19937 generator(test::random_seed + 1);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  std::uniform_int_distribution<std::size_t> size(1, 200);
  for (int trial = 0; trial < 12; ++trial) {
    const Shape shape{size(generator), size(generator), size(generator)};
    for (const auto step : kSteps) {
      if (kernels::available(step)) check(step, shape, trial % 2 == 1, generator);
    }
  }
}

TEST(Gemm, LongInnerDimensionFromTheMlpProjection) {
  std::mt19937 generator(test::random_seed + 2);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  for (const auto step : {GemmStep::packed, GemmStep::avx2}) {
    if (kernels::available(step)) check(step, {9, 33, 3072}, false, generator);
  }
}

TEST(Gemm, OverwritesOutputAndHandlesEmptyRows) {
  for (const auto step : kSteps) {
    if (!kernels::available(step)) continue;
    SCOPED_TRACE(std::string(kernels::name(step)));
    const std::vector<float> a{1, 2, 3, 4}, b{1, 0, 0, 1};
    std::vector<float> c(4, 99.0F);
    kernels::gemm(step, a, b, 2, 2, 2, c);
    EXPECT_EQ(c, (std::vector<float>{1, 2, 3, 4}));
    kernels::gemm(step, {}, b, 0, 2, 2, {});
  }
}

TEST(Gemm, RejectsShapeMismatches) {
  std::vector<float> values(6), output(4);
  EXPECT_THROW(kernels::gemm(GemmStep::packed, values, values, 2, 2, 2, output),
               std::invalid_argument);
  EXPECT_THROW(kernels::gemm(GemmStep::packed, values, values, 2, 3, 0, output),
               std::invalid_argument);
}

TEST(Gemm, DispatchPrefersAvx2OnlyWhenTheCpuSupportsIt) {
  const auto& features = kernels::cpu_features();
  GTEST_LOG_(INFO) << "avx2=" << features.avx2 << " fma=" << features.fma
                   << " avx512f=" << features.avx512f
                   << " best=" << kernels::name(kernels::best_step());
  EXPECT_TRUE(kernels::available(kernels::best_step()));
  EXPECT_TRUE(kernels::available(GemmStep::packed));
  if (!(features.avx2 && features.fma)) {
    EXPECT_FALSE(kernels::available(GemmStep::avx2));
    EXPECT_EQ(kernels::best_step(), GemmStep::packed);
    std::vector<float> one{1};
    EXPECT_THROW(kernels::gemm(GemmStep::avx2, one, one, 1, 1, 1, one), std::runtime_error);
  }
}
}  // namespace
}  // namespace nanoinfer
