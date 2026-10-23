#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "nanoinfer/gemm.hpp"
#include "nanoinfer/gemv.hpp"
#include "nanoinfer/ref_ops.hpp"
#include "nanoinfer/thread_pool.hpp"

namespace nanoinfer::test {
extern std::uint32_t random_seed;
}

namespace nanoinfer {
namespace {
using kernels::GemmStep;
using kernels::GemvKernel;
using kernels::QuantizedMatrix;

// A random int8 matrix in the given layout, its scales, and its exact float32
// dequantization (float(q) * scale), which is what the reference multiplies.
struct Fixture {
  std::vector<std::int8_t> q;
  std::vector<float> scales, dequantized;
  QuantizedMatrix matrix() const { return {q, scales, group}; }
  std::size_t group;
};

Fixture make(std::size_t n, std::size_t k, std::size_t group, bool transpose_b,
             std::mt19937& generator) {
  std::uniform_int_distribution<int> value(-127, 127);
  std::uniform_real_distribution<float> scale(0.0F, 0.05F);
  Fixture f{std::vector<std::int8_t>(n * k), std::vector<float>(n * k / group),
            std::vector<float>(n * k), group};
  for (auto& q : f.q) q = static_cast<std::int8_t>(value(generator));
  for (auto& s : f.scales) s = scale(generator);
  for (std::size_t i = 0; i < n * k; ++i) {
    // transposed: [N, K] grouped along K (axis 1); otherwise [K, N] along K (axis 0).
    const auto row = transpose_b ? i / k : i / n, column = transpose_b ? i % k : i % n;
    const float s = transpose_b ? f.scales[row * (k / group) + column / group]
                                : f.scales[row / group * n + column];
    f.dequantized[i] = static_cast<float>(f.q[i]) * s;
  }
  return f;
}

std::vector<float> random_values(std::size_t count, std::mt19937& generator) {
  std::uniform_real_distribution<float> distribution(-1, 1);
  std::vector<float> values(count);
  for (auto& value : values) value = distribution(generator);
  return values;
}

void expect_close(const std::vector<float>& a, const Fixture& f, const std::vector<float>& actual,
                  const std::vector<float>& expected, std::size_t m, std::size_t n, std::size_t k,
                  bool transpose_b) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double magnitude = 0;
      for (std::size_t p = 0; p < k; ++p) {
        const double w = transpose_b ? f.dequantized[j * k + p] : f.dequantized[p * n + j];
        magnitude += std::abs(a[i * k + p] * w);
      }
      ASSERT_TRUE(std::isfinite(actual[i * n + j])) << "output not written at " << i << "," << j;
      EXPECT_LE(std::abs(static_cast<double>(actual[i * n + j]) - expected[i * n + j]),
                1e-5 * magnitude + 1e-30)
          << i << "," << j;
    }
  }
}

struct Case {
  std::size_t n, k, group;
};
const std::vector<Case> kCases{{1, 8, 8},     {7, 64, 32},    {65, 32, 32},    {767, 64, 16},
                               {769, 96, 32}, {33, 768, 32},  {2304, 768, 32}, {768, 3072, 32},
                               {17, 128, 64}, {1025, 96, 32}, {5, 48, 16},     {3, 40, 8}};

TEST(Quantized, GemvMatchesReferenceOnDequantizedWeights) {
  std::mt19937 generator(test::random_seed);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  for (const auto kernel : {GemvKernel::scalar, GemvKernel::avx2}) {
    if (!kernels::available(kernel)) continue;
    for (const auto& [n, k, group] : kCases) {
      for (const bool transpose_b : {false, true}) {
        SCOPED_TRACE("n=" + std::to_string(n) + " k=" + std::to_string(k) +
                     " group=" + std::to_string(group) + " t=" + std::to_string(transpose_b));
        const auto f = make(n, k, group, transpose_b, generator);
        const auto x = random_values(k, generator);
        std::vector<float> expected(n), actual(n, std::numeric_limits<float>::quiet_NaN());
        ops::ref::matmul(x, f.dequantized, 1, n, k, expected, transpose_b);
        kernels::gemv(kernel, x, f.matrix(), n, k, actual, transpose_b);
        expect_close(x, f, actual, expected, 1, n, k, transpose_b);
      }
    }
  }
}

TEST(Quantized, GemmMatchesReferenceOnDequantizedWeights) {
  std::mt19937 generator(test::random_seed + 1);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  const std::vector<std::array<std::size_t, 4>> shapes{
      {2, 7, 64, 32}, {7, 65, 32, 16}, {13, 769, 96, 32}, {145, 33, 288, 32}, {6, 16, 256, 64}};
  for (const auto step : {GemmStep::packed, GemmStep::avx2}) {
    if (!kernels::available(step)) continue;
    for (const auto& [m, n, k, group] : shapes) {
      for (const bool transpose_b : {false, true}) {
        SCOPED_TRACE(std::string(kernels::name(step)) + " m=" + std::to_string(m) +
                     " n=" + std::to_string(n) + " k=" + std::to_string(k));
        const auto f = make(n, k, group, transpose_b, generator);
        const auto a = random_values(m * k, generator);
        std::vector<float> expected(m * n), actual(m * n, std::numeric_limits<float>::quiet_NaN());
        ops::ref::matmul(a, f.dequantized, m, n, k, expected, transpose_b);
        kernels::gemm(step, a, f.matrix(), m, n, k, actual, transpose_b);
        expect_close(a, f, actual, expected, m, n, k, transpose_b);
      }
    }
  }
}

TEST(Quantized, ResultsAreBitIdenticalForEveryThreadCount) {
  std::mt19937 generator(test::random_seed + 2);
  ThreadPool two(2, false), three(3, false), four(4, false);
  for (const bool transpose_b : {false, true}) {
    const auto f = make(1025, 96, 32, transpose_b, generator);
    const auto x = random_values(96, generator), a = random_values(9 * 96, generator);
    const auto gemv_kernel = kernels::best_gemv();
    const auto gemm_step = kernels::best_step();
    std::vector<float> gemv_one(1025), gemm_one(9 * 1025);
    kernels::gemv(gemv_kernel, x, f.matrix(), 1025, 96, gemv_one, transpose_b);
    kernels::gemm(gemm_step, a, f.matrix(), 9, 1025, 96, gemm_one, transpose_b);
    for (auto* pool : {&two, &three, &four}) {
      std::vector<float> gemv_threaded(1025), gemm_threaded(9 * 1025);
      kernels::gemv(gemv_kernel, x, f.matrix(), 1025, 96, gemv_threaded, transpose_b, pool);
      kernels::gemm(gemm_step, a, f.matrix(), 9, 1025, 96, gemm_threaded, transpose_b, pool);
      ASSERT_EQ(gemv_threaded, gemv_one) << "threads=" << pool->size();
      ASSERT_EQ(gemm_threaded, gemm_one) << "threads=" << pool->size();
    }
  }
}

TEST(Quantized, RejectsInconsistentGroupsAndShapes) {
  std::vector<std::int8_t> q(64);
  std::vector<float> scales(2), x(32), y(2);
  EXPECT_THROW(kernels::gemv(GemvKernel::scalar, x, QuantizedMatrix{q, scales, 0}, 2, 32, y),
               std::invalid_argument);
  EXPECT_THROW(kernels::gemv(GemvKernel::scalar, x, QuantizedMatrix{q, scales, 24}, 2, 32, y),
               std::invalid_argument);
  std::vector<float> wrong_scales(3);
  EXPECT_THROW(kernels::gemv(GemvKernel::scalar, x, QuantizedMatrix{q, wrong_scales, 32}, 2, 32, y),
               std::invalid_argument);
  std::vector<float> a(64), c(4);
  EXPECT_THROW(kernels::gemm(GemmStep::naive, a, QuantizedMatrix{q, scales, 32}, 2, 2, 32, c),
               std::invalid_argument);
}
}  // namespace
}  // namespace nanoinfer
