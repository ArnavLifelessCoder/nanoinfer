#include <gtest/gtest.h>

#include <array>
#include <cmath>
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

struct Case {
  std::size_t n, k;
};

const std::vector<Case> kCases{{1, 1},      {7, 63},    {63, 7},    {65, 769},   {767, 65},
                               {769, 767},  {33, 31},   {1025, 33}, {2304, 768}, {768, 3072},
                               {3072, 768}, {97, 1025}, {2049, 129}};

std::vector<float> random_values(std::size_t count, std::mt19937& generator) {
  std::uniform_real_distribution<float> distribution(-1, 1);
  std::vector<float> values(count);
  for (auto& value : values) value = distribution(generator);
  return values;
}

// Error relative to sum_p |x_p| |b_pj|, as in the GEMM tests.
void expect_close(const std::vector<float>& x, const std::vector<float>& b,
                  const std::vector<float>& actual, const std::vector<float>& expected,
                  std::size_t n, std::size_t k, bool transpose_b) {
  for (std::size_t j = 0; j < n; ++j) {
    double magnitude = 0;
    for (std::size_t p = 0; p < k; ++p) {
      magnitude +=
          std::abs(static_cast<double>(x[p]) * (transpose_b ? b[j * k + p] : b[p * n + j]));
    }
    ASSERT_TRUE(std::isfinite(actual[j])) << "output not written at " << j;
    EXPECT_LE(std::abs(static_cast<double>(actual[j]) - expected[j]), 1e-5 * magnitude + 1e-30)
        << "column " << j;
  }
}

TEST(Gemv, MatchesReferenceOnAwkwardShapesForBothLayouts) {
  std::mt19937 generator(test::random_seed);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  for (const auto kernel : {GemvKernel::scalar, GemvKernel::avx2}) {
    if (!kernels::available(kernel)) continue;
    for (const auto& [n, k] : kCases) {
      for (const bool transpose_b : {false, true}) {
        SCOPED_TRACE("kernel=" + std::to_string(static_cast<int>(kernel)) +
                     " n=" + std::to_string(n) + " k=" + std::to_string(k) +
                     " t=" + std::to_string(transpose_b));
        const auto x = random_values(k, generator), b = random_values(k * n, generator);
        std::vector<float> expected(n), actual(n, std::numeric_limits<float>::quiet_NaN());
        ops::ref::matmul(x, b, 1, n, k, expected, transpose_b);
        kernels::gemv(kernel, x, b, n, k, actual, transpose_b);
        expect_close(x, b, actual, expected, n, k, transpose_b);
      }
    }
  }
}

TEST(Gemv, ResultIsBitIdenticalForEveryThreadCount) {
  std::mt19937 generator(test::random_seed + 1);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  ThreadPool two(2, false), three(3, false), four(4, false);
  for (const auto kernel : {GemvKernel::scalar, GemvKernel::avx2}) {
    if (!kernels::available(kernel)) continue;
    for (const auto& [n, k] : kCases) {
      for (const bool transpose_b : {false, true}) {
        const auto x = random_values(k, generator), b = random_values(k * n, generator);
        std::vector<float> one(n);
        kernels::gemv(kernel, x, b, n, k, one, transpose_b);
        for (auto* pool : {&two, &three, &four}) {
          std::vector<float> threaded(n);
          kernels::gemv(kernel, x, b, n, k, threaded, transpose_b, pool);
          ASSERT_EQ(threaded, one) << "n=" << n << " k=" << k << " threads=" << pool->size();
        }
      }
    }
  }
}

TEST(Gemv, ParallelGemmIsBitIdenticalForEveryThreadCount) {
  std::mt19937 generator(test::random_seed + 2);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  ThreadPool two(2, false), three(3, false), five(5, false);
  const std::vector<std::array<std::size_t, 3>> shapes{
      {7, 65, 769}, {145, 33, 257}, {13, 3089, 31}, {300, 160, 513}};
  for (const auto step : {GemmStep::packed, GemmStep::avx2}) {
    if (!kernels::available(step)) continue;
    for (const auto& [m, n, k] : shapes) {
      for (const bool transpose_b : {false, true}) {
        const auto a = random_values(m * k, generator), b = random_values(k * n, generator);
        std::vector<float> one(m * n);
        kernels::gemm(step, a, b, m, n, k, one, transpose_b);
        for (auto* pool : {&two, &three, &five}) {
          std::vector<float> threaded(m * n, std::numeric_limits<float>::quiet_NaN());
          kernels::gemm(step, a, b, m, n, k, threaded, transpose_b, pool);
          ASSERT_EQ(threaded, one)
              << "m=" << m << " n=" << n << " k=" << k << " threads=" << pool->size();
        }
      }
    }
  }
}

TEST(Gemv, RejectsShapeMismatches) {
  std::vector<float> x(3), b(6), y(2);
  EXPECT_THROW(kernels::gemv(GemvKernel::scalar, x, b, 3, 3, y), std::invalid_argument);
  EXPECT_THROW(kernels::gemv(GemvKernel::scalar, x, b, 2, 0, y), std::invalid_argument);
}
}  // namespace
}  // namespace nanoinfer
