#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "nanoinfer/attention.hpp"
#include "nanoinfer/kv_cache.hpp"

namespace nanoinfer::test {
extern std::uint32_t random_seed;
}

namespace nanoinfer {
namespace {
using kernels::GemvKernel;

std::vector<float> random_values(std::size_t count, std::mt19937& generator, float range) {
  std::uniform_real_distribution<float> distribution(-range, range);
  std::vector<float> values(count);
  for (auto& value : values) value = distribution(generator);
  return values;
}

TEST(Attention, OptimizedTwinMatchesReferenceForPrefillAndDecodeShapes) {
  std::mt19937 generator(test::random_seed);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  // {heads, head_width, first_position, rows}: decode (one row after a
  // prefix), prefill from zero, awkward widths and row counts crossing the
  // 16-row unit size.
  const std::vector<std::array<std::size_t, 4>> cases{
      {1, 1, 0, 1},    {2, 7, 0, 7},     {3, 63, 5, 1},   {2, 65, 0, 17},
      {12, 64, 0, 33}, {12, 64, 159, 1}, {4, 16, 31, 16}, {1, 769, 2, 3}};
  for (const auto kernel : {GemvKernel::scalar, GemvKernel::avx2}) {
    if (!kernels::available(kernel)) continue;
    for (const auto& [heads, head_width, first, rows] : cases) {
      SCOPED_TRACE(std::to_string(heads) + "x" + std::to_string(head_width) +
                   " first=" + std::to_string(first) + " rows=" + std::to_string(rows));
      const auto width = heads * head_width, positions = first + rows;
      const auto qkv = random_values(rows * 3 * width, generator, 2.0F);
      const auto keys = random_values(positions * width, generator, 2.0F);
      const auto values = random_values(positions * width, generator, 1.0F);
      std::vector<float> expected(rows * width);
      std::vector<float> actual(rows * width, std::numeric_limits<float>::quiet_NaN());
      ops::ref::cached_attention(qkv, keys, values, first, width, heads, expected);
      // Split the units unevenly to show any split covers every output once.
      const auto units = kernels::attention_units(rows, heads);
      const auto middle = units / 3;
      kernels::cached_attention(kernel, qkv, keys, values, first, width, heads, actual, 0, middle);
      kernels::cached_attention(kernel, qkv, keys, values, first, width, heads, actual, middle,
                                units);
      for (std::size_t i = 0; i < actual.size(); ++i) {
        ASSERT_TRUE(std::isfinite(actual[i])) << "output not written at " << i;
        // Outputs are convex combinations of values in [-1, 1].
        EXPECT_NEAR(actual[i], expected[i], 2e-6) << i;
      }
    }
  }
}

TEST(Attention, RejectsUnitRangesBeyondTheShape) {
  std::vector<float> qkv(3 * 4), keys(4), values(4), output(4);
  EXPECT_THROW(kernels::cached_attention(GemvKernel::scalar, qkv, keys, values, 0, 4, 2, output, 0,
                                         kernels::attention_units(1, 2) + 1),
               std::invalid_argument);
}
}  // namespace
}  // namespace nanoinfer
