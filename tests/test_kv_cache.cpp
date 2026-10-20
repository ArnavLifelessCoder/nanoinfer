#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <random>
#include <vector>

#include "nanoinfer/kv_cache.hpp"
#include "nanoinfer/ref_ops.hpp"

namespace nanoinfer::test {
extern std::uint32_t random_seed;
}

namespace nanoinfer {
namespace {
TEST(KVCache, StoresFusedRowsInSeparateStableLayerBuffers) {
  KVCache cache(2, 7, 3, 7);
  const auto* keys = cache.keys(1).data();
  const auto* values = cache.values(1).data();
  std::vector<float> qkv(2 * 3 * 21);
  for (std::size_t i = 0; i < qkv.size(); ++i) qkv[i] = static_cast<float>(i);
  cache.store(1, 3, qkv);
  EXPECT_EQ(cache.keys(1).data(), keys);
  EXPECT_EQ(cache.values(1).data(), values);
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t channel = 0; channel < 21; ++channel) {
      EXPECT_EQ(cache.keys(1)[(3 + row) * 21 + channel], qkv[row * 63 + 21 + channel]);
      EXPECT_EQ(cache.values(1)[(3 + row) * 21 + channel], qkv[row * 63 + 42 + channel]);
      EXPECT_EQ(cache.keys(0)[(3 + row) * 21 + channel], 0);
    }
  }
  EXPECT_THROW(cache.store(1, 6, qkv), std::invalid_argument);
  EXPECT_THROW(cache.store(2, 0, qkv), std::out_of_range);
  EXPECT_THROW(cache.store(0, 0, std::span(qkv).first(1)), std::invalid_argument);
  EXPECT_THROW(KVCache(0, 1, 1, 1), std::invalid_argument);
  EXPECT_THROW(KVCache(1, 2, std::numeric_limits<std::size_t>::max(), 2), std::overflow_error);
}

TEST(KVCache, CachedAttentionExactlyMatchesFullCausalAttentionAtEveryPosition) {
  std::mt19937 generator(test::random_seed);
  std::uniform_real_distribution<float> distribution(-2, 2);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  for (const std::size_t length : {1, 7, 63, 65}) {
    for (const std::size_t head_width : {1, 7, 63, 65}) {
      const auto width = head_width * 3;
      SCOPED_TRACE("length=" + std::to_string(length) +
                   " head_width=" + std::to_string(head_width));
      std::vector<float> qkv(length * 3 * width), expected(length * width), actual(width);
      for (auto& value : qkv) value = distribution(generator);
      ops::ref::causal_attention(qkv, length, width, 3, expected);
      KVCache cache(1, length, 3, head_width);
      for (std::size_t position = 0; position < length; ++position) {
        const auto row = std::span(qkv).subspan(position * 3 * width, 3 * width);
        cache.store(0, position, row);
        ops::ref::cached_attention(row, cache.keys(0).first((position + 1) * width),
                                   cache.values(0).first((position + 1) * width), position, width,
                                   3, actual);
        for (std::size_t channel = 0; channel < width; ++channel) {
          ASSERT_EQ(actual[channel], expected[position * width + channel]);
        }
      }
      std::vector<float> prefill(expected.size());
      ops::ref::cached_attention(qkv, cache.keys(0), cache.values(0), 0, width, 3, prefill);
      EXPECT_EQ(prefill, expected);
    }
  }
}

TEST(KVCache, CachedAttentionRejectsInvalidPrefixBeforeIndexing) {
  std::vector<float> qkv(12), keys(8), values(8), output(4);
  EXPECT_THROW(ops::ref::cached_attention(qkv, keys, values, 2, 4, 2, output),
               std::invalid_argument);
  EXPECT_THROW(ops::ref::cached_attention(qkv, keys, values, 0, 4, 2, output),
               std::invalid_argument);
  EXPECT_THROW(ops::ref::cached_attention(qkv, keys, values, 1, 4, 3, output),
               std::invalid_argument);
  EXPECT_THROW(ops::ref::cached_attention({}, {}, {}, 0, 0, 1, {}), std::invalid_argument);
}
}  // namespace
}  // namespace nanoinfer
