#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <limits>
#include <random>

#include "nanoinfer/model.hpp"
#include "test_support.hpp"

namespace nanoinfer::test {
extern std::uint32_t random_seed;
}

namespace nanoinfer {
namespace {
template <typename T>
void put(std::vector<std::byte>& bytes, std::size_t offset, T value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void repair_metadata(std::vector<std::byte>& bytes) {
  put(bytes, 56, crc32(std::span(bytes).subspan(64, 256), crc32(std::span(bytes).first(56))));
}

std::vector<std::byte> fixture() {
  std::vector<std::byte> bytes(448);
  std::memcpy(bytes.data(), "GPT2RT01", 8);
  for (const auto offset : {8, 12, 20}) {
    put(bytes, offset, std::uint32_t{1});
  }
  for (const auto offset : {16, 24, 28}) {
    put(bytes, offset, std::uint32_t{7});
  }
  put(bytes, 32, std::uint32_t{2});
  put(bytes, 36, std::uint32_t{128});
  put(bytes, 40, std::uint64_t{320});
  put(bytes, 48, std::uint64_t{448});
  for (std::size_t index = 0; index < 2; ++index) {
    const auto base = 64 + index * 128;
    const auto offset = 320 + index * 64;
    bytes[base] = index == 0 ? std::byte{'a'} : std::byte{'b'};
    put(bytes, base + 64, std::uint32_t{1});
    put(bytes, base + 68, std::uint32_t{2});
    put(bytes, base + 72, std::uint64_t{1});
    put(bytes, base + 80, std::uint64_t{7});
    put(bytes, base + 104, static_cast<std::uint64_t>(offset));
    put(bytes, base + 112, std::uint64_t{28});
    for (std::size_t i = 0; i < 7; ++i) {
      put(bytes, offset + i * 4, static_cast<float>(index * 7 + i));
    }
    put(bytes, base + 120, crc32(std::span(bytes).subspan(offset, 28)));
  }
  repair_metadata(bytes);
  return bytes;
}

class LoaderTest : public ::testing::Test {
 protected:
  test::TempDirectory directory;
  std::filesystem::path path = directory.file("model.bin");

  void rejects(const std::vector<std::byte>& bytes) {
    test::write_file(path, bytes);
    EXPECT_THROW((Model(path, ModelValidation::layout_only)), std::runtime_error);
    // A failed constructor must release handles as well as successful ones.
    EXPECT_TRUE(std::filesystem::remove(path));
  }
};
}  // namespace

TEST(Checksum, MatchesStandardVectorAndChaining) {
  const std::string text = "123456789";
  const auto bytes = std::as_bytes(std::span(text));
  EXPECT_EQ(crc32(bytes), 0xcbf43926U);
  EXPECT_EQ(crc32({}), 0U);
  EXPECT_EQ(crc32(bytes.subspan(4), crc32(bytes.first(4))), crc32(bytes));
}

TEST_F(LoaderTest, ReadsShapesStridesAndZeroCopyValues) {
  test::write_file(path, fixture());
  const Model model(path, ModelValidation::layout_only);
  ASSERT_EQ(model.tensors().size(), 2);
  EXPECT_EQ(model.config().d_model, 7);
  const auto& view = model.tensor("a");
  EXPECT_EQ(view.shape, (std::vector<std::uint64_t>{1, 7}));
  EXPECT_EQ(view.strides, (std::vector<std::uint64_t>{7, 1}));
  EXPECT_EQ(reinterpret_cast<const std::byte*>(view.values.data()),
            model.mapped_bytes().data() + 320);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(view.values.data()) % 64, 0);
  EXPECT_EQ(view.values[6], 6.0F);
  EXPECT_EQ(model.tensor("b").values[6], 13.0F);
  EXPECT_THROW(static_cast<void>(model.tensor("missing")), std::out_of_range);
}

TEST_F(LoaderTest, RejectsEveryTruncationBoundary) {
  const auto valid = fixture();
  for (std::size_t length = 0; length < valid.size(); ++length) {
    SCOPED_TRACE(length);
    rejects(std::vector(valid.begin(), valid.begin() + static_cast<std::ptrdiff_t>(length)));
  }
}

TEST_F(LoaderTest, RejectsBitCorruptionAtEveryByte) {
  std::mt19937 generator(test::random_seed);
  SCOPED_TRACE("seed=" + std::to_string(test::random_seed));
  const auto valid = fixture();
  for (std::size_t index = 0; index < valid.size(); ++index) {
    SCOPED_TRACE(index);
    auto bytes = valid;
    bytes[index] ^= std::byte{static_cast<unsigned char>(1U << (generator() % 8))};
    rejects(bytes);
  }
}

TEST_F(LoaderTest, RejectsInvalidMetadataEvenWithCorrectChecksum) {
  const std::array<std::pair<std::size_t, std::uint32_t>, 15> invalid{{
      {8, 2},
      {12, 0},
      {20, 2},
      {32, 0},
      {32, 0xffffffffU},
      {36, 64},
      {60, 1},
      {64 + 64, 2},
      {64 + 68, 0},
      {64 + 68, 5},
      {64 + 72, 0},
      {64 + 88, 1},
      {64 + 112, 4},
      {64 + 124, 1},
      {64 + 120, 0},
  }};
  for (const auto& [offset, value] : invalid) {
    SCOPED_TRACE(offset);
    auto bytes = fixture();
    put(bytes, offset, value);
    repair_metadata(bytes);
    rejects(bytes);
  }
}

TEST_F(LoaderTest, RejectsOverflowOverlapsOffsetsAndTrailingData) {
  const std::array<std::pair<std::size_t, std::uint64_t>, 8> invalid{{
      {40, 64},
      {48, 449},
      {64 + 104, 321},
      {64 + 104, 64},
      {192 + 104, 320},
      {64 + 104, std::numeric_limits<std::uint64_t>::max()},
      {64 + 72, std::numeric_limits<std::uint64_t>::max()},
      {64 + 112, std::numeric_limits<std::uint64_t>::max()},
  }};
  for (const auto& [offset, value] : invalid) {
    SCOPED_TRACE(offset);
    auto bytes = fixture();
    put(bytes, offset, value);
    repair_metadata(bytes);
    rejects(bytes);
  }
  auto bytes = fixture();
  bytes.push_back(std::byte{0});
  put(bytes, 48, static_cast<std::uint64_t>(bytes.size()));
  repair_metadata(bytes);
  rejects(bytes);
}

TEST_F(LoaderTest, RejectsDuplicateUnterminatedAndInvalidNames) {
  for (int variant = 0; variant < 5; ++variant) {
    auto bytes = fixture();
    if (variant == 0) {
      bytes[192] = std::byte{'a'};
    } else if (variant == 1) {
      std::fill(bytes.begin() + 64, bytes.begin() + 128, std::byte{'x'});
    } else if (variant == 2) {
      bytes[64] = std::byte{0};
    } else if (variant == 3) {
      bytes[64] = std::byte{255};
    } else {
      bytes[66] = std::byte{'x'};
    }
    repair_metadata(bytes);
    rejects(bytes);
  }
}
}  // namespace nanoinfer
