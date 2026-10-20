#include <gtest/gtest.h>

#include <array>
#include <utility>

#include "nanoinfer/mapped_file.hpp"
#include "test_support.hpp"

namespace nanoinfer {
TEST(MappedFile, ReadsExactBytesAndSupportsUnicodePaths) {
  test::TempDirectory directory;
  const auto path = directory.file(std::filesystem::path(u"weights \u03bb.bin"));
  const std::array bytes{std::byte{0}, std::byte{127}, std::byte{255}};
  test::write_file(path, bytes);
  const MappedFile mapping(path);
  ASSERT_EQ(mapping.bytes().size(), bytes.size());
  EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), mapping.bytes().begin()));
}

TEST(MappedFile, MissingFileFailsCleanly) {
  test::TempDirectory directory;
  EXPECT_THROW((MappedFile(directory.file("missing.bin"))), std::system_error);
}

TEST(MappedFile, EmptyFileFailsCleanly) {
  test::TempDirectory directory;
  const auto path = directory.file("empty.bin");
  test::write_file(path, {});
  EXPECT_THROW((MappedFile(path)), std::runtime_error);
}

TEST(MappedFile, DirectoryFailsCleanly) {
  test::TempDirectory directory;
  const auto path = directory.file("directory");
  std::filesystem::create_directory(path);
  EXPECT_THROW((MappedFile(path)), std::runtime_error);
}

TEST(MappedFile, MoveTransfersMappingAndAssignmentReleasesPreviousFile) {
  test::TempDirectory directory;
  const auto first = directory.file("first.bin");
  const auto second = directory.file("second.bin");
  const std::array data{std::byte{42}};
  test::write_file(first, data);
  test::write_file(second, data);
  MappedFile original(first);
  const auto* address = original.bytes().data();
  MappedFile moved(std::move(original));
  EXPECT_TRUE(original.bytes().empty());
  EXPECT_EQ(moved.bytes().data(), address);
  {
    MappedFile assigned(second);
    assigned = std::move(moved);
    EXPECT_TRUE(moved.bytes().empty());
    EXPECT_EQ(assigned.bytes().data(), address);
    EXPECT_TRUE(std::filesystem::remove(second));
  }
  EXPECT_TRUE(std::filesystem::remove(first));
}
}  // namespace nanoinfer
