#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

namespace nanoinfer {

// Read-only mapping. Views remain valid until this owner is destroyed or overwritten.
class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path);
  ~MappedFile();
  MappedFile(MappedFile&&) noexcept;
  MappedFile& operator=(MappedFile&&) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nanoinfer
