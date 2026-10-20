#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <limits>
#include <stdexcept>
#include <system_error>

#include "nanoinfer/mapped_file.hpp"

namespace nanoinfer {
namespace {
[[noreturn]] void fail(const char* operation) {
  throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
}
}  // namespace

struct MappedFile::Impl {
  HANDLE file = INVALID_HANDLE_VALUE;
  HANDLE mapping = nullptr;
  const void* data = nullptr;
  std::size_t size = 0;

  ~Impl() {
    if (data != nullptr) {
      UnmapViewOfFile(data);
    }
    if (mapping != nullptr) {
      CloseHandle(mapping);
    }
    if (file != INVALID_HANDLE_VALUE) {
      CloseHandle(file);
    }
  }
};

MappedFile::MappedFile(const std::filesystem::path& path) : impl_(std::make_unique<Impl>()) {
  impl_->file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (impl_->file == INVALID_HANDLE_VALUE) {
    fail("open model");
  }
  LARGE_INTEGER length{};
  if (!GetFileSizeEx(impl_->file, &length)) {
    fail("read model length");
  }
  if (length.QuadPart <= 0 ||
      static_cast<unsigned long long>(length.QuadPart) > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("model file is empty or too large");
  }
  impl_->size = static_cast<std::size_t>(length.QuadPart);
  impl_->mapping = CreateFileMappingW(impl_->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (impl_->mapping == nullptr) {
    fail("create model mapping");
  }
  impl_->data = MapViewOfFile(impl_->mapping, FILE_MAP_READ, 0, 0, 0);
  if (impl_->data == nullptr) {
    fail("map model view");
  }
}

MappedFile::~MappedFile() = default;
MappedFile::MappedFile(MappedFile&&) noexcept = default;
MappedFile& MappedFile::operator=(MappedFile&&) noexcept = default;

std::span<const std::byte> MappedFile::bytes() const noexcept {
  if (!impl_) {
    return {};
  }
  return {static_cast<const std::byte*>(impl_->data), impl_->size};
}
}  // namespace nanoinfer
