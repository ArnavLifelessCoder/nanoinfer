#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <limits>
#include <stdexcept>
#include <system_error>

#include "nanoinfer/mapped_file.hpp"

namespace nanoinfer {
namespace {
[[noreturn]] void fail(const char* operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}
}  // namespace

struct MappedFile::Impl {
  int file = -1;
  void* data = MAP_FAILED;
  std::size_t size = 0;

  ~Impl() {
    if (data != MAP_FAILED) {
      munmap(data, size);
    }
    if (file >= 0) {
      close(file);
    }
  }
};

MappedFile::MappedFile(const std::filesystem::path& path) : impl_(std::make_unique<Impl>()) {
  impl_->file = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (impl_->file < 0) {
    fail("open model");
  }
  struct stat info{};
  if (fstat(impl_->file, &info) != 0) {
    fail("read model length");
  }
  if (!S_ISREG(info.st_mode) || info.st_size <= 0 ||
      static_cast<std::uintmax_t>(info.st_size) > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("model file is not regular, is empty, or is too large");
  }
  impl_->size = static_cast<std::size_t>(info.st_size);
  impl_->data = mmap(nullptr, impl_->size, PROT_READ, MAP_PRIVATE, impl_->file, 0);
  if (impl_->data == MAP_FAILED) {
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
