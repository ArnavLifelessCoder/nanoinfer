#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>

namespace nanoinfer::test {
class TempDirectory {
 public:
  TempDirectory() {
    static unsigned counter = 0;
    path_ = std::filesystem::current_path() / "test-files" /
            (std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
             std::to_string(counter++));
    std::filesystem::create_directories(path_);
  }
  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  TempDirectory(const TempDirectory&) = delete;
  TempDirectory& operator=(const TempDirectory&) = delete;
  [[nodiscard]] std::filesystem::path file(const std::filesystem::path& name) const {
    return path_ / name;
  }

 private:
  std::filesystem::path path_;
};

inline void write_file(const std::filesystem::path& path, std::span<const std::byte> data) {
  std::ofstream stream(path, std::ios::binary);
  stream.exceptions(std::ios::badbit | std::ios::failbit);
  stream.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
}
}  // namespace nanoinfer::test
