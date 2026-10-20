#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "nanoinfer/mapped_file.hpp"

namespace nanoinfer {

class FormatError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct ModelConfig {
  std::uint32_t layers;
  std::uint32_t d_model;
  std::uint32_t heads;
  std::uint32_t vocab;
  std::uint32_t ctx;
};

struct TensorView {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint64_t> strides;
  std::span<const float> values;
  std::uint64_t offset;
  std::uint32_t checksum;
};

// Matches zlib.crc32, including its optional seed for chained ranges.
[[nodiscard]] std::uint32_t crc32(std::span<const std::byte> bytes, std::uint32_t seed = 0);

class Model {
 public:
  explicit Model(const std::filesystem::path& path);
  [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
  [[nodiscard]] std::span<const TensorView> tensors() const noexcept { return tensors_; }
  [[nodiscard]] const TensorView& tensor(std::string_view name) const;
  [[nodiscard]] std::span<const std::byte> mapped_bytes() const noexcept { return file_.bytes(); }

 private:
  MappedFile file_;
  ModelConfig config_{};
  std::vector<TensorView> tensors_;
};

}  // namespace nanoinfer
