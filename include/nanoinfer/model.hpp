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

enum class DType : std::uint32_t { f32 = 1, q8 = 2 };

// Views into the read-only mapping. f32 tensors expose `values`. q8 tensors
// (format version 2) expose int8 `quantized` values and one f32 scale per
// `group` consecutive values along `axis`; value = quantized * scale.
struct TensorView {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint64_t> strides;
  std::span<const float> values;
  std::uint64_t offset;
  std::uint32_t checksum;
  DType dtype = DType::f32;
  std::span<const std::byte> payload;
  std::span<const std::int8_t> quantized;
  std::span<const float> scales;
  std::uint32_t group = 0;
  std::uint32_t axis = 0;
};

// Matches zlib.crc32, including its optional seed for chained ranges.
[[nodiscard]] std::uint32_t crc32(std::span<const std::byte> bytes, std::uint32_t seed = 0);

enum class ModelValidation { gpt2, layout_only };

class Model {
 public:
  explicit Model(const std::filesystem::path& path,
                 ModelValidation validation = ModelValidation::gpt2);
  void validate_gpt2() const;
  [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
  [[nodiscard]] std::uint32_t version() const noexcept { return version_; }
  [[nodiscard]] std::span<const TensorView> tensors() const noexcept { return tensors_; }
  [[nodiscard]] const TensorView& tensor(std::string_view name) const;
  [[nodiscard]] std::span<const std::byte> mapped_bytes() const noexcept { return file_.bytes(); }

 private:
  MappedFile file_;
  ModelConfig config_{};
  std::uint32_t version_ = 0;
  std::vector<TensorView> tensors_;
};

}  // namespace nanoinfer
