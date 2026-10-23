#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

#include "nanoinfer/model.hpp"

namespace nanoinfer {
namespace {
constexpr std::size_t kHeader = 64;
constexpr std::size_t kEntry = 128;
constexpr auto kCrcTable = [] {
  std::array<std::uint32_t, 256> table{};
  for (std::uint32_t i = 0; i < table.size(); ++i) {
    auto value = i;
    for (int bit = 0; bit < 8; ++bit) {
      value = (value >> 1) ^ ((value & 1U) != 0U ? 0xedb88320U : 0U);
    }
    table[i] = value;
  }
  return table;
}();

void require(bool condition, const char* message) {
  if (!condition) {
    throw FormatError(message);
  }
}

template <typename T>
T read(std::span<const std::byte> bytes, std::size_t offset) {
  require(offset <= bytes.size() && sizeof(T) <= bytes.size() - offset, "truncated field");
  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

bool all_zero(std::span<const std::byte> bytes) {
  return std::all_of(bytes.begin(), bytes.end(), [](std::byte b) { return b == std::byte{0}; });
}
}  // namespace

std::uint32_t crc32(std::span<const std::byte> bytes, std::uint32_t seed) {
  auto crc = seed ^ 0xffffffffU;
  for (const auto byte : bytes) {
    crc = kCrcTable[(crc ^ std::to_integer<std::uint32_t>(byte)) & 0xffU] ^ (crc >> 8);
  }
  return crc ^ 0xffffffffU;
}

Model::Model(const std::filesystem::path& path, ModelValidation validation) : file_(path) {
  require(std::endian::native == std::endian::little, "little-endian host required");
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  const auto bytes = file_.bytes();
  require(bytes.size() >= kHeader, "truncated header");
  require(std::memcmp(bytes.data(), "GPT2RT01", 8) == 0, "invalid magic");
  // Version 2 adds q8 tensors; version 1 files contain only f32.
  const auto version = read<std::uint32_t>(bytes, 8);
  require(version == 1 || version == 2, "unsupported format version");
  version_ = version;
  config_ = {read<std::uint32_t>(bytes, 12), read<std::uint32_t>(bytes, 16),
             read<std::uint32_t>(bytes, 20), read<std::uint32_t>(bytes, 24),
             read<std::uint32_t>(bytes, 28)};
  require(config_.layers > 0 && config_.d_model > 0 && config_.heads > 0 && config_.vocab > 0 &&
              config_.ctx > 0,
          "zero config dimension");
  require(config_.d_model % config_.heads == 0, "model width is not divisible by heads");
  const auto count = read<std::uint32_t>(bytes, 32);
  require(count > 0 && count <= (bytes.size() - kHeader) / kEntry, "invalid tensor count");
  require(read<std::uint32_t>(bytes, 36) == kEntry, "invalid table entry size");
  const auto data_start = kHeader + static_cast<std::size_t>(count) * kEntry;
  require(read<std::uint64_t>(bytes, 40) == data_start, "invalid data section offset");
  require(read<std::uint64_t>(bytes, 48) == bytes.size(), "file length mismatch");
  require(read<std::uint32_t>(bytes, 60) == 0, "nonzero header reserved field");
  const auto metadata_crc =
      crc32(bytes.subspan(kHeader, data_start - kHeader), crc32(bytes.first(56)));
  require(read<std::uint32_t>(bytes, 56) == metadata_crc, "metadata checksum mismatch");

  std::unordered_set<std::string> names;
  tensors_.reserve(count);
  auto expected_offset = data_start;
  for (std::size_t index = 0; index < count; ++index) {
    const auto entry = bytes.subspan(kHeader + index * kEntry, kEntry);
    const auto name_bytes = entry.first(64);
    const auto end = std::find(name_bytes.begin(), name_bytes.end(), std::byte{0});
    require(end != name_bytes.begin() && end != name_bytes.end(), "invalid tensor name");
    const auto length = static_cast<std::size_t>(end - name_bytes.begin());
    require(all_zero(name_bytes.subspan(length)), "nonzero name padding");
    require(std::all_of(name_bytes.begin(), end,
                        [](std::byte b) { return std::to_integer<unsigned>(b) < 128; }),
            "non-ASCII tensor name");
    TensorView view;
    view.name.assign(reinterpret_cast<const char*>(name_bytes.data()), length);
    require(names.insert(view.name).second, "duplicate tensor name");
    const auto dtype = read<std::uint32_t>(entry, 64);
    require(dtype == 1 || (dtype == 2 && version == 2), "unsupported tensor dtype");
    view.dtype = dtype == 1 ? DType::f32 : DType::q8;
    const auto rank = read<std::uint32_t>(entry, 68);
    require(rank >= 1 && rank <= 4, "invalid tensor rank");
    std::uint64_t elements = 1;
    for (std::uint32_t axis = 0; axis < 4; ++axis) {
      const auto dimension = read<std::uint64_t>(entry, 72 + axis * 8);
      if (axis < rank) {
        require(dimension > 0 && dimension <= std::numeric_limits<std::uint64_t>::max() / elements,
                "zero dimension or shape overflow");
        elements *= dimension;
        view.shape.push_back(dimension);
      } else {
        require(dimension == 0, "nonzero unused dimension");
      }
    }
    require(elements <= std::numeric_limits<std::uint64_t>::max() / sizeof(float),
            "tensor byte size overflow");
    view.offset = read<std::uint64_t>(entry, 104);
    const auto size = read<std::uint64_t>(entry, 112);
    const auto extra = read<std::uint32_t>(entry, 124);
    std::uint64_t value_bytes = elements * sizeof(float);
    if (view.dtype == DType::q8) {
      // extra = group | axis << 16. int8 values, zero padding to 4 bytes, then
      // one f32 scale per group of `group` values along `axis`.
      view.group = extra & 0xffffU;
      view.axis = extra >> 16;
      require(
          rank == 2 && view.group > 0 && view.axis <= 1 && view.shape[view.axis] % view.group == 0,
          "invalid q8 grouping");
      value_bytes = elements;
      const auto padded = (value_bytes + 3) / 4 * 4;
      require(size == padded + elements / view.group * sizeof(float),
              "tensor shape and byte size disagree");
    } else {
      require(extra == 0, "nonzero tensor reserved field");
      require(size == value_bytes, "tensor shape and byte size disagree");
    }
    require(view.offset % 64 == 0, "unaligned tensor");
    require(view.offset == expected_offset, "overlapping, reordered or noncanonical tensor offset");
    require(view.offset <= bytes.size() && size <= bytes.size() - view.offset,
            "tensor payload outside file");
    const auto payload =
        bytes.subspan(static_cast<std::size_t>(view.offset), static_cast<std::size_t>(size));
    view.checksum = crc32(payload);
    require(read<std::uint32_t>(entry, 120) == view.checksum, "tensor checksum mismatch");
    view.payload = payload;
    if (view.dtype == DType::f32) {
      view.values = {reinterpret_cast<const float*>(payload.data()),
                     static_cast<std::size_t>(elements)};
    } else {
      view.quantized = {reinterpret_cast<const std::int8_t*>(payload.data()),
                        static_cast<std::size_t>(elements)};
      const auto scales_offset = static_cast<std::size_t>((value_bytes + 3) / 4 * 4);
      require(all_zero(payload.subspan(static_cast<std::size_t>(value_bytes),
                                       scales_offset - static_cast<std::size_t>(value_bytes))),
              "nonzero q8 padding");
      // Payloads start 64-byte aligned and the scales 4-byte aligned after it.
      view.scales = {reinterpret_cast<const float*>(payload.data() + scales_offset),
                     static_cast<std::size_t>(elements / view.group)};
      require(std::all_of(view.scales.begin(), view.scales.end(),
                          [](float scale) { return std::isfinite(scale) && scale >= 0; }),
              "invalid q8 scale");
    }
    view.strides.resize(rank);
    std::uint64_t stride = 1;
    for (std::size_t axis = rank; axis > 0; --axis) {
      view.strides[axis - 1] = stride;
      stride *= view.shape[axis - 1];
    }
    expected_offset += static_cast<std::size_t>(size);
    const auto padding = (64 - expected_offset % 64) % 64;
    require(padding <= bytes.size() - expected_offset, "truncated tensor padding");
    require(all_zero(bytes.subspan(expected_offset, padding)), "nonzero tensor padding");
    expected_offset += padding;
    tensors_.push_back(std::move(view));
  }
  require(expected_offset == bytes.size(), "unexpected trailing data");
  if (validation == ModelValidation::gpt2) {
    validate_gpt2();
  }
}

void Model::validate_gpt2() const {
  const std::uint64_t d = config_.d_model;
  require(tensors_.size() == 4ULL + 12ULL * config_.layers, "GPT-2 tensor count mismatch");
  // q8_axis: the input-dimension axis along which a matrix may be int8
  // quantized (-1: must stay f32). Conv1D matrices are [in, out]; the tied
  // embedding is used as [N, K] for the logits.
  auto shape = [this](const std::string& name, std::vector<std::uint64_t> expected,
                      int q8_axis = -1) {
    const auto found = std::find_if(tensors_.begin(), tensors_.end(),
                                    [&name](const auto& view) { return view.name == name; });
    if (found == tensors_.end()) {
      throw FormatError("missing GPT-2 tensor: " + name);
    }
    if (found->shape != expected) {
      throw FormatError("tensor shape disagrees with header config: " + name);
    }
    if (found->dtype == DType::q8 && static_cast<int>(found->axis) != q8_axis) {
      throw FormatError("tensor may not be int8 quantized along this axis: " + name);
    }
  };
  shape("wte.weight", {config_.vocab, d}, 1);
  shape("wpe.weight", {config_.ctx, d});
  shape("ln_f.weight", {d});
  shape("ln_f.bias", {d});
  for (std::uint32_t layer = 0; layer < config_.layers; ++layer) {
    const auto prefix = "h." + std::to_string(layer) + ".";
    for (const auto* norm : {"ln_1", "ln_2"}) {
      shape(prefix + norm + ".weight", {d});
      shape(prefix + norm + ".bias", {d});
    }
    shape(prefix + "attn.c_attn.weight", {d, 3 * d}, 0);
    shape(prefix + "attn.c_attn.bias", {3 * d});
    shape(prefix + "attn.c_proj.weight", {d, d}, 0);
    shape(prefix + "attn.c_proj.bias", {d});
    shape(prefix + "mlp.c_fc.weight", {d, 4 * d}, 0);
    shape(prefix + "mlp.c_fc.bias", {4 * d});
    shape(prefix + "mlp.c_proj.weight", {4 * d, d}, 0);
    shape(prefix + "mlp.c_proj.bias", {d});
  }
}

const TensorView& Model::tensor(std::string_view name) const {
  const auto found = std::find_if(tensors_.begin(), tensors_.end(),
                                  [name](const auto& view) { return view.name == name; });
  if (found == tensors_.end()) {
    throw std::out_of_range("tensor not found: " + std::string(name));
  }
  return *found;
}
}  // namespace nanoinfer
