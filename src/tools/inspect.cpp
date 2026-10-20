#include "tools/inspect.hpp"

#include <iostream>
#include <string_view>

#include "nanoinfer/model.hpp"

namespace nanoinfer {
namespace {
void json_string(std::string_view value) {
  constexpr std::string_view hex = "0123456789abcdef";
  std::cout << '"';
  for (const unsigned char ch : value) {
    if (ch == '"' || ch == '\\') {
      std::cout << '\\' << static_cast<char>(ch);
    } else if (ch < 32) {
      std::cout << "\\u00" << hex[ch >> 4] << hex[ch & 15];
    } else {
      std::cout << static_cast<char>(ch);
    }
  }
  std::cout << '"';
}
}  // namespace

int inspect_model(const std::filesystem::path& path, bool layout_only) {
  try {
    const Model model(path, layout_only ? ModelValidation::layout_only : ModelValidation::gpt2);
    const auto& config = model.config();
    std::cout << "{\"format_version\":1,\"config\":{\"layers\":" << config.layers
              << ",\"d_model\":" << config.d_model << ",\"heads\":" << config.heads
              << ",\"vocab\":" << config.vocab << ",\"ctx\":" << config.ctx << "},\"tensors\":[";
    bool first = true;
    for (const auto& tensor : model.tensors()) {
      if (!first) {
        std::cout << ',';
      }
      first = false;
      std::cout << "{\"name\":";
      json_string(tensor.name);
      std::cout << ",\"dtype\":\"f32\",\"shape\":[";
      for (std::size_t axis = 0; axis < tensor.shape.size(); ++axis) {
        if (axis != 0) {
          std::cout << ',';
        }
        std::cout << tensor.shape[axis];
      }
      std::cout << "],\"offset\":" << tensor.offset
                << ",\"byte_size\":" << tensor.values.size_bytes()
                << ",\"crc32\":" << crc32(std::as_bytes(tensor.values)) << '}';
    }
    std::cout << "]}\n" << std::flush;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "model error: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace nanoinfer
