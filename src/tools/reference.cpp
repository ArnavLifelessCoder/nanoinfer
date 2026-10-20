#include "tools/reference.hpp"

#include <charconv>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "nanoinfer/reference.hpp"

namespace nanoinfer {
namespace {
std::uint32_t number(const std::string& text) {
  std::uint32_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw std::invalid_argument("expected an unsigned 32-bit integer: " + text);
  }
  return value;
}

std::vector<std::uint32_t> read_ids(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open prompt IDs");
  }
  std::vector<std::uint32_t> ids;
  std::string token;
  while (input >> token) {
    ids.push_back(number(token));
  }
  if (!input.eof()) {
    throw std::runtime_error("cannot read prompt IDs");
  }
  return ids;
}

void write_floats(const std::filesystem::path& path, std::span<const float> values) {
  if (values.size_bytes() > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error("output is too large for file stream");
  }
  std::ofstream output;
  output.exceptions(std::ios::badbit | std::ios::failbit);
  output.open(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size_bytes()));
}
}  // namespace

int reference_main(std::span<const std::filesystem::path> arguments) {
  if (arguments.size() != 3 && arguments.size() != 4) {
    std::cerr << "usage: gpt2ref MODEL PROMPT_IDS OUTPUT_DIRECTORY [NEW_TOKENS]\n";
    return 2;
  }
  try {
    const auto new_tokens = arguments.size() == 4 ? number(arguments[3].string()) : 32U;
    const ReferenceTransformer transformer(arguments[0]);
    const auto ids = read_ids(arguments[1]);
    // Validate the full requested length before doing any work or writing output.
    if (ids.size() > transformer.config().ctx ||
        new_tokens > transformer.config().ctx - ids.size()) {
      throw std::invalid_argument("prompt and continuation exceed model context");
    }
    const auto result = transformer.forward(ids);
    const auto& directory = arguments[2];
    std::filesystem::create_directories(directory);
    write_floats(directory / "logits.f32", result.logits);
    write_floats(directory / "final_ln.f32", result.final_ln);
    for (std::size_t layer = 0; layer < result.hidden_states.size(); ++layer) {
      write_floats(directory / ("layer_" + std::to_string(layer) + ".f32"),
                   result.hidden_states[layer]);
    }
    std::cout << "PASS forward, positions=" << result.positions << "; generating " << new_tokens
              << " tokens by full-prefix recomputation\n"
              << std::flush;
    const auto generated = transformer.greedy(ids, new_tokens);
    std::ofstream manifest;
    manifest.exceptions(std::ios::badbit | std::ios::failbit);
    manifest.open(directory / "output.json");
    manifest << "{\"positions\":" << result.positions << ",\"vocab\":" << transformer.config().vocab
             << ",\"d_model\":" << transformer.config().d_model
             << ",\"layers\":" << transformer.config().layers << ",\"greedy_ids\":[";
    for (std::size_t i = 0; i < generated.size(); ++i) {
      if (i != 0) manifest << ',';
      manifest << generated[i];
    }
    manifest << "]}\n";
    std::cout << "PASS generated " << generated.size() << " tokens\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "reference error: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace nanoinfer
