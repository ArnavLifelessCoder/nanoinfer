#include "tools/generate.hpp"

#include <charconv>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

#include "nanoinfer/cached.hpp"
#include "nanoinfer/sampling.hpp"

namespace nanoinfer {
namespace {
template <typename T>
T number(const std::string& text) {
  T value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    throw std::invalid_argument("invalid numeric value: " + text);
  }
  return value;
}
}  // namespace

int generate_main(std::span<const std::filesystem::path> arguments) {
  if (arguments.size() == 1 && arguments[0] == "--help") {
    std::cout << "gpt2rt --model MODEL --prompt-ids IDS [--max-new 128] [--temp 0]\n"
                 "       [--top-k 0] [--top-p 1] [--seed 42]\n"
                 "stdout: JSON prompt_ids and generated_ids; temp=0 selects greedy.\n"
              << std::flush;
    return 0;
  }
  try {
    std::filesystem::path model_path, ids_path;
    std::size_t new_tokens = 128;
    std::uint64_t seed = 42;
    SamplingOptions options;
    std::set<std::string> seen;
    if (arguments.size() % 2 != 0) throw std::invalid_argument("options require values");
    for (std::size_t i = 0; i < arguments.size(); i += 2) {
      const auto name = arguments[i].string();
      const auto& value = arguments[i + 1];
      if (!seen.insert(name).second) throw std::invalid_argument("duplicate option: " + name);
      if (name == "--model")
        model_path = value;
      else if (name == "--prompt-ids")
        ids_path = value;
      else if (name == "--max-new")
        new_tokens = number<std::size_t>(value.string());
      else if (name == "--seed")
        seed = number<std::uint64_t>(value.string());
      else if (name == "--temp")
        options.temperature = number<double>(value.string());
      else if (name == "--top-k")
        options.top_k = number<std::size_t>(value.string());
      else if (name == "--top-p")
        options.top_p = number<double>(value.string());
      else
        throw std::invalid_argument("unknown option: " + name);
    }
    if (model_path.empty() || ids_path.empty()) {
      throw std::invalid_argument("--model and --prompt-ids are required");
    }
    Sampler sampler(options, seed);
    std::ifstream input(ids_path);
    if (!input) throw std::runtime_error("cannot open prompt IDs");
    std::vector<std::uint32_t> ids;
    std::string token;
    while (input >> token) ids.push_back(number<std::uint32_t>(token));
    if (!input.eof()) throw std::runtime_error("cannot read prompt IDs");
    CachedTransformer transformer(model_path);
    if (ids.size() > transformer.config().ctx ||
        new_tokens > transformer.config().ctx - ids.size()) {
      throw std::invalid_argument("prompt and continuation exceed model context");
    }
    auto state = transformer.prefill(ids);
    std::vector<std::uint32_t> generated;
    generated.reserve(new_tokens);
    for (std::size_t i = 0; i < new_tokens; ++i) {
      generated.push_back(sampler.sample(std::span(state.logits).last(transformer.config().vocab)));
      if (i + 1 < new_tokens) state = transformer.decode(generated.back());
    }
    auto write_ids = [](std::span<const std::uint32_t> values) {
      std::cout << '[';
      for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << values[i];
      }
      std::cout << ']';
    };
    std::cout << "{\"prompt_ids\":";
    write_ids(ids);
    std::cout << ",\"generated_ids\":";
    write_ids(generated);
    std::cout << "}\n" << std::flush;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "generation error: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace nanoinfer
