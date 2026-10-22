// Runs the cached forward with an optimized GEMM kernel and writes the same
// capture files as gpt2ref, so the harness can compare it with both PyTorch and
// the scalar reference. Uses gpt2ref's positional interface and entry point.
#include <charconv>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nanoinfer/cached.hpp"
#include "nanoinfer/fp_env.hpp"
#include "nanoinfer/sampling.hpp"
#include "tools/reference.hpp"

namespace nanoinfer {
namespace {
std::uint32_t number(const std::string& text) {
  std::uint32_t value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    throw std::invalid_argument("expected unsigned integer");
  }
  return value;
}

kernels::GemmStep step_named(const std::string& text) {
  for (const auto step :
       {kernels::GemmStep::naive, kernels::GemmStep::reordered, kernels::GemmStep::blocked,
        kernels::GemmStep::packed, kernels::GemmStep::avx2}) {
    if (text == kernels::name(step)) return step;
  }
  throw std::invalid_argument("unknown GEMM kernel: " + text);
}

void write_floats(const std::filesystem::path& path, std::span<const float> values) {
  std::ofstream output;
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.open(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size_bytes()));
}
}  // namespace

int reference_main(std::span<const std::filesystem::path> arguments) {
  try {
    if (arguments.size() != 5 && arguments.size() != 6) {
      throw std::invalid_argument("gpt2opt MODEL IDS OUTPUT_DIRECTORY NEW_TOKENS KERNEL [THREADS]");
    }
    enable_flush_to_zero();
    const auto count = number(arguments[3].string());
    const auto threads = arguments.size() == 6 ? number(arguments[5].string()) : 1U;
    CachedTransformer transformer(arguments[0], step_named(arguments[4].string()), threads);
    std::ifstream input(arguments[1]);
    if (!input) throw std::runtime_error("cannot open prompt IDs");
    std::vector<std::uint32_t> tokens;
    std::string text;
    while (input >> text) tokens.push_back(number(text));
    if (!input.eof()) throw std::runtime_error("cannot read prompt IDs");
    const auto& config = transformer.config();
    if (tokens.size() > config.ctx || count > config.ctx - tokens.size()) {
      throw std::invalid_argument("context exceeded");
    }
    auto state = transformer.prefill(tokens, true);
    const auto& directory = arguments[2];
    std::filesystem::create_directories(directory);
    write_floats(directory / "logits.f32", state.logits);
    write_floats(directory / "final_ln.f32", state.final_ln);
    for (std::size_t layer = 0; layer < state.hidden_states.size(); ++layer) {
      write_floats(directory / ("layer_" + std::to_string(layer) + ".f32"),
                   state.hidden_states[layer]);
    }
    Sampler greedy;
    std::vector<std::uint32_t> generated;
    for (std::uint32_t step = 0; step < count; ++step) {
      generated.push_back(greedy.sample(std::span(state.logits).last(config.vocab)));
      if (step + 1 < count) state = transformer.decode(generated.back());
    }
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(directory / "output.json");
    output << "{\"positions\":" << tokens.size() << ",\"vocab\":" << config.vocab
           << ",\"d_model\":" << config.d_model << ",\"layers\":" << config.layers
           << ",\"kernel\":\"" << arguments[4].string() << "\",\"greedy_ids\":[";
    for (std::size_t i = 0; i < generated.size(); ++i) {
      if (i != 0) output << ',';
      output << generated[i];
    }
    output << "]}\n";
    std::cout << "PASS " << arguments[4].string() << " forward and " << count
              << " cached greedy tokens\n"
              << std::flush;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "optimized check error: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace nanoinfer
