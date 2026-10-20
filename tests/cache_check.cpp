// Uses gpt2ref's positional interface and platform entry point for golden checks.
#include <charconv>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nanoinfer/cached.hpp"
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

void require_equal(std::span<const float> actual, std::span<const float> expected) {
  if (actual.size() != expected.size()) throw std::runtime_error("comparison shape mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || actual[i] != expected[i]) {
      throw std::runtime_error("cache differs from full recomputation at element " +
                               std::to_string(i));
    }
  }
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
    if (arguments.size() != 4) {
      throw std::invalid_argument("gpt2cachecheck MODEL IDS OUTPUT_DIRECTORY NEW_TOKENS");
    }
    const auto count = number(arguments[3].string());
    ReferenceTransformer reference(arguments[0]);
    CachedTransformer cached(arguments[0]);
    Sampler sampler;
    std::ifstream input(arguments[1]);
    if (!input) throw std::runtime_error("cannot open prompt IDs");
    std::vector<std::uint32_t> tokens;
    std::string text;
    while (input >> text) tokens.push_back(number(text));
    if (!input.eof()) throw std::runtime_error("cannot read prompt IDs");
    const auto prompt_size = tokens.size();
    const auto& config = reference.config();
    if (tokens.size() > config.ctx || count > config.ctx - tokens.size()) {
      throw std::invalid_argument("context exceeded");
    }
    auto state = cached.prefill(tokens, true);
    auto full = reference.forward(tokens);
    require_equal(state.logits, full.logits);
    require_equal(state.final_ln, full.final_ln);
    for (std::size_t layer = 0; layer < full.hidden_states.size(); ++layer) {
      require_equal(state.hidden_states[layer], full.hidden_states[layer]);
    }
    const auto& directory = arguments[2];
    std::filesystem::create_directories(directory);
    write_floats(directory / "logits.f32", state.logits);
    write_floats(directory / "final_ln.f32", state.final_ln);
    for (std::size_t layer = 0; layer < state.hidden_states.size(); ++layer) {
      write_floats(directory / ("layer_" + std::to_string(layer) + ".f32"),
                   state.hidden_states[layer]);
    }
    std::ofstream checks;
    checks.exceptions(std::ios::failbit | std::ios::badbit);
    checks.open(directory / "cache-check.json");
    checks << "{\"comparison\":\"exact float equality, every logit and hidden channel\","
              "\"vocab\":"
           << config.vocab << ",\"layers\":" << config.layers << ",\"d_model\":" << config.d_model
           << ",\"positions\":[";
    auto record = [&checks](std::size_t position, bool prefill) {
      if (position != 0) checks << ',';
      checks << "{\"absolute_position\":" << position << ",\"phase\":\""
             << (prefill ? "prefill" : "decode")
             << "\",\"max_abs_logit_diff\":0,\"max_abs_hidden_diff\":0}";
    };
    for (std::size_t position = 0; position < prompt_size; ++position) record(position, true);
    std::vector<std::uint32_t> generated;
    for (std::uint32_t step = 0; step < count; ++step) {
      const auto token = sampler.sample(std::span(state.logits).last(config.vocab));
      generated.push_back(token);
      tokens.push_back(token);
      state = cached.decode(token, true);
      full = reference.forward(tokens, true, true);
      require_equal(state.logits, full.logits);
      require_equal(state.final_ln, std::span(full.final_ln).last(config.d_model));
      for (std::size_t layer = 0; layer < full.hidden_states.size(); ++layer) {
        require_equal(state.hidden_states[layer],
                      std::span(full.hidden_states[layer]).last(config.d_model));
      }
      record(tokens.size() - 1, false);
      std::cout << "PASS absolute position " << tokens.size() - 1
                << ": all logits and hidden states equal full recomputation\n"
                << std::flush;
    }
    checks << "],\"passed\":true}\n";
    std::ofstream output;
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.open(directory / "output.json");
    output << "{\"positions\":" << prompt_size << ",\"vocab\":" << config.vocab
           << ",\"d_model\":" << config.d_model << ",\"layers\":" << config.layers
           << ",\"greedy_ids\":[";
    for (std::size_t i = 0; i < generated.size(); ++i) {
      if (i != 0) output << ',';
      output << generated[i];
    }
    output << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "cache check error: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace nanoinfer
