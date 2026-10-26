#include "tools/generate.hpp"

#include <charconv>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

#include "nanoinfer/cached.hpp"
#include "nanoinfer/fp_env.hpp"
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

// "reference" selects the scalar reference products; "auto" the fastest kernel.
std::optional<kernels::GemmStep> gemm_option(const std::string& text) {
  if (text == "reference") return std::nullopt;
  if (text == "auto") return kernels::best_step();
  for (const auto step :
       {kernels::GemmStep::naive, kernels::GemmStep::reordered, kernels::GemmStep::blocked,
        kernels::GemmStep::packed, kernels::GemmStep::avx2}) {
    if (text == kernels::name(step)) return step;
  }
  throw std::invalid_argument("unknown GEMM kernel: " + text);
}
}  // namespace

int generate_main(std::span<const std::filesystem::path> arguments) {
  if (arguments.size() == 1 && arguments[0] == "--help") {
    std::cout << "gpt2rt --model MODEL --prompt-ids IDS [--max-new 128] [--temp 0]\n"
                 "       [--top-k 0] [--top-p 1] [--seed 42] [--gemm auto]\n"
                 "       [--threads PHYSICAL_CORES] [--stream 0]\n"
                 "stdout: JSON prompt_ids and generated_ids; temp=0 selects greedy.\n"
                 "--threads: worker count for optimized kernels; 1 with --gemm reference.\n"
                 "--stream 1: one JSON line per event (prefill, token, done) as it happens.\n"
                 "--gemm: auto, reference, naive, reordered, blocked, packed or avx2.\n"
              << std::flush;
    return 0;
  }
  try {
    std::filesystem::path model_path, ids_path;
    std::size_t new_tokens = 128;
    std::uint64_t seed = 42;
    std::optional<std::size_t> threads;
    bool stream = false;
    SamplingOptions options;
    std::optional<kernels::GemmStep> gemm = kernels::best_step();
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
      else if (name == "--stream")
        stream = number<unsigned>(value.string()) != 0;
      else if (name == "--threads")
        threads = number<std::size_t>(value.string());
      else if (name == "--gemm")
        gemm = gemm_option(value.string());
      else
        throw std::invalid_argument("unknown option: " + name);
    }
    if (model_path.empty() || ids_path.empty()) {
      throw std::invalid_argument("--model and --prompt-ids are required");
    }
    enable_flush_to_zero();
    Sampler sampler(options, seed);
    std::ifstream input(ids_path);
    if (!input) throw std::runtime_error("cannot open prompt IDs");
    std::vector<std::uint32_t> ids;
    std::string token;
    while (input >> token) ids.push_back(number<std::uint32_t>(token));
    if (!input.eof()) throw std::runtime_error("cannot read prompt IDs");
    const auto workers = threads.value_or(gemm ? physical_core_cpus().size() : 1);
    CachedTransformer transformer(model_path, gemm, workers);
    if (ids.size() > transformer.config().ctx ||
        new_tokens > transformer.config().ctx - ids.size()) {
      throw std::invalid_argument("prompt and continuation exceed model context");
    }
    using clock = std::chrono::steady_clock;
    const auto milliseconds = [](clock::time_point start) {
      return std::chrono::duration<double, std::milli>(clock::now() - start).count();
    };
    auto start = clock::now();
    auto state = transformer.prefill(ids);
    // Each token is reported with the time of the forward pass that produced
    // its logits: the prefill for the first, one decode step for the rest.
    double forward_ms = milliseconds(start);
    if (stream) {
      std::cout << "{\"event\":\"prefill\",\"tokens\":" << ids.size() << ",\"ms\":" << forward_ms
                << ",\"threads\":" << transformer.threads() << "}\n"
                << std::flush;
    }
    std::vector<std::uint32_t> generated;
    generated.reserve(new_tokens);
    for (std::size_t i = 0; i < new_tokens; ++i) {
      generated.push_back(sampler.sample(std::span(state.logits).last(transformer.config().vocab)));
      if (stream) {
        std::cout << "{\"event\":\"token\",\"id\":" << generated.back() << ",\"ms\":" << forward_ms
                  << "}\n"
                  << std::flush;
      }
      if (i + 1 < new_tokens) {
        start = clock::now();
        state = transformer.decode(generated.back());
        forward_ms = milliseconds(start);
      }
    }
    if (stream) {
      std::cout << "{\"event\":\"done\",\"generated\":" << generated.size() << "}\n" << std::flush;
      return 0;
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
