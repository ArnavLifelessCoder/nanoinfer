// End-to-end prefill and decode timing of the cached runtime; prints JSON.
//
// e2e_bench MODEL IDS_FILE THREADS GEMM REPETITIONS PREFILL_LENGTHS DECODE_PROMPT DECODE_TOKENS
//   PREFILL_LENGTHS is comma separated, e.g. 128,512. Token IDs are the first
//   tokens of IDS_FILE. Each measurement is preceded by one untimed warm-up.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanoinfer/cached.hpp"
#include "nanoinfer/fp_env.hpp"
#include "nanoinfer/sampling.hpp"

namespace {
using clock_type = std::chrono::steady_clock;

double seconds_since(clock_type::time_point start) {
  return std::chrono::duration<double>(clock_type::now() - start).count();
}

std::vector<std::size_t> numbers(const std::string& text) {
  std::vector<std::size_t> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) values.push_back(std::stoul(item));
  return values;
}

void print_list(const std::vector<double>& values) {
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
  std::cout << ']';
}

std::optional<nanoinfer::kernels::GemmStep> step_named(const std::string& text) {
  if (text == "reference") return std::nullopt;
  for (const auto step :
       {nanoinfer::kernels::GemmStep::packed, nanoinfer::kernels::GemmStep::avx2}) {
    if (text == nanoinfer::kernels::name(step)) return step;
  }
  throw std::invalid_argument("benchmark kernel must be reference, packed or avx2");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 9) {
      throw std::invalid_argument(
          "e2e_bench MODEL IDS THREADS GEMM REPETITIONS PREFILL_LENGTHS DECODE_PROMPT "
          "DECODE_TOKENS");
    }
    nanoinfer::enable_flush_to_zero();
    std::ifstream input(argv[2]);
    std::vector<std::uint32_t> ids;
    for (std::uint32_t id; input >> id;) ids.push_back(id);
    const auto threads = std::stoul(argv[3]);
    const auto repetitions = std::stoul(argv[5]);
    const auto prefill_lengths = numbers(argv[6]);
    const auto decode_prompt = std::stoul(argv[7]);
    const auto decode_tokens = std::stoul(argv[8]);
    nanoinfer::CachedTransformer model(argv[1], step_named(argv[4]), threads);
    std::cout << "{\"threads\":" << model.threads() << ",\"gemm\":\"" << argv[4]
              << "\",\"repetitions\":" << repetitions << ",\"prefill\":[";
    for (std::size_t index = 0; index < prefill_lengths.size(); ++index) {
      const auto length = prefill_lengths[index];
      if (length > ids.size()) throw std::invalid_argument("not enough prompt IDs");
      const auto tokens = std::span(ids).first(length);
      static_cast<void>(model.prefill(tokens));  // Warm-up.
      std::vector<double> times;
      for (std::size_t r = 0; r < repetitions; ++r) {
        const auto start = clock_type::now();
        static_cast<void>(model.prefill(tokens));
        times.push_back(seconds_since(start));
      }
      std::cout << (index ? "," : "") << "{\"tokens\":" << length << ",\"seconds\":";
      print_list(times);
      std::cout << '}';
    }
    std::cout << "],\"decode\":{\"prompt_tokens\":" << decode_prompt
              << ",\"new_tokens\":" << decode_tokens << ",\"runs\":[";
    nanoinfer::Sampler greedy;
    for (std::size_t r = 0; r <= repetitions; ++r) {  // Run 0 is the warm-up.
      auto state = model.prefill(std::span(ids).first(decode_prompt));
      std::vector<double> step_times;
      for (std::size_t step = 0; step < decode_tokens; ++step) {
        const auto token = greedy.sample(std::span(state.logits).last(model.config().vocab));
        const auto start = clock_type::now();
        state = model.decode(token);
        step_times.push_back(seconds_since(start));
      }
      if (r == 0) continue;
      std::cout << (r > 1 ? "," : "");
      print_list(step_times);
    }
    std::cout << "]}}\n" << std::flush;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "e2e_bench error: " << error.what() << '\n';
    return 1;
  }
}
