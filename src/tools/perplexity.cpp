// Perplexity of a token sequence with sliding windows; prints JSON.
//
// gpt2ppl MODEL IDS_FILE WINDOW STRIDE THREADS GEMM
//
// Windows start every STRIDE tokens and hold up to WINDOW tokens. In each
// window, token t is scored from the prefix inside the window, for every t not
// already scored by an earlier window (the first token of the sequence has no
// prefix and is never scored). python/perplexity.py applies the same rule to
// PyTorch, so the two numbers are directly comparable.
#include <cmath>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanoinfer/cached.hpp"
#include "nanoinfer/fp_env.hpp"

namespace {
std::optional<nanoinfer::kernels::GemmStep> step_named(const std::string& text) {
  if (text == "reference") return std::nullopt;
  for (const auto step :
       {nanoinfer::kernels::GemmStep::packed, nanoinfer::kernels::GemmStep::avx2}) {
    if (text == nanoinfer::kernels::name(step)) return step;
  }
  throw std::invalid_argument("kernel must be reference, packed or avx2");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 7) {
      throw std::invalid_argument("gpt2ppl MODEL IDS_FILE WINDOW STRIDE THREADS GEMM");
    }
    nanoinfer::enable_flush_to_zero();
    std::ifstream input(argv[2]);
    std::vector<std::uint32_t> ids;
    for (std::uint32_t id; input >> id;) ids.push_back(id);
    const auto window = std::stoul(argv[3]), stride = std::stoul(argv[4]);
    if (ids.size() < 2 || window < 2 || stride == 0 || stride > window) {
      throw std::invalid_argument("need at least two tokens and 0 < stride <= window");
    }
    nanoinfer::CachedTransformer model(argv[1], step_named(argv[6]), std::stoul(argv[5]));
    const auto vocab = model.config().vocab;
    double nll = 0;
    std::size_t scored = 0, windows = 0, previous_end = 0;
    for (std::size_t begin = 0; begin < ids.size(); begin += stride) {
      const auto end = std::min(ids.size(), begin + window);
      const auto logits = model.prefill(std::span(ids).subspan(begin, end - begin)).logits;
      for (auto t = std::max(begin + 1, previous_end); t < end; ++t) {
        // Row t - begin - 1 predicts token t; log-softmax in double.
        const float* row = logits.data() + (t - begin - 1) * vocab;
        double maximum = row[0];
        for (std::size_t v = 1; v < vocab; ++v) maximum = std::max(maximum, double{row[v]});
        double sum = 0;
        for (std::size_t v = 0; v < vocab; ++v) sum += std::exp(double{row[v]} - maximum);
        nll += maximum + std::log(sum) - row[ids[t]];
        ++scored;
      }
      previous_end = end;
      ++windows;
      if (end == ids.size()) break;
    }
    std::cout.precision(17);
    std::cout << "{\"tokens\":" << ids.size() << ",\"scored\":" << scored
              << ",\"windows\":" << windows << ",\"window\":" << window << ",\"stride\":" << stride
              << ",\"nll_sum\":" << nll
              << ",\"perplexity\":" << std::exp(nll / static_cast<double>(scored)) << "}\n"
              << std::flush;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "perplexity error: " << error.what() << '\n';
    return 1;
  }
}
