// Decode GEMV bandwidth: 768 -> 3072 (an MLP input projection, B is [K, N]) and
// 768 -> 50257 (the tied output embedding, B is [N, K]). The counted bytes are
// the weights read per call; the vector and output are negligible.
#include <benchmark/benchmark.h>

#include <memory>
#include <random>
#include <vector>

#include "nanoinfer/gemv.hpp"
#include "nanoinfer/thread_pool.hpp"

namespace {
using nanoinfer::kernels::GemvKernel;

void gemv_f32(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto threads = static_cast<std::size_t>(state.range(1));
  constexpr std::size_t k = 768;
  const bool transpose_b = n > 4096;
  const auto kernel = nanoinfer::kernels::best_gemv();
  std::mt19937 generator(42);
  std::uniform_real_distribution<float> distribution(-1, 1);
  std::vector<float> x(k), b(k * n), y(n);
  for (auto& value : x) value = distribution(generator);
  for (auto& value : b) value = distribution(generator);
  auto pool = std::make_unique<nanoinfer::ThreadPool>(threads);
  for (auto _ : state) {
    nanoinfer::kernels::gemv(kernel, x, b, n, k, y, transpose_b, pool.get());
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.counters["bytes"] = benchmark::Counter(4.0 * static_cast<double>(k * n),
                                               benchmark::Counter::kIsIterationInvariantRate);
  state.SetLabel(transpose_b ? "f32 transposed" : "f32");
}
BENCHMARK(gemv_f32)
    ->ArgNames({"n", "threads"})
    ->ArgsProduct({{3072, 50257}, {1, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->MinTime(0.5);
}  // namespace
