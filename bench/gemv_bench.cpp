// Decode GEMV bandwidth: 768 -> 3072 (an MLP input projection, B is [K, N]) and
// 768 -> 50257 (the tied output embedding, B is [N, K]). The counted bytes are
// the weights read per call; the vector and output are negligible.
#include <benchmark/benchmark.h>

#include <cmath>
#include <cstdint>
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
// int8 weights with group-32 f32 scales: counted bytes are the int8 values
// plus the scales, about 1.125 bytes per weight.
void gemv_int8(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto threads = static_cast<std::size_t>(state.range(1));
  constexpr std::size_t k = 768, group = 32;
  const bool transpose_b = n > 4096;
  const auto kernel = nanoinfer::kernels::best_gemv();
  std::mt19937 generator(42);
  std::uniform_real_distribution<float> distribution(-1, 1);
  std::uniform_int_distribution<int> value(-127, 127);
  std::vector<float> x(k), scales(k * n / group), y(n);
  std::vector<std::int8_t> q(k * n);
  for (auto& item : x) item = distribution(generator);
  for (auto& item : scales) item = 0.01F * std::abs(distribution(generator));
  for (auto& item : q) item = static_cast<std::int8_t>(value(generator));
  const nanoinfer::kernels::QuantizedMatrix matrix{q, scales, group};
  auto pool = std::make_unique<nanoinfer::ThreadPool>(threads);
  for (auto _ : state) {
    nanoinfer::kernels::gemv(kernel, x, matrix, n, k, y, transpose_b, pool.get());
    benchmark::DoNotOptimize(y.data());
    benchmark::ClobberMemory();
  }
  state.counters["bytes"] =
      benchmark::Counter(static_cast<double>(q.size() + scales.size() * sizeof(float)),
                         benchmark::Counter::kIsIterationInvariantRate);
  state.SetLabel(transpose_b ? "int8 transposed" : "int8");
}
BENCHMARK(gemv_int8)
    ->ArgNames({"n", "threads"})
    ->ArgsProduct({{3072, 50257}, {1, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->MinTime(0.5);

BENCHMARK(gemv_f32)
    ->ArgNames({"n", "threads"})
    ->ArgsProduct({{3072, 50257}, {1, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->MinTime(0.5);
}  // namespace
