// Single-thread GEMM throughput for every optimization step on the spec shapes.
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "nanoinfer/gemm.hpp"

namespace {
using nanoinfer::kernels::GemmStep;

void gemm(benchmark::State& state) {
  const auto step = static_cast<GemmStep>(state.range(0));
  const auto m = static_cast<std::size_t>(state.range(1));
  const auto n = static_cast<std::size_t>(state.range(2));
  const auto k = static_cast<std::size_t>(state.range(3));
  if (!nanoinfer::kernels::available(step)) {
    state.SkipWithError("kernel not supported on this CPU");
    return;
  }
  std::mt19937 generator(42);
  std::uniform_real_distribution<float> distribution(-1, 1);
  std::vector<float> a(m * k), b(k * n), c(m * n);
  for (auto& value : a) value = distribution(generator);
  for (auto& value : b) value = distribution(generator);
  for (auto _ : state) {
    nanoinfer::kernels::gemm(step, a, b, m, n, k, c);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
  state.SetLabel(std::string(nanoinfer::kernels::name(step)));
  state.counters["flops"] = benchmark::Counter(2.0 * static_cast<double>(m * n * k),
                                               benchmark::Counter::kIsIterationInvariantRate);
}

void shapes(benchmark::internal::Benchmark* benchmark) {
  for (int step = 0; step <= static_cast<int>(GemmStep::avx2); ++step) {
    benchmark->Args({step, 768, 768, 768});
    benchmark->Args({step, 1024, 3072, 768});
  }
}
BENCHMARK(gemm)
    ->Apply(shapes)
    ->ArgNames({"step", "m", "n", "k"})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->MinTime(0.5);
}  // namespace
