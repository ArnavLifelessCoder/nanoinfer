// Measures one core's AVX2 FMA throughput and its clock under FMA load. The
// clock comes from a dependent FMA chain: chain steps per second times the FMA
// latency in cycles (bench/run_gemm.py applies the latency it records).
#include <benchmark/benchmark.h>

#include <cstddef>
#include <vector>

#include "nanoinfer/gemm.hpp"
#include "nanoinfer/thread_pool.hpp"

namespace nanoinfer::bench {
float fma_throughput(std::size_t iterations);
float fma_chain(std::size_t iterations);
}  // namespace nanoinfer::bench

namespace {
constexpr std::size_t kIterations = 1 << 24;

bool supported(benchmark::State& state) {
  const auto& features = nanoinfer::kernels::cpu_features();
  if (features.avx2 && features.fma) return true;
  state.SkipWithError("AVX2/FMA not supported on this CPU");
  return false;
}

void fma_peak(benchmark::State& state) {
  if (!supported(state)) return;
  for (auto _ : state) benchmark::DoNotOptimize(nanoinfer::bench::fma_throughput(kIterations));
  // 12 FMAs x 8 lanes x 2 flops per iteration.
  state.counters["flops"] =
      benchmark::Counter(12.0 * 8 * 2 * kIterations, benchmark::Counter::kIsIterationInvariantRate);
}

void fma_latency_chain(benchmark::State& state) {
  if (!supported(state)) return;
  for (auto _ : state) benchmark::DoNotOptimize(nanoinfer::bench::fma_chain(kIterations));
  state.counters["chain_steps"] = benchmark::Counter(static_cast<double>(kIterations),
                                                     benchmark::Counter::kIsIterationInvariantRate);
}
// Every physical core at once (one pinned pool thread each): the all-core
// throughput and the all-core clock under FMA load, for the machine-wide
// compute ceiling. chain_steps is per thread.
void fma_all_cores(benchmark::State& state) {
  if (!supported(state)) return;
  const auto threads = nanoinfer::physical_core_cpus().size();
  nanoinfer::ThreadPool pool(threads);
  std::vector<float> sink(threads * 16);
  for (auto _ : state) {
    pool.parallel_for(threads, [&](std::size_t begin, std::size_t end) {
      for (auto t = begin; t < end; ++t) {
        sink[t * 16] = nanoinfer::bench::fma_throughput(kIterations);
      }
    });
    benchmark::DoNotOptimize(sink.data());
  }
  // 12 FMAs x 8 lanes x 2 flops per iteration per thread.
  state.counters["flops"] = benchmark::Counter(12.0 * 8 * 2 * kIterations * threads,
                                               benchmark::Counter::kIsIterationInvariantRate);
  state.counters["threads"] = static_cast<double>(threads);
}

void fma_all_cores_chain(benchmark::State& state) {
  if (!supported(state)) return;
  const auto threads = nanoinfer::physical_core_cpus().size();
  nanoinfer::ThreadPool pool(threads);
  std::vector<float> sink(threads * 16);
  for (auto _ : state) {
    pool.parallel_for(threads, [&](std::size_t begin, std::size_t end) {
      for (auto t = begin; t < end; ++t) sink[t * 16] = nanoinfer::bench::fma_chain(kIterations);
    });
    benchmark::DoNotOptimize(sink.data());
  }
  state.counters["chain_steps"] = benchmark::Counter(static_cast<double>(kIterations),
                                                     benchmark::Counter::kIsIterationInvariantRate);
  state.counters["threads"] = static_cast<double>(threads);
}

BENCHMARK(fma_peak)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(fma_all_cores)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(fma_all_cores_chain)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(fma_latency_chain)->Unit(benchmark::kMillisecond)->UseRealTime();
}  // namespace
