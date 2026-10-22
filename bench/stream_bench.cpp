// STREAM-style triad a[i] = b[i] + s * c[i] over arrays far larger than the
// caches, split across the thread pool. Bytes counted per STREAM: 3 x 4 x N
// per pass (write-allocate traffic for a is not counted).
#include <benchmark/benchmark.h>

#include <memory>
#include <vector>

#include "nanoinfer/thread_pool.hpp"

namespace nanoinfer::bench {
float sum_avx2(const float* values, std::size_t count);
}  // namespace nanoinfer::bench

namespace {
constexpr std::size_t kElements = std::size_t{1} << 26;  // 256 MiB per array.

void triad(benchmark::State& state) {
  const auto threads = static_cast<std::size_t>(state.range(0));
  nanoinfer::ThreadPool pool(threads);
  std::vector<float> a(kElements), b(kElements, 1.0F), c(kElements, 2.0F);
  // First touch by the threads that will use each range.
  pool.parallel_for(kElements, [&](std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; ++i) a[i] = 0.0F;
  });
  const float scale = 3.0F;
  for (auto _ : state) {
    pool.parallel_for(kElements, [&](std::size_t begin, std::size_t end) {
      float* __restrict out = a.data();
      const float* __restrict left = b.data();
      const float* __restrict right = c.data();
      for (auto i = begin; i < end; ++i) out[i] = left[i] + scale * right[i];
    });
    benchmark::DoNotOptimize(a.data());
    benchmark::ClobberMemory();
  }
  state.counters["bytes"] = benchmark::Counter(3.0 * sizeof(float) * kElements,
                                               benchmark::Counter::kIsIterationInvariantRate);
  state.counters["threads"] = static_cast<double>(threads);
}
// Read-only stream: sum one large array. Decode reads weights and writes
// almost nothing, so this is the ceiling the GEMV is compared with; the triad
// also pays for writes.
void read(benchmark::State& state) {
  const auto threads = static_cast<std::size_t>(state.range(0));
  nanoinfer::ThreadPool pool(threads);
  std::vector<float> data(kElements, 1.0F);
  std::vector<float> partial(threads * 16, 0.0F);  // One cache line per participant.
  for (auto _ : state) {
    pool.parallel_for(threads, [&](std::size_t begin, std::size_t end) {
      for (auto chunk = begin; chunk < end; ++chunk) {
        // Chunk boundaries rounded to 64 floats; kElements is a multiple of 64 x 16.
        const auto first = kElements / threads / 64 * 64 * chunk;
        const auto last = chunk + 1 == threads ? kElements : first + kElements / threads / 64 * 64;
        partial[chunk * 16] = nanoinfer::bench::sum_avx2(data.data() + first, last - first);
      }
    });
    benchmark::DoNotOptimize(partial.data());
    benchmark::ClobberMemory();
  }
  state.counters["bytes"] = benchmark::Counter(sizeof(float) * static_cast<double>(kElements),
                                               benchmark::Counter::kIsIterationInvariantRate);
  state.counters["threads"] = static_cast<double>(threads);
}
BENCHMARK(read)
    ->ArgName("threads")
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->MinTime(1.0);

BENCHMARK(triad)
    ->ArgName("threads")
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime()
    ->MinTime(1.0);
}  // namespace
