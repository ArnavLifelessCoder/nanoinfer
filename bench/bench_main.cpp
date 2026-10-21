// Shared benchmark entry point: pins the benchmark thread to one core and
// enables flush-to-zero before running the registered benchmarks.
#include <benchmark/benchmark.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "nanoinfer/fp_env.hpp"

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace {
bool pin_to_core(unsigned core) {
#if defined(_WIN32)
  return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << core) != 0;
#else
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#endif
}
}  // namespace

int main(int argc, char** argv) {
  // --pin-core=N chooses the core; the default avoids core 0, which services
  // most interrupts.
  unsigned core = 2;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--pin-core=", 11) == 0) {
      core = static_cast<unsigned>(std::strtoul(argv[i] + 11, nullptr, 10));
      for (int j = i; j + 1 < argc; ++j) argv[j] = argv[j + 1];
      --argc;
      --i;
    }
  }
  if (!pin_to_core(core)) {
    std::cerr << "warning: could not pin the benchmark thread to core " << core << '\n';
  }
  nanoinfer::enable_flush_to_zero();
  benchmark::AddCustomContext("pinned_core", std::to_string(core));
  benchmark::AddCustomContext("flush_to_zero", "on");
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
