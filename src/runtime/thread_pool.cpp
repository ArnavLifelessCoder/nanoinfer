#include "nanoinfer/thread_pool.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "nanoinfer/fp_env.hpp"

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>

#include <fstream>
#include <string>
#endif

namespace nanoinfer {
namespace {
// About a hundred microseconds of spinning before an idle worker sleeps.
constexpr int kSpinsBeforeSleep = 4000;

inline void relax() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

void pin_current_thread(unsigned cpu) {
#if defined(_WIN32)
  if (cpu < 64) SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu);
#else
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}
}  // namespace

std::vector<unsigned> physical_core_cpus() {
  std::vector<unsigned> cpus;
#if defined(_WIN32)
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
  std::vector<unsigned char> buffer(length);
  auto* records = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());
  if (length != 0 && GetLogicalProcessorInformationEx(RelationProcessorCore, records, &length)) {
    for (DWORD offset = 0; offset < length;) {
      const auto* record =
          reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
      const auto& group = record->Processor.GroupMask[0];
      // Single processor group only: up to 64 logical CPUs.
      if (group.Group == 0 && group.Mask != 0) {
        unsigned bit = 0;
        while (((group.Mask >> bit) & 1) == 0) ++bit;
        cpus.push_back(bit);
      }
      offset += record->Size;
    }
  }
#else
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  sched_getaffinity(0, sizeof(allowed), &allowed);
  std::set<std::string> seen;
  for (unsigned cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed)) continue;
    std::ifstream siblings("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                           "/topology/thread_siblings_list");
    std::string list;
    if (!(siblings >> list)) list = std::to_string(cpu);
    if (seen.insert(list).second) cpus.push_back(cpu);
  }
#endif
  std::sort(cpus.begin(), cpus.end());
  if (cpus.empty()) {
    for (unsigned cpu = 0; cpu < std::max(1U, std::thread::hardware_concurrency()); ++cpu) {
      cpus.push_back(cpu);
    }
  }
  return cpus;
}

ThreadPool::ThreadPool(std::size_t threads, bool pin) {
  if (threads == 0) throw std::invalid_argument("a thread pool needs at least one thread");
  const auto cpus = pin ? physical_core_cpus() : std::vector<unsigned>{};
  if (pin) pin_current_thread(cpus[0]);
  workers_.reserve(threads - 1);
  for (std::size_t index = 1; index < threads; ++index) {
    // Copy the CPU id: the worker may start after this constructor returns.
    const unsigned cpu = pin ? cpus[index % cpus.size()] : 0;
    workers_.emplace_back([this, index, pin, cpu] {
      if (pin) pin_current_thread(cpu);
      enable_flush_to_zero();
      worker(index);
    });
  }
}

ThreadPool::~ThreadPool() {
  stop_.store(true);
  {
    // Taking the lock orders the store before any sleeper's predicate check.
    const std::lock_guard lock(mutex_);
  }
  start_.notify_all();
  for (auto& thread : workers_) thread.join();
}

void ThreadPool::run_chunk(std::size_t participant, std::size_t count,
                           const std::function<void(std::size_t, std::size_t)>& fn) {
  const auto begin = count * participant / size();
  const auto end = count * (participant + 1) / size();
  if (begin == end) return;
  try {
    fn(begin, end);
  } catch (...) {
    const std::lock_guard lock(mutex_);
    if (!error_) error_ = std::current_exception();
  }
}

void ThreadPool::worker(std::size_t index) {
  std::size_t seen = 0;
  for (;;) {
    bool ready = false;
    for (int spin = 0; spin < kSpinsBeforeSleep && !ready; ++spin) {
      ready = generation_.load(std::memory_order_acquire) != seen || stop_.load();
      if (!ready) relax();
    }
    if (!ready) {
      std::unique_lock lock(mutex_);
      // Sequentially consistent with the caller's generation increment and
      // sleeper check: either the caller sees this sleeper and notifies, or
      // this predicate sees the new generation. No wake-up is lost.
      sleepers_.fetch_add(1);
      start_.wait(lock, [&] { return stop_.load() || generation_.load() != seen; });
      sleepers_.fetch_sub(1);
    }
    if (stop_.load()) return;
    // The caller cannot publish another generation until this one is done.
    seen = generation_.load(std::memory_order_acquire);
    run_chunk(index, count_, *job_);
    pending_.fetch_sub(1, std::memory_order_release);
  }
}

void ThreadPool::parallel_for(std::size_t count,
                              const std::function<void(std::size_t, std::size_t)>& fn) {
  if (count == 0) return;
  if (workers_.empty()) {
    fn(0, count);
    return;
  }
  job_ = &fn;
  count_ = count;
  pending_.store(workers_.size(), std::memory_order_relaxed);
  generation_.fetch_add(1);  // Publishes job_, count_ and pending_.
  if (sleepers_.load() != 0) {
    const std::lock_guard lock(mutex_);
    start_.notify_all();
  }
  run_chunk(0, count, fn);
  for (std::size_t spin = 0; pending_.load(std::memory_order_acquire) != 0; ++spin) {
    if (spin < (std::size_t{1} << 20)) {
      relax();
    } else {
      std::this_thread::yield();
    }
  }
  std::exception_ptr error;
  {
    const std::lock_guard lock(mutex_);
    error = error_;
    error_ = nullptr;
  }
  job_ = nullptr;
  if (error) std::rethrow_exception(error);
}
}  // namespace nanoinfer
