#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace nanoinfer {

// One logical CPU per physical core, in topology order (SMT siblings skipped).
[[nodiscard]] std::vector<unsigned> physical_core_cpus();

// Fixed set of worker threads. parallel_for statically splits [0, count) into
// one contiguous chunk per participant; the calling thread runs the first
// chunk and the workers the rest, and the call returns once every chunk is
// done. Static partitioning keeps scheduling deterministic: chunk boundaries
// depend only on count and the thread count. Calls must not overlap.
//
// Decode issues about a hundred short parallel regions per token, so idle
// workers spin briefly on an atomic generation counter before sleeping on a
// condition variable; the caller only takes the lock to wake sleepers.
class ThreadPool {
 public:
  // threads counts the caller, so ThreadPool(1) runs everything inline. With
  // pin set, participant i runs on physical core i (the caller included).
  explicit ThreadPool(std::size_t threads, bool pin = true);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  [[nodiscard]] std::size_t size() const noexcept { return workers_.size() + 1; }

  // fn(begin, end) for each non-empty chunk. The first exception thrown by any
  // chunk is rethrown on the caller after all chunks have finished.
  void parallel_for(std::size_t count, const std::function<void(std::size_t, std::size_t)>& fn);

 private:
  void worker(std::size_t index);
  void run_chunk(std::size_t participant, std::size_t count,
                 const std::function<void(std::size_t, std::size_t)>& fn);

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable start_;
  // job_ and count_ are written before generation_ is released and read after
  // it is acquired, so plain members are safe.
  const std::function<void(std::size_t, std::size_t)>* job_ = nullptr;
  std::size_t count_ = 0;
  std::atomic<std::size_t> generation_{0};
  std::atomic<std::size_t> pending_{0};
  std::atomic<std::size_t> sleepers_{0};
  std::atomic<bool> stop_{false};
  std::exception_ptr error_;  // Guarded by mutex_.
};

}  // namespace nanoinfer
