#include <gtest/gtest.h>

#include <atomic>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

#include "nanoinfer/thread_pool.hpp"

namespace nanoinfer {
namespace {
TEST(ThreadPool, CoversEveryIndexExactlyOnceForAnyCountAndSize) {
  for (const std::size_t threads : {1, 2, 3, 4, 7}) {
    ThreadPool pool(threads, false);
    EXPECT_EQ(pool.size(), threads);
    for (const std::size_t count : {0, 1, 2, 5, 6, 7, 64, 1001}) {
      SCOPED_TRACE("threads=" + std::to_string(threads) + " count=" + std::to_string(count));
      std::vector<std::atomic<int>> hits(count);
      pool.parallel_for(count, [&](std::size_t begin, std::size_t end) {
        ASSERT_LT(begin, end);
        for (auto i = begin; i < end; ++i) hits[i].fetch_add(1);
      });
      for (std::size_t i = 0; i < count; ++i) EXPECT_EQ(hits[i].load(), 1) << i;
    }
  }
}

TEST(ThreadPool, ChunkBoundariesDependOnlyOnCountAndSize) {
  ThreadPool pool(4, false);
  for (int repeat = 0; repeat < 50; ++repeat) {
    std::vector<std::atomic<std::size_t>> ends(10);  // Indexed by chunk start.
    pool.parallel_for(10, [&](std::size_t begin, std::size_t end) { ends[begin].store(end); });
    std::vector<std::size_t> chunks;
    for (std::size_t begin = 0; begin < 10; ++begin) {
      if (ends[begin].load() != 0) chunks.push_back(begin * 100 + ends[begin].load());
    }
    // [0,2) [2,5) [5,7) [7,10), encoded as begin * 100 + end.
    EXPECT_EQ(chunks, (std::vector<std::size_t>{2, 205, 507, 710}));
  }
}

TEST(ThreadPool, ReusesWorkersAcrossManyCallsAndSumsCorrectly) {
  ThreadPool pool(4, false);
  for (int call = 0; call < 2000; ++call) {
    std::vector<long long> partial(4, 0);
    pool.parallel_for(400, [&](std::size_t begin, std::size_t end) {
      long long sum = 0;
      for (auto i = begin; i < end; ++i) sum += static_cast<long long>(i);
      partial[begin / 100] = sum;
    });
    ASSERT_EQ(std::accumulate(partial.begin(), partial.end(), 0LL), 399LL * 400 / 2);
  }
}

TEST(ThreadPool, PropagatesTheFirstExceptionAfterAllChunksFinish) {
  ThreadPool pool(4, false);
  std::atomic<int> finished{0};
  EXPECT_THROW(pool.parallel_for(4,
                                 [&](std::size_t begin, std::size_t) {
                                   if (begin % 2 == 1) throw std::runtime_error("chunk failed");
                                   finished.fetch_add(1);
                                 }),
               std::runtime_error);
  EXPECT_EQ(finished.load(), 2);
  // The pool stays usable after an exception.
  std::atomic<int> after{0};
  pool.parallel_for(8, [&](std::size_t begin, std::size_t end) {
    after.fetch_add(static_cast<int>(end - begin));
  });
  EXPECT_EQ(after.load(), 8);
}

TEST(ThreadPool, PinnedPoolUsesDistinctPhysicalCores) {
  const auto cpus = physical_core_cpus();
  ASSERT_FALSE(cpus.empty());
  EXPECT_EQ(std::set<unsigned>(cpus.begin(), cpus.end()).size(), cpus.size());
  ThreadPool pool(std::min<std::size_t>(cpus.size(), 4));
  std::atomic<int> total{0};
  pool.parallel_for(100, [&](std::size_t begin, std::size_t end) {
    total.fetch_add(static_cast<int>(end - begin));
  });
  EXPECT_EQ(total.load(), 100);
}

TEST(ThreadPool, RejectsZeroThreads) { EXPECT_THROW(ThreadPool(0, false), std::invalid_argument); }
}  // namespace
}  // namespace nanoinfer
