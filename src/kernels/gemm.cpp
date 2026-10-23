// Portable GEMM steps, panel packing, the blocked driver and runtime CPU
// dispatch. Built with the baseline instruction set; only the micro-kernel in
// gemm_avx2.cpp uses AVX2/FMA, and it is called only after a runtime check.
#include "nanoinfer/gemm.hpp"

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>

#include "kernels/micro_kernel.hpp"
#include "nanoinfer/ref_ops.hpp"
#include "nanoinfer/thread_pool.hpp"

#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
#include <intrin.h>
#endif

namespace nanoinfer::kernels {
namespace {
using detail::kKc;
using detail::kMc;
using detail::kMr;
using detail::kNc;
using detail::kNr;

void validate(std::span<const float> a, std::span<const float> b, std::size_t rows,
              std::size_t columns, std::size_t inner, std::span<float> output) {
  if (inner == 0 || columns == 0 || a.size() != ops::ref::elements(rows, inner) ||
      b.size() != ops::ref::elements(inner, columns) ||
      output.size() != ops::ref::elements(rows, columns)) {
    throw std::invalid_argument("gemm shape mismatch");
  }
}

// B(p, j) for either storage order.
inline float b_at(const float* b, std::size_t p, std::size_t j, std::size_t inner,
                  std::size_t columns, bool transpose_b) {
  return transpose_b ? b[j * inner + p] : b[p * columns + j];
}

// y[0..count) += x * source[0..count). Steps 2-4 rely on the compiler to
// vectorize this one contiguous loop; keeping it in one place, with no
// surrounding index arithmetic, keeps every step vectorizable.
inline void axpy(float* __restrict y, const float* __restrict source, float x, std::size_t count) {
  for (std::size_t j = 0; j < count; ++j) y[j] += x * source[j];
}

void gemm_naive(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                std::size_t k, bool transpose_b) {
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float sum = 0.0F;
      for (std::size_t p = 0; p < k; ++p) sum += a[i * k + p] * b_at(b, p, j, k, n, transpose_b);
      c[i * n + j] = sum;
    }
  }
}

void gemm_reordered(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                    std::size_t k, bool transpose_b) {
  std::fill(c, c + m * n, 0.0F);
  for (std::size_t i = 0; i < m; ++i) {
    float* row = c + i * n;
    for (std::size_t p = 0; p < k; ++p) {
      const float x = a[i * k + p];
      if (transpose_b) {
        for (std::size_t j = 0; j < n; ++j) row[j] += x * b[j * k + p];
      } else {
        axpy(row, b + p * n, x, n);
      }
    }
  }
}

void gemm_blocked(const float* a, const float* b, float* c, std::size_t m, std::size_t n,
                  std::size_t k, bool transpose_b) {
  // Scalar tiles sized for the same caches as the packed kernel: a 64 x 256
  // block of C and a 256 x 256 block of B.
  constexpr std::size_t block_m = 64, block_n = 256, block_k = 256;
  std::fill(c, c + m * n, 0.0F);
  for (std::size_t jc = 0; jc < n; jc += block_n) {
    const auto j_end = std::min(n, jc + block_n);
    for (std::size_t pc = 0; pc < k; pc += block_k) {
      const auto p_end = std::min(k, pc + block_k);
      for (std::size_t ic = 0; ic < m; ic += block_m) {
        const auto i_end = std::min(m, ic + block_m);
        for (std::size_t i = ic; i < i_end; ++i) {
          float* row = c + i * n;
          for (std::size_t p = pc; p < p_end; ++p) {
            const float x = a[i * k + p];
            if (transpose_b) {
              for (std::size_t j = jc; j < j_end; ++j) row[j] += x * b[j * k + p];
            } else {
              axpy(row + jc, b + p * n + jc, x, j_end - jc);
            }
          }
        }
      }
    }
  }
}

// Reusable 64-byte aligned packing buffers, one pair per thread.
struct AlignedDelete {
  void operator()(float* pointer) const noexcept {
    ::operator delete[](pointer, std::align_val_t{64});
  }
};

class Workspace {
 public:
  float* get(std::size_t count) {
    if (count > capacity_) {
      data_.reset(
          static_cast<float*>(::operator new[](count * sizeof(float), std::align_val_t{64})));
      capacity_ = count;
    }
    return data_.get();
  }

 private:
  std::unique_ptr<float[], AlignedDelete> data_;
  std::size_t capacity_ = 0;
};

// MR-row panels of an mc x kc block of A, column by column, zero padded.
void pack_a(const float* a, std::size_t lda, std::size_t row0, std::size_t mc, std::size_t col0,
            std::size_t kc, float* destination) {
  for (std::size_t ir = 0; ir < mc; ir += kMr) {
    const auto valid = std::min(kMr, mc - ir);
    for (std::size_t p = 0; p < kc; ++p) {
      for (std::size_t r = 0; r < kMr; ++r) {
        *destination++ = r < valid ? a[(row0 + ir + r) * lda + col0 + p] : 0.0F;
      }
    }
  }
}

// Element sources for B: float32, or int8 dequantized while packing. Packed
// panels are float32 either way, so the micro-kernels are shared.
struct FloatSource {
  const float* b;
  std::size_t inner, columns;
  bool transpose_b;
  float operator()(std::size_t p, std::size_t j) const {
    return b_at(b, p, j, inner, columns, transpose_b);
  }
};

struct QuantizedSource {
  const std::int8_t* q;
  const float* scales;
  std::size_t inner, columns, group;
  bool transpose_b;
  // float(q) * scale, the exact dequantization of python/quantize.py.
  float operator()(std::size_t p, std::size_t j) const {
    if (transpose_b) {
      return static_cast<float>(q[j * inner + p]) * scales[j * (inner / group) + p / group];
    }
    return static_cast<float>(q[p * columns + j]) * scales[(p / group) * columns + j];
  }
};

// One NR-column panel (number `panel`) of a kc x nc block of B, row by row,
// zero padded. Panels are independent, so they can be packed in parallel.
template <typename Source>
void pack_b_panel(const Source& source, std::size_t row0, std::size_t kc, std::size_t col0,
                  std::size_t nc, std::size_t panel, float* packed) {
  const auto jr = panel * kNr;
  const auto valid = std::min(kNr, nc - jr);
  float* destination = packed + jr * kc;
  for (std::size_t p = 0; p < kc; ++p) {
    for (std::size_t j = 0; j < kNr; ++j) {
      *destination++ = j < valid ? source(row0 + p, col0 + jr + j) : 0.0F;
    }
  }
}

template <typename Fn>
void run(ThreadPool* pool, std::size_t count, const Fn& fn) {
  if (pool == nullptr) {
    if (count != 0) fn(std::size_t{0}, count);
  } else {
    pool->parallel_for(count, fn);
  }
}

void micro_kernel_scalar(std::size_t depth, const float* a, const float* b, float* c,
                         std::size_t ldc, std::size_t rows, std::size_t cols, bool accumulate) {
  // One output row at a time: its 16 sums stay in registers across the whole
  // depth, instead of the tile round-tripping through memory every step.
  float tile[kMr * kNr];
  for (std::size_t i = 0; i < kMr; ++i) {
    float sums[kNr] = {};
    for (std::size_t p = 0; p < depth; ++p) axpy(sums, b + p * kNr, a[p * kMr + i], kNr);
    for (std::size_t j = 0; j < kNr; ++j) tile[i * kNr + j] = sums[j];
  }
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      c[i * ldc + j] = accumulate ? c[i * ldc + j] + tile[i * kNr + j] : tile[i * kNr + j];
    }
  }
}

// NR panels per work unit: one unit is an MC-row block of A times 8 panels
// (128 columns) of B. A shape constant, never derived from the thread count.
constexpr std::size_t kPanelsPerUnit = 8;

template <typename Source>
void gemm_packed(detail::MicroKernel kernel, const float* a, const Source& source, float* c,
                 std::size_t m, std::size_t n, std::size_t k, ThreadPool* pool) {
  // The caller's packed B block is shared read-only by every thread.
  thread_local Workspace b_space;
  float* packed_b = b_space.get(kKc * ((std::min(n, kNc) + kNr - 1) / kNr * kNr));
  const auto row_blocks = (m + kMc - 1) / kMc;
  for (std::size_t jc = 0; jc < n; jc += kNc) {
    const auto nc = std::min(kNc, n - jc);
    const auto panels = (nc + kNr - 1) / kNr;
    const auto groups = (panels + kPanelsPerUnit - 1) / kPanelsPerUnit;
    for (std::size_t pc = 0; pc < k; pc += kKc) {
      const auto kc = std::min(kKc, k - pc);
      run(pool, panels, [&](std::size_t begin, std::size_t end) {
        for (auto panel = begin; panel < end; ++panel) {
          pack_b_panel(source, pc, kc, jc, nc, panel, packed_b);
        }
      });
      // Units are ordered by row block, so a thread's contiguous range of
      // units repacks its A block only when the row block changes.
      run(pool, row_blocks * groups, [&](std::size_t begin, std::size_t end) {
        thread_local Workspace a_space;
        float* packed_a = a_space.get(kKc * ((std::min(m, kMc) + kMr - 1) / kMr * kMr));
        std::size_t packed_block = row_blocks;
        for (auto unit = begin; unit < end; ++unit) {
          const auto block = unit / groups, group = unit % groups;
          const auto ic = block * kMc;
          const auto mc = std::min(kMc, m - ic);
          if (block != packed_block) {
            pack_a(a, k, ic, mc, pc, kc, packed_a);
            packed_block = block;
          }
          const auto panel_end = std::min(panels, (group + 1) * kPanelsPerUnit);
          for (auto panel = group * kPanelsPerUnit; panel < panel_end; ++panel) {
            const auto jr = panel * kNr;
            for (std::size_t ir = 0; ir < mc; ir += kMr) {
              kernel(kc, packed_a + ir * kc, packed_b + jr * kc, c + (ic + ir) * n + jc + jr, n,
                     std::min(kMr, mc - ir), std::min(kNr, nc - jr), pc != 0);
            }
          }
        }
      });
    }
  }
}

CpuFeatures detect() noexcept {
  CpuFeatures features;
#if defined(_MSC_VER) && !defined(__clang__) && defined(_M_X64)
  int registers[4];
  __cpuid(registers, 0);
  const int highest = registers[0];
  __cpuid(registers, 1);
  const bool osxsave = (registers[2] >> 27) & 1;
  const bool avx = (registers[2] >> 28) & 1;
  const bool fma = (registers[2] >> 12) & 1;
  const unsigned long long xcr0 = osxsave ? _xgetbv(0) : 0;
  const bool ymm_state = (xcr0 & 0x6) == 0x6;
  const bool zmm_state = (xcr0 & 0xe6) == 0xe6;
  bool avx2 = false, avx512f = false;
  if (highest >= 7) {
    __cpuidex(registers, 7, 0);
    avx2 = (registers[1] >> 5) & 1;
    avx512f = (registers[1] >> 16) & 1;
  }
  features.avx2 = avx && avx2 && ymm_state;
  features.fma = avx && fma && ymm_state;
  features.avx512f = avx512f && zmm_state;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
  // libgcc also checks that the operating system saves the vector state.
  __builtin_cpu_init();
  features.avx2 = __builtin_cpu_supports("avx2");
  features.fma = __builtin_cpu_supports("fma");
  features.avx512f = __builtin_cpu_supports("avx512f");
#endif
  return features;
}
}  // namespace

std::string_view name(GemmStep step) noexcept {
  switch (step) {
    case GemmStep::naive:
      return "naive";
    case GemmStep::reordered:
      return "reordered";
    case GemmStep::blocked:
      return "blocked";
    case GemmStep::packed:
      return "packed";
    case GemmStep::avx2:
      return "avx2";
  }
  return "unknown";
}

const CpuFeatures& cpu_features() noexcept {
  static const CpuFeatures features = detect();
  return features;
}

bool available(GemmStep step) noexcept {
  if (step != GemmStep::avx2) return true;
#if defined(NANOINFER_HAVE_AVX2_KERNEL)
  return cpu_features().avx2 && cpu_features().fma;
#else
  return false;
#endif
}

GemmStep best_step() noexcept {
  return available(GemmStep::avx2) ? GemmStep::avx2 : GemmStep::packed;
}

void gemm(GemmStep step, std::span<const float> a, std::span<const float> b, std::size_t rows,
          std::size_t columns, std::size_t inner, std::span<float> output, bool transpose_b,
          ThreadPool* pool) {
  validate(a, b, rows, columns, inner, output);
  if (rows == 0) return;
  const float* left = a.data();
  const float* right = b.data();
  float* destination = output.data();
  switch (step) {
    case GemmStep::naive:
      return gemm_naive(left, right, destination, rows, columns, inner, transpose_b);
    case GemmStep::reordered:
      return gemm_reordered(left, right, destination, rows, columns, inner, transpose_b);
    case GemmStep::blocked:
      return gemm_blocked(left, right, destination, rows, columns, inner, transpose_b);
    case GemmStep::packed:
      return gemm_packed(micro_kernel_scalar, left, FloatSource{right, inner, columns, transpose_b},
                         destination, rows, columns, inner, pool);
    case GemmStep::avx2:
#if defined(NANOINFER_HAVE_AVX2_KERNEL)
      if (!available(GemmStep::avx2)) {
        throw std::runtime_error("the AVX2 GEMM needs a CPU with AVX2 and FMA");
      }
      return gemm_packed(detail::micro_kernel_avx2, left,
                         FloatSource{right, inner, columns, transpose_b}, destination, rows,
                         columns, inner, pool);
#else
      throw std::runtime_error("this build has no AVX2 GEMM");
#endif
  }
  throw std::invalid_argument("unknown GEMM step");
}
void gemm(GemmStep step, std::span<const float> a, const QuantizedMatrix& b, std::size_t rows,
          std::size_t columns, std::size_t inner, std::span<float> output, bool transpose_b,
          ThreadPool* pool) {
  if (inner == 0 || columns == 0 || a.size() != ops::ref::elements(rows, inner) ||
      output.size() != ops::ref::elements(rows, columns)) {
    throw std::invalid_argument("gemm shape mismatch");
  }
  validate(b, columns, inner);
  if (rows == 0) return;
  const QuantizedSource source{b.values.data(), b.scales.data(), inner,
                               columns,         b.group,         transpose_b};
  if (step == GemmStep::packed) {
    return gemm_packed(micro_kernel_scalar, a.data(), source, output.data(), rows, columns, inner,
                       pool);
  }
#if defined(NANOINFER_HAVE_AVX2_KERNEL)
  if (step == GemmStep::avx2) {
    if (!available(GemmStep::avx2)) {
      throw std::runtime_error("the AVX2 GEMM needs a CPU with AVX2 and FMA");
    }
    return gemm_packed(detail::micro_kernel_avx2, a.data(), source, output.data(), rows, columns,
                       inner, pool);
  }
#endif
  throw std::invalid_argument("int8 GEMM supports the packed and avx2 steps");
}
}  // namespace nanoinfer::kernels
