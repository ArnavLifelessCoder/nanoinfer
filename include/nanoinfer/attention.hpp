#pragma once

#include <cstddef>
#include <span>

#include "nanoinfer/gemv.hpp"

namespace nanoinfer::kernels {

// Optimized twin of ops::ref::cached_attention, same arguments and result up
// to float rounding: dot products and the weighted sum of V use the AVX2 (or
// portable) vector primitives, and the work is split into units of one head
// times kAttentionRows query rows so that it parallelizes past the head count.
inline constexpr std::size_t kAttentionRows = 16;

[[nodiscard]] std::size_t attention_units(std::size_t rows, std::size_t heads) noexcept;

// Computes units [unit_begin, unit_end). Units write disjoint parts of output.
void cached_attention(GemvKernel kernel, std::span<const float> fused_qkv,
                      std::span<const float> keys, std::span<const float> values,
                      std::size_t first_position, std::size_t width, std::size_t heads,
                      std::span<float> output, std::size_t unit_begin, std::size_t unit_end);

}  // namespace nanoinfer::kernels
