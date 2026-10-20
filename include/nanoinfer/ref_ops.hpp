#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace nanoinfer::ops::ref {

// Contiguous row-major buffers. Outputs must not overlap inputs, except for
// the explicitly in-place bias, residual, GELU and softmax operations.
[[nodiscard]] std::size_t elements(std::size_t rows, std::size_t columns);
void embedding(std::span<const std::uint32_t> ids, std::span<const float> weights,
               std::size_t vocabulary, std::size_t width, std::span<float> output);
void matmul(std::span<const float> a, std::span<const float> b, std::size_t rows,
            std::size_t columns, std::size_t inner, std::span<float> output,
            bool transpose_b = false);
void bias(std::span<float> values, std::span<const float> bias_values);
void residual(std::span<float> values, std::span<const float> addition);
void layer_norm(std::span<const float> input, std::span<const float> scale,
                std::span<const float> shift, std::span<float> output);
void gelu(std::span<float> values);
void softmax(std::span<float> values);
// QKV is [positions, 3 * width], with each Q/K/V segment split into heads.
void causal_attention(std::span<const float> qkv, std::size_t positions, std::size_t width,
                      std::size_t heads, std::span<float> output);

}  // namespace nanoinfer::ops::ref
