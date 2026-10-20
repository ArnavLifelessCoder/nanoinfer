#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "nanoinfer/ref_ops.hpp"

namespace nanoinfer::ops::ref {
namespace {
void require(bool condition) {
  if (!condition) {
    throw std::invalid_argument("reference op shape mismatch");
  }
}
}  // namespace

std::size_t elements(std::size_t rows, std::size_t columns) {
  if (columns != 0 && rows > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::overflow_error("tensor element count overflow");
  }
  return rows * columns;
}

void embedding(std::span<const std::uint32_t> ids, std::span<const float> weights,
               std::size_t vocabulary, std::size_t width, std::span<float> output) {
  require(width > 0 && weights.size() == elements(vocabulary, width) &&
          output.size() == elements(ids.size(), width));
  for (std::size_t row = 0; row < ids.size(); ++row) {
    if (ids[row] >= vocabulary) {
      throw std::out_of_range("token or position ID outside embedding");
    }
    const auto* source = weights.data() + static_cast<std::size_t>(ids[row]) * width;
    std::copy_n(source, width, output.data() + row * width);
  }
}

void matmul(std::span<const float> a, std::span<const float> b, std::size_t rows,
            std::size_t columns, std::size_t inner, std::span<float> output, bool transpose_b) {
  require(inner > 0 && columns > 0 && a.size() == elements(rows, inner) &&
          b.size() == elements(inner, columns) && output.size() == elements(rows, columns));
  const auto* left = a.data();
  const auto* right = b.data();
  auto* destination = output.data();
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < columns; ++j) {
      float sum = 0.0F;
      for (std::size_t k = 0; k < inner; ++k) {
        const auto index = transpose_b ? j * inner + k : k * columns + j;
        sum += left[i * inner + k] * right[index];
      }
      destination[i * columns + j] = sum;
    }
  }
}

void bias(std::span<float> values, std::span<const float> bias_values) {
  require(!bias_values.empty() && values.size() % bias_values.size() == 0);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] += bias_values[i % bias_values.size()];
  }
}

void residual(std::span<float> values, std::span<const float> addition) {
  require(values.size() == addition.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] += addition[i];
  }
}

void layer_norm(std::span<const float> input, std::span<const float> scale,
                std::span<const float> shift, std::span<float> output) {
  const auto width = scale.size();
  require(width > 0 && shift.size() == width && input.size() % width == 0 &&
          output.size() == input.size());
  for (std::size_t row = 0; row < input.size() / width; ++row) {
    const auto* source = input.data() + row * width;
    float mean = 0.0F;
    for (std::size_t j = 0; j < width; ++j) {
      mean += source[j];
    }
    mean /= static_cast<float>(width);
    float variance = 0.0F;
    for (std::size_t j = 0; j < width; ++j) {
      const float centered = source[j] - mean;
      variance += centered * centered;
    }
    variance /= static_cast<float>(width);
    const float inverse = 1.0F / std::sqrt(variance + 1e-5F);
    for (std::size_t j = 0; j < width; ++j) {
      output[row * width + j] = (source[j] - mean) * inverse * scale[j] + shift[j];
    }
  }
}

void gelu(std::span<float> values) {
  for (auto& value : values) {
    const float cube = value * value * value;
    value = 0.5F * value * (1.0F + std::tanh(0.7978845608028654F * (value + 0.044715F * cube)));
  }
}

void softmax(std::span<float> values) {
  require(!values.empty());
  const auto maximum = *std::max_element(values.begin(), values.end());
  float sum = 0.0F;
  for (auto& value : values) {
    value = std::exp(value - maximum);
    sum += value;
  }
  for (auto& value : values) {
    value /= sum;
  }
}

void causal_attention(std::span<const float> qkv, std::size_t positions, std::size_t width,
                      std::size_t heads, std::span<float> output) {
  require(heads > 0 && width > 0 && width % heads == 0 &&
          qkv.size() == elements(positions, elements(3, width)) &&
          output.size() == elements(positions, width));
  const auto head_width = width / heads;
  const auto scale = 1.0F / std::sqrt(static_cast<float>(head_width));
  std::vector<float> scores(positions);
  const auto* fused = qkv.data();
  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t query = 0; query < positions; ++query) {
      for (std::size_t key = 0; key <= query; ++key) {
        float dot = 0.0F;
        for (std::size_t k = 0; k < head_width; ++k) {
          dot += fused[query * 3 * width + head * head_width + k] *
                 fused[key * 3 * width + width + head * head_width + k];
        }
        scores[key] = dot * scale;
      }
      softmax(std::span(scores).first(query + 1));
      for (std::size_t k = 0; k < head_width; ++k) {
        float sum = 0.0F;
        for (std::size_t key = 0; key <= query; ++key) {
          sum += scores[key] * fused[key * 3 * width + 2 * width + head * head_width + k];
        }
        output[query * width + head * head_width + k] = sum;
      }
    }
  }
}
}  // namespace nanoinfer::ops::ref
