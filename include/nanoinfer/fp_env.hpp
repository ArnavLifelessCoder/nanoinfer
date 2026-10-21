#pragma once

namespace nanoinfer {
// Sets flush-to-zero and denormals-are-zero for the calling thread, so
// subnormal intermediates cannot slow the kernels down. MXCSR is per thread:
// every worker thread that runs kernels calls this too.
void enable_flush_to_zero() noexcept;
}  // namespace nanoinfer
