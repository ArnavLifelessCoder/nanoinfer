#include "nanoinfer/fp_env.hpp"

#if defined(_M_X64) || defined(__x86_64__) || defined(__SSE3__)
#include <pmmintrin.h>
#include <xmmintrin.h>
#endif

namespace nanoinfer {
void enable_flush_to_zero() noexcept {
#if defined(_M_X64) || defined(__x86_64__) || defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif
}
}  // namespace nanoinfer
