#include <limits>

int main() {
  // Deliberate UB verifies that the sanitizer gate rejects a reported violation.
  volatile int maximum = std::numeric_limits<int>::max();
  volatile int result = maximum + 1;
  (void)result;
  return 0;
}
