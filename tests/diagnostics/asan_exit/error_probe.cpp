#include <cstdio>
#include <string_view>

__declspec(noinline) int read_value(const int* pointer) { return pointer[0]; }

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string_view mode(argv[1]);
  if (mode == "heap-overflow") {
    volatile int* values = new int[1];
    values[argc] = 17;
    delete[] values;
  } else if (mode == "use-after-free") {
    auto* values = new int[8];
    values[0] = 17;
    delete[] values;
    volatile int result = read_value(values);
    static_cast<void>(result);
  } else {
    return 2;
  }
  std::fputs("ERROR: memory violation was not caught\n", stderr);
  return 0;
}
