#include <cstdio>

int main(int argc, char**) {
  std::puts("probe complete");
#ifdef PROBE_FLUSH
  if (std::fflush(stdout) != 0) return 3;
#endif
  return argc > 1 ? 2 : 0;
}
