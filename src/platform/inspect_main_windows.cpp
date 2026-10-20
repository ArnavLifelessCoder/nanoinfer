#include <iostream>

#include "tools/inspect.hpp"

int wmain(int argc, wchar_t** argv) {
  if (argc != 2) {
    std::cerr << "usage: gpt2inspect MODEL\n";
    return 2;
  }
  return nanoinfer::inspect_model(std::filesystem::path(argv[1]));
}
