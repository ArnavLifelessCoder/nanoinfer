#include <iostream>
#include <string_view>

#include "tools/inspect.hpp"

int main(int argc, char** argv) {
  const bool layout_only = argc == 3 && std::string_view(argv[2]) == "--layout-only";
  if (argc != 2 && !layout_only) {
    std::cerr << "usage: gpt2inspect MODEL [--layout-only]\n";
    return 2;
  }
  return nanoinfer::inspect_model(std::filesystem::path(argv[1]), layout_only);
}
