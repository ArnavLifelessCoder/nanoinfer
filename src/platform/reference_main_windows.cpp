#include <filesystem>
#include <vector>

#include "tools/reference.hpp"

int wmain(int argc, wchar_t** argv) {
  std::vector<std::filesystem::path> arguments;
  for (int i = 1; i < argc; ++i) arguments.emplace_back(argv[i]);
  return nanoinfer::reference_main(arguments);
}
