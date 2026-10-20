#include <filesystem>
#include <vector>

#include "tools/generate.hpp"

int main(int argc, char** argv) {
  std::vector<std::filesystem::path> arguments;
  for (int i = 1; i < argc; ++i) arguments.emplace_back(argv[i]);
  return nanoinfer::generate_main(arguments);
}
