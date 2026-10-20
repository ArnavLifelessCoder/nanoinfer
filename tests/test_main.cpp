#include <gtest/gtest.h>

#include <charconv>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace nanoinfer::test {
std::uint32_t random_seed = 42;
}

int main(int argc, char** argv) {
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg(argv[index]);
    if (arg.starts_with("--seed=")) {
      const auto value = arg.substr(7);
      const auto result =
          std::from_chars(value.data(), value.data() + value.size(), nanoinfer::test::random_seed);
      if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        std::cerr << "invalid --seed\n";
        return 2;
      }
      for (int next = index; next + 1 < argc; ++next) {
        argv[next] = argv[next + 1];
      }
      --argc;
      --index;
    }
  }
  std::cout << "Random seed: " << nanoinfer::test::random_seed << '\n';
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
