#pragma once

#include <filesystem>
#include <span>

namespace nanoinfer {
int generate_main(std::span<const std::filesystem::path> arguments);
}
