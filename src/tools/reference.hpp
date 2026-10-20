#pragma once

#include <filesystem>
#include <span>

namespace nanoinfer {
int reference_main(std::span<const std::filesystem::path> arguments);
}
