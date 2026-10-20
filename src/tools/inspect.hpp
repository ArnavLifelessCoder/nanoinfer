#pragma once

#include <filesystem>

namespace nanoinfer {
int inspect_model(const std::filesystem::path& path, bool layout_only = false);
}
