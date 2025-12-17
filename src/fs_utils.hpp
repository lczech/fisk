#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

inline fs::path ensure_output_dir(std::string const& dir)
{
    fs::path p(dir);

    std::error_code ec;
    if (fs::exists(p, ec)) {
        if (!fs::is_directory(p, ec)) {
            throw std::runtime_error(
                "Output path exists but is not a directory: " + p.string()
            );
        }
    } else {
        if (!fs::create_directories(p, ec)) {
            throw std::runtime_error(
                "Failed to create output directory: " + p.string()
            );
        }
    }

    return p;
}
