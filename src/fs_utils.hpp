#pragma once

#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>  // isatty, fileno

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

inline std::ofstream get_ofstream( fs::path path, std::string filename )
{
    auto const target = path / filename;
    std::ofstream os(target.string());
    if (!os) {
        throw std::runtime_error("bench_pext: cannot open output file: " + target.string());
    }
    return os;
}

inline bool stdout_is_terminal()
{
    return isatty(fileno(stdout));
}
