#pragma once

#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>  // isatty, fileno
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

// ------------------------------------------------------------------------
//     Path handling
// ------------------------------------------------------------------------

inline std::filesystem::path parent_directory(std::filesystem::path p)
{
    namespace fs = std::filesystem;

    // If it's relative, anchor it to the current directory so canonical() works.
    if (p.is_relative()) {
        p = fs::current_path() / p;
    }

    // Normalizes ., .., symlinks if possible.
    auto canon = fs::weakly_canonical(p);

    return canon.parent_path();
}

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

// ------------------------------------------------------------------------
//     File handling
// ------------------------------------------------------------------------

inline std::vector<std::string> load_lines(std::string const& path)
{
    std::ifstream in(path);
    if(!in) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::vector<std::string> lines;
    lines.reserve(1024); // arbitrary; grows automatically

    std::string line;
    while(std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
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
