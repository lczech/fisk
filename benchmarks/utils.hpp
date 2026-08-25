#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>  // isatty, fileno
#include <vector>

// =================================================================================================
//     File System
// =================================================================================================

// ------------------------------------------------------------------------
//     Path handling
// ------------------------------------------------------------------------

/**
 * @brief Get the parent directory of a given path @p p.
 */
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

/**
 * @brief Ensure that a given @p dir is a directory.
 *
 * If the path already exists, it checks that it is actually a directory, and throws otherwise.
 * If the path does not exist, the directory and its parents are created.
 */
inline std::filesystem::path ensure_output_dir(std::string const& dir)
{
    namespace fs = std::filesystem;
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

/**
 * @brief Load the lines in a file into a vector of strings.
 */
inline std::vector<std::string> load_lines(std::string const& path)
{
    std::ifstream in(path);
    if(!in) {
        throw std::runtime_error("load_lines() cannot open input file: " + path);
    }

    std::vector<std::string> lines;
    lines.reserve(1024); // arbitrary; grows automatically

    std::string line;
    while(std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

/**
 * @brief Get the ofstream object to write to a given file path.
 */
inline std::ofstream get_ofstream( std::filesystem::path path, std::string filename )
{
    auto const target = path / filename;
    std::ofstream os(target.string());
    if (!os) {
        throw std::runtime_error("get_ofstream() cannot open output file: " + target.string());
    }
    return os;
}

/**
 * @brief Check if `stdout` is a terminal.
 */
inline bool stdout_is_terminal()
{
    return isatty(fileno(stdout));
}
