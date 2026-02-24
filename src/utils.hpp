#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>  // isatty, fileno
#include <fstream>
#include <vector>

// =================================================================================================
//     Random numbers
// =================================================================================================

/**
 * @brief Fast deterministic random bit generator via the splitmix64 algorithm.
 *
 * Splitmix64 is a pseudo-random number generator, which uses a fairly simple algorithm that,
 * though it is considered to be poor for cryptographic purposes, is very fast to calculate,
 * and is "good enough" for many random number needs. It passes several fairly rigorous PRNG
 * "fitness" tests that some more complex algorithms fail.
 *
 * Here, we implement a fixed-increment version of it. If you need a full implementation,
 * see for instance https://gist.github.com/imneme/6179748664e88ef3c34860f44309fc71
 *
 * @see http://dx.doi.org/10.1145/2714064.2660195
 */
class Splitmix64
{
public:

    // -------------------------------------------------
    //     Constructors and Rule of Five
    // -------------------------------------------------

    Splitmix64() = default;
    Splitmix64( std::uint64_t seed )
        : state_(seed)
    {}

    // -------------------------------------------------
    //     Member Functions
    // -------------------------------------------------

    inline void set_seed( std::uint64_t seed )
    {
        state_ = seed;
    }

    inline std::uint64_t get_uint64()
    {
        state_ += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = state_;
        z = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9ULL;
        z = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111ebULL;
        return z ^ ( z >> 31 );
    }

    inline double get_double()
    {
        return static_cast<double>( get_uint64() ) / two_pow_64_;
    }

    // -------------------------------------------------
    //     Private Members
    // -------------------------------------------------

private:
    std::uint64_t state_ = 0;
    double const two_pow_64_ = std::pow( 2.0, 64 );
};

// =================================================================================================
//     File System
// =================================================================================================

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

inline std::ofstream get_ofstream( std::filesystem::path path, std::string filename )
{
    auto const target = path / filename;
    std::ofstream os(target.string());
    if (!os) {
        throw std::runtime_error("get_ofstream() cannot open output file: " + target.string());
    }
    return os;
}

inline bool stdout_is_terminal()
{
    return isatty(fileno(stdout));
}
