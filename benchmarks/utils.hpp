#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "sink.hpp"

#if defined(_MSC_VER)
    #include <io.h>  // _isatty, _fileno
#else
    #include <unistd.h>  // isatty, fileno
#endif

#if defined(__linux__)
    #include <sched.h>  // sched_getcpu
#endif

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
    #if defined(_MSC_VER)
        return _isatty(_fileno(stdout)) != 0;
    #else
        return isatty(fileno(stdout));
    #endif
}

// =================================================================================================
//     Build Info
// =================================================================================================

/**
 * @brief Print the benchmark-specific build settings that affect results, complementing the
 * platform and compiler info from fisk/util/cpu_runtime.hpp.
 */
inline void info_print_benchmark_build(std::ostream& os)
{
    os << "Benchmark build:\n";
    #if defined(FISK_BENCHMARK_ALIGN_CODE)
    os << "  code align : 64 (-falign-functions=64 -falign-loops=64)\n";
    #else
    os << "  code align : compiler default\n";
    #endif
    os << "  sink mode  : " << kSinkName << "\n";
}

// =================================================================================================
//     Power Info
// =================================================================================================

/**
 * @brief Print the power settings of Linux that set the clock speed of the CPU that runs the
 * benchmark, such as its frequency governor and turbo, and whether mains power is connected.
 * Settings that the system does not have are left out; on other systems, the whole section.
 */
inline void info_print_power(std::ostream& os)
{
    #if defined(__linux__)
        auto const read = [](std::filesystem::path const& path) {
            std::ifstream in(path);
            std::string line;
            std::getline(in, line);
            return line;
        };
        auto const print = [&](char const* name, std::string const& value) {
            if (!value.empty()) {
                os << "  " << name << " : " << value << "\n";
            }
        };

        int const cpu = sched_getcpu();
        std::filesystem::path const cpufreq =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu < 0 ? 0 : cpu) + "/cpufreq";
        os << "Power:\n";
        os << "  cpu        : " << cpu << "\n";
        print("driver    ", read(cpufreq / "scaling_driver"));
        print("governor  ", read(cpufreq / "scaling_governor"));
        print("energy    ", read(cpufreq / "energy_performance_preference"));

        // Turbo is off with intel_pstate/no_turbo 1, else with cpufreq/boost 0.
        std::string turbo;
        auto const no_turbo = read("/sys/devices/system/cpu/intel_pstate/no_turbo");
        auto const boost = read("/sys/devices/system/cpu/cpufreq/boost");
        if (!no_turbo.empty()) {
            turbo = no_turbo == "1" ? "off" : "on";
        } else if (!boost.empty()) {
            turbo = boost == "1" ? "on" : "off";
        }
        print("turbo     ", turbo);
        print("profile   ", read("/sys/firmware/acpi/platform_profile"));

        // The power supplies of type Mains, such as the AC adapter of a laptop.
        std::error_code ec;
        std::filesystem::path const supplies = "/sys/class/power_supply";
        for (auto const& entry : std::filesystem::directory_iterator(supplies, ec)) {
            if (read(entry.path() / "type") == "Mains") {
                print("mains     ", read(entry.path() / "online") == "1" ? "online" : "offline");
            }
        }
    #else
        (void)os;
    #endif
}
