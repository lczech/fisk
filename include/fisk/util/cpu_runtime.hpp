#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <string>

#include "fisk/util/intrinsics.hpp"

#if defined(SYSTEM_X86_64_GNU_CLANG)
    #include <cpuid.h>
#elif defined(PLATFORM_ARM64)
    #include <sys/sysctl.h>
#elif defined(_MSC_VER)
    #include <intrin.h>
#endif

namespace fisk {

// =================================================================================================
//     Hardware Info
// =================================================================================================

inline std::string info_platform_name()
{
    #if defined _WIN64
        return "Win64";
    #elif defined _WIN32
        return "Win32";
    #elif defined __linux__
        return "Linux";
    #elif defined __APPLE__
        return "Apple";
    #elif defined __unix__
        return "Unix";
    #else
        return "Unknown";
    #endif
}

inline std::string info_platform_arch()
{
    #if defined(__x86_64__) || defined(_M_X64)
        return "x86-64";
    #elif defined(__i386__) || defined(_M_IX86)
        return "x86-32";
    #elif defined(__aarch64__) || defined(_M_ARM64)
        return "ARM64";
    #elif defined(__arm__) || defined(_M_ARM)
        return "ARM32";
    #else
        return "Unknown";
    #endif
}

inline void info_print_platform(std::ostream& os)
{
    os << "Platform:\n";
    os << "  name    : " << info_platform_name() << "\n";
    os << "  arch    : " << info_platform_arch() << "\n";
}

inline std::string info_cpu_vendor()
{
    #if defined(PLATFORM_ARM64)

        char buf[512];
        size_t size = sizeof(buf);
        if (sysctlbyname("machdep.cpu.vendor", buf, &size, nullptr, 0) == 0) {
            return std::string(buf, size - 1);
        }
        return "Unknown";

    #else

        // Vendor string (leaf 0)
        std::array<std::uint32_t,4> regs{};
        #ifdef SYSTEM_X86_64_GNU_CLANG
            __cpuid(0, regs[0], regs[1], regs[2], regs[3]);
        #elif defined(_MSC_VER)
            int r[4]; __cpuid(r, 0);
            regs = { (uint32_t)r[0], (uint32_t)r[1], (uint32_t)r[2], (uint32_t)r[3] };
        #endif

        char vendor_str[13]{};
        std::memcpy(vendor_str + 0, &regs[1], 4); // EBX
        std::memcpy(vendor_str + 4, &regs[3], 4); // EDX
        std::memcpy(vendor_str + 8, &regs[2], 4); // ECX
        vendor_str[12] = '\0';

        return vendor_str;

    #endif
}

inline std::string info_cpu_model()
{
    #if defined(PLATFORM_ARM64)

        char buf[512];
        size_t size = sizeof(buf);
        if (sysctlbyname("machdep.cpu.brand_string", buf, &size, nullptr, 0) == 0) {
            return std::string(buf, size - 1);
        }
        return "Unknown";

    #else

        // Brand string (0x80000002-0x80000004)
        char brand[49]{};
        std::uint32_t regs[4];

        for (std::uint32_t i = 0; i < 3; ++i) {
            std::uint32_t leaf = 0x80000002u + i;
            #if defined(SYSTEM_X86_64_GNU_CLANG)
                __cpuid(leaf, regs[0], regs[1], regs[2], regs[3]);
            #elif defined(_MSC_VER)
                int r[4]; __cpuid(r, leaf);
                regs[0] = (uint32_t)r[0];
                regs[1] = (uint32_t)r[1];
                regs[2] = (uint32_t)r[2];
                regs[3] = (uint32_t)r[3];
            #endif
            std::memcpy(brand + 16*i +  0, &regs[0], 4);
            std::memcpy(brand + 16*i +  4, &regs[1], 4);
            std::memcpy(brand + 16*i +  8, &regs[2], 4);
            std::memcpy(brand + 16*i + 12, &regs[3], 4);
        }
        brand[48] = '\0';

        // Trim leading spaces in brand:
        std::string brand_str = brand;
        while (!brand_str.empty() && brand_str.front() == ' ') {
            brand_str.erase(brand_str.begin());
        }
        return brand_str;

    #endif
}

/**
 * @brief The x86 CPU family, as the base family plus, for base family 0xF, the extended family
 * (CPUID leaf 1). 0 on ARM.
 */
inline unsigned info_cpu_family()
{
    #if defined(PLATFORM_ARM64)

        return 0;

    #else

        std::uint32_t eax = 0;
        #if defined(SYSTEM_X86_64_GNU_CLANG)
            std::uint32_t ebx, ecx, edx;
            __cpuid(1, eax, ebx, ecx, edx);
        #elif defined(_MSC_VER)
            int r[4]; __cpuid(r, 1);
            eax = static_cast<std::uint32_t>(r[0]);
        #endif

        unsigned const base = (eax >> 8) & 0xFu;
        unsigned const extended = (eax >> 20) & 0xFFu;
        return base == 0xFu ? base + extended : base;

    #endif
}

inline void info_print_cpu(std::ostream& os)
{
    os << "CPU:\n";
    os << "  vendor  : " << info_cpu_vendor() << "\n";
    os << "  model   : " << info_cpu_model() << "\n";
    os << "  family  : 0x" << std::hex << info_cpu_family() << std::dec << "\n";
}

// =================================================================================================
//     Cache Info
// =================================================================================================

/**
 * @brief Sizes of the data caches of the CPU, in bytes; 0 for a level that the CPU does not have,
 * or that is not known.
 */
struct CacheSizes
{
    std::size_t l1d = 0;
    std::size_t l2 = 0;
    std::size_t l3 = 0;
};

/**
 * @brief The data cache sizes of the CPU, see CacheSizes.
 *
 * On x86, from the cache parameters of CPUID (leaf 4 on Intel, leaf 0x8000001D on AMD and Hygon),
 * which describe the core that runs the call: on CPUs with performance and efficiency cores, these
 * differ. On ARM, from sysctl, for the performance cores of Apple Silicon.
 */
inline CacheSizes info_cache_sizes()
{
    CacheSizes sizes;

    #if defined(PLATFORM_ARM64)

        auto const read = [](char const* name, char const* fallback) -> std::size_t {
            for (char const* key : {name, fallback}) {
                std::uint64_t value = 0;
                size_t size = sizeof(value);
                if (sysctlbyname(key, &value, &size, nullptr, 0) == 0 && value > 0) {
                    return static_cast<std::size_t>(value);
                }
            }
            return 0;
        };
        sizes.l1d = read("hw.perflevel0.l1dcachesize", "hw.l1dcachesize");
        sizes.l2  = read("hw.perflevel0.l2cachesize",  "hw.l2cachesize");
        sizes.l3  = read("hw.perflevel0.l3cachesize",  "hw.l3cachesize");

    #elif defined(SYSTEM_X86_64_GNU_CLANG) || defined(_MSC_VER)

        std::uint32_t regs[4]{};
        auto const cpuid = [&](std::uint32_t leaf, std::uint32_t subleaf) {
            #if defined(SYSTEM_X86_64_GNU_CLANG)
                __cpuid_count(leaf, subleaf, regs[0], regs[1], regs[2], regs[3]);
            #else
                int r[4];
                __cpuidex(r, static_cast<int>(leaf), static_cast<int>(subleaf));
                for (int i = 0; i < 4; ++i) {
                    regs[i] = static_cast<std::uint32_t>(r[i]);
                }
            #endif
        };

        auto const vendor = info_cpu_vendor();
        std::uint32_t const leaf =
            (vendor == "AuthenticAMD" || vendor == "HygonGenuine") ? 0x8000001Du : 4u;

        // The largest leaf of the range of the cache leaf, as the cache leaf needs to exist.
        cpuid(leaf & 0x80000000u, 0);
        if (regs[0] < leaf) {
            return sizes;
        }

        // One subleaf per cache, until one of type 0.
        for (std::uint32_t i = 0; i < 16; ++i) {
            cpuid(leaf, i);
            std::uint32_t const type = regs[0] & 0x1Fu;
            if (type == 0) {
                break;
            }
            if (type == 2) {
                continue; // instruction cache
            }
            std::size_t const ways       = ((regs[1] >> 22) & 0x3FFu) + 1;
            std::size_t const partitions = ((regs[1] >> 12) & 0x3FFu) + 1;
            std::size_t const line_size  = (regs[1] & 0xFFFu) + 1;
            std::size_t const sets       = static_cast<std::size_t>(regs[2]) + 1;
            std::size_t const size = ways * partitions * line_size * sets;
            switch ((regs[0] >> 5) & 0x7u) {
                case 1: sizes.l1d = size; break;
                case 2: sizes.l2  = size; break;
                case 3: sizes.l3  = size; break;
                default: break;
            }
        }

    #endif

    return sizes;
}

inline void info_print_caches(std::ostream& os)
{
    auto const sizes = info_cache_sizes();
    auto const print = [&](char const* name, std::size_t bytes) {
        os << "  " << name << " : ";
        if (bytes == 0) {
            os << "unknown\n";
        } else {
            os << bytes / 1024 << " KiB\n";
        }
    };
    os << "Caches:\n";
    print("L1d    ", sizes.l1d);
    print("L2     ", sizes.l2);
    print("L3     ", sizes.l3);
}

// =================================================================================================
//     Compiler Info
// =================================================================================================

inline std::string info_compiler_family()
{
    #if defined(__clang__)
        return "clang";
    #elif defined(__ICC) || defined(__INTEL_COMPILER)
        return "icc";
    #elif defined(__GNUC__) || defined(__GNUG__)
        return "gcc";
    #elif defined(__HP_cc) || defined(__HP_aCC)
        return "hp";
    #elif defined(__IBMCPP__)
        return "ilecpp";
    #elif defined(_MSC_VER)
        return "msvc";
    #elif defined(__PGI)
        return "pgcpp";
    #elif defined(__SUNPRO_CC)
        return "sunpro";
    #else
        return "unknown";
    #endif
}

inline std::string info_compiler_version()
{
    #if defined(__clang__)
        return __clang_version__;
    #elif defined(__ICC) || defined(__INTEL_COMPILER)
        return std::to_string(__INTEL_COMPILER);
    #elif defined(__GNUC__) || defined(__GNUG__)
        return
            std::to_string(__GNUC__)            + "." +
            std::to_string(__GNUC_MINOR__)      + "." +
            std::to_string(__GNUC_PATCHLEVEL__)
        ;
    #elif defined(__HP_cc) || defined(__HP_aCC)
        return "";
    #elif defined(__IBMCPP__)
        return std::to_string(__IBMCPP__);
    #elif defined(_MSC_VER)
        return std::to_string(_MSC_VER);
    #elif defined(__PGI)
        return std::to_string(__PGI);
    #elif defined(__SUNPRO_CC)
        return std::to_string(__SUNPRO_CC);
    #else
        return "unknown";
    #endif
}

inline void info_print_compiler(std::ostream& os)
{
    os << "Compiler:\n";
    os << "  family  : " << info_compiler_family() << "\n";
    os << "  version : " << info_compiler_version() << "\n";
}

// =================================================================================================
//     CPU Intrinsics
// =================================================================================================

// -----------------------------------------------------------------
// GCC / Clang x86 helpers
// -----------------------------------------------------------------

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))

inline void ensure_cpu_init_() noexcept
{
    static bool initialized = [] {
        __builtin_cpu_init();
        return true;
    }();
    (void)initialized;
}

#endif

// -----------------------------------------------------------------
// MSVC x86 helpers
// -----------------------------------------------------------------

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))

inline void cpuidex_(int out[4], int leaf, int subleaf) noexcept
{
    __cpuidex(out, leaf, subleaf);
}

inline std::uint64_t xgetbv0_() noexcept
{
    return _xgetbv(0);
}

inline bool os_avx_state_enabled_() noexcept
{
    int regs[4] = {};
    cpuidex_(regs, 1, 0);

    bool const osxsave = (regs[2] & (1 << 27)) != 0;
    bool const avx     = (regs[2] & (1 << 28)) != 0;

    if (!osxsave || !avx) {
        return false;
    }

    std::uint64_t const xcr0 = xgetbv0_();

    // XMM state (bit 1) and YMM state (bit 2) must be enabled by the OS.
    return (xcr0 & 0x6) == 0x6;
}

inline bool os_avx512_state_enabled_() noexcept
{
    if (!os_avx_state_enabled_()) {
        return false;
    }

    std::uint64_t const xcr0 = xgetbv0_();

    // XMM/YMM + opmask + upper ZMM state
    return (xcr0 & 0xE6) == 0xE6;
}

inline bool msvc_cpu_bmi2_() noexcept
{
    int regs[4] = {};
    cpuidex_(regs, 7, 0);
    return (regs[1] & (1 << 8)) != 0; // EBX.BMI2
}

inline bool msvc_cpu_sse2_() noexcept
{
    int regs[4] = {};
    cpuidex_(regs, 1, 0);
    return (regs[3] & (1 << 26)) != 0; // EDX.SSE2
}

inline bool msvc_cpu_ssse3_() noexcept
{
    int regs[4] = {};
    cpuidex_(regs, 1, 0);
    return (regs[2] & (1 << 9)) != 0; // ECX.SSSE3
}

inline bool msvc_cpu_sse41_() noexcept
{
    int regs[4] = {};
    cpuidex_(regs, 1, 0);
    return (regs[2] & (1 << 19)) != 0; // ECX.SSE4_1
}

inline bool msvc_cpu_avx2_() noexcept
{
    if (!os_avx_state_enabled_()) {
        return false;
    }

    int regs[4] = {};
    cpuidex_(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0; // EBX.AVX2
}

inline bool msvc_cpu_avx512f_() noexcept
{
    if (!os_avx512_state_enabled_()) {
        return false;
    }

    int regs[4] = {};
    cpuidex_(regs, 7, 0);
    return (regs[1] & (1 << 16)) != 0; // EBX.AVX512F
}

#endif

// -----------------------------------------------------------------
//     Compile-time checks
// -----------------------------------------------------------------

inline bool compiled_bmi2() noexcept
{
#ifdef FISK_HAS_BMI2
    return true;
#else
    return false;
#endif
}

inline bool compiled_sse2() noexcept
{
#ifdef FISK_HAS_SSE2
    return true;
#else
    return false;
#endif
}

inline bool compiled_ssse3() noexcept
{
#ifdef FISK_HAS_SSSE3
    return true;
#else
    return false;
#endif
}

inline bool compiled_sse41() noexcept
{
#ifdef FISK_HAS_SSE41
    return true;
#else
    return false;
#endif
}

inline bool compiled_avx2() noexcept
{
#ifdef FISK_HAS_AVX2
    return true;
#else
    return false;
#endif
}

inline bool compiled_avx512() noexcept
{
#ifdef FISK_HAS_AVX512
    return true;
#else
    return false;
#endif
}

inline bool compiled_neon() noexcept
{
#ifdef FISK_HAS_NEON
    return true;
#else
    return false;
#endif
}

// -----------------------------------------------------------------
//     Runtime CPU checks
// -----------------------------------------------------------------

inline bool cpu_bmi2() noexcept
{
    #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        ensure_cpu_init_();
        return __builtin_cpu_supports("bmi2");
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        return msvc_cpu_bmi2_();
    #else
        return false;
    #endif
}

inline bool cpu_sse2() noexcept
{
    #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        ensure_cpu_init_();
        return __builtin_cpu_supports("sse2");
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        return msvc_cpu_sse2_();
    #else
        return false;
    #endif
}

inline bool cpu_ssse3() noexcept
{
    #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        ensure_cpu_init_();
        return __builtin_cpu_supports("ssse3");
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        return msvc_cpu_ssse3_();
    #else
        return false;
    #endif
}

inline bool cpu_sse41() noexcept
{
    #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        ensure_cpu_init_();
        return __builtin_cpu_supports("sse4.1");
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        return msvc_cpu_sse41_();
    #else
        return false;
    #endif
}

inline bool cpu_avx2() noexcept
{
    #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        ensure_cpu_init_();
        return __builtin_cpu_supports("avx2");
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        return msvc_cpu_avx2_();
    #else
        return false;
    #endif
}

inline bool cpu_avx512() noexcept
{
    #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        ensure_cpu_init_();
        return __builtin_cpu_supports("avx512f");
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        return msvc_cpu_avx512f_();
    #else
        return false;
    #endif
}

inline bool cpu_neon() noexcept
{
    #if defined(__aarch64__) || defined(_M_ARM64)
        return true;
    #else
        return false;
    #endif
}

// -----------------------------------------------------------------
//     Combined checks: "safe to use in this build"
// -----------------------------------------------------------------

inline bool bmi2_enabled() noexcept
{
    static bool const enabled = compiled_bmi2() && cpu_bmi2();
    return enabled;
}

inline bool ssse3_enabled() noexcept
{
    static bool const enabled = compiled_ssse3() && cpu_ssse3();
    return enabled;
}

inline bool sse41_enabled() noexcept
{
    static bool const enabled = compiled_sse41() && cpu_sse41();
    return enabled;
}

// SimdSSE2 uses SSSE3 and SSE4.1 where these are compiled in, so that it then needs a CPU that
// has them.
inline bool sse2_enabled() noexcept
{
    static bool const enabled = compiled_sse2() && cpu_sse2()
        && (!compiled_ssse3() || cpu_ssse3()) && (!compiled_sse41() || cpu_sse41());
    return enabled;
}

inline bool avx2_enabled() noexcept
{
    static bool const enabled = compiled_avx2() && cpu_avx2();
    return enabled;
}

inline bool avx512_enabled() noexcept
{
    static bool const enabled = compiled_avx512() && cpu_avx512();
    return enabled;
}

inline bool neon_enabled() noexcept
{
    static bool const enabled = compiled_neon() && cpu_neon();
    return enabled;
}

// -----------------------------------------------------------------
//     Heuristics
// -----------------------------------------------------------------

/**
 * @brief Whether PEXT is likely fast on a CPU of this vendor and family, by a list of known CPUs
 * rather than a measurement.
 *
 * AMD Zen 1, Zen+ and Zen 2 (family 0x17) and Hygon Dhyana (family 0x18) implement PEXT in
 * microcode, with a latency that grows with the number of set bits in the mask; there, it is
 * slower than its software replacements. All other CPUs with BMI2 are assumed to have it in
 * hardware: Intel since Haswell, and AMD since Zen 3 (family 0x19).
 */
inline bool pext_likely_fast(std::string const& vendor, unsigned family) noexcept
{
    if (vendor == "AuthenticAMD" && family == 0x17) {
        return false;
    }
    if (vendor == "HygonGenuine" && family == 0x18) {
        return false;
    }
    return true;
}

/**
 * @brief Whether PEXT can be used in this build and on this CPU, and is likely fast there.
 *
 * This is a heuristic, see the overload above: a CPU that is not on its list is assumed to have a
 * fast PEXT. Measure, when in doubt.
 */
inline bool pext_likely_fast()
{
    static bool const fast =
        bmi2_enabled() && pext_likely_fast(info_cpu_vendor(), info_cpu_family());
    return fast;
}

// -----------------------------------------------------------------
//     Reporting
// -----------------------------------------------------------------

inline void info_print_intrinsics(std::ostream& os)
{
    auto print_one_feature = [&os](char const* name, bool compiled, bool cpu)
    {
        os << "  " << name;
        for (std::size_t i = std::char_traits<char>::length(name); i < 8; ++i) {
            os << ' ';
        }
        os << ": ";
        os << "compiled=" << (compiled ? "yes, " : "no,  ");
        os << "cpu=" << (cpu ? "yes" : "no") << '\n';
    };

    os << "Instruction sets:\n";
    print_one_feature("BMI2",   compiled_bmi2(),   cpu_bmi2());
    print_one_feature("SSE2",   compiled_sse2(),   cpu_sse2());
    print_one_feature("SSSE3",  compiled_ssse3(),  cpu_ssse3());
    print_one_feature("SSE4.1", compiled_sse41(),  cpu_sse41());
    print_one_feature("AVX2",   compiled_avx2(),   cpu_avx2());
    print_one_feature("AVX512", compiled_avx512(), cpu_avx512());
    print_one_feature("Neon",   compiled_neon(),   cpu_neon());
}

// Print the platform, CPU, caches, compiler, and the instruction sets as compiled and as present on
// the CPU.
inline void info_print_system(std::ostream& os)
{
    info_print_platform(os);
    info_print_cpu(os);
    info_print_caches(os);
    info_print_compiler(os);
    info_print_intrinsics(os);
}

} // namespace fisk
