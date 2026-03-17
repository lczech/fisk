#include "sys_info.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <string_view>

#if defined(SYSTEM_X86_64_GNU_CLANG)
    #include <cpuid.h>
#elif defined(PLATFORM_ARM64)
    #include <sys/sysctl.h>
#elif defined(_MSC_VER)
    #include <intrin.h>
#endif

// =================================================================================================
//     Hardware Info
// =================================================================================================

std::string info_platform_name()
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

std::string info_platform_arch()
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

void info_print_platform(std::ostream& os)
{
    os << "Platform:\n";
    os << "  name    : " << info_platform_name() << "\n";
    os << "  arch    : " << info_platform_arch() << "\n";
}

std::string info_cpu_vendor()
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

std::string info_cpu_model()
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

void info_print_cpu(std::ostream& os)
{
    os << "CPU:\n";
    os << "  vendor  : " << info_cpu_vendor() << "\n";
    os << "  model   : " << info_cpu_model() << "\n";
}

// =================================================================================================
//     Compiler Info
// =================================================================================================

std::string info_compiler_family()
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

std::string info_compiler_version()
{
    #if defined(__clang__)
        return __clang_version__;
    #elif defined(__ICC) || defined(__INTEL_COMPILER)
        return __INTEL_COMPILER;
    #elif defined(__GNUC__) || defined(__GNUG__)
        return
            std::to_string(__GNUC__)            + "." +
            std::to_string(__GNUC_MINOR__)      + "." +
            std::to_string(__GNUC_PATCHLEVEL__)
        ;
    #elif defined(__HP_cc) || defined(__HP_aCC)
        return "";
    #elif defined(__IBMCPP__)
        return __IBMCPP__;
    #elif defined(_MSC_VER)
        return _MSC_VER;
    #elif defined(__PGI)
        return __PGI;
    #elif defined(__SUNPRO_CC)
        return __SUNPRO_CC;
    #else
        return "unknown";
    #endif
}

void info_print_compiler(std::ostream& os)
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

inline void ensure_cpu_init() noexcept
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

void cpuidex(int out[4], int leaf, int subleaf) noexcept
{
    __cpuidex(out, leaf, subleaf);
}

std::uint64_t xgetbv0() noexcept
{
    return _xgetbv(0);
}

bool os_avx_state_enabled() noexcept
{
    int regs[4] = {};
    cpuidex(regs, 1, 0);

    bool const osxsave = (regs[2] & (1 << 27)) != 0;
    bool const avx     = (regs[2] & (1 << 28)) != 0;

    if (!osxsave || !avx) {
        return false;
    }

    std::uint64_t const xcr0 = xgetbv0();

    // XMM state (bit 1) and YMM state (bit 2) must be enabled by the OS.
    return (xcr0 & 0x6) == 0x6;
}

bool os_avx512_state_enabled() noexcept
{
    if (!os_avx_state_enabled()) {
        return false;
    }

    std::uint64_t const xcr0 = xgetbv0();

    // XMM/YMM + opmask + upper ZMM state
    return (xcr0 & 0xE6) == 0xE6;
}

bool msvc_cpu_bmi2() noexcept
{
    int regs[4] = {};
    cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 8)) != 0; // EBX.BMI2
}

bool msvc_cpu_sse2() noexcept
{
    int regs[4] = {};
    cpuidex(regs, 1, 0);
    return (regs[3] & (1 << 26)) != 0; // EDX.SSE2
}

bool msvc_cpu_pclmul() noexcept
{
    int regs[4] = {};
    cpuidex(regs, 1, 0);
    return (regs[2] & (1 << 1)) != 0; // ECX.PCLMULQDQ
}

bool msvc_cpu_avx2() noexcept
{
    if (!os_avx_state_enabled()) {
        return false;
    }

    int regs[4] = {};
    cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0; // EBX.AVX2
}

bool msvc_cpu_avx512f() noexcept
{
    if (!os_avx512_state_enabled()) {
        return false;
    }

    int regs[4] = {};
    cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 16)) != 0; // EBX.AVX512F
}

#endif

// -----------------------------------------------------------------
//     Compile-time checks
// -----------------------------------------------------------------

bool compiled_bmi2() noexcept
{
#ifdef FISK_HAS_BMI2
    return true;
#else
    return false;
#endif
}

bool compiled_sse2() noexcept
{
#ifdef FISK_HAS_SSE2
    return true;
#else
    return false;
#endif
}

bool compiled_avx2() noexcept
{
#ifdef FISK_HAS_AVX2
    return true;
#else
    return false;
#endif
}

bool compiled_avx512() noexcept
{
#ifdef FISK_HAS_AVX512
    return true;
#else
    return false;
#endif
}

bool compiled_neon() noexcept
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

bool cpu_bmi2() noexcept
{
    #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        ensure_cpu_init();
        return __builtin_cpu_supports("bmi2");
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        return msvc_cpu_bmi2();
    #else
        return false;
    #endif
}

bool cpu_sse2() noexcept
{
    #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        ensure_cpu_init();
        return __builtin_cpu_supports("sse2");
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        return msvc_cpu_sse2();
    #else
        return false;
    #endif
}

bool cpu_avx2() noexcept
{
    #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        ensure_cpu_init();
        return __builtin_cpu_supports("avx2");
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        return msvc_cpu_avx2();
    #else
        return false;
    #endif
}

bool cpu_avx512() noexcept
{
    #if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
        ensure_cpu_init();
        return __builtin_cpu_supports("avx512f");
    #elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
        return msvc_cpu_avx512f();
    #else
        return false;
    #endif
}

bool cpu_neon() noexcept
{
    #if defined(__aarch64__) || defined(_M_ARM64)
        return true;
    #else
        return false;
    #endif
}

// -----------------------------------------------------------------
//     Combined checks
// -----------------------------------------------------------------

bool bmi2_enabled() noexcept
{
    static bool const enabled = compiled_bmi2() && cpu_bmi2();
    return enabled;
}

bool sse2_enabled() noexcept
{
    static bool const enabled = compiled_sse2() && cpu_sse2();
    return enabled;
}

bool avx2_enabled() noexcept
{
    static bool const enabled = compiled_avx2() && cpu_avx2();
    return enabled;
}

bool avx512_enabled() noexcept
{
    static bool const enabled = compiled_avx512() && cpu_avx512();
    return enabled;
}

bool neon_enabled() noexcept
{
    static bool const enabled = compiled_neon() && cpu_neon();
    return enabled;
}

// -----------------------------------------------------------------
//     Print for user output
// -----------------------------------------------------------------


void info_print_intrinsics(std::ostream& os)
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
    print_one_feature("AVX2",   compiled_avx2(),   cpu_avx2());
    print_one_feature("AVX512", compiled_avx512(), cpu_avx512());
    print_one_feature("NEON",   compiled_neon(),   cpu_neon());
}
