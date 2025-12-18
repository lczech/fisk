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

void info_print_intrinsics(std::ostream& os)
{
    os << "Instruction sets:\n";

    // -----------------------------
    // BMI2
    // -----------------------------
    bool cmake_bmi2 =
    #ifdef HAVE_BMI2
        true;
    #else
        false;
    #endif

    bool cpu_bmi2 = false;
    #ifdef SYSTEM_X86_64_GNU_CLANG
        __builtin_cpu_init();
        cpu_bmi2 = __builtin_cpu_supports("bmi2");
    #endif

    os << "  BMI2    : ";
    os << "compiled=" << (cmake_bmi2 ? "yes, " : "no,  ");
    os << "cpu=" << (cpu_bmi2 ? "yes" : "no") << "\n";

    // -----------------------------
    // CLMUL (PCLMUL)
    // -----------------------------
    bool cmake_clmul =
    #ifdef HAVE_CLMUL
        true;
    #else
        false;
    #endif

    bool cpu_clmul = false;
    #ifdef SYSTEM_X86_64_GNU_CLANG
        __builtin_cpu_init();
        cpu_clmul = __builtin_cpu_supports("pclmul");
    #endif

    os << "  CLMUL   : ";
    os << "compiled=" << (cmake_clmul ? "yes, " : "no,  ");
    os << "cpu=" << (cpu_clmul ? "yes" : "no") << "\n";
}
