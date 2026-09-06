#include "fisk/core/cpu_runtime.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/random.hpp"

#include <cstdint>
#include "testing.hpp"

// =================================================================================================
//     CPU Runtime Tests
// =================================================================================================

// cpu_runtime.hpp is mostly hardware/platform strings with no fixed "correct" answer to assert
// on portably (whatever info_cpu_vendor() etc. return depends on the machine running the test).
// What we can check regardless of machine is the invariants between its tiers.

TEST(CpuRuntime, CompiledFlagsMatchMacros)
{
    // compiled_*() must reflect exactly the FISK_HAS_* macros set up by intrinsics.hpp, whatever
    // this particular build was configured with.
#ifdef FISK_HAS_BMI2
    EXPECT_TRUE(compiled_bmi2());
#else
    EXPECT_FALSE(compiled_bmi2());
#endif

#ifdef FISK_HAS_SSE2
    EXPECT_TRUE(compiled_sse2());
#else
    EXPECT_FALSE(compiled_sse2());
#endif

#ifdef FISK_HAS_AVX2
    EXPECT_TRUE(compiled_avx2());
#else
    EXPECT_FALSE(compiled_avx2());
#endif

#ifdef FISK_HAS_AVX512
    EXPECT_TRUE(compiled_avx512());
#else
    EXPECT_FALSE(compiled_avx512());
#endif

#ifdef FISK_HAS_NEON
    EXPECT_TRUE(compiled_neon());
#else
    EXPECT_FALSE(compiled_neon());
#endif
}

TEST(CpuRuntime, EnabledImpliesCompiled)
{
    // The "safe to use in this build" checks may only report enabled when the corresponding
    // code path was actually compiled in, regardless of what the CPU running the test supports.
    EXPECT_TRUE(!bmi2_enabled()   || compiled_bmi2());
    EXPECT_TRUE(!sse2_enabled()   || compiled_sse2());
    EXPECT_TRUE(!avx2_enabled()   || compiled_avx2());
    EXPECT_TRUE(!avx512_enabled() || compiled_avx512());
    EXPECT_TRUE(!neon_enabled()   || compiled_neon());
}

// =================================================================================================
//     Intrinsics Tests
// =================================================================================================

TEST(Intrinsics, ByteSwap16KnownValues)
{
    EXPECT_EQ(byte_swap_16(0x0000), 0x0000);
    EXPECT_EQ(byte_swap_16(0x0001), 0x0100);
    EXPECT_EQ(byte_swap_16(0x0123), 0x2301);
    EXPECT_EQ(byte_swap_16(0xffff), 0xffff);
}

TEST(Intrinsics, ByteSwap16IsInvolution)
{
    // byte_swap_16() applied twice must be the identity, for any input.
    Splitmix64 rng(2024);
    for (int i = 0; i < 1000; ++i) {
        std::uint16_t const x = static_cast<std::uint16_t>(rng.get_uint64());
        EXPECT_EQ(byte_swap_16(byte_swap_16(x)), x);
    }
}

TEST(Intrinsics, ByteSwap32KnownValues)
{
    EXPECT_EQ(byte_swap_32(0x00000000u), 0x00000000u);
    EXPECT_EQ(byte_swap_32(0x00000001u), 0x01000000u);
    EXPECT_EQ(byte_swap_32(0x01234567u), 0x67452301u);
    EXPECT_EQ(byte_swap_32(0xffffffffu), 0xffffffffu);
}

TEST(Intrinsics, ByteSwap32IsInvolution)
{
    // byte_swap_32() applied twice must be the identity, for any input.
    Splitmix64 rng(2024);
    for (int i = 0; i < 1000; ++i) {
        std::uint32_t const x = static_cast<std::uint32_t>(rng.get_uint64());
        EXPECT_EQ(byte_swap_32(byte_swap_32(x)), x);
    }
}

TEST(Intrinsics, ByteSwap64KnownValues)
{
    EXPECT_EQ(byte_swap_64(0x0000000000000000ULL), 0x0000000000000000ULL);
    EXPECT_EQ(byte_swap_64(0x0000000000000001ULL), 0x0100000000000000ULL);
    EXPECT_EQ(byte_swap_64(0x0123456789abcdefULL), 0xefcdab8967452301ULL);
    EXPECT_EQ(byte_swap_64(0xffffffffffffffffULL), 0xffffffffffffffffULL);
}

TEST(Intrinsics, ByteSwap64IsInvolution)
{
    // byte_swap_64() applied twice must be the identity, for any input.
    Splitmix64 rng(2024);
    for (int i = 0; i < 1000; ++i) {
        std::uint64_t const x = rng.get_uint64();
        EXPECT_EQ(byte_swap_64(byte_swap_64(x)), x);
    }
}

// =================================================================================================
//     Random Tests
// =================================================================================================

TEST(Random, Splitmix64DeterministicForSameSeed)
{
    Splitmix64 a(12345);
    Splitmix64 b(12345);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(a.get_uint64(), b.get_uint64());
    }
}

TEST(Random, Splitmix64DifferentSeedsDiffer)
{
    Splitmix64 a(1);
    Splitmix64 b(2);
    EXPECT_NE(a.get_uint64(), b.get_uint64());
}

TEST(Random, Splitmix64SetSeedResetsSequence)
{
    Splitmix64 rng(42);
    std::uint64_t const first = rng.get_uint64();
    std::uint64_t const second = rng.get_uint64();

    rng.set_seed(42);
    EXPECT_EQ(rng.get_uint64(), first);
    EXPECT_EQ(rng.get_uint64(), second);
}

TEST(Random, Splitmix64GetDoubleInUnitRange)
{
    Splitmix64 rng(7);
    for (int i = 0; i < 1000; ++i) {
        double const d = rng.get_double();
        EXPECT_TRUE(d >= 0.0);
        EXPECT_TRUE(d < 1.0);
    }
}
