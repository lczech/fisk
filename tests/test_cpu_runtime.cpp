#include "fisk/core/cpu_runtime.hpp"
#include "testing.hpp"

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
