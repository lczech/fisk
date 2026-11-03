#include <cstddef>
#include <string>

#include "fisk/core/types.hpp"
#include "fisk/simd/simd.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/kmer_extract/ascii_rolling_simd.hpp"
#include "kmer_extract/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// for_each_kmer_ascii_checked_rolling_simd() with the arch `Simd` for the checks of
// kmer_extract/common.hpp.
template <SimdArch Simd>
static constexpr auto extract = []<Encoding E, Layout L>(
    std::string const& seq, std::size_t k, auto const& callback
) {
    for_each_kmer_ascii_checked_rolling_simd<Simd, E, L>(seq, k, callback);
};

// The checks of the checked extraction from ASCII with the arch `Simd`: oracle correctness on the
// valid-only sequences in all four conventions, k and length boundaries, invalid characters, the
// block boundaries of every arch, and case insensitivity.
template <SimdArch Simd>
static void check_rolling_simd()
{
    check_ascii_oracle(extract<Simd>);

    // k must be in [1, 32]; k == 0 or k > 32 is a documented precondition violation.
    check_invalid_k_throws(32, [](std::string const& seq, std::size_t k) {
        extract<Simd>.template operator()<Encoding::kACGT, Layout::kMSB>(
            seq, k, [](KmerAcgtMsb) {}
        );
    });
    check_ascii_length_boundaries(extract<Simd>);

    check_ascii_invalid_character_placement(extract<Simd>);
    check_ascii_invalid_injection_fuzz(extract<Simd>);
    check_ascii_block_boundaries(extract<Simd>);
    check_ascii_mixed_case(extract<Simd>);
}

// =================================================================================================
//     SIMD Archs
// =================================================================================================

TEST(KmerExtractAsciiRollingSimd, Scalar)
{
    check_rolling_simd<SimdScalar>();
}

#if defined(FISK_HAS_SSE2)
TEST(KmerExtractAsciiRollingSimd, Sse2)
{
    check_rolling_simd<SimdSSE2>();
}
#endif

#if defined(FISK_HAS_AVX2)
TEST(KmerExtractAsciiRollingSimd, Avx2)
{
    check_rolling_simd<SimdAVX2>();
}
#endif

#if defined(FISK_HAS_AVX512)
TEST(KmerExtractAsciiRollingSimd, Avx512)
{
    check_rolling_simd<SimdAVX512>();
}
#endif

#if defined(FISK_HAS_NEON)
TEST(KmerExtractAsciiRollingSimd, Neon)
{
    check_rolling_simd<SimdNEON>();
}
#endif

// =================================================================================================
//     Conventions
// =================================================================================================

// Without Encoding and Layout, the extractor keeps emitting ACGT/MSB. The callback only accepts
// KmerAcgtMsb, so any other default fails to compile.
TEST(KmerExtractAsciiRollingSimd, DefaultConventions)
{
    std::string const seq = "ACGTA";
    std::size_t count = 0;
    auto const callback = [&](KmerAcgtMsb) { ++count; };
    for_each_kmer_ascii_checked_rolling_simd<SimdScalar>(seq, 3, callback);
    EXPECT_EQ(count, std::size_t{3});
}
