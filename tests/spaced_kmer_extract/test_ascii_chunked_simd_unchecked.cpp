#include "spaced_kmer_extract/ascii_chunked_simd_common.hpp"
#include "testing.hpp"

// =================================================================================================
//     for_each_spaced_kmer_ascii_unchecked_chunked_simd()
// =================================================================================================

TEST(SpacedKmerExtractAsciiChunkedSimd, UncheckedScalar)
{
    check_chunked_simd<fisk::SimdScalar, false>();
}

#if defined(FISK_HAS_SSE2)
TEST(SpacedKmerExtractAsciiChunkedSimd, UncheckedSse2)
{
    check_chunked_simd<fisk::SimdSSE2, false>();
}
#endif

#if defined(FISK_HAS_AVX2)
TEST(SpacedKmerExtractAsciiChunkedSimd, UncheckedAvx2)
{
    check_chunked_simd<fisk::SimdAVX2, false>();
}
#endif

#if defined(FISK_HAS_AVX512)
TEST(SpacedKmerExtractAsciiChunkedSimd, UncheckedAvx512)
{
    check_chunked_simd<fisk::SimdAVX512, false>();
}
#endif

#if defined(FISK_HAS_NEON)
TEST(SpacedKmerExtractAsciiChunkedSimd, UncheckedNeon)
{
    check_chunked_simd<fisk::SimdNEON, false>();
}
#endif

TEST(SpacedKmerExtractAsciiChunkedSimd, UncheckedErrorContract)
{
    check_chunked_simd_error_contract<false, 64>();
    check_chunked_simd_error_contract<false, 128>();
}
