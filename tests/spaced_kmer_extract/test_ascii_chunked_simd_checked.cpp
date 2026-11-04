#include "spaced_kmer_extract/ascii_chunked_simd_common.hpp"
#include "testing.hpp"

// =================================================================================================
//     for_each_spaced_kmer_ascii_checked_chunked_simd()
// =================================================================================================

TEST(SpacedKmerExtractAsciiChunkedSimd, CheckedScalar)
{
    check_chunked_simd<fisk::SimdScalar, true>();
}

#if defined(FISK_HAS_SSE2)
TEST(SpacedKmerExtractAsciiChunkedSimd, CheckedSse2)
{
    check_chunked_simd<fisk::SimdSSE2, true>();
}
#endif

#if defined(FISK_HAS_AVX2)
TEST(SpacedKmerExtractAsciiChunkedSimd, CheckedAvx2)
{
    check_chunked_simd<fisk::SimdAVX2, true>();
}
#endif

#if defined(FISK_HAS_AVX512)
TEST(SpacedKmerExtractAsciiChunkedSimd, CheckedAvx512)
{
    check_chunked_simd<fisk::SimdAVX512, true>();
}
#endif

#if defined(FISK_HAS_NEON)
TEST(SpacedKmerExtractAsciiChunkedSimd, CheckedNeon)
{
    check_chunked_simd<fisk::SimdNEON, true>();
}
#endif

TEST(SpacedKmerExtractAsciiChunkedSimd, CheckedErrorContract)
{
    check_chunked_simd_error_contract<true, 64>();
    check_chunked_simd_error_contract<true, 128>();
}
