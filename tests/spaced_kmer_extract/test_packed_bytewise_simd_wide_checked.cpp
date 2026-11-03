#include "fisk/core/types.hpp"
#include "spaced_kmer_extract/packed_simd_common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SpacedKmerExtractPackedBytewiseSimd, WideCheckedMsb)
{
    check_simd_wide_extraction<Layout::kMSB, true>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, WideCheckedLsb)
{
    check_simd_wide_extraction<Layout::kLSB, true>();
}
