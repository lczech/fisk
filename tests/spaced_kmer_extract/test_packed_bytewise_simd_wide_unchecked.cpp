#include "fisk/core/types.hpp"
#include "spaced_kmer_extract/packed_simd_common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SpacedKmerExtractPackedBytewiseSimd, WideUncheckedMsb)
{
    check_simd_wide_extraction<Layout::kMSB, false>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, WideUncheckedLsb)
{
    check_simd_wide_extraction<Layout::kLSB, false>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, WideErrorContract)
{
    check_simd_error_contract<Layout::kMSB, 64>();
    check_simd_error_contract<Layout::kLSB, 64>();
    check_simd_error_contract<Layout::kMSB, 128>();
    check_simd_error_contract<Layout::kLSB, 128>();
}
