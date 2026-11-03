#include "fisk/core/types.hpp"
#include "spaced_kmer_extract/packed_scalar_common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SpacedKmerExtractPackedBytewiseScalar, WideCheckedMsb)
{
    check_wide_extraction<Layout::kMSB, true>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, WideCheckedLsb)
{
    check_wide_extraction<Layout::kLSB, true>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, CheckedDefaultMaxSpan)
{
    check_default_max_span<Layout::kMSB, true>();
    check_default_max_span<Layout::kLSB, true>();
}
