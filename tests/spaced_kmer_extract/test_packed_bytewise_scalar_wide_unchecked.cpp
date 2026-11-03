#include "fisk/core/types.hpp"
#include "spaced_kmer_extract/packed_scalar_common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SpacedKmerExtractPackedBytewiseScalar, WideUncheckedMsb)
{
    check_wide_extraction<Layout::kMSB, false>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, WideUncheckedLsb)
{
    check_wide_extraction<Layout::kLSB, false>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, UncheckedDefaultMaxSpan)
{
    check_default_max_span<Layout::kMSB, false>();
    check_default_max_span<Layout::kLSB, false>();
}

TEST(SpacedKmerExtractPackedBytewiseScalar, WideErrorContract)
{
    check_error_contract<Layout::kMSB, 64>();
    check_error_contract<Layout::kLSB, 64>();
    check_error_contract<Layout::kMSB, 128>();
    check_error_contract<Layout::kLSB, 128>();
}
