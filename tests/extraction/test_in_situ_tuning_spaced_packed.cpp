#include "extraction/in_situ_spaced_common.hpp"

using namespace fisk;

// The in-situ tuning of the spaced k-mer extractor for packed input, see
// in_situ_spaced_common.hpp.

TEST(InSituTuning, SpacedPackedChecked)
{
    check_in_situ_spaced<InputFormat::kPacked, InputCheck::kChecked, Layout::kMSB>();
}

TEST(InSituTuning, SpacedPackedUnchecked)
{
    check_in_situ_spaced<InputFormat::kPacked, InputCheck::kUnchecked, Layout::kLSB>();
}
