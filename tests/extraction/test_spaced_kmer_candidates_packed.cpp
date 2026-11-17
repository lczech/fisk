#include "extraction/spaced_common.hpp"
#include "testing.hpp"

using namespace fisk;

// The candidates of the estimate of the spaced k-mers from packed sequences, in their own
// translation unit, as each compiles all of them.

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kLsb = Layout::kLSB;
constexpr auto kChecked = InputCheck::kChecked;
constexpr auto kUnchecked = InputCheck::kUnchecked;

TEST(SpacedKmerCandidates, PackedChecked)
{
    check_spaced_kmer_candidates<InputFormat::kPacked, kChecked, kAcgt, kLsb>();
}

TEST(SpacedKmerCandidates, PackedUnchecked)
{
    check_spaced_kmer_candidates<InputFormat::kPacked, kUnchecked, kAcgt, kMsb>();
}
