#include "extractor/spaced_common.hpp"
#include "testing.hpp"

using namespace fisk;

// The candidates of the estimate of the spaced k-mers from ASCII, in their own translation unit, as
// each compiles all of them.

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kActg = Encoding::kACTG;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kLsb = Layout::kLSB;
constexpr auto kChecked = InputCheck::kChecked;
constexpr auto kUnchecked = InputCheck::kUnchecked;

TEST(SpacedKmerCandidates, AsciiChecked)
{
    check_spaced_kmer_candidates<InputFormat::kAscii, kChecked, kAcgt, kMsb>();
}

TEST(SpacedKmerCandidates, AsciiUnchecked)
{
    check_spaced_kmer_candidates<InputFormat::kAscii, kUnchecked, kActg, kLsb>();
}
