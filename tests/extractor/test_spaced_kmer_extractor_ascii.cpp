#include "extractor/spaced_common.hpp"
#include "testing.hpp"

using namespace fisk;

// The algorithms of the spaced k-mer extractors from ASCII, in their own translation unit, as
// each compiles all of them.

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kActg = Encoding::kACTG;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kLsb = Layout::kLSB;
constexpr auto kChecked = InputCheck::kChecked;
constexpr auto kUnchecked = InputCheck::kUnchecked;

TEST(SpacedKmerExtractor, AsciiChecked)
{
    check_spaced_kmer_extractor_algorithms<InputFormat::kAscii, kChecked, kAcgt, kMsb>();
}

TEST(SpacedKmerExtractor, AsciiUnchecked)
{
    check_spaced_kmer_extractor_algorithms<InputFormat::kAscii, kUnchecked, kActg, kLsb>();
}
