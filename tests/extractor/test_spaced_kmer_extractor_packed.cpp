#include "extractor/spaced_common.hpp"
#include "testing.hpp"

using namespace fisk;

// The algorithms of the spaced k-mer extractors from packed sequences, in their own translation
// unit, as each compiles all of them.

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kLsb = Layout::kLSB;
constexpr auto kChecked = InputCheck::kChecked;
constexpr auto kUnchecked = InputCheck::kUnchecked;

TEST(SpacedKmerExtractor, PackedChecked)
{
    check_spaced_kmer_extractor_algorithms<InputFormat::kPacked, kChecked, kAcgt, kLsb>();
}

TEST(SpacedKmerExtractor, PackedUnchecked)
{
    check_spaced_kmer_extractor_algorithms<InputFormat::kPacked, kUnchecked, kAcgt, kMsb>();
}
