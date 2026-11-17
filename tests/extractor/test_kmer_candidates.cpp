#include "fisk/core/types.hpp"
#include "extractor/kmer_common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Candidates
// =================================================================================================

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kActg = Encoding::kACTG;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kLsb = Layout::kLSB;
constexpr auto kChecked = InputCheck::kChecked;
constexpr auto kUnchecked = InputCheck::kUnchecked;

TEST(KmerCandidates, AsciiChecked)
{
    check_kmer_candidates<InputFormat::kAscii, kChecked, kAcgt, kMsb>();
    check_kmer_candidates<InputFormat::kAscii, kChecked, kActg, kLsb>();
}

TEST(KmerCandidates, AsciiUnchecked)
{
    check_kmer_candidates<InputFormat::kAscii, kUnchecked, kAcgt, kMsb>();
}

TEST(KmerCandidates, PackedChecked)
{
    check_kmer_candidates<InputFormat::kPacked, kChecked, kAcgt, kMsb>();
}

TEST(KmerCandidates, PackedUnchecked)
{
    check_kmer_candidates<InputFormat::kPacked, kUnchecked, kAcgt, kLsb>();
}
