#include <cstddef>
#include <string>
#include <vector>

#include "fisk/core/types.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "extraction/common.hpp"
#include "extraction/spaced_common.hpp"
#include "testing.hpp"

using namespace fisk;

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kAscii = InputFormat::kAscii;
constexpr auto kChecked = InputCheck::kChecked;

// =================================================================================================
//     Largest Mask Sets
// =================================================================================================

// The most masks of the default MaxMasks, with every candidate, which keeps the compile time of all
// mask counts down to one test.
TEST(SpacedKmerCandidates, DefaultMaxMasks)
{
    check_spaced_kmer_candidates<kAscii, kChecked, kAcgt, kMsb, kDefaultMaxMasks>(
        {random_masks(kDefaultMaxMasks, 32, 2)}
    );
}

// =================================================================================================
//     Maximum Span
// =================================================================================================

// A larger maximum span than the default, 128, with every candidate, for masks of four and five
// pieces, the most of that span.
TEST(SpacedKmerCandidates, MaxSpan)
{
    std::vector<std::vector<std::string>> mask_sets;
    for (std::size_t span : {85u, 128u}) {
        mask_sets.push_back(random_masks(1, span, span));
    }
    check_spaced_kmer_candidates<kAscii, kChecked, kAcgt, kMsb, 1, 128>(mask_sets);
}
