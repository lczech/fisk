// See forced_inline_common.hpp: undefined before any fisk header.
#undef FISK_ALWAYS_INLINE_FOR_EACH

#include "extractor/forced_inline_common.hpp"

// =================================================================================================
//     Spaced K-mers from Packed Sequences
// =================================================================================================

TEST(ForcedInline, SpacedKmerExtractorPacked)
{
    check_spaced_kmer_extractor_algorithms<kPacked, kChecked, kAcgt, kMsb>();
    check_spaced_kmer_extractor_algorithms<kPacked, kUnchecked, kAcgt, kMsb>();
}
