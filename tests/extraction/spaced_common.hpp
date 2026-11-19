#pragma once

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extraction/spaced_kmer_estimate.hpp"
#include "extraction/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

// =================================================================================================
//     Spaced K-mer Candidate Checks
// =================================================================================================

// Every candidate of the estimate of for_each_spaced_kmer() that this CPU can run, run directly,
// for each of the `mask_sets`, with a callback of single spaced k-mers, and with one that also
// takes vectors. The masks are prepared for up to `MaxMasks` masks of up to `MaxSpan`.
template <
    fisk::InputFormat I, fisk::InputCheck C, fisk::Encoding E, fisk::Layout L,
    std::size_t MaxMasks = kMaxTestMasks, std::size_t MaxSpan = fisk::kDefaultMaxSpan
>
inline void check_spaced_kmer_candidates(
    std::vector<std::vector<std::string>> const& mask_sets
) {
    using namespace fisk;
    using Candidates = typename detail::SpacedKmerEstimate<I, C, E, L, MaxMasks>::type;
    auto const seqs = extraction_test_sequences<C>();

    for (auto const& masks : mask_sets) {
        SpacedKmerMaskForms<L, MaxMasks, MaxSpan> const mask_forms(masks);
        std::size_t count = 0;
        for_each_available_candidate(Candidates{}, [&](auto const& candidate) {
            ++count;
            auto const name = std::remove_cvref_t<decltype(candidate)>::name();

            for (auto const& seq : seqs) {
                auto const expected = by_mask(oracle_spaced_events<E, L>(seq, masks));
                auto const input = extraction_test_input<I, C, E, L>(seq);

                SpacedKmerRecorder single;
                candidate.run(mask_forms, input, single);
                check_spaced_events(by_mask(single.events), expected);

                // The SIMD algorithms hand their vectors to a callback that takes them.
                SpacedKmerRecorder unpacked;
                std::size_t vectors = 0;
                auto both = callback_overload{
                    [&](std::size_t pos, std::size_t mask_idx, SpacedKmer<E, L> kmer) {
                        unpacked(pos, mask_idx, kmer);
                    },
                    [&](std::size_t pos, std::size_t mask_idx, auto kmers, auto valid) {
                        ++vectors;
                        for_each_valid_lane(
                            pos, kmers, valid, [&](std::size_t lane_pos, auto const& kmer) {
                                unpacked(lane_pos, mask_idx, kmer);
                            }
                        );
                    }
                };
                candidate.run(mask_forms, input, both);
                check_spaced_events(by_mask(unpacked.events), expected);
                bool const simd = name.find("simd") != std::string::npos;
                EXPECT_TRUE(simd ? (vectors > 0 || expected.empty()) : vectors == 0);
            }
        });
        EXPECT_NE(count, 0u);
    }
}

// The candidates as above, for sets of one and three masks, and for the wide sets of
// extraction_wide_mask_sets(), which keeps the compile time down; the largest sets are checked on
// their own, see test_spaced_kmer_candidates.cpp.
template <fisk::InputFormat I, fisk::InputCheck C, fisk::Encoding E, fisk::Layout L>
inline void check_spaced_kmer_candidates()
{
    std::vector<std::vector<std::string>> mask_sets = {
        {"1101101"},
        random_masks(kMaxTestMasks, 11, 1),
    };
    for (auto& set : extraction_wide_mask_sets()) {
        mask_sets.push_back(std::move(set));
    }
    check_spaced_kmer_candidates<I, C, E, L>(mask_sets);
}
