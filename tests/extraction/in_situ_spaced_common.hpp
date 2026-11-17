#pragma once

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extraction/spaced_kmer_extractor.hpp"
#include "extraction/common.hpp"
#include "extraction/in_situ_common.hpp"
#include "oracle.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "testing.hpp"

// =================================================================================================
//     In-Situ Tuning Checks of the Spaced K-mer Extractor
// =================================================================================================

// The checks of test_in_situ_tuning_spaced_ascii.cpp and test_in_situ_tuning_spaced_packed.cpp,
// each in its own translation unit, as each compiles all algorithms of its input format.


inline bool in_situ_same_events(
    std::vector<SpacedEvent> const& got, std::vector<SpacedEvent> const& expected
) {
    if (got.size() != expected.size()) {
        return false;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (got[i].pos != expected[i].pos || got[i].mask_idx != expected[i].mask_idx ||
            got[i].value != expected[i].value) {
            return false;
        }
    }
    return true;
}

// The spaced k-mers of all calls of for_each() while and after tuning, for inputs whose tuning
// ends at each piece of their long sequence, see in_situ_test_input(), with a callback of single
// spaced k-mers and with one that also takes vectors.
template <fisk::InputFormat I, fisk::InputCheck C, fisk::Layout L>
inline void check_in_situ_spaced()
{
    using namespace fisk;
    constexpr auto E = Encoding::kACGT;
    using Extractor = SpacedKmerExtractor<I, C, E, L, kMaxTestMasks>;
    Splitmix64 rng(4343);

    std::vector<std::vector<std::string>> mask_sets;
    for (std::size_t span : {3u, 31u, 33u, 64u}) {
        mask_sets.push_back(random_masks(1, span, span));
    }
    mask_sets.push_back(random_masks(kMaxTestMasks, 11, 11));
    mask_sets.push_back(random_masks(kMaxTestMasks, 58, 58));

    for (auto const& masks : mask_sets) {
        std::size_t const overlap = masks.front().size() - 1;
        auto const lengths = in_situ_test_lengths(overlap, rng);
        for (std::size_t rounds = 1; rounds <= 3; ++rounds) {
            for (std::size_t prefix = 0; prefix <= 5; ++prefix) {
                for (std::size_t length : lengths) {
                    auto const seqs = in_situ_test_input<C>(prefix, length, rng);
                    Extractor single(masks);
                    Extractor both(masks);
                    single.tune(in_situ_test_options(rounds));
                    both.tune(in_situ_test_options(rounds));
                    for (std::size_t s = 0; s < seqs.size(); ++s) {
                        auto const expected = by_mask(oracle_spaced_events<E, L>(seqs[s], masks));
                        auto const input = extraction_test_input<Extractor>(seqs[s]);

                        SpacedKmerRecorder got;
                        single.for_each(input, got);
                        SpacedKmerRecorder unpacked;
                        auto const callback = callback_overload{
                            [&](std::size_t pos, std::size_t mask_idx, SpacedKmer<E, L> kmer) {
                                unpacked(pos, mask_idx, kmer);
                            },
                            [&](std::size_t pos, std::size_t mask_idx, auto kmers, auto valid) {
                                for_each_valid_lane(
                                    pos, kmers, valid, [&](std::size_t lane_pos, auto const& kmer) {
                                        unpacked(lane_pos, mask_idx, kmer);
                                    }
                                );
                            }
                        };
                        both.for_each(input, callback);
                        if (!in_situ_same_events(by_mask(got.events), expected) ||
                            !in_situ_same_events(by_mask(unpacked.events), expected)) {
                            std::cerr << "    in-situ mismatch: span " << overlap + 1 << ", "
                                      << masks.size() << " masks, rounds " << rounds << ", prefix "
                                      << prefix << ", length " << length << ", sequence " << s
                                      << "\n";
                            EXPECT_TRUE(false);
                        }
                    }
                }
            }
        }
    }
}

