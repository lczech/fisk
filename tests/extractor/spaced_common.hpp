#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extractor/spaced_kmer_extractor.hpp"
#include "extractor/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

// =================================================================================================
//     Spaced K-mer Extractor Checks
// =================================================================================================

// Every algorithm of the extractor, for sets of one and three masks, run with a callback of single
// spaced k-mers, and with one that also takes vectors. The extractor takes up to kMaxTestMasks
// masks, which keeps the compile time down; the largest sets are checked on their own, see
// test_spaced_kmer_extractor.cpp.
template <fisk::InputFormat I, fisk::InputCheck C, fisk::Encoding E, fisk::Layout L>
inline void check_spaced_kmer_extractor_algorithms()
{
    using namespace fisk;
    using Extractor = SpacedKmerExtractor<I, C, E, L, kMaxTestMasks>;
    auto const seqs = extractor_test_sequences<C>();
    std::vector<std::vector<std::string>> const mask_sets = {
        {"1101101"},
        random_masks(kMaxTestMasks, 11, 1),
    };

    for (auto const& masks : mask_sets) {
        Extractor extractor(masks);
        auto const names = extractor.algorithm_names();
        EXPECT_FALSE(names.empty());

        for (auto const& name : names) {
            extractor.set_algorithm(name);
            EXPECT_EQ(extractor.algorithm_name(), name);

            for (auto const& seq : seqs) {
                auto const expected = by_mask(oracle_spaced_events<E, L>(seq, masks));
                auto const input = extractor_test_input<Extractor>(seq);

                SpacedKmerRecorder single;
                extractor.for_each(input, single);
                check_spaced_events(by_mask(single.events), expected);

                // The SIMD algorithms hand their vectors to a callback that takes them.
                SpacedKmerRecorder unpacked;
                std::size_t vectors = 0;
                auto const both = callback_overload{
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
                extractor.for_each(input, both);
                check_spaced_events(by_mask(unpacked.events), expected);
                bool const simd = name.find("simd") != std::string::npos;
                EXPECT_TRUE(simd ? (vectors > 0 || expected.empty()) : vectors == 0);
            }
        }
    }
}
