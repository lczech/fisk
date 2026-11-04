#include <cstddef>
#include <string>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extractor/for_each.hpp"
#include "fisk/extractor/spaced_kmer_extractor.hpp"
#include "fisk/extractor/spaced_kmer_masks.hpp"
#include "extractor/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kActg = Encoding::kACTG;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kLsb = Layout::kLSB;
constexpr auto kAscii = InputFormat::kAscii;
constexpr auto kPacked = InputFormat::kPacked;
constexpr auto kChecked = InputCheck::kChecked;
constexpr auto kUnchecked = InputCheck::kUnchecked;

// =================================================================================================
//     Spaced K-mers
// =================================================================================================

// for_each_spaced_kmer() extracts all spaced k-mers of each mask, for sets of one and of the most
// masks, as for_each_kmer(), see test_for_each_kmer.cpp.
template <InputFormat I, InputCheck C, Encoding E, Layout L>
static void check_for_each_spaced_kmer()
{
    using Extractor = SpacedKmerExtractor<I, C, E, L, kMaxTestMasks>;
    auto const seqs = extractor_test_sequences<C>();
    std::vector<std::vector<std::string>> const mask_sets = {
        {"1101101"},
        random_masks(kMaxTestMasks, 11, 1),
    };

    for (auto const& strings : mask_sets) {
        SpacedKmerMasks<L, kMaxTestMasks> const masks(strings);
        Extractor const extractor(masks);
        for (auto const& seq : seqs) {
            auto const expected = by_mask(oracle_spaced_events<E, L>(seq, strings));
            auto const input = extractor_test_input<Extractor>(seq);

            SpacedKmerRecorder single;
            for_each_spaced_kmer<C, E, L>(input, masks, single);
            check_spaced_events(by_mask(single.events), expected);
            auto const single_name =
                for_each_spaced_kmer_algorithm_name<C, E, L>(input, masks, single);
            EXPECT_EQ(single_name, extractor.algorithm_name(single));

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
            for_each_spaced_kmer<C, E, L>(input, masks, both);
            check_spaced_events(by_mask(unpacked.events), expected);
            auto const name = for_each_spaced_kmer_algorithm_name<C, E, L>(input, masks, both);
            EXPECT_EQ(name, extractor.algorithm_name(both));
            bool const simd = name.find("simd") != std::string::npos;
            EXPECT_TRUE(simd ? (vectors > 0 || expected.empty()) : vectors == 0);

            // Packed input gives the template arguments by its type.
            if constexpr (I == kPacked) {
                SpacedKmerRecorder deduced;
                for_each_spaced_kmer(input, masks, deduced);
                check_spaced_events(deduced.events, single.events);
                EXPECT_EQ(
                    for_each_spaced_kmer_algorithm_name(input, masks, deduced),
                    extractor.algorithm_name()
                );
            }
        }
    }
}

TEST(ForEachSpacedKmer, AsciiChecked)
{
    check_for_each_spaced_kmer<kAscii, kChecked, kAcgt, kMsb>();
    check_for_each_spaced_kmer<kAscii, kChecked, kActg, kLsb>();
}

TEST(ForEachSpacedKmer, AsciiUnchecked)
{
    check_for_each_spaced_kmer<kAscii, kUnchecked, kAcgt, kMsb>();
}

TEST(ForEachSpacedKmer, PackedChecked)
{
    check_for_each_spaced_kmer<kPacked, kChecked, kAcgt, kMsb>();
}

TEST(ForEachSpacedKmer, PackedUnchecked)
{
    check_for_each_spaced_kmer<kPacked, kUnchecked, kAcgt, kLsb>();
}
