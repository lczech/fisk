#include <cstddef>
#include <string>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extraction/spaced_kmer_extractor.hpp"
#include "extraction/common.hpp"
#include "extraction/piece_common.hpp"
#include "oracle.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "testing.hpp"

using namespace fisk;

// The packed spaced k-mer algorithms of the extractor on pieces of packed sequences, see
// make_extractor_packed_piece() and piece_common.hpp: each piece hands out exactly the spaced
// k-mers of the whole sequence that start in it, at their positions in the sequence, with a
// callback of single spaced k-mers and with one that also takes vectors.

namespace {

bool same_events(std::vector<SpacedEvent> const& got, std::vector<SpacedEvent> const& expected)
{
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

template <InputCheck C, Layout L>
void check_spaced_packed_pieces()
{
    constexpr auto E = Encoding::kACGT;
    using Extractor = SpacedKmerExtractor<InputFormat::kPacked, C, E, L, kMaxTestMasks>;
    using Candidates = typename SpacedKmerExtractorCandidates<
        InputFormat::kPacked, C, E, L, kMaxTestMasks
    >::type;
    Splitmix64 rng(C == InputCheck::kChecked ? 27 : 28);

    // Single masks of spans around the words of the loops, up to the widest, and sets of several
    // masks, of one piece and of three, see extraction_wide_mask_sets().
    std::vector<std::vector<std::string>> mask_sets;
    for (std::size_t span : {3u, 31u, 32u, 33u, 64u}) {
        mask_sets.push_back(random_masks(1, span, span));
    }
    mask_sets.push_back(random_masks(kMaxTestMasks, 11, 11));
    mask_sets.push_back(random_masks(kMaxTestMasks, 58, 58));

    for (auto const& masks : mask_sets) {
        Extractor const extractor(masks);
        std::size_t const overlap = masks.front().size() - 1;
        std::string const what =
            std::to_string(masks.size()) + " masks of span " + std::to_string(overlap + 1);
        for (std::size_t length = 0; length <= kPieceTestMaxLength; ++length) {
            auto const seqs = piece_test_sequences<C>(length, overlap, rng);
            for (std::size_t s = 0; s < seqs.size(); ++s) {
                auto const all = oracle_spaced_events<E, L>(seqs[s], masks);
                auto const packed = extraction_test_input<Extractor>(seqs[s]);

                for (auto const& piece : piece_test_pieces(length, overlap)) {
                    std::vector<SpacedEvent> expected;
                    for (auto const& e : all) {
                        if (e.pos >= piece.start && e.pos < piece.start + piece.starts) {
                            expected.push_back(e);
                        }
                    }
                    expected = by_mask(std::move(expected));
                    auto const part = make_extractor_packed_piece(
                        packed, piece.start, piece.starts, overlap
                    );

                    piece_test_for_each_type(Candidates{}, [&](auto candidate) {
                        if (!candidate.available()) {
                            return;
                        }

                        // Single spaced k-mers, at their positions in the sequence.
                        SpacedKmerRecorder single;
                        auto shifted = [&](
                            std::size_t pos, std::size_t mask_idx, SpacedKmer<E, L> kmer
                        ) {
                            single(piece.start + pos, mask_idx, kmer);
                        };
                        candidate.run(extractor.mask_forms(), part, shifted);
                        if (!same_events(by_mask(single.events), expected)) {
                            piece_test_fail(candidate.name(), length, s, what, piece, "single");
                        }

                        // A callback that also takes the vectors of the SIMD algorithms.
                        SpacedKmerRecorder unpacked;
                        auto both = callback_overload{
                            [&](std::size_t pos, std::size_t mask_idx, SpacedKmer<E, L> kmer) {
                                unpacked(piece.start + pos, mask_idx, kmer);
                            },
                            [&](std::size_t pos, std::size_t mask_idx, auto kmers, auto valid) {
                                for_each_valid_lane(
                                    pos, kmers, valid, [&](std::size_t lane_pos, auto const& kmer) {
                                        unpacked(piece.start + lane_pos, mask_idx, kmer);
                                    }
                                );
                            }
                        };
                        candidate.run(extractor.mask_forms(), part, both);
                        if (!same_events(by_mask(unpacked.events), expected)) {
                            piece_test_fail(candidate.name(), length, s, what, piece, "vector");
                        }
                    });
                }
            }
        }
    }
}

} // namespace

TEST(PackedPieces, SpacedChecked)
{
    check_spaced_packed_pieces<InputCheck::kChecked, Layout::kMSB>();
    check_spaced_packed_pieces<InputCheck::kChecked, Layout::kLSB>();
}

TEST(PackedPieces, SpacedUnchecked)
{
    check_spaced_packed_pieces<InputCheck::kUnchecked, Layout::kMSB>();
    check_spaced_packed_pieces<InputCheck::kUnchecked, Layout::kLSB>();
}
