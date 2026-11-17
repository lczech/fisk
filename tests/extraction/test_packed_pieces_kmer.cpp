#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extraction/kmer_extractor.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "extraction/common.hpp"
#include "extraction/piece_common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// The packed k-mer algorithms of the extractor on pieces of packed sequences, see
// make_extractor_packed_piece() and piece_common.hpp: each piece hands out exactly the k-mers of
// the whole sequence that start in it, at their positions in the sequence, with a callback of
// single k-mers and with one that also takes vectors.

namespace {

template <InputCheck C, Layout L>
void check_kmer_packed_pieces()
{
    constexpr auto E = Encoding::kACGT;
    using Candidates = typename KmerExtractorCandidates<InputFormat::kPacked, C, E, L>::type;
    Splitmix64 rng(C == InputCheck::kChecked ? 17 : 18);

    for (std::size_t k : std::initializer_list<std::size_t>{1, 2, 5, 31, 32}) {
        std::size_t const overlap = k - 1;
        for (std::size_t length = 0; length <= kPieceTestMaxLength; ++length) {
            auto const seqs = piece_test_sequences<C>(length, overlap, rng);
            for (std::size_t s = 0; s < seqs.size(); ++s) {
                auto const all = C == InputCheck::kChecked
                    ? oracle_kmers_skipping_invalid<E, L>(seqs[s], k)
                    : oracle_kmers_all_windows<E, L>(seqs[s], k);
                auto const packed =
                    extraction_test_input<KmerExtractor<InputFormat::kPacked, C, E, L>>(seqs[s]);

                for (auto const& piece : piece_test_pieces(length, overlap)) {
                    std::vector<std::size_t> expected_positions;
                    std::vector<std::uint64_t> expected_values;
                    for (auto const& e : all) {
                        if (e.pos >= piece.start && e.pos < piece.start + piece.starts) {
                            expected_positions.push_back(e.pos);
                            expected_values.push_back(*e.value);
                        }
                    }
                    auto const part = make_extractor_packed_piece(
                        packed, piece.start, piece.starts, overlap
                    );

                    piece_test_for_each_type(Candidates{}, [&](auto candidate) {
                        if (!candidate.available()) {
                            return;
                        }
                        std::string const what = "k " + std::to_string(k);

                        // Single k-mers, at their positions in the sequence.
                        KmerRecorder single;
                        auto shifted = [&](std::size_t pos, Kmer<E, L> kmer) {
                            single(piece.start + pos, kmer);
                        };
                        candidate.run(k, part, shifted);
                        if (single.positions != expected_positions ||
                            single.values != expected_values) {
                            piece_test_fail(candidate.name(), length, s, what, piece, "single");
                        }

                        // A callback that also takes the vectors of the SIMD algorithms.
                        KmerRecorder unpacked;
                        auto both = callback_overload{
                            [&](std::size_t pos, Kmer<E, L> kmer) {
                                unpacked(piece.start + pos, kmer);
                            },
                            [&](std::size_t pos, auto kmers, auto valid) {
                                for_each_valid_lane(
                                    pos, kmers, valid, [&](std::size_t lane_pos, auto const& kmer) {
                                        unpacked(piece.start + lane_pos, kmer);
                                    }
                                );
                            }
                        };
                        candidate.run(k, part, both);
                        if (unpacked.positions != expected_positions ||
                            unpacked.values != expected_values) {
                            piece_test_fail(candidate.name(), length, s, what, piece, "vector");
                        }
                    });
                }
            }
        }
    }
}

} // namespace

TEST(PackedPieces, KmerChecked)
{
    check_kmer_packed_pieces<InputCheck::kChecked, Layout::kMSB>();
    check_kmer_packed_pieces<InputCheck::kChecked, Layout::kLSB>();
}

TEST(PackedPieces, KmerUnchecked)
{
    check_kmer_packed_pieces<InputCheck::kUnchecked, Layout::kMSB>();
    check_kmer_packed_pieces<InputCheck::kUnchecked, Layout::kLSB>();
}
