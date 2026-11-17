#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "fisk/extraction/extractor_plan.hpp"
#include "fisk/util/random.hpp"
#include "corpus.hpp"
#include "testing.hpp"

// =================================================================================================
//     Packed Piece Test Helpers
// =================================================================================================

// The tests of the pieces of packed sequences, see make_extractor_packed_piece(), run every packed
// algorithm of the extractors on every piece of sequences of all lengths up to kPieceTestMaxLength,
// and compare its k-mers with those of the oracle that start in the piece.
inline constexpr std::size_t kPieceTestMaxLength = 1024;

// A piece of a sequence: the k-mers that start at the `starts` positions from `start`.
struct PieceTestPiece
{
    std::size_t start;
    std::size_t starts;
};

// Every piece of a sequence of `length` nucleotides that make_extractor_packed_piece() takes:
// starting at each multiple of kExtractorPieceUnit, with each multiple of kExtractorPieceUnit of
// starts within the sequence, and with all starts up to its end. A sequence without k-mers has
// one piece of no starts.
inline std::vector<PieceTestPiece> piece_test_pieces(std::size_t length, std::size_t overlap)
{
    constexpr std::size_t unit = fisk::kExtractorPieceUnit;
    std::size_t const total = length > overlap ? length - overlap : 0;
    std::vector<PieceTestPiece> pieces;
    for (std::size_t start = 0; start == 0 || start < total; start += unit) {
        for (std::size_t starts = unit; start + starts < total; starts += unit) {
            pieces.push_back({start, starts});
        }
        pieces.push_back({start, total - start});
    }
    return pieces;
}

// The sequences of `length` nucleotides that the pieces are tested on: one of valid nucleotides,
// and, when checked, one with invalid ones at and around the boundaries of the pieces and those of
// their overlap, with a few more at random, and one with many at random.
template <fisk::InputCheck C>
inline std::vector<std::string> piece_test_sequences(
    std::size_t length, std::size_t overlap, fisk::Splitmix64& rng
) {
    std::vector<std::string> seqs;
    seqs.push_back(random_sequence(rng, length, "ACGT"));
    if constexpr (C == fisk::InputCheck::kChecked) {
        auto seq = random_sequence(rng, length, "ACGT");
        auto const set_invalid = [&](std::size_t pos) {
            if (pos < seq.size()) {
                seq[pos] = 'N';
            }
        };
        set_invalid(0);
        set_invalid(length - 1);
        for (std::size_t b = fisk::kExtractorPieceUnit; b <= kPieceTestMaxLength;
             b += fisk::kExtractorPieceUnit) {
            for (std::size_t pos : {b - 1, b, b + 1, b + overlap - 1, b + overlap}) {
                set_invalid(pos);
            }
        }
        inject_invalid(seq, rng, 0.005, "N");
        seqs.push_back(std::move(seq));

        auto many = random_sequence(rng, length, "ACGT");
        inject_invalid(many, rng, 0.1, "N");
        seqs.push_back(std::move(many));
    }
    return seqs;
}

// Report the case of a failed check of a piece, and fail the test.
inline void piece_test_fail(
    std::string const& algorithm, std::size_t length, std::size_t seq_index,
    std::string const& what, PieceTestPiece const& piece, char const* callback
) {
    std::cerr << "    piece mismatch: " << algorithm << ", " << what << ", length " << length
              << ", sequence " << seq_index << ", start " << piece.start << ", starts "
              << piece.starts << ", " << callback << " callback\n";
    EXPECT_TRUE(false);
}

// Call `f` with a default-constructed object of each type of a TypeList.
template <typename... Ts, typename F>
inline void piece_test_for_each_type(fisk::TypeList<Ts...>, F&& f)
{
    (f(Ts{}), ...);
}
