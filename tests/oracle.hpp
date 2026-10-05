#pragma once

// Shared ground truth, and the checks built on it, for the whole test suite.
//
// The oracle is built from characters with plain per-character rules, deliberately independent of
// every encoder, packer and extractor under test, so that a mistake in the bit tricks cannot
// cancel itself out against the reference. It only uses the library's convention tags (Encoding,
// Layout), its data types (Kmer, PackedSequence), and the kInvalidNucleotide constant, all of
// which are part of the contracts being tested, not of their implementations.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/types.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "corpus.hpp"
#include "testing.hpp"

// =================================================================================================
//     Characters and Windows
// =================================================================================================

// Two-bit code of `c` under E, case-insensitive, or kInvalidNucleotide for anything that is not a
// nucleotide, matching the contract of the library's validating char encoders. Ends in a
// static_assert for the same reason the encoders themselves do: a third Encoding must break the
// oracle loudly rather than be silently treated as one of the existing two.
template <fisk::Encoding E>
inline std::uint8_t oracle_code(char c)
{
    if constexpr (E == fisk::Encoding::kACGT) {
        switch (c) {
            case 'A': case 'a': return 0;
            case 'C': case 'c': return 1;
            case 'G': case 'g': return 2;
            case 'T': case 't': return 3;
            default:            return fisk::kInvalidNucleotide;
        }
    } else if constexpr (E == fisk::Encoding::kACTG) {
        switch (c) {
            case 'A': case 'a': return 0;
            case 'C': case 'c': return 1;
            case 'T': case 't': return 2;
            case 'G': case 'g': return 3;
            default:            return fisk::kInvalidNucleotide;
        }
    } else {
        static_assert(fisk::dependent_false_v<E>, "Unhandled Encoding in the test oracle");
        return fisk::kInvalidNucleotide;
    }
}

// Encode a whole window into a raw word, base by base, honoring both conventions; nullopt if any
// character in it is not a nucleotide. The window must be at most 32 characters.
template <fisk::Encoding E, fisk::Layout L>
inline std::optional<std::uint64_t> oracle_encode(std::string_view window)
{
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < window.size(); ++i) {
        std::uint8_t const code = oracle_code<E>(window[i]);
        if (code >= fisk::kInvalidNucleotide) {
            return std::nullopt;
        }
        std::size_t shift = 0;
        if constexpr (L == fisk::Layout::kMSB) {
            shift = 2 * (window.size() - 1 - i);
        } else if constexpr (L == fisk::Layout::kLSB) {
            shift = 2 * i;
        } else {
            static_assert(fisk::dependent_false_v<L>, "Unhandled Layout in the test oracle");
        }
        value |= std::uint64_t{code} << shift;
    }
    return value;
}

// =================================================================================================
//     Expected K-mers
// =================================================================================================

// One expected emission of a k-mer extractor: the window start position, and its value, or
// nullopt if the window overlaps a character that is not a nucleotide.
struct ExpectedKmer
{
    std::size_t pos;
    std::optional<std::uint64_t> value;
};

// Every window of length k in `seq`, in order, valid or not. This is what the extractors that
// assume valid input emit: one k-mer per window, whose value is only specified for windows that
// do not overlap an invalid character.
template <fisk::Encoding E, fisk::Layout L>
inline std::vector<ExpectedKmer> oracle_kmers_all_windows(std::string_view seq, std::size_t k)
{
    std::vector<ExpectedKmer> out;
    for (std::size_t pos = 0; pos + k <= seq.size(); ++pos) {
        out.push_back({pos, oracle_encode<E, L>(seq.substr(pos, k))});
    }
    return out;
}

// Only the windows of length k in `seq` that consist of nucleotides only, in order. This is what
// the validating extractors emit: every window that overlaps an invalid character is skipped.
template <fisk::Encoding E, fisk::Layout L>
inline std::vector<ExpectedKmer> oracle_kmers_skipping_invalid(std::string_view seq, std::size_t k)
{
    auto out = oracle_kmers_all_windows<E, L>(seq, k);
    std::erase_if(out, [](ExpectedKmer const& e) { return !e.value.has_value(); });
    return out;
}

// =================================================================================================
//     K-mer Checks
// =================================================================================================

// Checks one extractor call's output, in emission order, against `expected`: the values always,
// and the positions if given (they are not, for the callback form without `pos`). Windows the
// oracle has no value for are those overlapping an invalid character, fed to an extractor that
// assumes valid input; their value is unspecified, but must still be a valid k-mer of width k.
inline void check_emitted_kmers(
    std::vector<ExpectedKmer> const& expected, std::size_t k,
    std::vector<std::uint64_t> const& values, std::vector<std::size_t> const* positions
) {
    EXPECT_EQ(values.size(), expected.size());
    if (positions) {
        EXPECT_EQ(positions->size(), values.size());
    }

    std::size_t const n = std::min(values.size(), expected.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (positions && i < positions->size()) {
            EXPECT_EQ((*positions)[i], expected[i].pos);
        }
        if (expected[i].value) {
            EXPECT_EQ(values[i], *expected[i].value);
        } else if (k < 32) {
            EXPECT_EQ(values[i] >> (2 * k), std::uint64_t{0});
        }
    }
}

// Runs `extract`, called as `extract(callback)`, once with a `callback(pos, kmer)` and once with a
// `callback(kmer)`, and checks both against `expected`. Both callback forms are accepted by every
// scalar extractor (see invoke_kmer_callback()), so both are checked for every one of them.
template <fisk::Encoding E, fisk::Layout L, typename Extract>
inline void check_kmer_callbacks(
    std::vector<ExpectedKmer> const& expected, std::size_t k, Extract&& extract
) {
    std::vector<std::size_t> positions;
    std::vector<std::uint64_t> values;
    extract([&](std::size_t pos, fisk::Kmer<E, L> kmer) {
        positions.push_back(pos);
        values.push_back(fisk::kmer_value(kmer));
    });
    check_emitted_kmers(expected, k, values, &positions);

    values.clear();
    extract([&](fisk::Kmer<E, L> kmer) { values.push_back(fisk::kmer_value(kmer)); });
    check_emitted_kmers(expected, k, values, nullptr);
}

// =================================================================================================
//     Error Contract
// =================================================================================================

// Checks that `extract`, called as `extract(seq, k)`, rejects every invalid k with
// std::invalid_argument: 0, one past `max_k`, and a value large enough to overflow any `k - 1` or
// `2 * k` arithmetic done before the range check. Each on empty input too, so that an early return
// for sequences shorter than k cannot skip the check.
template <typename Extract>
inline void check_invalid_k_throws(std::size_t max_k, Extract&& extract)
{
    for (std::string const& seq : {std::string{}, std::string(129, 'T')}) {
        for (std::size_t const k : {
            std::size_t{0}, max_k + 1, std::numeric_limits<std::size_t>::max()
        }) {
            EXPECT_THROW(extract(seq, k), std::invalid_argument);
        }
    }
}

// =================================================================================================
//     Packed Sequences
// =================================================================================================

// The bits of the last byte beyond `length` must be zero (see PackedSequence in core/types.hpp).
template <fisk::Encoding E, fisk::Layout L>
inline void check_packed_padding_zero(fisk::PackedSequence<E, L> const& packed)
{
    std::size_t const used = packed.length % 4;
    if (used == 0 || packed.data.empty()) {
        return;
    }

    std::uint8_t const last = packed.data.back();
    unsigned padding = 0;
    if constexpr (L == fisk::Layout::kMSB) {
        padding = last & (0xFFu >> (2 * used));
    } else if constexpr (L == fisk::Layout::kLSB) {
        padding = static_cast<unsigned>(last) >> (2 * used);
    } else {
        static_assert(fisk::dependent_false_v<L>, "Unhandled Layout in the test oracle");
    }
    EXPECT_EQ(padding, 0u);
}

// Runs `check_one(seq, packed, k, expected)` for every sequence in `seqs`, plus a fixed set of
// edge cases, packed with `encoder`, and every k in [1, max_k], with `expected` the oracle's k-mers
// for that sequence and k. Shared by the scalar and SIMD packed extractor tests, so that both are
// held to the same inputs. Also checks the padding of every sequence it packs.
template <typename Encoder, typename CheckOne>
inline void sweep_packed_extractors(
    Encoder const& encoder, std::vector<std::string> const& seqs, std::size_t max_k,
    CheckOne&& check_one
) {
    constexpr auto E = Encoder::encoding;
    constexpr auto L = Encoder::layout;

    auto check_sequence = [&](std::string const& seq) {
        auto const packed = fisk::pack_sequence(seq, encoder);
        check_packed_padding_zero(packed);
        for (std::size_t k = 1; k <= max_k; ++k) {
            check_one(seq, packed, k, oracle_kmers_all_windows<E, L>(seq, k));
        }
    };
    for (auto const& seq : seqs) {
        check_sequence(seq);
    }

    // All-zero/all-one windows and isolated nonzero bases expose lost high bits and bad masks.
    check_sequence(std::string(129, 'A'));
    check_sequence(std::string(129, 'T'));
    for (std::size_t p = 0; p < 40; ++p) {
        std::string seq(40, 'A');
        seq[p] = 'T';
        check_sequence(seq);
    }
}
