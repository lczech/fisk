#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_assume_valid.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// The oracle (oracle.hpp) decodes directly from the ASCII input, independent of pack_sequence()/
// for_each_kmer_packed_aligned(), since this file's whole point is to also exercise the packing
// step.

// Sequence lengths 0..300, each with random per-position content -- deliberately covering many
// exact multiples, one-more, and one-less relationships against every chunk_size in
// test_chunk_sizes() below, rather than hand-picking a handful of specific (length, chunk_size)
// pairs.
static std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(90125);
        for (std::size_t len = 0; len <= 300; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        return out;
    }();
    return seqs;
}

// Sequences with invalid characters. The function assumes valid input, so each invalid character
// is silently encoded as some nucleotide; but that must only change the k-mers overlapping it,
// not their count or positions, nor any other k-mer. Hand-picked characters at the start, end, and
// around the word and chunk boundaries of the packing step, plus random bytes from the full range.
static std::vector<std::string> const& invalid_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(90126);
        std::size_t marker_idx = 0;
        for (std::size_t const len : std::vector<std::size_t>{1, 8, 33, 100}) {
            for (std::size_t const pos : {std::size_t{0}, len / 2, len - 1}) {
                std::string seq = random_sequence(rng, len);
                seq[pos] = kHandPickedInvalid[marker_idx++ % kHandPickedInvalid.size()];
                out.push_back(std::move(seq));
            }
        }
        for (auto& seq : invalid_injected_sequences(30, 150, 0.05, 90127)) {
            out.push_back(std::move(seq));
        }
        return out;
    }();
    return seqs;
}

// Every valid k: each one is dispatched to its own compile-time specialization, so each needs its
// own coverage.
static std::vector<std::size_t> const& all_ks()
{
    static std::vector<std::size_t> const ks = [] {
        std::vector<std::size_t> out;
        for (std::size_t k = 1; k <= 32; ++k) {
            out.push_back(k);
        }
        return out;
    }();
    return ks;
}

// Chunk sizes spanning: smaller than most k values above (each chunk then emits fewer k-mers than
// it looks ahead by), comparable to some of them, and much larger than the longest test sequence
// (degenerating to a single chunk).
static std::vector<std::size_t> const& test_chunk_sizes()
{
    static std::vector<std::size_t> const sizes = {1, 2, 3, 7, 8, 16, 31, 32, 33, 64, 1000};
    return sizes;
}

// Checks for_each_kmer_ascii_assume_valid() with the word encoder of the given conventions, for
// every sequence in `seqs`, every valid k, and every chunk size: exact positions for every window,
// and exact values for every window that consists of nucleotides only.
template <Encoding E, Layout L>
static void check_assume_valid(std::vector<std::string> const& seqs)
{
    WordEncoderButterfly<E, L> encoder;
    PackedSequence<E, L> scratch;
    for (auto const& seq : seqs) {
        for (auto const k : all_ks()) {
            auto const expected = oracle_kmers_all_windows<E, L>(seq, k);
            for (auto const chunk_size : test_chunk_sizes()) {
                check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
                    for_each_kmer_ascii_assume_valid(seq, k, encoder, chunk_size, callback);
                });
                check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
                    for_each_kmer_ascii_assume_valid(
                        seq, k, encoder, chunk_size, scratch, callback
                    );
                });
            }
        }
    }
}

// Smaller sweeps for ACTG: these exist to isolate the Encoding axis on its own, not to re-cover
// every (length, k, chunk_size) combination the ACGT ones already do.
static std::vector<std::string> const& actg_test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        for (std::size_t const len : std::vector<std::size_t>{0, 1, 31, 63, 64, 200}) {
            out.push_back(test_sequences()[len]);
        }
        return out;
    }();
    return seqs;
}

// =================================================================================================
//     for_each_kmer_ascii_assume_valid()
// =================================================================================================

TEST(KmerExtractAsciiAssumeValid, Msb)
{
    check_assume_valid<Encoding::kACGT, Layout::kMSB>(test_sequences());
}

TEST(KmerExtractAsciiAssumeValid, Lsb)
{
    check_assume_valid<Encoding::kACGT, Layout::kLSB>(test_sequences());
}

TEST(KmerExtractAsciiAssumeValid, ActgMsb)
{
    check_assume_valid<Encoding::kACTG, Layout::kMSB>(actg_test_sequences());
}

TEST(KmerExtractAsciiAssumeValid, ActgLsb)
{
    check_assume_valid<Encoding::kACTG, Layout::kLSB>(actg_test_sequences());
}

// Invalid characters, under all four conventions, since which nucleotide an invalid character is
// silently encoded as depends on the encoding's bit tricks.
TEST(KmerExtractAsciiAssumeValid, InvalidInput)
{
    check_assume_valid<Encoding::kACGT, Layout::kMSB>(invalid_sequences());
    check_assume_valid<Encoding::kACGT, Layout::kLSB>(invalid_sequences());
    check_assume_valid<Encoding::kACTG, Layout::kMSB>(invalid_sequences());
    check_assume_valid<Encoding::kACTG, Layout::kLSB>(invalid_sequences());
}

// The convenience overload (no explicit chunk_size, so kDefaultAsciiChunkSize) must be exactly
// correct as well, in both callback forms.
TEST(KmerExtractAsciiAssumeValid, ConvenienceOverload)
{
    WordEncoderButterfly<Encoding::kACGT, Layout::kMSB> encoder;
    PackedSequence<Encoding::kACGT, Layout::kMSB> scratch;
    for (auto const& seq : test_sequences()) {
        for (auto const k : all_ks()) {
            auto const expected = oracle_kmers_all_windows<Encoding::kACGT, Layout::kMSB>(seq, k);
            check_kmer_callbacks<Encoding::kACGT, Layout::kMSB>(expected, k, [&](auto const& cb) {
                for_each_kmer_ascii_assume_valid(seq, k, encoder, cb);
            });
            check_kmer_callbacks<Encoding::kACGT, Layout::kMSB>(expected, k, [&](auto const& cb) {
                for_each_kmer_ascii_assume_valid(seq, k, encoder, scratch, cb);
            });
        }
    }
}

// Separate caller-owned buffers make same-encoder extraction inside a callback safe.
TEST(KmerExtractAsciiAssumeValid, ScratchEnablesNestedSameEncoder)
{
    constexpr Encoding E = Encoding::kACGT;
    constexpr Layout L = Layout::kMSB;
    constexpr std::size_t k = 21;
    WordEncoderButterfly<E, L> encoder;
    std::string const& outer = test_sequences()[100];
    std::string const& inner = test_sequences()[63];
    auto const outer_expected = oracle_kmers_all_windows<E, L>(outer, k);
    auto const inner_expected = oracle_kmers_all_windows<E, L>(inner, k);
    PackedSequence<E, L> outer_scratch;
    PackedSequence<E, L> inner_scratch;
    std::vector<std::size_t> outer_positions;
    std::vector<std::uint64_t> outer_values;

    for_each_kmer_ascii_assume_valid(
        outer, k, encoder, outer_scratch,
        [&](std::size_t outer_pos, Kmer<E, L> outer_kmer) {
            std::vector<std::size_t> inner_positions;
            std::vector<std::uint64_t> inner_values;
            for_each_kmer_ascii_assume_valid(
                inner, k, encoder, inner_scratch,
                [&](std::size_t inner_pos, Kmer<E, L> inner_kmer) {
                    inner_positions.push_back(inner_pos);
                    inner_values.push_back(kmer_value(inner_kmer));
                }
            );
            check_emitted_kmers(inner_expected, k, inner_values, &inner_positions);
            outer_positions.push_back(outer_pos);
            outer_values.push_back(kmer_value(outer_kmer));
        }
    );
    check_emitted_kmers(outer_expected, k, outer_values, &outer_positions);
}

// =================================================================================================
//     Error Contract
// =================================================================================================

TEST(KmerExtractAsciiAssumeValid, InvalidKThrows)
{
    WordEncoderButterfly<Encoding::kACGT, Layout::kMSB> encoder;
    PackedSequence<Encoding::kACGT, Layout::kMSB> scratch;
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        for_each_kmer_ascii_assume_valid(seq, k, encoder, [](KmerAcgtMsb) {});
    });
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        for_each_kmer_ascii_assume_valid(seq, k, encoder, std::size_t{7}, [](KmerAcgtMsb) {});
    });
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        for_each_kmer_ascii_assume_valid(
            seq, k, encoder, std::size_t{7}, scratch, [](KmerAcgtMsb) {}
        );
    });
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        for_each_kmer_ascii_assume_valid(seq, k, encoder, scratch, [](KmerAcgtMsb) {});
    });
}

TEST(KmerExtractAsciiAssumeValid, InvalidChunkSizeThrows)
{
    WordEncoderButterfly<Encoding::kACGT, Layout::kMSB> encoder;
    PackedSequence<Encoding::kACGT, Layout::kMSB> scratch;
    EXPECT_THROW(
        for_each_kmer_ascii_assume_valid(
            "ACGTACGT", 3, encoder, std::size_t{0}, [](KmerAcgtMsb) {}
        ),
        std::invalid_argument
    );
    EXPECT_THROW(
        for_each_kmer_ascii_assume_valid(
            "ACGTACGT", 3, encoder, std::size_t{0}, scratch, [](KmerAcgtMsb) {}
        ),
        std::invalid_argument
    );
}
