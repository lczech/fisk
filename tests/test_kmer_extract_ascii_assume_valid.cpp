#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_assume_valid.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Oracles decode directly from the ASCII input, independent of pack_sequence()/
// for_each_kmer_packed_aligned() (unlike test_kmer_extract_packed.cpp's oracle, which only needs
// to be independent of the packed extractors, since this file's whole point is to also exercise
// the packing step). Deliberately not reused from anywhere else in the test suite, so that a
// mistake shared between production code and its oracle cannot cancel out.

static int code_acgt(char c)
{
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return -1;
    }
}

static int code_actg(char c)
{
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'T': case 't': return 2;
        case 'G': case 'g': return 3;
        default:            return -1;
    }
}

static std::uint64_t oracle_msb_acgt(std::string const& seq, std::size_t start, std::size_t k)
{
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < k; ++i) {
        v = (v << 2) | static_cast<std::uint64_t>(code_acgt(seq[start + i]));
    }
    return v;
}

static std::uint64_t oracle_lsb_acgt(std::string const& seq, std::size_t start, std::size_t k)
{
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < k; ++i) {
        v |= static_cast<std::uint64_t>(code_acgt(seq[start + i])) << (2 * i);
    }
    return v;
}

static std::uint64_t oracle_msb_actg(std::string const& seq, std::size_t start, std::size_t k)
{
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < k; ++i) {
        v = (v << 2) | static_cast<std::uint64_t>(code_actg(seq[start + i]));
    }
    return v;
}

static std::uint64_t oracle_lsb_actg(std::string const& seq, std::size_t start, std::size_t k)
{
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < k; ++i) {
        v |= static_cast<std::uint64_t>(code_actg(seq[start + i])) << (2 * i);
    }
    return v;
}

// Sequence lengths 0..300, each with random per-position content -- deliberately covering many
// exact multiples, one-more, and one-less relationships against every chunk_size in
// test_chunk_sizes() below, rather than hand-picking a handful of specific (length, chunk_size)
// pairs.
static std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        char const bases[] = "ACGTacgt";
        Splitmix64 rng(90125);

        auto random_seq = [&](std::size_t len) {
            std::string s;
            s.reserve(len);
            for (std::size_t i = 0; i < len; ++i) {
                s += bases[rng.get_uint64() % 8];
            }
            return s;
        };

        for (std::size_t len = 0; len <= 300; ++len) {
            out.push_back(random_seq(len));
        }
        return out;
    }();
    return seqs;
}

// Every valid k: each one is dispatched to its own compile-time specialization, so each needs its
// own coverage.
static std::vector<std::size_t> const& test_ks()
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

// Checks `got` (one call's emitted k-mers for `seq`/`k`) against the base-by-base oracle.
template <typename OracleFn>
static void check_kmers(
    std::vector<std::uint64_t> const& got, std::string const& seq, std::size_t k, OracleFn&& oracle
) {
    std::vector<std::uint64_t> exp;
    if (seq.size() >= k) {
        for (std::size_t start = 0; start + k <= seq.size(); ++start) {
            exp.push_back(oracle(seq, start, k));
        }
    }

    ASSERT_EQ(got.size(), exp.size());
    for (std::size_t i = 0; i < exp.size(); ++i) {
        EXPECT_EQ(got[i], exp[i]);
    }
}

// =================================================================================================
//     for_each_kmer_ascii_assume_valid()
// =================================================================================================

TEST(KmerExtractAsciiAssumeValid, Msb)
{
    WordEncoderButterfly<Encoding::kACGT, Layout::kMSB> encoder;
    for (auto const& seq : test_sequences()) {
        for (auto const k : test_ks()) {
            for (auto const chunk_size : test_chunk_sizes()) {
                std::vector<std::uint64_t> got;
                for_each_kmer_ascii_assume_valid(
                    seq, k, encoder, chunk_size,
                    [&](KmerAcgtMsb kmer) { got.push_back(kmer_value(kmer)); }
                );
                check_kmers(got, seq, k, oracle_msb_acgt);
            }
        }
    }
}

TEST(KmerExtractAsciiAssumeValid, Lsb)
{
    WordEncoderButterfly<Encoding::kACGT, Layout::kLSB> encoder;
    for (auto const& seq : test_sequences()) {
        for (auto const k : test_ks()) {
            for (auto const chunk_size : test_chunk_sizes()) {
                std::vector<std::uint64_t> got;
                for_each_kmer_ascii_assume_valid(
                    seq, k, encoder, chunk_size,
                    [&](KmerAcgtLsb kmer) { got.push_back(kmer_value(kmer)); }
                );
                check_kmers(got, seq, k, oracle_lsb_acgt);
            }
        }
    }
}

// Smaller sweeps than Msb/Lsb above: these exist to isolate the Encoding axis (ACTG instead of
// ACGT) on its own, not to re-cover every (length, k, chunk_size) combination those already do.
static std::vector<std::size_t> const& actg_test_lengths()
{
    static std::vector<std::size_t> const lens = {0, 1, 31, 63, 64, 200};
    return lens;
}

TEST(KmerExtractAsciiAssumeValid, ActgMsb)
{
    WordEncoderButterfly<Encoding::kACTG, Layout::kMSB> encoder;
    for (auto const len : actg_test_lengths()) {
        std::string const& seq = test_sequences()[len];
        for (auto const k : test_ks()) {
            for (auto const chunk_size : test_chunk_sizes()) {
                std::vector<std::uint64_t> got;
                for_each_kmer_ascii_assume_valid(
                    seq, k, encoder, chunk_size,
                    [&](KmerActgMsb kmer) { got.push_back(kmer_value(kmer)); }
                );
                check_kmers(got, seq, k, oracle_msb_actg);
            }
        }
    }
}

TEST(KmerExtractAsciiAssumeValid, ActgLsb)
{
    WordEncoderButterfly<Encoding::kACTG, Layout::kLSB> encoder;
    for (auto const len : actg_test_lengths()) {
        std::string const& seq = test_sequences()[len];
        for (auto const k : test_ks()) {
            for (auto const chunk_size : test_chunk_sizes()) {
                std::vector<std::uint64_t> got;
                for_each_kmer_ascii_assume_valid(
                    seq, k, encoder, chunk_size,
                    [&](KmerActgLsb kmer) { got.push_back(kmer_value(kmer)); }
                );
                check_kmers(got, seq, k, oracle_lsb_actg);
            }
        }
    }
}

// The convenience overload (no explicit chunk_size) must agree with the full overload called
// with kDefaultAsciiChunkSize -- exactly, not just in aggregate.
TEST(KmerExtractAsciiAssumeValid, ConvenienceOverloadMatchesDefault)
{
    WordEncoderButterfly<Encoding::kACGT, Layout::kMSB> encoder;
    for (auto const& seq : test_sequences()) {
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> with_default, with_explicit;
            for_each_kmer_ascii_assume_valid(
                seq, k, encoder,
                [&](KmerAcgtMsb kmer) { with_default.push_back(kmer_value(kmer)); }
            );
            for_each_kmer_ascii_assume_valid(
                seq, k, encoder, kDefaultAsciiChunkSize,
                [&](KmerAcgtMsb kmer) { with_explicit.push_back(kmer_value(kmer)); }
            );
            EXPECT_EQ(with_default, with_explicit);
        }
    }
}

// The callback may also take a leading `pos` (see invoke_kmer_callback()). Positions are reported
// per chunk internally, so this checks that they come out relative to the whole sequence, across
// chunk boundaries: every k-mer start 0..n-k, in order, each with the k-mer value at that start.
TEST(KmerExtractAsciiAssumeValid, PositionMatchesOracle)
{
    WordEncoderButterfly<Encoding::kACGT, Layout::kMSB> encoder;
    for (auto const& seq : test_sequences()) {
        for (auto const k : test_ks()) {
            for (auto const chunk_size : test_chunk_sizes()) {
                std::vector<std::size_t> positions;
                std::vector<std::uint64_t> values;
                for_each_kmer_ascii_assume_valid(
                    seq, k, encoder, chunk_size,
                    [&](std::size_t pos, KmerAcgtMsb kmer) {
                        positions.push_back(pos);
                        values.push_back(kmer_value(kmer));
                    }
                );

                std::size_t const n_exp = seq.size() >= k ? seq.size() - k + 1 : 0;
                ASSERT_EQ(positions.size(), n_exp);
                for (std::size_t i = 0; i < n_exp; ++i) {
                    EXPECT_EQ(positions[i], i);
                    EXPECT_EQ(values[i], oracle_msb_acgt(seq, positions[i], k));
                }
            }
        }
    }
}

// =================================================================================================
//     Error Contract
// =================================================================================================

TEST(KmerExtractAsciiAssumeValid, InvalidKThrows)
{
    WordEncoderButterfly<Encoding::kACGT, Layout::kMSB> encoder;
    EXPECT_THROW(
        for_each_kmer_ascii_assume_valid("ACGTACGT", 0, encoder, [](KmerAcgtMsb) {}),
        std::invalid_argument
    );
    EXPECT_THROW(
        for_each_kmer_ascii_assume_valid("ACGTACGT", 33, encoder, [](KmerAcgtMsb) {}),
        std::invalid_argument
    );
}

TEST(KmerExtractAsciiAssumeValid, InvalidChunkSizeThrows)
{
    WordEncoderButterfly<Encoding::kACGT, Layout::kMSB> encoder;
    EXPECT_THROW(
        for_each_kmer_ascii_assume_valid(
            "ACGTACGT", 3, encoder, std::size_t{0}, [](KmerAcgtMsb) {}
        ),
        std::invalid_argument
    );
}
