#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/core/seq_enc.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "fisk/kmer_extract/simd.hpp"
#include "testing.hpp"

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Ground truth codes, independent of every encoder under test. ACGT ordering.
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

// Same, but ACTG ordering -- used to prove the generic extractors do not hardcode ACGT.
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

// Ground truth for for_each_kmer_rolling()'s contract, built from scratch: MSB/left-rolling k-mers,
// skipping every window that overlaps a symbol `code_of` reports as invalid (< 0), and requiring
// a full re-accumulation of k valid symbols before emitting again.
template <typename CodeFn>
static std::vector<std::uint64_t> oracle_kmers(std::string const& seq, std::size_t k, CodeFn&& code_of)
{
    std::vector<std::uint64_t> out;
    if (seq.size() < k) {
        return out;
    }

    std::uint64_t const mask = (k == 32) ? ~std::uint64_t{0} : ((std::uint64_t{1} << (2 * k)) - 1u);
    std::uint64_t kmer = 0;
    std::size_t valid = 0;
    for (std::size_t i = 0; i < seq.size(); ++i) {
        int const c = code_of(seq[i]);
        kmer = ((kmer << 2) & mask) | static_cast<std::uint64_t>(c < 0 ? 0 : c);
        valid = (c >= 0) ? (valid + 1) : 0;
        if (valid >= k) {
            out.push_back(kmer);
        }
    }
    return out;
}

// Checks `got` (one extractor's emitted k-mers) against the oracle, in both count and content.
template <typename CodeFn>
static void check_kmers(
    std::vector<std::uint64_t> const& got, std::string const& seq, std::size_t k, CodeFn&& code_of
) {
    auto const exp = oracle_kmers(seq, k, code_of);
    EXPECT_EQ(got.size(), exp.size());
    std::size_t const n = std::min(got.size(), exp.size());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(got[i], exp[i]);
    }
}

// Collectors, one per implementation, so call sites read the same regardless of which extractor
// is under test.
template <typename Enc>
static std::vector<std::uint64_t> collect_rolling(std::string const& seq, std::size_t k, Enc&& enc)
{
    std::vector<std::uint64_t> out;
    for_each_kmer_rolling(seq, k, enc, [&](std::uint64_t kmer) { out.push_back(kmer); });
    return out;
}

template <typename Enc>
static std::vector<std::uint64_t> collect_reextract(std::string const& seq, std::size_t k, Enc&& enc)
{
    std::vector<std::uint64_t> out;
    for_each_kmer_reextract(seq, k, enc, [&](std::uint64_t kmer) { out.push_back(kmer); });
    return out;
}

static std::vector<std::uint64_t> collect_simd(std::string const& seq, std::size_t k)
{
    std::vector<std::uint64_t> out;
    for_each_kmer_simd(seq, k, [&](std::uint64_t kmer) { out.push_back(kmer); });
    return out;
}

static std::vector<std::uint64_t> collect_simd_scalar(std::string const& seq, std::size_t k)
{
    std::vector<std::uint64_t> out;
    for_each_kmer_simd_scalar(seq, k, [&](std::uint64_t kmer) { out.push_back(kmer); });
    return out;
}

// Checks all four for_each_kmer variants against the oracle for one (seq, k), using ACGT-table
// encoding throughout. Required whenever for_each_kmer_simd()/for_each_kmer_simd_scalar() are in
// scope, since both hardcode ACGT-ascii encoding and cannot be checked against any other ordering.
static void check_all_impls_acgt(std::string const& seq, std::size_t k)
{
    check_kmers(collect_rolling(seq, k, char_to_nt_table_acgt), seq, k, code_acgt);
    check_kmers(collect_reextract(seq, k, char_to_nt_table_acgt), seq, k, code_acgt);
    check_kmers(collect_simd(seq, k), seq, k, code_acgt);
    check_kmers(collect_simd_scalar(seq, k), seq, k, code_acgt);
}

// =================================================================================================
//     Random Sequences
// =================================================================================================

// Valid-only sequences (pure ACGT/acgt), shared across the oracle sweep, the cross-implementation
// differential test, and the invalid-sentinel test below (whose custom encoder repurposes the
// lowercase half of this alphabet as "invalid"). Lengths 0..70 plus a batch of longer random
// lengths.
static std::vector<std::string> const& valid_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        char const bases[] = "ACGTacgt";
        Splitmix64 rng(90210);

        auto random_seq = [&](std::size_t len) {
            std::string s;
            s.reserve(len);
            for (std::size_t i = 0; i < len; ++i) {
                s += bases[rng.get_uint64() % 8];
            }
            return s;
        };

        for (std::size_t len = 0; len <= 70; ++len) {
            out.push_back(random_seq(len));
        }
        for (int i = 0; i < 20; ++i) {
            std::size_t const len = static_cast<std::size_t>(rng.get_uint64() % 300);
            out.push_back(random_seq(len));
        }
        return out;
    }();
    return seqs;
}

// k values spanning the full supported range [1, 32] shared by every for_each_kmer*() variant.
static std::vector<std::size_t> const& test_ks()
{
    static std::vector<std::size_t> const ks = {
        1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 27, 28, 29, 30, 31, 32
    };
    return ks;
}

// Sequences with invalid characters injected at a controlled, explicit rate.
// Used to stress the window-reset logic at different densities.
static std::vector<std::string> invalid_injected_sequences(double rate, std::uint64_t seed)
{
    std::vector<std::string> out;
    char const bases[] = "ACGTacgt";
    char const invalid_markers[] = "Nn-.";
    Splitmix64 rng(seed);

    auto random_seq = [&](std::size_t len) {
        std::string s;
        s.reserve(len);
        for (std::size_t i = 0; i < len; ++i) {
            if (rng.get_double() < rate) {
                s += invalid_markers[rng.get_uint64() % 4];
            } else {
                s += bases[rng.get_uint64() % 8];
            }
        }
        return s;
    };

    for (int i = 0; i < 40; ++i) {
        std::size_t const len = static_cast<std::size_t>(rng.get_uint64() % 150);
        out.push_back(random_seq(len));
    }
    return out;
}

// =================================================================================================
//     Oracle Correctness
// =================================================================================================

// Correct k-mers, correct order, correct MSB encoding, for every length/k combination in the
// shared sequence set. ACGT-table encoding, checked against all four implementations.
TEST(KmerExtract, OracleAcgtTable)
{
    for (auto const& seq : valid_sequences()) {
        for (auto const k : test_ks()) {
            check_all_impls_acgt(seq, k);
        }
    }
}

// Same, but with an ACTG-ordered encoder, to prove for_each_kmer_rolling()/for_each_kmer_reextract()
// do not hardcode ACGT ordering. for_each_kmer_simd()/for_each_kmer_simd_scalar() cannot be checked
// here, since they hardcode ACGT-ascii encoding; neither can for_each_kmer(), which fixes its
// encoder to the ACGT lookup table (see the Convenience Wrapper section below).
TEST(KmerExtract, OracleActgTable)
{
    for (auto const& seq : valid_sequences()) {
        for (auto const k : test_ks()) {
            check_kmers(collect_rolling(seq, k, char_to_nt_table_actg), seq, k, code_actg);
            check_kmers(collect_reextract(seq, k, char_to_nt_table_actg), seq, k, code_actg);
        }
    }
}

// =================================================================================================
//     Convenience Wrapper (for_each_kmer)
// =================================================================================================

// for_each_kmer() is a thin forward to for_each_kmer_rolling() with the ACGT lookup-table encoder
// fixed in; this checks it against the oracle directly (rather than relying purely on the rolling
// tests above), plus its own k-validity and a couple of invalid-character sanity cases. The
// underlying skip/reset logic itself is exhaustively covered by the for_each_kmer_rolling() tests.
TEST(KmerExtract, ConvenienceWrapperMatchesRolling)
{
    for (auto const& seq : valid_sequences()) {
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer(seq, k, [&](std::uint64_t kmer) { got.push_back(kmer); });
            check_kmers(got, seq, k, code_acgt);
        }
    }

    std::vector<std::string> const invalid_seqs = { "NACGTACGT", "ACGTNNNACGT" };
    std::vector<std::size_t> const invalid_ks = { 1, 4, 8 };
    for (auto const& seq : invalid_seqs) {
        for (auto const k : invalid_ks) {
            std::vector<std::uint64_t> got;
            for_each_kmer(seq, k, [&](std::uint64_t kmer) { got.push_back(kmer); });
            check_kmers(got, seq, k, code_acgt);
        }
    }

    EXPECT_ANY_THROW(for_each_kmer("ACGT", 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer("ACGT", 33, [](std::uint64_t) {}));
}

// =================================================================================================
//     k and Length Boundaries
// =================================================================================================

// k must be in [1, 32]; k == 0 or k > 32 is a documented precondition violation for every variant.
TEST(KmerExtract, InvalidKThrows)
{
    EXPECT_ANY_THROW(for_each_kmer_rolling("ACGT", 0, char_to_nt_table_acgt, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_rolling("ACGT", 33, char_to_nt_table_acgt, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_reextract("ACGT", 0, char_to_nt_table_acgt, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_reextract("ACGT", 33, char_to_nt_table_acgt, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_simd("ACGT", 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_simd("ACGT", 33, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_simd_scalar("ACGT", 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_simd_scalar("ACGT", 33, [](std::uint64_t) {}));
}

// Empty input, and sequences one shorter than / exactly / one longer than k: zero, one, and two
// k-mers respectively, with no out-of-bounds access at any of these edges.
TEST(KmerExtract, LengthBoundaries)
{
    std::string seq;
    while (seq.size() < 40) {
        seq += "ACGT";
    }

    for (auto const k : test_ks()) {
        check_all_impls_acgt("", k);
        check_all_impls_acgt(seq.substr(0, k - 1), k);
        check_all_impls_acgt(seq.substr(0, k), k);
        check_all_impls_acgt(seq.substr(0, k + 1), k);
    }
}

// =================================================================================================
//     Invalid Characters
// =================================================================================================

// Hand-crafted placements: invalid at the very start/middle/end, a run of several in a row,
// alternating valid/invalid, and a sequence that is entirely invalid. Every k-mer overlapping an
// invalid symbol must be suppressed, and the window must fully refill before emitting again.
TEST(KmerExtract, InvalidCharacterPlacement)
{
    std::vector<std::string> const seqs = {
        "NACGTACGTACGT",
        "ACGTACGNTACGT",
        "ACGTACGTACGTN",
        "ACGTNNNACGTACGT",
        "ANCNGNTNACNGNTN",
        "NNNNNNNNNNNNNNN",
        "N",
        "A",
        // Longer placements, so that k=16 and k=32 below exercise invalid-run-skipping and the
        // window refill too, rather than degenerating into a "sequence shorter than k" check.
        "N" + std::string(45, 'A'),
        std::string(45, 'A') + "N",
        std::string(20, 'A') + "N" + std::string(20, 'A'),
        std::string(20, 'A') + "NNN" + std::string(20, 'A'),
    };
    std::vector<std::size_t> const ks = { 1, 3, 4, 8, 16, 32 };

    for (auto const& seq : seqs) {
        for (auto const k : ks) {
            check_all_impls_acgt(seq, k);
        }
    }
}

// Custom encoder whose invalid sentinel is not exactly 4 (255 here, and lowercase bases are
// deliberately "invalid" under it), checking that for_each_kmer_rolling()/for_each_kmer_reextract()
// honor the documented ">= 4 is invalid" contract rather than special-casing the value 4. Only
// applies to these two, since neither for_each_kmer_simd()/for_each_kmer_simd_scalar() nor the
// for_each_kmer() convenience wrapper take a custom encoder.
TEST(KmerExtract, InvalidSentinelNotFour)
{
    auto const enc = [](char c) -> std::uint8_t {
        switch (c) {
            case 'A': return 0;
            case 'C': return 1;
            case 'G': return 2;
            case 'T': return 3;
            default:  return 255;
        }
    };
    auto const code_of = [](char c) -> int {
        switch (c) {
            case 'A': return 0;
            case 'C': return 1;
            case 'G': return 2;
            case 'T': return 3;
            default:  return -1;
        }
    };

    for (auto const& seq : valid_sequences()) {
        for (auto const k : test_ks()) {
            check_kmers(collect_rolling(seq, k, enc), seq, k, code_of);
            check_kmers(collect_reextract(seq, k, enc), seq, k, code_of);
        }
    }
}

// Randomized sequences with invalid characters injected at a controlled rate, swept across
// sparse, moderate, and heavy regimes.
TEST(KmerExtract, InvalidInjectionFuzz)
{
    struct Rate { double p; std::uint64_t seed; };
    std::vector<Rate> const rates = {
        {0.02, 111111},
        {0.10, 222222},
        {0.30, 333333},
    };

    for (auto const& r : rates) {
        for (auto const& seq : invalid_injected_sequences(r.p, r.seed)) {
            for (auto const k : test_ks()) {
                check_all_impls_acgt(seq, k);
            }
        }
    }
}

// =================================================================================================
//     SIMD Block Boundaries
// =================================================================================================

// for_each_kmer_simd() processes 32-character AVX2 blocks (falling back to a scalar tail), and
// for_each_kmer_simd_scalar() processes 8-character blocks; both then finish with a byte-at-a-time
// remainder. 16 and 64 are included pre-emptively, matching the SSE2/AVX512 block widths already
// used elsewhere in the codebase, so this generator needs no changes if k-mer extraction grows
// those variants too.
//
// For each block size, builds sequences one-under/on/one-over one and two block widths, with an
// invalid marker at the last character of a block, the first character of the next, or straddling
// both -- exactly where a block-boundary bug would hide.
TEST(KmerExtract, SimdBlockBoundaries)
{
    std::vector<std::size_t> const block_sizes = { 8, 16, 32, 64 };
    Splitmix64 rng(424242);
    char const bases[] = "ACGT";

    auto random_valid = [&](std::size_t len) {
        std::string s;
        s.reserve(len);
        for (std::size_t i = 0; i < len; ++i) {
            s += bases[rng.get_uint64() % 4];
        }
        return s;
    };

    for (auto const bs : block_sizes) {
        std::vector<std::size_t> const lens = {
            1 * bs - 1, 1 * bs, 1 * bs + 1,
            2 * bs - 1, 2 * bs, 2 * bs + 1,
            3 * bs - 1, 3 * bs, 3 * bs + 1,
            4 * bs - 1, 4 * bs, 4 * bs + 1
        };
        std::vector<std::vector<std::size_t>> const marker_sets = {
            { bs - 1 },             // last char of the first block
            { bs },                 // first char of the second block
            { bs - 1, bs },         // straddling the boundary
            { bs - 2, bs - 1, bs, bs + 1 }, // a run straddling the boundary
        };

        for (auto const len : lens) {
            std::string const base = random_valid(len);

            for (auto const& markers : marker_sets) {
                std::string seq = base;
                bool any_in_range = false;
                for (auto const pos : markers) {
                    if (pos < seq.size()) {
                        seq[pos] = 'N';
                        any_in_range = true;
                    }
                }
                if (!any_in_range) {
                    continue;
                }

                for (auto const k : test_ks()) {
                    check_all_impls_acgt(seq, k);
                }
            }
        }
    }
}

// =================================================================================================
//     Cross-Implementation Differential
// =================================================================================================

// All four implementations must emit the exact same k-mer sequence for the same input. ACGT-table
// encoding throughout, since the SIMD variants hardcode that convention.
TEST(KmerExtract, DifferentialAllImplementationsAgree)
{
    for (auto const& seq : valid_sequences()) {
        for (auto const k : test_ks()) {
            auto const core        = collect_rolling(seq, k, char_to_nt_table_acgt);
            auto const reextract   = collect_reextract(seq, k, char_to_nt_table_acgt);
            auto const simd        = collect_simd(seq, k);
            auto const simd_scalar = collect_simd_scalar(seq, k);

            EXPECT_EQ(reextract, core);
            EXPECT_EQ(simd, core);
            EXPECT_EQ(simd_scalar, core);
        }
    }
}

// =================================================================================================
//     Case Insensitivity
// =================================================================================================

// A single deterministic, easy-to-eyeball mixed-case sequence, checked end-to-end through every
// implementation. Per-character case handling is already exhaustively covered in test_seq_enc.cpp;
// this just confirms it survives all the way through k-mer assembly.
TEST(KmerExtract, MixedCase)
{
    std::string const seq = "AcGtACgtacGTAcGtACgtacGTAcGtACgt";
    std::vector<std::size_t> const ks = { 1, 4, 8, 16, 32 };
    for (auto const k : ks) {
        check_all_impls_acgt(seq, k);
    }
}

// =================================================================================================
//     decode_kmer_2bit()
// =================================================================================================

// Round-trips decode_kmer_2bit() against the oracle: decoding an oracle-computed k-mer must give
// back the (uppercased) substring it was extracted from.
TEST(KmerExtract, DecodeKmer2Bit)
{
    for (auto const& seq : valid_sequences()) {
        for (auto const k : test_ks()) {
            if (seq.size() < k) {
                continue;
            }
            auto const kmers = oracle_kmers(seq, k, code_acgt);
            for (std::size_t i = 0; i < kmers.size(); ++i) {
                std::string upper = seq.substr(i, k);
                for (auto& c : upper) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                }
                EXPECT_EQ(decode_kmer_2bit(kmers[i], k), upper);
            }
        }
    }
}
