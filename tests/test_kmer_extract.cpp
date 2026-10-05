#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "fisk/kmer_extract/simd.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// Checks for_each_kmer_rolling() and for_each_kmer_reextract(), the two extractors that take an
// encoder, against `expected` for one (seq, k), in both callback forms.
template <typename Enc>
static void check_encoder_impls(
    std::string const& seq, std::size_t k, Enc const& enc, std::vector<ExpectedKmer> const& expected
) {
    constexpr auto E = Enc::encoding;
    check_kmer_callbacks<E, Layout::kMSB>(expected, k, [&](auto const& callback) {
        for_each_kmer_rolling(seq, k, enc, callback);
    });
    check_kmer_callbacks<E, Layout::kMSB>(expected, k, [&](auto const& callback) {
        for_each_kmer_reextract(seq, k, enc, callback);
    });
}

// Checks all five for_each_kmer variants against the oracle for one (seq, k): exact positions and
// values, in order, skipping every window that overlaps an invalid character. ACGT-table encoding
// throughout, since for_each_kmer_simd()/for_each_kmer_simd_scalar() hardcode ACGT-ascii encoding
// and for_each_kmer() fixes its encoder to the ACGT lookup table, so neither can be checked
// against any other ordering.
static void check_all_impls_acgt(std::string const& seq, std::size_t k)
{
    auto const expected = oracle_kmers_skipping_invalid<Encoding::kACGT, Layout::kMSB>(seq, k);
    check_encoder_impls(seq, k, CharEncoderTable<Encoding::kACGT>{}, expected);
    check_kmer_callbacks<Encoding::kACGT, Layout::kMSB>(expected, k, [&](auto const& callback) {
        for_each_kmer_simd(seq, k, callback);
    });
    check_kmer_callbacks<Encoding::kACGT, Layout::kMSB>(expected, k, [&](auto const& callback) {
        for_each_kmer_simd_scalar(seq, k, callback);
    });
    check_kmer_callbacks<Encoding::kACGT, Layout::kMSB>(expected, k, [&](auto const& callback) {
        for_each_kmer(seq, k, callback);
    });
}

// =================================================================================================
//     Random Sequences
// =================================================================================================

// Valid-only sequences (pure ACGT/acgt), shared across the oracle sweep and the invalid-sentinel
// test below (whose custom encoder repurposes the lowercase half of this alphabet as "invalid").
// Lengths 0..70 plus a batch of longer random lengths.
static std::vector<std::string> const& valid_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(90210);
        for (std::size_t len = 0; len <= 70; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        for (int i = 0; i < 20; ++i) {
            std::size_t const len = static_cast<std::size_t>(rng.get_uint64() % 300);
            out.push_back(random_sequence(rng, len));
        }
        return out;
    }();
    return seqs;
}

// =================================================================================================
//     Oracle Correctness
// =================================================================================================

// Correct k-mers, correct positions, correct order, correct MSB encoding, for every length/k
// combination in the shared sequence set. ACGT-table encoding, checked against all five
// implementations.
TEST(KmerExtract, OracleAcgtTable)
{
    for (auto const& seq : valid_sequences()) {
        for (auto const k : test_ks()) {
            check_all_impls_acgt(seq, k);
        }
    }
}

// Same, but with an ACTG-ordered encoder, to prove for_each_kmer_rolling()/for_each_kmer_reextract()
// do not hardcode ACGT ordering. The other three cannot be checked here, since they hardcode ACGT
// (see check_all_impls_acgt()).
TEST(KmerExtract, OracleActgTable)
{
    for (auto const& seq : valid_sequences()) {
        for (auto const k : test_ks()) {
            check_encoder_impls(
                seq, k, CharEncoderTable<Encoding::kACTG>{},
                oracle_kmers_skipping_invalid<Encoding::kACTG, Layout::kMSB>(seq, k)
            );
        }
    }
}

// =================================================================================================
//     k and Length Boundaries
// =================================================================================================

// k must be in [1, 32]; k == 0 or k > 32 is a documented precondition violation for every variant.
TEST(KmerExtract, InvalidKThrows)
{
    CharEncoderTable<Encoding::kACGT> const table;
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        for_each_kmer_rolling(seq, k, table, [](KmerAcgtMsb) {});
    });
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        for_each_kmer_reextract(seq, k, table, [](KmerAcgtMsb) {});
    });
    check_invalid_k_throws(32, [](std::string const& seq, std::size_t k) {
        for_each_kmer_simd(seq, k, [](KmerAcgtMsb) {});
    });
    check_invalid_k_throws(32, [](std::string const& seq, std::size_t k) {
        for_each_kmer_simd_scalar(seq, k, [](KmerAcgtMsb) {});
    });
    check_invalid_k_throws(32, [](std::string const& seq, std::size_t k) {
        for_each_kmer(seq, k, [](KmerAcgtMsb) {});
    });
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
// Written with 'N' for readability, and then checked with every hand-picked invalid character in
// its place.
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

    for (auto const marker : kHandPickedInvalid) {
        for (auto seq : seqs) {
            std::replace(seq.begin(), seq.end(), 'N', marker);
            for (auto const k : ks) {
                check_all_impls_acgt(seq, k);
            }
        }
    }
}

// A user-defined encoder has to state which Encoding its codes are in, so that the loops can tag the
// k-mers they build from them. A bare callable carries no such statement, and must be rejected at
// compile time rather than silently producing k-mers of an assumed convention.
namespace {

template <typename Enc>
concept RollingAccepts = requires(Enc enc) {
    for_each_kmer_rolling(std::string_view{}, std::size_t{1}, enc, [](auto) {});
};

using UntaggedEncoder = decltype([](char) -> std::uint8_t { return 0; });

} // namespace

static_assert( RollingAccepts<CharEncoderTable<Encoding::kACGT>>);
static_assert( RollingAccepts<CharEncoderAscii<Encoding::kACTG>>);
static_assert(!RollingAccepts<UntaggedEncoder>);
static_assert(!RollingAccepts<std::uint8_t(*)(char)>);

// Custom encoder whose invalid sentinel is not exactly 4 (255 here, and lowercase bases are
// deliberately "invalid" under it), checking that for_each_kmer_rolling()/for_each_kmer_reextract()
// honor the documented ">= 4 is invalid" contract rather than special-casing the value 4. Only
// applies to these two, since neither for_each_kmer_simd()/for_each_kmer_simd_scalar() nor the
// for_each_kmer() convenience wrapper take a custom encoder. Also shows the shape a user-defined
// encoder takes: a functor that states its encoding.
namespace {

struct SentinelNotFourEncoder
{
    static constexpr Encoding encoding = Encoding::kACGT;

    constexpr std::uint8_t operator()(char c) const noexcept
    {
        switch (c) {
            case 'A': return 0;
            case 'C': return 1;
            case 'G': return 2;
            case 'T': return 3;
            default:  return 255;
        }
    }
};

} // namespace

TEST(KmerExtract, InvalidSentinelNotFour)
{
    SentinelNotFourEncoder const enc;
    for (auto const& seq : valid_sequences()) {
        // What the encoder sees as invalid, spelled out for the oracle: every lowercase base.
        std::string as_seen = seq;
        for (auto& c : as_seen) {
            if (std::islower(static_cast<unsigned char>(c))) {
                c = 'N';
            }
        }
        for (auto const k : test_ks()) {
            auto const expected =
                oracle_kmers_skipping_invalid<Encoding::kACGT, Layout::kMSB>(as_seen, k);
            check_encoder_impls(seq, k, enc, expected);
        }
    }
}

// Randomized sequences with invalid characters from the full byte range injected at a controlled
// rate, swept across sparse, moderate, and heavy regimes.
TEST(KmerExtract, InvalidInjectionFuzz)
{
    struct Rate { double p; std::uint64_t seed; };
    std::vector<Rate> const rates = {
        {0.02, 111111},
        {0.10, 222222},
        {0.30, 333333},
    };

    for (auto const& r : rates) {
        for (auto const& seq : invalid_injected_sequences(40, 149, r.p, r.seed)) {
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
// both -- exactly where a block-boundary bug would hide. The marker cycles through the hand-picked
// invalid characters from one sequence to the next.
TEST(KmerExtract, SimdBlockBoundaries)
{
    std::vector<std::size_t> const block_sizes = { 8, 16, 32, 64 };
    Splitmix64 rng(424242);
    std::size_t marker_idx = 0;

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
            std::string const base = random_sequence(rng, len, "ACGT");

            for (auto const& markers : marker_sets) {
                std::string seq = base;
                char const marker = kHandPickedInvalid[marker_idx++ % kHandPickedInvalid.size()];
                bool any_in_range = false;
                for (auto const pos : markers) {
                    if (pos < seq.size()) {
                        seq[pos] = marker;
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
//     Case Insensitivity
// =================================================================================================

// A single deterministic, easy-to-eyeball mixed-case sequence, checked end-to-end through every
// implementation. Per-character case handling is already exhaustively covered in test_char_encoder.cpp;
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
//     kmer_decode()
// =================================================================================================

// Round-trips kmer_decode() against the oracle: decoding an oracle-computed k-mer must give back the
// (uppercased) substring it was extracted from.
TEST(KmerExtract, KmerDecode)
{
    for (auto const& seq : valid_sequences()) {
        for (auto const k : test_ks()) {
            auto const windows = oracle_kmers_all_windows<Encoding::kACGT, Layout::kMSB>(seq, k);
            for (auto const& window : windows) {
                std::string upper = seq.substr(window.pos, k);
                for (auto& c : upper) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                }
                auto const kmer = kmer_cast<Encoding::kACGT, Layout::kMSB>(window.value.value(), k);
                EXPECT_EQ(kmer_decode(kmer, k), upper);
            }
        }
    }
}
