#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/kmer_extract/packed.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "testing.hpp"

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

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

// Ground truth, Msb/left-rolling convention -- matches for_each_kmer_rolling() in kmer_extract.hpp.
static std::uint64_t oracle_msb(std::string const& seq, std::size_t start, std::size_t k)
{
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < k; ++i) {
        v = (v << 2) | static_cast<std::uint64_t>(code_acgt(seq[start + i]));
    }
    return v;
}

// Ground truth, Lsb/right-rolling convention -- earliest base in the low bits instead.
static std::uint64_t oracle_lsb(std::string const& seq, std::size_t start, std::size_t k)
{
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < k; ++i) {
        v |= static_cast<std::uint64_t>(code_acgt(seq[start + i])) << (2 * i);
    }
    return v;
}

// Sequence lengths 0..512, each with random per-position content.
static std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        char const bases[] = "ACGTacgt";
        Splitmix64 rng(5005);

        auto random_seq = [&](std::size_t len) {
            std::string s;
            s.reserve(len);
            for (std::size_t k = 0; k < len; ++k) {
                s += bases[rng.get_uint64() % 8];
            }
            return s;
        };

        for (std::size_t len = 0; len <= 512; ++len) {
            out.push_back(random_seq(len));
        }
        return out;
    }();
    return seqs;
}

// k values spanning for_each_kmer_packed_narrow_blockwise()'s supported range and up to the documented max.
static std::vector<std::size_t> const& test_ks()
{
    static std::vector<std::size_t> const ks = {
        1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 27, 28, 29, 30, 31, 32
    };
    return ks;
}

// Checks `got` (one extractor's emitted k-mers for `seq`/`k`) against `oracle`.
template <typename OracleFn>
static void check_kmers(
    std::vector<std::uint64_t> const& got, std::string const& seq, std::size_t k, OracleFn&& oracle
) {
    std::vector<std::uint64_t> exp;
    if (seq.size() >= k) {
        for (std::size_t e = k - 1; e < seq.size(); ++e) {
            exp.push_back(oracle(seq, e - k + 1, k));
        }
    }

    EXPECT_EQ(got.size(), exp.size());
    std::size_t const n = std::min(got.size(), exp.size());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(got[i], exp[i]);
    }
}

// Reuse the base-by-base oracle for every new variant, covering all k and every short length.
template <typename Extractor, typename Encoder, typename OracleFn>
static void check_aligned_variant(
    Extractor extract, Encoder encoder, OracleFn oracle, std::size_t max_k
) {
    auto check_sequence = [&](std::string const& seq) {
        auto const packed = pack_sequence(seq, encoder);
        for (std::size_t k = 1; k <= max_k; ++k) {
            std::vector<std::uint64_t> got;
            extract(packed, k, [&](std::uint64_t v) { got.push_back(v); });
            check_kmers(got, seq, k, oracle);
        }
    };
    for (auto const& seq : test_sequences()) {
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

    // Reject invalid k before arithmetic, even for empty input or a value too large for unsigned.
    for (std::size_t length : {std::size_t{0}, std::size_t{129}}) {
        auto const packed = pack_sequence(std::string(length, 'T'), encoder);
        for (std::size_t k : {std::size_t{0}, max_k + 1, std::numeric_limits<std::size_t>::max()}) {
            EXPECT_THROW(extract(packed, k, [](std::uint64_t) {}), std::runtime_error);
        }
    }
}

// =================================================================================================
//     for_each_kmer_packed_wide_rolling()
// =================================================================================================

TEST(KmerExtractPacked, WideRollingMsb)
{
    EncodeAcgt8ButterflyMsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_rolling(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_msb);
        }
    }
}

TEST(KmerExtractPacked, WideRollingLsb)
{
    EncodeAcgt8ButterflyLsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_rolling(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_lsb);
        }
    }
}

// =================================================================================================
//     for_each_kmer_packed_narrow_blockwise()
// =================================================================================================

TEST(KmerExtractPacked, NarrowBlockwiseMsb)
{
    EncodeAcgt8ButterflyMsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            if (k > 29) {
                continue;
            }
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_narrow_blockwise(packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); });
            check_kmers(got, seq, k, oracle_msb);
        }
    }
}

TEST(KmerExtractPacked, NarrowBlockwiseLsb)
{
    EncodeAcgt8ButterflyLsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            if (k > 29) {
                continue;
            }
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_narrow_blockwise(packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); });
            check_kmers(got, seq, k, oracle_lsb);
        }
    }
}

// =================================================================================================
//     for_each_kmer_packed_narrow_fixed_k()
// =================================================================================================

// for_each_kmer_packed_narrow_fixed_k() dispatches through all 29 compile-time-k instantiations, so
// this uses every k in [1, 29], not just test_ks()'s sampled subset, to exercise each one.
TEST(KmerExtractPacked, NarrowFixedKMsb)
{
    EncodeAcgt8ButterflyMsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (std::size_t k = 1; k <= 29; ++k) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_narrow_fixed_k(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_msb);
        }
    }
}

TEST(KmerExtractPacked, NarrowFixedKLsb)
{
    EncodeAcgt8ButterflyLsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (std::size_t k = 1; k <= 29; ++k) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_narrow_fixed_k(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_lsb);
        }
    }
}

// =================================================================================================
//     for_each_kmer_packed_narrow_rolling()
// =================================================================================================

TEST(KmerExtractPacked, NarrowRollingMsb)
{
    EncodeAcgt8ButterflyMsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            if (k > 29) {
                continue;
            }
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_narrow_rolling(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_msb);
        }
    }
}

TEST(KmerExtractPacked, NarrowRollingLsb)
{
    EncodeAcgt8ButterflyLsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            if (k > 29) {
                continue;
            }
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_narrow_rolling(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_lsb);
        }
    }
}

TEST(KmerExtractPacked, WideBlockwiseMsb)
{
    EncodeAcgt8ButterflyMsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_blockwise(packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); });
            check_kmers(got, seq, k, oracle_msb);
        }
    }
}

TEST(KmerExtractPacked, WideBlockwiseLsb)
{
    EncodeAcgt8ButterflyLsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_blockwise(packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); });
            check_kmers(got, seq, k, oracle_lsb);
        }
    }
}

// for_each_kmer_packed_wide_128() only exists where `unsigned __int128` does -- see the
// __SIZEOF_INT128__ guard around its definition in kmer_extract/packed.hpp.
#ifdef __SIZEOF_INT128__

TEST(KmerExtractPacked, Wide128Msb)
{
    EncodeAcgt8ButterflyMsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_128(packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); });
            check_kmers(got, seq, k, oracle_msb);
        }
    }
}

TEST(KmerExtractPacked, Wide128Lsb)
{
    EncodeAcgt8ButterflyLsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_128(packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); });
            check_kmers(got, seq, k, oracle_lsb);
        }
    }
}

#endif // __SIZEOF_INT128__

TEST(KmerExtractPacked, WideHybridMsb)
{
    EncodeAcgt8ButterflyMsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_hybrid(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_msb);
        }
    }
}

TEST(KmerExtractPacked, WideHybridLsb)
{
    EncodeAcgt8ButterflyLsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_hybrid(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_lsb);
        }
    }
}

TEST(KmerExtractPacked, WideHoistedMsb)
{
    EncodeAcgt8ButterflyMsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_hoisted(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_msb);
        }
    }
}

TEST(KmerExtractPacked, WideHoistedLsb)
{
    EncodeAcgt8ButterflyLsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_hoisted(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_lsb);
        }
    }
}

TEST(KmerExtractPacked, WideHybridHoistedMsb)
{
    EncodeAcgt8ButterflyMsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_hybrid_hoisted(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_msb);
        }
    }
}

TEST(KmerExtractPacked, WideHybridHoistedLsb)
{
    EncodeAcgt8ButterflyLsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (auto const k : test_ks()) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_hybrid_hoisted(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_lsb);
        }
    }
}

// for_each_kmer_packed_wide_fixed_k() dispatches through all 32 compile-time-k instantiations, so
// this uses every k in [1, 32], not just test_ks()'s sampled subset, to exercise each one.
TEST(KmerExtractPacked, WideFixedKMsb)
{
    EncodeAcgt8ButterflyMsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (std::size_t k = 1; k <= 32; ++k) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_fixed_k(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_msb);
        }
    }
}

TEST(KmerExtractPacked, WideFixedKLsb)
{
    EncodeAcgt8ButterflyLsb ex;
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, ex);
        for (std::size_t k = 1; k <= 32; ++k) {
            std::vector<std::uint64_t> got;
            for_each_kmer_packed_wide_fixed_k(
                packed, k, [&](std::uint64_t kmer) { got.push_back(kmer); }
            );
            check_kmers(got, seq, k, oracle_lsb);
        }
    }
}

// =================================================================================================
//     Error Contract
// =================================================================================================

TEST(KmerExtractPacked, InvalidKThrows)
{
    TwoBitSequence<BitOrder::Msb> empty;
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_rolling(empty, 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_rolling(empty, 33, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_narrow_blockwise(empty, 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_narrow_blockwise(empty, 30, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_narrow_fixed_k(empty, 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_narrow_fixed_k(empty, 30, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_narrow_rolling(empty, 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_narrow_rolling(empty, 30, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_blockwise(empty, 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_blockwise(empty, 33, [](std::uint64_t) {}));
#ifdef __SIZEOF_INT128__
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_128(empty, 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_128(empty, 33, [](std::uint64_t) {}));
#endif // __SIZEOF_INT128__
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_hybrid(empty, 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_hybrid(empty, 33, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_hoisted(empty, 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_hoisted(empty, 33, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_hybrid_hoisted(empty, 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_hybrid_hoisted(empty, 33, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_fixed_k(empty, 0, [](std::uint64_t) {}));
    EXPECT_ANY_THROW(for_each_kmer_packed_wide_fixed_k(empty, 33, [](std::uint64_t) {}));
}

// Experimental variants: compare the complete ordered output, not merely the benchmark sum.

TEST(KmerExtractPacked, NarrowAlignedMsb)
{
    check_aligned_variant(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_narrow_aligned(seq, k, func);
        },
        EncodeAcgt8ButterflyMsb{}, oracle_msb, 29
    );
}

TEST(KmerExtractPacked, WideAlignedMsb)
{
    check_aligned_variant(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_wide_aligned(seq, k, func);
        },
        EncodeAcgt8ButterflyMsb{}, oracle_msb, 32
    );
}

TEST(KmerExtractPacked, WideSplitKMsb)
{
    check_aligned_variant(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_wide_split_k(seq, k, func);
        },
        EncodeAcgt8ButterflyMsb{}, oracle_msb, 32
    );
}

TEST(KmerExtractPacked, NarrowAlignedLsb)
{
    check_aligned_variant(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_narrow_aligned(seq, k, func);
        },
        EncodeAcgt8ButterflyLsb{}, oracle_lsb, 29
    );
}

TEST(KmerExtractPacked, WideAlignedLsb)
{
    check_aligned_variant(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_wide_aligned(seq, k, func);
        },
        EncodeAcgt8ButterflyLsb{}, oracle_lsb, 32
    );
}

TEST(KmerExtractPacked, WideSplitKLsb)
{
    check_aligned_variant(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_wide_split_k(seq, k, func);
        },
        EncodeAcgt8ButterflyLsb{}, oracle_lsb, 32
    );
}
