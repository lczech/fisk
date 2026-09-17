#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/kmer_extract/packed_simd.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Deliberately duplicated from test_kmer_extract_packed.cpp (same helpers, same shape) rather than
// shared, since these are file-local `static` there. Might refactor later to avoid code duplication.

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

// Ground truth, MSB/left-rolling convention -- matches for_each_kmer_rolling() in kmer_extract.hpp.
static std::uint64_t oracle_msb(std::string const& seq, std::size_t start, std::size_t k)
{
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < k; ++i) {
        v = (v << 2) | static_cast<std::uint64_t>(code_acgt(seq[start + i]));
    }
    return v;
}

// Ground truth, LSB/right-rolling convention -- earliest base in the low bits instead.
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

// Checks `got` (one extractor's emitted k-mers for `seq`/`k`, already flattened and in order)
// against `oracle`.
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

// =================================================================================================
//     Vector -> scalar unpacking
// =================================================================================================

#if defined(FISK_HAS_SSE2)
static void store_kmer_vec(__m128i v, std::uint64_t* buf)
{
    _mm_storeu_si128(reinterpret_cast<__m128i*>(buf), v);
}
#endif // FISK_HAS_SSE2

#if defined(FISK_HAS_AVX2)
static void store_kmer_vec(__m256i v, std::uint64_t* buf)
{
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf), v);
}
#endif // FISK_HAS_AVX2

#if defined(FISK_HAS_AVX512)
static void store_kmer_vec(__m512i v, std::uint64_t* buf)
{
    _mm512_storeu_si512(buf, v);
}
#endif // FISK_HAS_AVX512

#if defined(FISK_HAS_NEON)
static void store_kmer_vec(uint64x2_t v, std::uint64_t* buf)
{
    vst1q_u64(buf, v);
}
#endif // FISK_HAS_NEON

// =================================================================================================
//     Generic SIMD Variant Check
// =================================================================================================

template <std::size_t Lanes, typename Extractor, typename Encoder, typename OracleFn>
static void check_simd_variant(
    Extractor extract, Encoder encoder, OracleFn oracle, std::size_t max_k
) {
    auto check_sequence = [&](std::string const& seq) {
        auto const packed = pack_sequence(seq, encoder);
        for (std::size_t k = 1; k <= max_k; ++k) {
            std::size_t const expected_kmers = seq.size() >= k ? seq.size() - k + 1 : 0;
            std::size_t const expected_callbacks =
                (expected_kmers + Lanes - 1) / Lanes;
            std::size_t callback_index = 0;
            std::vector<std::uint64_t> got;
            extract(packed, k, [&](auto v, std::size_t n) {
                EXPECT_TRUE(n > std::size_t{0});
                EXPECT_TRUE(n <= Lanes);

                std::size_t const expected_count =
                    callback_index + 1 < expected_callbacks || expected_kmers % Lanes == 0
                    ? Lanes
                    : expected_kmers % Lanes;
                EXPECT_EQ(n, expected_count);

                alignas(64) std::uint64_t buf[Lanes];
                store_kmer_vec(v, buf);
                for (std::size_t i = 0; i < n; ++i) {
                    got.push_back(buf[i]);
                }
                if (n < Lanes) {
                    for (std::size_t i = n; i < Lanes; ++i) {
                        EXPECT_EQ(buf[i], std::uint64_t{0});
                    }
                }
                ++callback_index;
            });
            EXPECT_EQ(callback_index, expected_callbacks);
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
            EXPECT_THROW(
                extract(packed, k, [](auto, std::size_t) {}), std::runtime_error
            );
        }
    }
}

// =================================================================================================
//     SSE2
// =================================================================================================

#if defined(FISK_HAS_SSE2)

TEST(KmerExtractPackedSimd, DispatcherSse2)
{
    check_simd_variant<2>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_sse2(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, oracle_msb, 32
    );
    check_simd_variant<2>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_sse2(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, oracle_lsb, 32
    );
}

#endif // FISK_HAS_SSE2

// =================================================================================================
//     NEON
// =================================================================================================

#if defined(FISK_HAS_NEON)

TEST(KmerExtractPackedSimd, NarrowNeonMsb)
{
    check_simd_variant<2>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_narrow_neon_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, oracle_msb, 29
    );
}

TEST(KmerExtractPackedSimd, NarrowNeonLsb)
{
    check_simd_variant<2>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_narrow_neon_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, oracle_lsb, 29
    );
}

TEST(KmerExtractPackedSimd, WideNeonMsb)
{
    check_simd_variant<2>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_wide_neon_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, oracle_msb, 32
    );
}

TEST(KmerExtractPackedSimd, WideNeonLsb)
{
    check_simd_variant<2>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_wide_neon_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, oracle_lsb, 32
    );
}

TEST(KmerExtractPackedSimd, DispatcherNeon)
{
    check_simd_variant<2>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_neon(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, oracle_msb, 32
    );
    check_simd_variant<2>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_neon(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, oracle_lsb, 32
    );
}

#endif // FISK_HAS_NEON

// =================================================================================================
//     AVX2
// =================================================================================================

#if defined(FISK_HAS_AVX2)

TEST(KmerExtractPackedSimd, NarrowAvx2Msb)
{
    check_simd_variant<4>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_narrow_avx2_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, oracle_msb, 29
    );
}

TEST(KmerExtractPackedSimd, NarrowAvx2Lsb)
{
    check_simd_variant<4>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_narrow_avx2_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, oracle_lsb, 29
    );
}

TEST(KmerExtractPackedSimd, WideAvx2Msb)
{
    check_simd_variant<4>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_wide_avx2_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, oracle_msb, 32
    );
}

TEST(KmerExtractPackedSimd, WideAvx2Lsb)
{
    check_simd_variant<4>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_wide_avx2_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, oracle_lsb, 32
    );
}

TEST(KmerExtractPackedSimd, DispatcherAvx2)
{
    check_simd_variant<4>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_avx2(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, oracle_msb, 32
    );
    check_simd_variant<4>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_avx2(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, oracle_lsb, 32
    );
}

#endif // FISK_HAS_AVX2

// =================================================================================================
//     AVX-512
// =================================================================================================

#if defined(FISK_HAS_AVX512)

TEST(KmerExtractPackedSimd, NarrowAvx512Msb)
{
    check_simd_variant<8>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_narrow_avx512_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, oracle_msb, 29
    );
}

TEST(KmerExtractPackedSimd, NarrowAvx512Lsb)
{
    check_simd_variant<8>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_narrow_avx512_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, oracle_lsb, 29
    );
}

TEST(KmerExtractPackedSimd, WideAvx512Msb)
{
    check_simd_variant<8>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_wide_avx512_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, oracle_msb, 32
    );
}

TEST(KmerExtractPackedSimd, WideAvx512Lsb)
{
    check_simd_variant<8>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_wide_avx512_(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, oracle_lsb, 32
    );
}

TEST(KmerExtractPackedSimd, DispatcherAvx512)
{
    check_simd_variant<8>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_avx512(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{}, oracle_msb, 32
    );
    check_simd_variant<8>(
        [](auto const& seq, std::size_t k, auto func) {
            for_each_kmer_packed_simd_avx512(seq, k, func);
        },
        WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{}, oracle_lsb, 32
    );
}

#endif // FISK_HAS_AVX512
