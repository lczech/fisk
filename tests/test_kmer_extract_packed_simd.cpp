#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/kmer_extract/packed_simd.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// Same inputs as test_kmer_extract_packed.cpp: sequence lengths 0..512, each with random
// per-position content, plus the edge cases of sweep_packed_extractors().
static std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(5005);
        for (std::size_t len = 0; len <= 512; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        return out;
    }();
    return seqs;
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

#if defined(FISK_HAS_SSE2)   || \
    defined(FISK_HAS_AVX2)   || \
    defined(FISK_HAS_AVX512) || \
    defined(FISK_HAS_NEON)

// Runs `extract`, called as `extract(callback)`, once with a `callback(pos, vec, n)` and once with
// a `callback(vec, n)`, and checks both against `expected`, lane by lane: `pos` is the start of
// the k-mer in lane 0, and lane `j` holds the one starting at `pos + j`. Also checks the vector
// structure: every vector is full except possibly the last, which zero-pads its unused lanes.
template <std::size_t Lanes, typename Extract>
static void check_simd_callbacks(
    std::vector<ExpectedKmer> const& expected, std::size_t k, Extract&& extract
) {
    std::size_t const expected_kmers = expected.size();
    std::size_t const expected_callbacks = (expected_kmers + Lanes - 1) / Lanes;

    std::vector<std::size_t> positions;
    std::vector<std::uint64_t> values;
    std::size_t callbacks = 0;
    auto unpack = [&](auto vec, std::size_t n) {
        std::size_t const expected_count =
            callbacks + 1 < expected_callbacks || expected_kmers % Lanes == 0
            ? Lanes
            : expected_kmers % Lanes;
        EXPECT_EQ(n, expected_count);

        alignas(64) std::uint64_t buf[Lanes];
        store_kmer_vec(vec, buf);
        for (std::size_t i = 0; i < n && i < Lanes; ++i) {
            values.push_back(buf[i]);
        }
        for (std::size_t i = n; i < Lanes; ++i) {
            EXPECT_EQ(buf[i], std::uint64_t{0});
        }
        ++callbacks;
    };

    extract([&](std::size_t pos, auto vec, std::size_t n) {
        for (std::size_t i = 0; i < n && i < Lanes; ++i) {
            positions.push_back(pos + i);
        }
        unpack(vec, n);
    });
    EXPECT_EQ(callbacks, expected_callbacks);
    check_emitted_kmers(expected, k, values, &positions);

    values.clear();
    callbacks = 0;
    extract([&](auto vec, std::size_t n) { unpack(vec, n); });
    EXPECT_EQ(callbacks, expected_callbacks);
    check_emitted_kmers(expected, k, values, nullptr);
}

// Checks the narrow (k in [1, 29]) and wide (k in [1, 32]) internal helpers and the public
// dispatcher of one ISA, each called as `extract(packed, k, callback)`, under the conventions of E
// and L, against the same oracle output, plus their invalid-k contracts.
template <std::size_t Lanes, Encoding E, Layout L, typename Narrow, typename Wide, typename Dispatch>
static void check_simd_isa(Narrow const& narrow, Wide const& wide, Dispatch const& dispatch)
{
    WordEncoderButterfly<E, L> const encoder;
    sweep_packed_extractors(
        encoder, test_sequences(), 32,
        [&](std::string const&, auto const& packed, std::size_t k, auto const& expected) {
            if (k <= 29) {
                check_simd_callbacks<Lanes>(expected, k, [&](auto const& callback) {
                    narrow(packed, k, callback);
                });
            }
            check_simd_callbacks<Lanes>(expected, k, [&](auto const& callback) {
                wide(packed, k, callback);
            });
            check_simd_callbacks<Lanes>(expected, k, [&](auto const& callback) {
                dispatch(packed, k, callback);
            });
        }
    );

    auto const throws_for = [&](auto const& extract, std::size_t max_k) {
        check_invalid_k_throws(max_k, [&](std::string const& seq, std::size_t k) {
            extract(pack_sequence(seq, encoder), k, [](auto, std::size_t) {});
        });
    };
    throws_for(narrow, 29);
    throws_for(wide, 32);
    throws_for(dispatch, 32);
}

#endif

// =================================================================================================
//     SSE2
// =================================================================================================

#if defined(FISK_HAS_SSE2)

static constexpr auto sse2_narrow = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_narrow_sse2_(seq, k, func);
};
static constexpr auto sse2_wide = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_wide_sse2_(seq, k, func);
};
static constexpr auto sse2_dispatch = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_sse2(seq, k, func);
};

TEST(KmerExtractPackedSimd, Sse2AcgtMsb)
{
    check_simd_isa<2, Encoding::kACGT, Layout::kMSB>(sse2_narrow, sse2_wide, sse2_dispatch);
}

TEST(KmerExtractPackedSimd, Sse2AcgtLsb)
{
    check_simd_isa<2, Encoding::kACGT, Layout::kLSB>(sse2_narrow, sse2_wide, sse2_dispatch);
}

TEST(KmerExtractPackedSimd, Sse2ActgMsb)
{
    check_simd_isa<2, Encoding::kACTG, Layout::kMSB>(sse2_narrow, sse2_wide, sse2_dispatch);
}

TEST(KmerExtractPackedSimd, Sse2ActgLsb)
{
    check_simd_isa<2, Encoding::kACTG, Layout::kLSB>(sse2_narrow, sse2_wide, sse2_dispatch);
}

#endif // FISK_HAS_SSE2

// =================================================================================================
//     NEON
// =================================================================================================

#if defined(FISK_HAS_NEON)

static constexpr auto neon_narrow = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_narrow_neon_(seq, k, func);
};
static constexpr auto neon_wide = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_wide_neon_(seq, k, func);
};
static constexpr auto neon_dispatch = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_neon(seq, k, func);
};

TEST(KmerExtractPackedSimd, NeonAcgtMsb)
{
    check_simd_isa<2, Encoding::kACGT, Layout::kMSB>(neon_narrow, neon_wide, neon_dispatch);
}

TEST(KmerExtractPackedSimd, NeonAcgtLsb)
{
    check_simd_isa<2, Encoding::kACGT, Layout::kLSB>(neon_narrow, neon_wide, neon_dispatch);
}

TEST(KmerExtractPackedSimd, NeonActgMsb)
{
    check_simd_isa<2, Encoding::kACTG, Layout::kMSB>(neon_narrow, neon_wide, neon_dispatch);
}

TEST(KmerExtractPackedSimd, NeonActgLsb)
{
    check_simd_isa<2, Encoding::kACTG, Layout::kLSB>(neon_narrow, neon_wide, neon_dispatch);
}

#endif // FISK_HAS_NEON

// =================================================================================================
//     AVX2
// =================================================================================================

#if defined(FISK_HAS_AVX2)

static constexpr auto avx2_narrow = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_narrow_avx2_(seq, k, func);
};
static constexpr auto avx2_wide = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_wide_avx2_(seq, k, func);
};
static constexpr auto avx2_dispatch = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_avx2(seq, k, func);
};

TEST(KmerExtractPackedSimd, Avx2AcgtMsb)
{
    check_simd_isa<4, Encoding::kACGT, Layout::kMSB>(avx2_narrow, avx2_wide, avx2_dispatch);
}

TEST(KmerExtractPackedSimd, Avx2AcgtLsb)
{
    check_simd_isa<4, Encoding::kACGT, Layout::kLSB>(avx2_narrow, avx2_wide, avx2_dispatch);
}

TEST(KmerExtractPackedSimd, Avx2ActgMsb)
{
    check_simd_isa<4, Encoding::kACTG, Layout::kMSB>(avx2_narrow, avx2_wide, avx2_dispatch);
}

TEST(KmerExtractPackedSimd, Avx2ActgLsb)
{
    check_simd_isa<4, Encoding::kACTG, Layout::kLSB>(avx2_narrow, avx2_wide, avx2_dispatch);
}

#endif // FISK_HAS_AVX2

// =================================================================================================
//     AVX-512
// =================================================================================================

#if defined(FISK_HAS_AVX512)

static constexpr auto avx512_narrow = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_narrow_avx512_(seq, k, func);
};
static constexpr auto avx512_wide = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_wide_avx512_(seq, k, func);
};
static constexpr auto avx512_dispatch = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_simd_avx512(seq, k, func);
};

TEST(KmerExtractPackedSimd, Avx512AcgtMsb)
{
    check_simd_isa<8, Encoding::kACGT, Layout::kMSB>(avx512_narrow, avx512_wide, avx512_dispatch);
}

TEST(KmerExtractPackedSimd, Avx512AcgtLsb)
{
    check_simd_isa<8, Encoding::kACGT, Layout::kLSB>(avx512_narrow, avx512_wide, avx512_dispatch);
}

TEST(KmerExtractPackedSimd, Avx512ActgMsb)
{
    check_simd_isa<8, Encoding::kACTG, Layout::kMSB>(avx512_narrow, avx512_wide, avx512_dispatch);
}

TEST(KmerExtractPackedSimd, Avx512ActgLsb)
{
    check_simd_isa<8, Encoding::kACTG, Layout::kLSB>(avx512_narrow, avx512_wide, avx512_dispatch);
}

#endif // FISK_HAS_AVX512
