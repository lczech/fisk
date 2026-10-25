#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
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
//     Generic SIMD Arch Check
// =================================================================================================

// Runs `extract`, called as `extract(callback)`, with a `callback(pos, vec, n)`, and checks it
// against `expected`, lane by lane: `pos` is the start of the k-mer in lane 0, and lane `j` holds
// the one starting at `pos + j`. Also checks the vector structure: every vector is full except
// possibly the last, whose lanes past its count are unspecified.
template <SimdArch Simd, typename Extract>
static void check_simd_callbacks(
    std::vector<ExpectedKmer> const& expected, std::size_t k, Extract&& extract
) {
    using simd_vector = typename Simd::simd_vector;
    constexpr std::size_t lanes = Simd::lanes;
    std::size_t const expected_kmers = expected.size();
    std::size_t const expected_callbacks = (expected_kmers + lanes - 1) / lanes;

    std::vector<std::size_t> positions;
    std::vector<std::uint64_t> values;
    std::size_t callbacks = 0;
    auto unpack = [&](simd_vector vec, std::size_t n) {
        std::size_t const expected_count =
            callbacks + 1 < expected_callbacks || expected_kmers % lanes == 0
            ? lanes
            : expected_kmers % lanes;
        EXPECT_EQ(n, expected_count);

        alignas(64) std::uint64_t buf[lanes];
        Simd::store_aligned(vec, buf);
        for (std::size_t i = 0; i < n && i < lanes; ++i) {
            values.push_back(buf[i]);
        }
        ++callbacks;
    };

    extract([&](std::size_t pos, simd_vector vec, std::size_t n) {
        for (std::size_t i = 0; i < n && i < lanes; ++i) {
            positions.push_back(pos + i);
        }
        unpack(vec, n);
    });
    EXPECT_EQ(callbacks, expected_callbacks);
    check_emitted_kmers(expected, k, values, &positions);
}

// Checks the narrow-width (k in [1, 29]) and full-width (k in [1, 32]) internal helpers, also
// without masking, and the public dispatcher for one SIMD arch, under the conventions of E and L,
// against the same oracle output, plus their invalid-k contracts.
template <SimdArch Simd, Encoding E, Layout L>
static void check_simd_arch()
{
    using simd_vector = typename Simd::simd_vector;
    auto const narrow_width = [](auto const& seq, std::size_t k, auto const& func) {
        for_each_kmer_packed_bytewise_simd_narrow_width_<Simd>(seq, k, func);
    };
    auto const full_width = [](auto const& seq, std::size_t k, auto const& func) {
        for_each_kmer_packed_bytewise_simd_full_width_<Simd>(seq, k, func);
    };
    // The dispatcher hands on the vectors as KmerVector<E, L, Simd>, checked here as raw vectors.
    auto const dispatch = [](auto const& seq, std::size_t k, auto const& func) {
        for_each_kmer_packed_unchecked_bytewise_simd<Simd>(
            seq, k, [&](std::size_t pos, KmerVector<E, L, Simd> kmers, std::size_t n) {
                func(pos, kmer_value(kmers), n);
            }
        );
    };

    // Without masking, the bits above the k-mers hold neighboring bases. Masking them in the
    // callback instead has to give the k-mers.
    auto const masked_after = [](auto const& extract) {
        return [extract](auto const& seq, std::size_t k, auto const& func) {
            std::uint64_t const mask = (k == 32) ? ~std::uint64_t{0}
                                                 : (std::uint64_t{1} << (2 * k)) - 1u;
            simd_vector const mask_v = Simd::broadcast(mask);
            extract(seq, k, [&](std::size_t pos, simd_vector vec, std::size_t n) {
                func(pos, Simd::bitwise_and(vec, mask_v), n);
            });
        };
    };
    auto const narrow_width_unmasked = masked_after(
        [](auto const& seq, std::size_t k, auto const& func) {
            for_each_kmer_packed_bytewise_simd_narrow_width_<Simd, false>(seq, k, func);
        }
    );
    auto const full_width_unmasked = masked_after(
        [](auto const& seq, std::size_t k, auto const& func) {
            for_each_kmer_packed_bytewise_simd_full_width_<Simd, false>(seq, k, func);
        }
    );

    WordEncoderButterfly<E, L> const encoder;
    sweep_packed_extractors(
        encoder, test_sequences(), 32,
        [&](std::string const&, auto const& packed, std::size_t k, auto const& expected) {
            if (k <= 29) {
                check_simd_callbacks<Simd>(expected, k, [&](auto const& callback) {
                    narrow_width(packed, k, callback);
                });
                check_simd_callbacks<Simd>(expected, k, [&](auto const& callback) {
                    narrow_width_unmasked(packed, k, callback);
                });
            }
            check_simd_callbacks<Simd>(expected, k, [&](auto const& callback) {
                full_width(packed, k, callback);
            });
            check_simd_callbacks<Simd>(expected, k, [&](auto const& callback) {
                full_width_unmasked(packed, k, callback);
            });
            check_simd_callbacks<Simd>(expected, k, [&](auto const& callback) {
                dispatch(packed, k, callback);
            });

            // The single k-mers of the dispatcher, from for_each_valid_lane() in both callback
            // forms, and from store_valid_lanes().
            check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
                for_each_kmer_packed_unchecked_bytewise_simd<Simd>(
                    packed, k, [&](std::size_t pos, auto kmers, std::size_t n) {
                        for_each_valid_lane(pos, kmers, n, callback);
                    }
                );
            });
            check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
                for_each_kmer_packed_unchecked_bytewise_simd<Simd>(
                    packed, k, [&](std::size_t pos, auto kmers, std::size_t n) {
                        Kmer<E, L> stored[Simd::lanes];
                        std::size_t const count = store_valid_lanes(kmers, n, stored);
                        EXPECT_EQ(count, n);
                        for (std::size_t i = 0; i < count; ++i) {
                            invoke_kmer_callback<E, L>(callback, pos + i, kmer_value(stored[i]));
                        }
                    }
                );
            });
        }
    );

    auto const throws_for = [&](auto const& extract, std::size_t max_k) {
        check_invalid_k_throws(max_k, [&](std::string const& seq, std::size_t k) {
            extract(
                pack_sequence_unchecked(seq, encoder), k, [](std::size_t, auto, std::size_t) {}
            );
        });
    };
    throws_for(narrow_width, 29);
    throws_for(full_width, 32);
    throws_for(dispatch, 32);
}

// =================================================================================================
//     Tests
// =================================================================================================

TEST(KmerExtractPackedSimd, ScalarAcgtMsb)
{
    check_simd_arch<SimdScalar, Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedSimd, ScalarAcgtLsb)
{
    check_simd_arch<SimdScalar, Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedSimd, ScalarActgMsb)
{
    check_simd_arch<SimdScalar, Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerExtractPackedSimd, ScalarActgLsb)
{
    check_simd_arch<SimdScalar, Encoding::kACTG, Layout::kLSB>();
}

#if defined(FISK_HAS_SSE2)

TEST(KmerExtractPackedSimd, Sse2AcgtMsb)
{
    check_simd_arch<SimdSSE2, Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedSimd, Sse2AcgtLsb)
{
    check_simd_arch<SimdSSE2, Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedSimd, Sse2ActgMsb)
{
    check_simd_arch<SimdSSE2, Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerExtractPackedSimd, Sse2ActgLsb)
{
    check_simd_arch<SimdSSE2, Encoding::kACTG, Layout::kLSB>();
}

#endif // FISK_HAS_SSE2

#if defined(FISK_HAS_AVX2)

TEST(KmerExtractPackedSimd, Avx2AcgtMsb)
{
    check_simd_arch<SimdAVX2, Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedSimd, Avx2AcgtLsb)
{
    check_simd_arch<SimdAVX2, Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedSimd, Avx2ActgMsb)
{
    check_simd_arch<SimdAVX2, Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerExtractPackedSimd, Avx2ActgLsb)
{
    check_simd_arch<SimdAVX2, Encoding::kACTG, Layout::kLSB>();
}

#endif // FISK_HAS_AVX2

#if defined(FISK_HAS_AVX512)

TEST(KmerExtractPackedSimd, Avx512AcgtMsb)
{
    check_simd_arch<SimdAVX512, Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedSimd, Avx512AcgtLsb)
{
    check_simd_arch<SimdAVX512, Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedSimd, Avx512ActgMsb)
{
    check_simd_arch<SimdAVX512, Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerExtractPackedSimd, Avx512ActgLsb)
{
    check_simd_arch<SimdAVX512, Encoding::kACTG, Layout::kLSB>();
}

#endif // FISK_HAS_AVX512

#if defined(FISK_HAS_NEON)

TEST(KmerExtractPackedSimd, NeonAcgtMsb)
{
    check_simd_arch<SimdNEON, Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedSimd, NeonAcgtLsb)
{
    check_simd_arch<SimdNEON, Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedSimd, NeonActgMsb)
{
    check_simd_arch<SimdNEON, Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerExtractPackedSimd, NeonActgLsb)
{
    check_simd_arch<SimdNEON, Encoding::kACTG, Layout::kLSB>();
}

#endif // FISK_HAS_NEON
