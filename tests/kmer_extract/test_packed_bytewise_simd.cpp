#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "fisk/core/cast.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/simd.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "kmer_extract/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Unchecked
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
        encoder, packed_test_sequences(), 32,
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

TEST(KmerExtractPackedBytewiseSimd, UncheckedScalarAcgtMsb)
{
    check_simd_arch<SimdScalar, Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedScalarAcgtLsb)
{
    check_simd_arch<SimdScalar, Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedScalarActgMsb)
{
    check_simd_arch<SimdScalar, Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedScalarActgLsb)
{
    check_simd_arch<SimdScalar, Encoding::kACTG, Layout::kLSB>();
}

#if defined(FISK_HAS_SSE2)

TEST(KmerExtractPackedBytewiseSimd, UncheckedSse2AcgtMsb)
{
    check_simd_arch<SimdSSE2, Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedSse2AcgtLsb)
{
    check_simd_arch<SimdSSE2, Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedSse2ActgMsb)
{
    check_simd_arch<SimdSSE2, Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedSse2ActgLsb)
{
    check_simd_arch<SimdSSE2, Encoding::kACTG, Layout::kLSB>();
}

#endif

#if defined(FISK_HAS_AVX2)

TEST(KmerExtractPackedBytewiseSimd, UncheckedAvx2AcgtMsb)
{
    check_simd_arch<SimdAVX2, Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedAvx2AcgtLsb)
{
    check_simd_arch<SimdAVX2, Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedAvx2ActgMsb)
{
    check_simd_arch<SimdAVX2, Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedAvx2ActgLsb)
{
    check_simd_arch<SimdAVX2, Encoding::kACTG, Layout::kLSB>();
}

#endif

#if defined(FISK_HAS_AVX512)

TEST(KmerExtractPackedBytewiseSimd, UncheckedAvx512AcgtMsb)
{
    check_simd_arch<SimdAVX512, Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedAvx512AcgtLsb)
{
    check_simd_arch<SimdAVX512, Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedAvx512ActgMsb)
{
    check_simd_arch<SimdAVX512, Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedAvx512ActgLsb)
{
    check_simd_arch<SimdAVX512, Encoding::kACTG, Layout::kLSB>();
}

#endif

#if defined(FISK_HAS_NEON)

TEST(KmerExtractPackedBytewiseSimd, UncheckedNeonAcgtMsb)
{
    check_simd_arch<SimdNEON, Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedNeonAcgtLsb)
{
    check_simd_arch<SimdNEON, Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedNeonActgMsb)
{
    check_simd_arch<SimdNEON, Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerExtractPackedBytewiseSimd, UncheckedNeonActgLsb)
{
    check_simd_arch<SimdNEON, Encoding::kACTG, Layout::kLSB>();
}

#endif

// =================================================================================================
//     Checked
// =================================================================================================

// Runs `extract`, called as `extract(callback)`, with a `callback(pos, kmers, valid_lanes)`, and
// checks the k-mers of the valid lanes against `expected`. The vectors have to cover the positions
// in order, `Simd::lanes` at a time, and lanes beyond the last k-mer must not be valid.
template <SimdArch Simd, typename Extract>
static void check_valid_lanes_callback(
    std::vector<ExpectedKmer> const& expected, std::size_t k, std::size_t num_kmers,
    Extract&& extract
) {
    constexpr std::size_t lanes = Simd::lanes;

    std::vector<std::size_t> positions;
    std::vector<std::uint64_t> values;
    std::size_t next_pos = 0;
    extract([&](std::size_t pos, auto kmers, SimdVectorValidLanes<Simd> valid_lanes) {
        EXPECT_EQ(pos, next_pos);
        next_pos = pos + lanes;

        alignas(64) std::uint64_t buf[lanes];
        Simd::store_aligned(kmer_value(kmers), buf);
        std::uint64_t const bits = valid_lanes.bits;
        EXPECT_EQ(bits >> lanes, std::uint64_t{0});
        for (std::size_t j = 0; j < lanes; ++j) {
            if ((bits >> j) & 1) {
                EXPECT_TRUE(pos + j < num_kmers);
                positions.push_back(pos + j);
                values.push_back(buf[j]);
            }
        }
    });
    EXPECT_TRUE(next_pos >= num_kmers);
    check_emitted_kmers(expected, k, values, &positions);
}

// The `check` of for_each_kmer_packed_checked_bytewise_simd() for the checks of
// kmer_extract/common.hpp, for every arch of the build: the vectors with their valid lanes, and the
// single valid k-mers, from for_each_valid_lane() in both callback forms, and from
// store_valid_lanes(), without their positions.
static constexpr auto checked = [](
    auto const& packed, std::size_t k, std::vector<ExpectedKmer> const& expected
) {
    using Seq = std::remove_cvref_t<decltype(packed)>;
    std::size_t const num_kmers = packed.length() >= k ? packed.length() - k + 1 : 0;

    auto const check_simd = [&]<typename Simd>() {
        constexpr Encoding E = Seq::encoding;
        constexpr Layout L = Seq::layout;
        check_valid_lanes_callback<Simd>(expected, k, num_kmers, [&](auto const& callback) {
            for_each_kmer_packed_checked_bytewise_simd<Simd>(packed, k, callback);
        });

        check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
            for_each_kmer_packed_checked_bytewise_simd<Simd>(
                packed, k, [&](std::size_t pos, auto kmers, auto valid_lanes) {
                    for_each_valid_lane(pos, kmers, valid_lanes, callback);
                }
            );
        });
        std::vector<std::uint64_t> stored_values;
        for_each_kmer_packed_checked_bytewise_simd<Simd>(
            packed, k, [&](std::size_t, auto kmers, auto valid_lanes) {
                Kmer<E, L> stored[Simd::lanes];
                std::size_t const count = store_valid_lanes(kmers, valid_lanes, stored);
                for (std::size_t i = 0; i < count; ++i) {
                    stored_values.push_back(kmer_value(stored[i]));
                }
            }
        );
        check_emitted_kmers(expected, k, stored_values, nullptr);
    };
    check_simd.template operator()<SimdScalar>();
    #if defined(FISK_HAS_SSE2)
    check_simd.template operator()<SimdSSE2>();
    #endif
    #if defined(FISK_HAS_AVX2)
    check_simd.template operator()<SimdAVX2>();
    #endif
    #if defined(FISK_HAS_AVX512)
    check_simd.template operator()<SimdAVX512>();
    #endif
    #if defined(FISK_HAS_NEON)
    check_simd.template operator()<SimdNEON>();
    #endif
};

// The unchecked extractor, with the scalar arch, for the sequences whose validity bits it ignores.
static constexpr auto unchecked = [](auto const& seq, std::size_t k, auto const& callback) {
    for_each_kmer_packed_unchecked_bytewise_simd<SimdScalar>(
        seq, k, [&](std::size_t pos, auto kmers, std::size_t valid_count) {
            for_each_valid_lane(pos, kmers, valid_count, callback);
        }
    );
};
static constexpr auto checked_throws = [](auto const& seq, std::size_t k) {
    for_each_kmer_packed_checked_bytewise_simd<SimdScalar>(seq, k, [](std::size_t, auto, auto) {});
};

TEST(KmerExtractPackedBytewiseSimd, CheckedRandomAcgtMsb)
{
    check_checked_random_sequences<Encoding::kACGT, Layout::kMSB>(checked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedRandomAcgtLsb)
{
    check_checked_random_sequences<Encoding::kACGT, Layout::kLSB>(checked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedRandomActgMsb)
{
    check_checked_random_sequences<Encoding::kACTG, Layout::kMSB>(checked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedRandomActgLsb)
{
    check_checked_random_sequences<Encoding::kACTG, Layout::kLSB>(checked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedSingleInvalidMsb)
{
    check_checked_single_invalid<Encoding::kACGT, Layout::kMSB>(checked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedSingleInvalidLsb)
{
    check_checked_single_invalid<Encoding::kACGT, Layout::kLSB>(checked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedValidRunsMsb)
{
    check_checked_valid_runs<Encoding::kACGT, Layout::kMSB>(checked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedValidRunsLsb)
{
    check_checked_valid_runs<Encoding::kACGT, Layout::kLSB>(checked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedAllValidEqualsUncheckedMsb)
{
    check_checked_all_valid_equals_unchecked<Encoding::kACGT, Layout::kMSB>(checked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedAllValidEqualsUncheckedLsb)
{
    check_checked_all_valid_equals_unchecked<Encoding::kACGT, Layout::kLSB>(checked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedInvalidInValidityOnlyMsb)
{
    check_checked_invalid_in_validity_only<Encoding::kACGT, Layout::kMSB>(checked, unchecked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedInvalidInValidityOnlyLsb)
{
    check_checked_invalid_in_validity_only<Encoding::kACGT, Layout::kLSB>(checked, unchecked);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedThrowsMsb)
{
    check_checked_throws<Encoding::kACGT, Layout::kMSB>(checked_throws);
}

TEST(KmerExtractPackedBytewiseSimd, CheckedThrowsLsb)
{
    check_checked_throws<Encoding::kACGT, Layout::kLSB>(checked_throws);
}
