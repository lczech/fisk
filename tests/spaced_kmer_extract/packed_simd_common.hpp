#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/util/random.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "simd_lanes.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "testing.hpp"

// =================================================================================================
//     Spaced K-mer Extraction from Packed Sequences, SIMD: Shared Test Helpers
// =================================================================================================

// The helpers of the tests of packed_bytewise_simd.hpp, which are split into
// test_packed_bytewise_simd_unchecked.cpp and _checked.cpp for spans up to 32, and
// test_packed_bytewise_simd_wide_unchecked.cpp and _wide_checked.cpp for spans above 32, so that
// each compiles with less memory.

// Random masks of the given span, which keep the first and last position.
template <std::size_t N>
inline std::array<std::string, N> random_masks(fisk::Splitmix64& rng, std::size_t span_k)
{
    std::array<std::string, N> masks;
    for (auto& mask : masks) {
        mask.assign(span_k, '0');
        for (auto& c : mask) {
            c = (rng.get_uint64() & 1u) ? '1' : '0';
        }
        mask.front() = '1';
        mask.back() = '1';
    }
    return masks;
}

// The kernels of the given masks, by default with the maximum span of the tests of spans up to 32.
template <typename Kernel, fisk::Layout L, std::size_t MaxSpan = kNarrowMaxSpan, std::size_t N>
inline std::array<fisk::SpacedKmerMask<L, Kernel, MaxSpan>, N> make_kernels(
    std::array<std::string, N> const& masks
) {
    std::array<fisk::SpacedKmerMask<L, Kernel, MaxSpan>, N> kernels;
    for (std::size_t m = 0; m < N; ++m) {
        kernels[m] = fisk::make_spaced_kmer_kernel<Kernel, L, MaxSpan>(masks[m]);
    }
    return kernels;
}

// Runs the extraction with a vector callback, and returns its spaced k-mers as events, taken out
// of the vectors with for_each_valid_lane(), in the order of the scalar callbacks. Checks the
// structure of the vectors on the way: consecutive positions per mask, and the masks in order per
// vector of positions. With `Shorthand`, the callback is the single-mask form without mask_idx.
template <typename Simd, fisk::Encoding E, fisk::Layout L, bool Shorthand = false, typename Extract>
inline std::vector<SpacedEvent> collect_vector_events(std::size_t mask_count, Extract&& extract)
{
    using V = fisk::SpacedKmerVector<E, L, Simd>;
    constexpr std::size_t lanes = Simd::lanes;
    std::vector<SpacedEvent> events;
    std::vector<std::size_t> next_pos(mask_count, 0);
    std::size_t expected_mask = 0;
    // Not inlined into each step of the unrolled loops, as that multiplies the compile time.
    auto const on_vector = [&](
        std::size_t pos, std::size_t mask_idx, V spaced_kmers, std::size_t valid_count
    ) FISK_NOINLINE_LAMBDA {
        EXPECT_EQ(mask_idx, expected_mask);
        expected_mask = (expected_mask + 1) % mask_count;
        EXPECT_TRUE(mask_idx < mask_count);
        if (mask_idx >= mask_count) {
            return;
        }
        EXPECT_EQ(pos, next_pos[mask_idx]);
        next_pos[mask_idx] = pos + valid_count;
        EXPECT_TRUE(valid_count >= 1 && valid_count <= lanes);
        fisk::for_each_valid_lane(
            pos, spaced_kmers, std::min(valid_count, lanes),
            [&](std::size_t p, fisk::SpacedKmer<E, L> const& spaced_kmer) {
                events.push_back({p, mask_idx, fisk::kmer_value(spaced_kmer)});
            }
        );
    };
    if constexpr (Shorthand) {
        extract([&](std::size_t pos, V spaced_kmers, std::size_t valid_count) {
            on_vector(pos, 0, spaced_kmers, valid_count);
        });
    } else {
        extract(on_vector);
    }
    EXPECT_EQ(expected_mask, std::size_t{0});
    std::sort(events.begin(), events.end(), [](auto const& lhs, auto const& rhs) {
        return std::tie(lhs.pos, lhs.mask_idx) < std::tie(rhs.pos, rhs.mask_idx);
    });
    return events;
}

struct NoopSpacedVectorCallback
{
    template <typename V>
    void operator()(std::size_t, std::size_t, V const&, std::size_t) const {}
};

struct NoopSpacedValidLanesCallback
{
    template <typename V, typename Lanes>
    void operator()(std::size_t, std::size_t, V const&, Lanes const&) const {}
};

// Checks that spans outside of [1, MaxSpan] and sets of kernels with different spans are rejected,
// unchecked and checked, through the single-kernel overload and a set of kernels, also for
// sequences shorter than the span; with spans above 32 among the different ones where MaxSpan
// allows them.
template <fisk::Layout L, std::size_t MaxSpan>
inline void check_simd_error_contract()
{
    using Kernel = fisk::BitExtractKernelButterfly<fisk::SimdScalar>;
    using Mask = fisk::SpacedKmerMask<L, Kernel, MaxSpan>;
    fisk::WordEncoderButterfly<fisk::Encoding::kACGT, L> const encoder;
    for (std::string const& seq : {std::string{}, std::string(2 * MaxSpan + 1, 'T')}) {
        auto const packed = fisk::pack_sequence_unchecked(seq, encoder);
        auto const checked = fisk::pack_sequence_checked(seq, encoder);
        auto const expect_throw = [&](auto const& kernel_or_kernels) {
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_packed_unchecked_bytewise_simd(
                    packed, kernel_or_kernels, NoopSpacedVectorCallback{}
                ),
                std::invalid_argument
            );
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_packed_checked_bytewise_simd(
                    checked, kernel_or_kernels, NoopSpacedValidLanesCallback{}
                ),
                std::invalid_argument
            );
        };

        // A kernel with an invalid span, which make_spaced_kmer_kernel() would reject.
        for (std::size_t const span_k : {std::size_t{0}, MaxSpan + 1}) {
            Mask kernel;
            kernel.mask_pieces[0] = Kernel(3);
            kernel.span_k = static_cast<std::uint16_t>(span_k);
            expect_throw(kernel);
            expect_throw(std::span<Mask const, 1>(&kernel, 1));
        }

        std::vector<std::array<std::string, 2>> mixed_sets = {{"101", "1001"}};
        if constexpr (MaxSpan > 32) {
            mixed_sets.push_back({
                std::string("1").append(38, '0').append("1"),
                std::string("1").append(39, '0').append("1")
            });
        }
        for (auto const& mask_strings : mixed_sets) {
            auto const mixed = make_kernels<Kernel, L, MaxSpan>(mask_strings);
            expect_throw(std::span<Mask const, 2>(mixed.data(), 2));
        }
    }
}

// =================================================================================================
//     Spans above 32
// =================================================================================================

// Checks kernels with spans above 32, see wide_test_mask_sets(), unchecked or checked, against the
// oracle: the butterfly kernel of every arch of the build and the PEXT kernel, with the default
// `MaxCompiledPieces`, through the overload for a set of kernels and, for a single kernel, also the
// single-kernel overload; and the butterfly kernel of the scalar arch with a copy of the loop for
// each piece count (`MaxCompiledPieces` 5) and for one piece only (1). The Encoding does not change
// the loop, so that ACGT suffices here.
template <fisk::Layout L, bool Checked>
inline void check_simd_wide_extraction()
{
    constexpr fisk::Encoding E = fisk::Encoding::kACGT;
    fisk::WordEncoderButterfly<E, L> const encoder;
    for (auto const& mask_strings : wide_test_mask_sets()) {
        std::size_t const span_k = mask_strings.front().size();
        auto const& seqs = Checked ? packed_checked_test_sequences() : packed_test_sequences();
        for (auto const& seq : seqs) {
            auto const packed = [&] {
                if constexpr (Checked) {
                    return fisk::pack_sequence_checked(seq, encoder);
                } else {
                    return fisk::pack_sequence_unchecked(seq, encoder);
                }
            }();
            auto const expected = oracle_spaced_events<E, L>(seq, mask_strings);
            std::size_t const positions = seq.size() < span_k ? 0 : seq.size() - span_k + 1;

            // The events of `extract`, which calls back with vectors of `mask_count` kernels.
            auto const collect = [&]<typename Simd>(std::size_t mask_count, auto const& extract) {
                if constexpr (Checked) {
                    return collect_valid_lanes_events<Simd, E, L, false, true>(
                        mask_count, positions, extract
                    );
                } else {
                    return collect_vector_events<Simd, E, L>(mask_count, extract);
                }
            };
            auto const run = [&]<std::size_t MaxCompiledPieces>(
                auto const& kernel_or_kernels, auto const& callback
            ) {
                if constexpr (Checked) {
                    fisk::for_each_spaced_kmer_packed_checked_bytewise_simd<MaxCompiledPieces>(
                        packed, kernel_or_kernels, callback
                    );
                } else {
                    fisk::for_each_spaced_kmer_packed_unchecked_bytewise_simd<MaxCompiledPieces>(
                        packed, kernel_or_kernels, callback
                    );
                }
            };

            // The single kernel, also as itself, or the set of three kernels.
            auto const check = [&]<
                typename Kernel, std::size_t MaxCompiledPieces, std::size_t N
            >() {
                using Simd = typename Kernel::simd_arch;
                std::array<std::string, N> masks;
                std::copy(mask_strings.begin(), mask_strings.end(), masks.begin());
                auto const kernels = make_kernels<Kernel, L, 128>(masks);
                check_spaced_events(collect.template operator()<Simd>(N, [&](auto const& cb) {
                    run.template operator()<MaxCompiledPieces>(std::span{kernels}, cb);
                }), expected);
                if constexpr (N == 1) {
                    check_spaced_events(collect.template operator()<Simd>(1, [&](auto const& cb) {
                        run.template operator()<MaxCompiledPieces>(kernels[0], cb);
                    }), expected);
                }
            };
            auto const check_sizes = [&]<typename Kernel, std::size_t MaxCompiledPieces>() {
                if (mask_strings.size() == 1) {
                    check.template operator()<Kernel, MaxCompiledPieces, 1>();
                } else {
                    check.template operator()<Kernel, MaxCompiledPieces, 3>();
                }
            };
            constexpr std::size_t D = fisk::kDefaultMaxCompiledPieces;
            check_sizes.template operator()<fisk::BitExtractKernelButterfly<fisk::SimdScalar>, D>();
            check_sizes.template operator()<fisk::BitExtractKernelButterfly<fisk::SimdScalar>, 1>();
            check_sizes.template operator()<fisk::BitExtractKernelButterfly<fisk::SimdScalar>, 5>();
            #if defined(FISK_HAS_SSE2)
            check_sizes.template operator()<fisk::BitExtractKernelButterfly<fisk::SimdSSE2>, D>();
            #endif
            #if defined(FISK_HAS_AVX2)
            check_sizes.template operator()<fisk::BitExtractKernelButterfly<fisk::SimdAVX2>, D>();
            #endif
            #if defined(FISK_HAS_AVX512)
            check_sizes.template operator()<fisk::BitExtractKernelButterfly<fisk::SimdAVX512>, D>();
            #endif
            #if defined(FISK_HAS_NEON)
            check_sizes.template operator()<fisk::BitExtractKernelButterfly<fisk::SimdNEON>, D>();
            #endif
            #if defined(FISK_HAS_BMI2)
            check_sizes.template operator()<fisk::BitExtractKernelPext, D>();
            #endif
        }
    }
}
