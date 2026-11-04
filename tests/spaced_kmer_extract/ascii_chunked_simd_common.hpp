#pragma once

#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/ascii_chunked_simd.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "testing.hpp"

// =================================================================================================
//     Spaced K-mer Extraction from ASCII, Chunked, SIMD: Shared Test Helpers
// =================================================================================================

// The checks of the tests of ascii_chunked_simd.hpp, which are split into
// test_ascii_chunked_simd_unchecked.cpp and test_ascii_chunked_simd_checked.cpp, so that each
// compiles with less memory.

// Vector callbacks that hand on the spaced k-mers of the valid lanes to a callback of single
// spaced k-mers, with for_each_valid_lane(): with mask_idx, and, for a single kernel, without it.
// Named types, for the reason given in spaced_kmer_extract/common.hpp.
template <typename Callback>
struct SpacedLaneCallback
{
    Callback const& callback;

    template <typename V, typename Valid>
    void operator()(std::size_t pos, std::size_t mask_idx, V spaced_kmers, Valid valid) const
    {
        fisk::for_each_valid_lane(
            pos, spaced_kmers, valid, [&](std::size_t p, auto const& spaced_kmer) {
                callback(p, mask_idx, spaced_kmer);
            }
        );
    }
};

template <typename Callback>
struct SpacedLaneShorthandCallback
{
    Callback const& callback;

    template <typename V, typename Valid>
    void operator()(std::size_t pos, V spaced_kmers, Valid valid) const
    {
        fisk::for_each_valid_lane(pos, spaced_kmers, valid, callback);
    }
};

// The unchecked or checked function, with the kernels `Kernel`, as an `extract` for the checks of
// the chunked loops, see spaced_kmer_extract/common.hpp. A callback with mask_idx gets its spaced
// k-mers from the vector form with mask_idx, the others from the vector shorthand without it.
template <bool Checked, typename Kernel>
inline constexpr auto extract = []<std::size_t ChunkSize, fisk::Encoding E, fisk::Layout L>(
    std::string const& seq, std::vector<std::string> const& mask_strings, auto single,
    auto* scratch, auto const& callback
) {
    using Callback = std::remove_cvref_t<decltype(callback)>;
    fisk::WordEncoderButterfly<E, L> const encoder;
    auto const kernels = fisk::make_spaced_kmer_kernels<Kernel>(
        fisk::make_spaced_kmer_masks<L, 128>(mask_strings)
    );
    using Mask = typename decltype(kernels)::value_type;
    auto const run = [&](auto const& kernel_or_kernels, auto const& vector_callback) {
        if constexpr (Checked) {
            if (scratch) {
                fisk::for_each_spaced_kmer_ascii_checked_chunked_simd<ChunkSize>(
                    seq, kernel_or_kernels, encoder, *scratch, vector_callback
                );
            } else {
                fisk::for_each_spaced_kmer_ascii_checked_chunked_simd<ChunkSize>(
                    seq, kernel_or_kernels, encoder, vector_callback
                );
            }
        } else {
            if (scratch) {
                fisk::for_each_spaced_kmer_ascii_unchecked_chunked_simd<ChunkSize>(
                    seq, kernel_or_kernels, encoder, *scratch, vector_callback
                );
            } else {
                fisk::for_each_spaced_kmer_ascii_unchecked_chunked_simd<ChunkSize>(
                    seq, kernel_or_kernels, encoder, vector_callback
                );
            }
        }
    };
    if constexpr (decltype(single)::value) {
        if constexpr (std::invocable<Callback, std::size_t, std::size_t, fisk::SpacedKmer<E, L>>) {
            run(kernels.front(), SpacedLaneCallback<Callback>{callback});
        } else {
            run(kernels.front(), SpacedLaneShorthandCallback<Callback>{callback});
        }
    } else if (kernels.size() == 1) {
        run(std::span<Mask const, 1>(kernels.data(), 1), SpacedLaneCallback<Callback>{callback});
    } else {
        run(std::span<Mask const, 3>(kernels.data(), 3), SpacedLaneCallback<Callback>{callback});
    }
};

// The unchecked or checked function with the butterfly kernel of the arch `Simd`, for spans up to
// 32 and above.
template <fisk::SimdArch Simd, bool Checked>
inline void check_chunked_simd()
{
    using Kernel = fisk::BitExtractKernelButterfly<Simd>;
    for (auto const* mask_sets : {&chunked_mask_sets(), &chunked_wide_mask_sets()}) {
        check_spaced_ascii_chunked_all<Checked, true>(extract<Checked, Kernel>, *mask_sets);
    }
}

// The checks of check_spaced_ascii_chunked_error_contract() (spaced_kmer_extract/common.hpp), with
// the butterfly kernel of SimdScalar.
template <bool Checked, std::size_t MaxSpan>
inline void check_chunked_simd_error_contract()
{
    constexpr fisk::Encoding E = fisk::Encoding::kACGT;
    constexpr fisk::Layout L = fisk::Layout::kMSB;
    using Scratch = std::conditional_t<
        Checked, fisk::PackedSequenceWithValidity<E, L>, fisk::PackedSequence<E, L>
    >;
    fisk::WordEncoderButterfly<E, L> const encoder;
    Scratch scratch;
    auto const noop = [](std::size_t, std::size_t, auto const&, auto const&) {};
    using Kernel = fisk::BitExtractKernelButterfly<fisk::SimdScalar>;
    auto const to_kernels = [](auto const& masks) {
        return fisk::make_spaced_kmer_kernels<Kernel>(masks);
    };
    check_spaced_ascii_chunked_error_contract<MaxSpan>(to_kernels, [&](
        std::string const& seq, auto const& kernel_or_kernels
    ) {
        if constexpr (Checked) {
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_ascii_checked_chunked_simd<7>(
                    seq, kernel_or_kernels, encoder, scratch, noop
                ),
                std::invalid_argument
            );
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_ascii_checked_chunked_simd<7>(
                    seq, kernel_or_kernels, encoder, noop
                ),
                std::invalid_argument
            );
        } else {
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_ascii_unchecked_chunked_simd<7>(
                    seq, kernel_or_kernels, encoder, scratch, noop
                ),
                std::invalid_argument
            );
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_ascii_unchecked_chunked_simd<7>(
                    seq, kernel_or_kernels, encoder, noop
                ),
                std::invalid_argument
            );
        }
    });
}
