#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/packed_simd.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "fisk/kmer_spaced/simd.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Extraction from a PackedSequence, SIMD
// =================================================================================================

// The SIMD counterpart of for_each_spaced_kmer_packed() (kmer_spaced/packed.hpp), for the bit
// extraction kernels of a SIMD arch (bit_extract/simd.hpp), such as
// BitExtractKernelButterfly<SimdAVX2>, created via make_spaced_kmer_kernel(). The vectors of the
// k-mer extraction from packed sequences (kmer_extract/packed_simd.hpp) hold the windows of
// consecutive spans, and each kernel extracts its spaced k-mers from all of them at once. As in
// for_each_spaced_kmer_packed(), the windows still hold the neighboring bases above the span,
// which the bit extraction does not take.
//
// The callback is called either with vectors, as `callback(pos, mask_idx, vec, valid_count)`: lane
// `j` of `vec` holds the spaced k-mer of mask `mask_idx` at position `pos + j`, for the first
// `valid_count` lanes, and the remaining lanes are zero. Per vector of positions, the masks are
// emitted in their order. As for for_each_kmer_packed_simd(), the vectors hold raw words, with the
// Encoding and Layout of the sequence. Or, if the callback does not take vectors, with one spaced
// k-mer at a time, in the same order as for_each_spaced_kmer_packed(): per position, all masks.
// This stores the vectors and emits their lanes, and is hence slower.

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// Extract and emit the spaced k-mers of all kernels from the `windows` of the spans at `pos` and
// after. `Vector` is deduced instead of taken from the kernel, as GCC warns about vector types
// that are spelled out as template arguments.
template <
    Encoding E, Layout L, typename Kernel, std::size_t N, typename Callback, typename Vector
>
FISK_ALWAYS_INLINE
inline void emit_spaced_kmers_packed_simd_(
    std::array<Kernel, N> const& kernels, std::size_t pos, Vector windows,
    std::size_t valid_count, Callback& callback
) {
    if constexpr (SpacedKmerVectorCallback<Callback&, Vector>) {
        [&]<std::size_t... M>(std::index_sequence<M...>) {
            (invoke_spaced_kmer_vector_callback(
                callback, M, pos, kernels[M].bit_extract(windows), valid_count
            ), ...);
        }(std::make_index_sequence<N>{});
    } else if constexpr (SpacedKmerCallback<Callback&, E, L, N == 1>) {
        alignas(64) std::uint64_t spaced[N][Kernel::lanes];
        [&]<std::size_t... M>(std::index_sequence<M...>) {
            (Kernel::store(kernels[M].bit_extract(windows), spaced[M]), ...);
        }(std::make_index_sequence<N>{});
        for (std::size_t lane = 0; lane < valid_count; ++lane) {
            for (std::size_t m = 0; m < N; ++m) {
                assert_fits_width_(spaced[m][lane], spaced_kmer_weight(kernels[m]));
                invoke_spaced_kmer_callback<E, L, N == 1>(
                    callback, m, pos + lane, spaced[m][lane]
                );
            }
        }
    } else {
        static_assert(
            SpacedKmerVectorCallback<Callback&, Vector> ||
                SpacedKmerCallback<Callback&, E, L, N == 1>,
            "Callback must be callable as callback(pos, mask_idx, vec, valid_count), as "
            "callback(pos, mask_idx, spaced_kmer), or, for a single kernel, as "
            "callback(pos, spaced_kmer)."
        );
    }
}

// -------------------------------------------------------------------------------------------------
//     Extraction
// -------------------------------------------------------------------------------------------------

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it with SIMD bit extraction, and
 * call a callback on them, for an array of kernels.
 *
 * As the sequence holds only valid bases, a spaced k-mer is emitted at every position, for every
 * kernel. It has the Encoding and Layout of the sequence. See above for the forms of the callback.
 *
 * @tparam Kernel   Bit extraction kernel of a SIMD arch, see BitExtractKernelWithArch, tagged as
 *                  spaced k-mer mask with the Layout of the sequence, see make_spaced_kmer_kernel().
 *                  The spans of all kernels must be the same, and in [1, 32].
 * @tparam N        Number of kernels in the array.
 * @tparam Callback Callback function, called with vectors or with single spaced k-mers.
 */
template <Encoding E, Layout L, typename Kernel, std::size_t N, typename Callback>
    requires BitExtractKernelWithArch<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_simd(
    PackedSequence<E, L> const& seq,
    std::array<SpacedKmerMask<L, Kernel>, N> const& kernels,
    Callback&& callback
) {
    static_assert(N > 0, "Need at least one kernel.");
    using Simd = typename Kernel::simd_arch;

    std::size_t const span_k = common_spaced_kmer_span_(kernels);
    for ([[maybe_unused]] auto const& kernel : kernels) {
        assert(is_valid_spaced_kmer_mask(kernel));
    }
    if (seq.length < span_k) {
        return;
    }

    // The windows come without the mask of the k-mers, which the kernels make redundant.
    auto const emit = [&](std::size_t pos, auto windows, std::size_t valid_count) {
        emit_spaced_kmers_packed_simd_<E, L>(kernels, pos, windows, valid_count, callback);
    };
    if (span_k <= 29) {
        for_each_kmer_packed_simd_narrow_width_<Simd, false>(seq, span_k, emit);
    } else {
        for_each_kmer_packed_simd_full_width_<Simd, false>(seq, span_k, emit);
    }
}

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it with SIMD bit extraction, and
 * call a callback on them, for a single kernel.
 *
 * Same as the overload above, for a single kernel, with `mask_idx` always 0. For single spaced
 * k-mers, the callback can also be called as callback(pos, spaced_kmer).
 */
template <Encoding E, Layout L, typename Kernel, typename Callback>
    requires BitExtractKernelWithArch<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_simd(
    PackedSequence<E, L> const& seq,
    SpacedKmerMask<L, Kernel> const& kernel,
    Callback&& callback
) {
    std::array<SpacedKmerMask<L, Kernel>, 1> const kernels{{kernel}};
    for_each_spaced_kmer_packed_simd(seq, kernels, std::forward<Callback>(callback));
}

} // namespace fisk
