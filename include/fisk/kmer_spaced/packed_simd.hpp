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

// The SIMD counterpart of for_each_spaced_kmer_packed_unchecked_bytewise_scalar()
// (kmer_spaced/packed.hpp), for the bit extraction kernels of a SIMD arch (bit_extract/simd.hpp),
// such as BitExtractKernelButterfly<SimdAVX2>, created via make_spaced_kmer_kernel(). The vectors
// of the k-mer extraction from packed sequences (kmer_extract/packed_simd.hpp) hold the windows of
// consecutive spans, and each kernel extracts its spaced k-mers from all of them at once. As in
// for_each_spaced_kmer_packed_unchecked_bytewise_scalar(), the windows still hold the neighboring
// bases above the span, which the bit extraction does not take.
//
// The callback is called with vectors, as `callback(pos, mask_idx, spaced_kmers, valid_count)`,
// or, for a single kernel, also as `callback(pos, spaced_kmers, valid_count)`: lane `j` of the
// SpacedKmerVector `spaced_kmers` holds the spaced k-mer of mask `mask_idx` at position `pos + j`,
// for the first `valid_count` lanes, and the remaining lanes are unspecified. Per vector of
// positions, the masks are emitted in their order. The spaced k-mers have the Encoding and Layout
// of the sequence. For single spaced k-mers, use for_each_valid_lane() in the callback.
//
// The checked counterpart, for_each_spaced_kmer_packed_checked_bytewise_simd(), takes a
// PackedSequenceWithValidity, and calls back with the valid lanes of each vector instead of a
// count, as `callback(pos, mask_idx, spaced_kmers, valid_lanes)`, or, for a single kernel, also as
// `callback(pos, spaced_kmers, valid_lanes)`, as for_each_spaced_kmer_ascii_checked_rolling_simd()
// (kmer_spaced/simd.hpp): lane `j` is valid if all positions that mask `mask_idx` keeps are valid
// nucleotides in the span at `pos + j`. The vectors cover all positions in their order, and those
// after the last spaced k-mer, up to the end of the last step of the loop, have no valid lanes.

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
inline void emit_spaced_kmers_packed_unchecked_bytewise_simd_(
    std::array<Kernel, N> const& kernels, std::size_t pos, Vector windows,
    std::size_t valid_count, Callback& callback
) {
    using Simd = typename Kernel::simd_arch;
    static_assert(
        SpacedKmerVectorCallback<Callback&, E, L, Simd, N == 1>,
        "Callback must be callable as callback(pos, mask_idx, spaced_kmers, valid_count), or, "
        "for a single kernel, as callback(pos, spaced_kmers, valid_count)."
    );
    [&]<std::size_t... M>(std::index_sequence<M...>) FISK_ALWAYS_INLINE_LAMBDA {
        (invoke_spaced_kmer_vector_callback<E, L, Simd, N == 1>(
            callback, M, pos,
            spaced_kmer_vector_cast_unchecked<E, L, Simd>(kernels[M].bit_extract(windows)),
            valid_count
        ), ...);
    }(std::make_index_sequence<N>{});
}

// As emit_spaced_kmers_packed_unchecked_bytewise_simd_(), with the valid lanes of each kernel, from
// the `validity` of the windows, see run_packed_bytewise_blocks_() (kmer_extract/packed.hpp):
// AllValid, or the validity window at `pos`, as PackedValidity::kNucleotides, against which each
// lane tests the positions that the mask keeps, its validity_mask, see SpacedKmerMask.
template <
    Encoding E, Layout L, typename Kernel, std::size_t N, typename Callback, typename Vector,
    typename Validity
>
FISK_ALWAYS_INLINE
inline void emit_spaced_kmers_packed_checked_bytewise_simd_(
    std::array<SpacedKmerMask<L, Kernel>, N> const& kernels, std::size_t pos, Vector windows,
    Validity validity, Callback& callback
) {
    using Simd = typename Kernel::simd_arch;
    constexpr std::size_t lanes = Simd::lanes;
    static_assert(
        SpacedKmerVectorValidLanesCallback<Callback&, E, L, Simd, N == 1>,
        "Callback must be callable as callback(pos, mask_idx, spaced_kmers, valid_lanes), or, "
        "for a single kernel, as callback(pos, spaced_kmers, valid_lanes)."
    );
    auto const lanes_of = [&](std::size_t m) FISK_ALWAYS_INLINE_LAMBDA -> std::uint64_t {
        if constexpr (std::is_same_v<Validity, AllValid>) {
            (void) m;
            return (std::uint64_t{1} << lanes) - 1u;
        } else {
            std::uint64_t const vm = kernels[m].validity_mask;
            std::uint64_t bits = 0;
            for (std::size_t j = 0; j < lanes; ++j) {
                bits |= static_cast<std::uint64_t>(((validity >> j) & vm) == vm) << j;
            }
            return bits;
        }
    };
    [&]<std::size_t... M>(std::index_sequence<M...>) FISK_ALWAYS_INLINE_LAMBDA {
        (invoke_spaced_kmer_vector_valid_lanes_callback<E, L, Simd, N == 1>(
            callback, M, pos,
            spaced_kmer_vector_cast_unchecked<E, L, Simd>(kernels[M].bit_extract(windows)),
            SimdVectorValidLanes<Simd>{lanes_of(M)}
        ), ...);
    }(std::make_index_sequence<N>{});
}

// -------------------------------------------------------------------------------------------------
//     Extraction
// -------------------------------------------------------------------------------------------------

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it with SIMD bit extraction, and
 * call a callback on them, for an array of kernels.
 *
 * As the sequence holds only valid bases, a spaced k-mer is emitted at every position, for every
 * kernel. It has the Encoding and Layout of the sequence. See above for the form of the callback.
 * A PackedSequenceWithValidity is taken as well, ignoring its validity bits.
 *
 * @tparam Kernel   Bit extraction kernel of a SIMD arch, see BitExtractKernel, tagged as spaced
 *                  k-mer mask with the Layout of the sequence, see make_spaced_kmer_kernel().
 *                  The spans of all kernels must be the same, and in [1, 32].
 * @tparam N        Number of kernels in the array.
 * @tparam Callback Callback function, called with vectors of spaced k-mers, see above.
 */
template <
    PackedSequenceType Seq, Layout L, typename Kernel, std::size_t N, typename Callback
>
    requires BitExtractKernel<Kernel> && (L == Seq::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_unchecked_bytewise_simd(
    Seq const& seq,
    std::array<SpacedKmerMask<L, Kernel>, N> const& kernels,
    Callback&& callback
) {
    static_assert(N > 0, "Need at least one kernel.");
    using Simd = typename Kernel::simd_arch;
    constexpr Encoding E = Seq::encoding;

    std::size_t const span_k = common_spaced_kmer_span_(kernels);
    for ([[maybe_unused]] auto const& kernel : kernels) {
        assert(is_valid_spaced_kmer_mask(kernel));
    }
    if (seq.length() < span_k) {
        return;
    }

    // The windows come without the mask of the k-mers, which the kernels make redundant.
    auto const emit = [&](
        std::size_t pos, auto windows, std::size_t valid_count
    ) FISK_ALWAYS_INLINE_LAMBDA {
        emit_spaced_kmers_packed_unchecked_bytewise_simd_<E, L>(
            kernels, pos, windows, valid_count, callback
        );
    };
    if (span_k <= 29) {
        for_each_kmer_packed_bytewise_simd_narrow_width_<Simd, false>(
            without_validity_(seq), span_k, emit
        );
    } else {
        for_each_kmer_packed_bytewise_simd_full_width_<Simd, false>(
            without_validity_(seq), span_k, emit
        );
    }
}

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it with SIMD bit extraction, and
 * call a callback on them, for a single kernel.
 *
 * Same as the overload above, for a single kernel, with `mask_idx` always 0. The callback can also
 * be called without it, as callback(pos, spaced_kmers, valid_count).
 */
template <PackedSequenceType Seq, Layout L, typename Kernel, typename Callback>
    requires BitExtractKernel<Kernel> && (L == Seq::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_unchecked_bytewise_simd(
    Seq const& seq,
    SpacedKmerMask<L, Kernel> const& kernel,
    Callback&& callback
) {
    std::array<SpacedKmerMask<L, Kernel>, 1> const kernels{{kernel}};
    for_each_spaced_kmer_packed_unchecked_bytewise_simd(
        seq, kernels, std::forward<Callback>(callback)
    );
}

/**
 * @brief Iterate a PackedSequenceWithValidity, extract its spaced k-mers with SIMD bit extraction,
 * and call a callback on them with their valid lanes, for an array of kernels.
 *
 * As for_each_spaced_kmer_packed_unchecked_bytewise_simd(), with the valid lanes of each vector
 * instead of its count, see above: a spaced k-mer is valid if the positions that its mask keeps are
 * valid by the validity bits of `seq`, see pack_sequence_checked(). As in
 * for_each_spaced_kmer_packed_checked_bytewise_scalar(), the loop tests a block of 32 positions at
 * once, and the blocks whose nucleotides are all valid run without a test per lane.
 *
 * @tparam Kernel   Bit extraction kernel of a SIMD arch, see BitExtractKernel, tagged as spaced
 *                  k-mer mask with the Layout of the sequence, see make_spaced_kmer_kernel().
 *                  The spans of all kernels must be the same, and in [1, 32].
 * @tparam N        Number of kernels in the array.
 * @tparam Callback Callback function, called with vectors of spaced k-mers, see above.
 */
template <Encoding E, Layout L, typename Kernel, std::size_t N, typename Callback>
    requires BitExtractKernel<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_checked_bytewise_simd(
    PackedSequenceWithValidity<E, L> const& seq,
    std::array<SpacedKmerMask<L, Kernel>, N> const& kernels,
    Callback&& callback
) {
    static_assert(N > 0, "Need at least one kernel.");
    using Simd = typename Kernel::simd_arch;

    std::size_t const span_k = common_spaced_kmer_span_(kernels);
    for ([[maybe_unused]] auto const& kernel : kernels) {
        assert(is_valid_spaced_kmer_mask(kernel));
    }
    if (seq.length() < span_k) {
        return;
    }

    auto const emit = [&](
        std::size_t pos, auto windows, auto validity
    ) FISK_ALWAYS_INLINE_LAMBDA {
        emit_spaced_kmers_packed_checked_bytewise_simd_<E, L>(
            kernels, pos, windows, validity, callback
        );
    };
    constexpr PackedValidity form = PackedValidity::kNucleotides;
    if (span_k <= 29) {
        for_each_kmer_packed_bytewise_simd_narrow_width_<Simd, false, form>(seq, span_k, emit);
    } else {
        for_each_kmer_packed_bytewise_simd_full_width_<Simd, false, form>(seq, span_k, emit);
    }
}

/**
 * @brief Iterate a PackedSequenceWithValidity, extract its spaced k-mers with SIMD bit extraction,
 * and call a callback on them with their valid lanes, for a single kernel.
 *
 * Same as the overload above, for a single kernel, with `mask_idx` always 0. The callback can also
 * be called without it, as callback(pos, spaced_kmers, valid_lanes).
 */
template <Encoding E, Layout L, typename Kernel, typename Callback>
    requires BitExtractKernel<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_checked_bytewise_simd(
    PackedSequenceWithValidity<E, L> const& seq,
    SpacedKmerMask<L, Kernel> const& kernel,
    Callback&& callback
) {
    std::array<SpacedKmerMask<L, Kernel>, 1> const kernels{{kernel}};
    for_each_spaced_kmer_packed_checked_bytewise_simd(
        seq, kernels, std::forward<Callback>(callback)
    );
}

} // namespace fisk
