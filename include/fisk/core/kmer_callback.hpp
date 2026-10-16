#pragma once

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/simd.hpp"

namespace fisk {

// =================================================================================================
//     Valid Lanes
// =================================================================================================

/**
 * @brief The lanes of a SIMD vector of k-mers that hold valid k-mers, for the extraction functions
 * that check the validity of the nucleotides, such as for_each_kmer_packed_checked_simd().
 *
 * `value` is the lane mask in the form native to the arch, see SimdArch: a vector with all bits
 * set in the valid lanes and zero in the others, or the mask register of AVX512, which the masked
 * instructions of AVX512 take directly. bits() has bit `j` set for a valid lane `j`.
 */
template <SimdArch Simd>
struct SimdValidLanes
{
    typename Simd::lane_mask value;

    std::uint64_t bits() const noexcept
    {
        return Simd::lane_mask_to_bits(value);
    }
};

// =================================================================================================
//     K-mer Extraction Callback Concepts
// =================================================================================================

// The forms of callbacks that the k-mer extraction functions accept, as dispatched by the
// invoke_*() functions below. A vector form takes vectors of one SIMD arch, which the concept
// receives as a deduced type: GCC warns about vector types that are spelled out as template
// arguments. The vector forms always take the position: the forms of single k-mers and of vectors
// are then told apart by their number of arguments alone, also for generic lambdas.

// -------------------------------------------------------------------------------------------------
//     Contiguous K-mers
// -------------------------------------------------------------------------------------------------

/**
 * @brief A callback for single k-mers, as `callback(pos, kmer)` or `callback(kmer)`.
 */
template <typename Callback, Encoding E, Layout L>
concept KmerCallback =
    std::invocable<Callback, std::size_t, Kmer<E, L> const&> ||
    std::invocable<Callback, Kmer<E, L> const&>;

/**
 * @brief A callback for vectors of k-mers, as `callback(pos, vec, valid_count)`.
 */
template <typename Callback, typename Vector>
concept KmerVectorCallback = std::invocable<Callback, std::size_t, Vector, std::size_t>;

/**
 * @brief A callback for vectors of k-mers with their valid lanes, as
 * `callback(pos, vec, valid_lanes)`, see SimdValidLanes.
 */
template <typename Callback, typename Vector, typename Simd>
concept KmerValidLanesCallback =
    std::invocable<Callback, std::size_t, Vector, SimdValidLanes<Simd>>;

// -------------------------------------------------------------------------------------------------
//     Spaced K-mers
// -------------------------------------------------------------------------------------------------

/**
 * @brief A callback for single spaced k-mers, as `callback(pos, mask_idx, spaced_kmer)`, or, when
 * `SingleMask` is true, also as `callback(pos, spaced_kmer)`.
 */
template <typename Callback, Encoding E, Layout L, bool SingleMask>
concept SpacedKmerCallback =
    std::invocable<Callback, std::size_t, std::size_t, SpacedKmer<E, L> const&> ||
    (SingleMask && std::invocable<Callback, std::size_t, SpacedKmer<E, L> const&>);

/**
 * @brief A callback for vectors of spaced k-mers, as `callback(pos, mask_idx, vec, valid_count)`.
 */
template <typename Callback, typename Vector>
concept SpacedKmerVectorCallback =
    std::invocable<Callback, std::size_t, std::size_t, Vector, std::size_t>;

// =================================================================================================
//     K-mer Extraction Callback Dispatch
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Contiguous K-mers
// -------------------------------------------------------------------------------------------------

/**
 * @brief Invoke a k-mer extraction callback with the k-mer of a raw word, with or without a leading
 * position argument.
 *
 * Accepts either `callback(pos, kmer)` or `callback(kmer)`, see KmerCallback. This is resolved at
 * compile time, so a callback without `pos` does not pay for it. The word must already be a valid
 * k-mer, see kmer_cast_unchecked().
 */
template <Encoding E, Layout L, typename Callback>
FISK_ALWAYS_INLINE
inline void invoke_kmer_callback(Callback&& callback, std::size_t pos, std::uint64_t raw)
{
    using Value = Kmer<E, L>;
    auto const value = kmer_cast_unchecked<E, L>(raw);
    if constexpr (std::invocable<Callback, std::size_t, Value const&>) {
        callback(pos, value);
    } else if constexpr (std::invocable<Callback, Value const&>) {
        callback(value);
    } else {
        static_assert(
            KmerCallback<Callback, E, L>,
            "Callback must be callable as callback(pos, kmer) or callback(kmer)."
        );
    }
}

/**
 * @brief Invoke a k-mer extraction callback that receives a whole SIMD register of consecutive
 * k-mers.
 *
 * Accepts `callback(pos, vec, valid_count)`, see KmerVectorCallback. `pos` is the start position
 * of the k-mer in lane 0; lane `j` holds the k-mer starting at `pos + j`, for `j < valid_count`.
 * As for invoke_kmer_callback(), a callback that ignores the position costs nothing.
 */
template <typename Callback, typename Vector>
FISK_ALWAYS_INLINE
inline void invoke_kmer_vector_callback(
    Callback&& callback, std::size_t pos, Vector vec, std::size_t valid_count
) {
    if constexpr (std::invocable<Callback, std::size_t, Vector, std::size_t>) {
        callback(pos, vec, valid_count);
    } else {
        static_assert(
            KmerVectorCallback<Callback, Vector>,
            "Callback must be callable as callback(pos, vec, valid_count)."
        );
    }
}

/**
 * @brief Invoke a k-mer extraction callback with a vector of consecutive k-mers and its valid
 * lanes, or with each of its valid k-mers.
 *
 * Accepts `callback(pos, vec, valid_lanes)`, see KmerValidLanesCallback, where lane `j` of `vec`
 * holds the k-mer starting at `pos + j`, valid if lane `j` of `valid_lanes` is. Or the forms of
 * single k-mers of invoke_kmer_callback(), called for each valid lane, in order. That stores the
 * vector, and is hence slower.
 */
template <Encoding E, Layout L, SimdArch Simd, typename Callback, typename Vector>
FISK_ALWAYS_INLINE
inline void invoke_kmer_valid_lanes_callback(
    Callback&& callback, std::size_t pos, Vector vec, SimdValidLanes<Simd> valid_lanes
) {
    if constexpr (KmerValidLanesCallback<Callback, Vector, Simd>) {
        callback(pos, vec, valid_lanes);
    } else if constexpr (KmerCallback<Callback, E, L>) {
        alignas(64) std::uint64_t kmers[Simd::lanes];
        Simd::store_aligned(vec, kmers);
        for (std::uint64_t bits = valid_lanes.bits(); bits != 0; bits &= bits - 1) {
            std::size_t const lane = static_cast<std::size_t>(std::countr_zero(bits));
            invoke_kmer_callback<E, L>(callback, pos + lane, kmers[lane]);
        }
    } else {
        static_assert(
            KmerValidLanesCallback<Callback, Vector, Simd> || KmerCallback<Callback, E, L>,
            "Callback must be callable as callback(pos, vec, valid_lanes), as "
            "callback(pos, kmer), or as callback(kmer)."
        );
    }
}

// -------------------------------------------------------------------------------------------------
//     Spaced K-mers
// -------------------------------------------------------------------------------------------------

/**
 * @brief Invoke a spaced k-mer extraction callback with the spaced k-mer of a raw word, with an
 * optional position-only shorthand.
 *
 * Accepts `callback(pos, mask_idx, spaced_kmer)`. When `SingleMask` is true (exactly one mask,
 * known at compile time), the shorthand `callback(pos, spaced_kmer)` is also accepted, as
 * `mask_idx` is always 0 then, see SpacedKmerCallback. The word must already be a valid spaced
 * k-mer, see spaced_kmer_cast_unchecked().
 *
 * @tparam SingleMask Whether the caller is known at compile time to iterate exactly one mask.
 */
template <Encoding E, Layout L, bool SingleMask, typename Callback>
FISK_ALWAYS_INLINE
inline void invoke_spaced_kmer_callback(
    Callback&& callback, std::size_t mask_idx, std::size_t pos, std::uint64_t raw
) {
    using Value = SpacedKmer<E, L>;
    auto const value = spaced_kmer_cast_unchecked<E, L>(raw);

    // There is no `callback(mask_idx, spaced_kmer)` shorthand for multiple masks: it would have the
    // same shape as the single-mask shorthand, so a `(std::size_t, Value)` callback would receive
    // either pos or mask_idx, depending on how many masks are used at the call site.
    if constexpr (std::invocable<Callback, std::size_t, std::size_t, Value const&>) {
        callback(pos, mask_idx, value);
    } else if constexpr (SingleMask && std::invocable<Callback, std::size_t, Value const&>) {
        callback(pos, value);
    } else if constexpr (SingleMask) {
        static_assert(
            SpacedKmerCallback<Callback, E, L, SingleMask>,
            "Callback must be callable as callback(pos, mask_idx, value) or callback(pos, value)."
        );
    } else {
        static_assert(
            SpacedKmerCallback<Callback, E, L, SingleMask>,
            "Callback must be callable as callback(pos, mask_idx, value). The 2-arg "
            "callback(pos, value) shorthand is only available for a single mask (known at "
            "compile time), since with multiple masks the mask index must be explicit."
        );
    }
}

/**
 * @brief Invoke a spaced k-mer extraction callback that receives a whole SIMD register of the
 * spaced k-mers of one mask at consecutive positions.
 *
 * Accepts `callback(pos, mask_idx, vec, valid_count)`, see SpacedKmerVectorCallback. Lane `j` holds
 * the spaced k-mer of mask `mask_idx` starting at `pos + j`, for `j < valid_count`; a vector never
 * mixes masks. There is no shorthand without `mask_idx` for a single mask: with SimdScalar, whose
 * vectors are words, `callback(pos, vec, valid_count)` would have the shape of the scalar
 * `callback(pos, mask_idx, spaced_kmer)`, so a callback could receive either.
 */
template <typename Callback, typename Vector>
FISK_ALWAYS_INLINE
inline void invoke_spaced_kmer_vector_callback(
    Callback&& callback, std::size_t mask_idx, std::size_t pos, Vector vec, std::size_t valid_count
) {
    if constexpr (std::invocable<Callback, std::size_t, std::size_t, Vector, std::size_t>) {
        callback(pos, mask_idx, vec, valid_count);
    } else {
        static_assert(
            SpacedKmerVectorCallback<Callback, Vector>,
            "Callback must be callable as callback(pos, mask_idx, vec, valid_count)."
        );
    }
}

} // namespace fisk
