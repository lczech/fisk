#pragma once

#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "fisk/core/cast.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/simd.hpp"
#include "fisk/core/types.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction Callback Concepts
// =================================================================================================

// The forms of callbacks that the k-mer extraction functions accept, as dispatched by the
// invoke_*() functions below. The vector forms of k-mers take a KmerVector. Those of spaced k-mers
// take the vectors of one SIMD arch, which the concept receives as a deduced type: GCC warns about
// vector types that are spelled out as template arguments. The vector forms always take the
// position: the forms of single k-mers and of vectors are then told apart by their number of
// arguments alone, also for generic lambdas.

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
 * @brief A callback for vectors of k-mers, as `callback(pos, kmers, valid_count)`, see KmerVector.
 */
template <typename Callback, Encoding E, Layout L, typename Simd>
concept KmerVectorCallback =
    std::invocable<Callback, std::size_t, KmerVector<E, L, Simd>, std::size_t>;

/**
 * @brief A callback for vectors of k-mers with their valid lanes, as
 * `callback(pos, kmers, valid_lanes)`, see KmerVector and KmerVectorValidLanes.
 */
template <typename Callback, Encoding E, Layout L, typename Simd>
concept KmerVectorValidLanesCallback =
    std::invocable<Callback, std::size_t, KmerVector<E, L, Simd>, KmerVectorValidLanes<Simd>>;

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
 * @brief Invoke a k-mer extraction callback with a vector of consecutive k-mers.
 *
 * Accepts `callback(pos, kmers, valid_count)`, see KmerVectorCallback. `pos` is the start position
 * of the k-mer in lane 0; lane `j` holds the k-mer starting at `pos + j`, for `j < valid_count`.
 * As for invoke_kmer_callback(), a callback that ignores the position costs nothing.
 */
template <Encoding E, Layout L, SimdArch Simd, typename Callback>
FISK_ALWAYS_INLINE
inline void invoke_kmer_vector_callback(
    Callback&& callback, std::size_t pos, KmerVector<E, L, Simd> kmers, std::size_t valid_count
) {
    if constexpr (KmerVectorCallback<Callback, E, L, Simd>) {
        callback(pos, kmers, valid_count);
    } else {
        static_assert(
            KmerVectorCallback<Callback, E, L, Simd>,
            "Callback must be callable as callback(pos, kmers, valid_count)."
        );
    }
}

/**
 * @brief Invoke a k-mer extraction callback with a vector of consecutive k-mers and its valid
 * lanes.
 *
 * Accepts `callback(pos, kmers, valid_lanes)`, see KmerVectorValidLanesCallback, where lane `j` of
 * `kmers` holds the k-mer starting at `pos + j`, valid if lane `j` of `valid_lanes` is.
 */
template <Encoding E, Layout L, SimdArch Simd, typename Callback>
FISK_ALWAYS_INLINE
inline void invoke_kmer_vector_valid_lanes_callback(
    Callback&& callback, std::size_t pos, KmerVector<E, L, Simd> kmers,
    KmerVectorValidLanes<Simd> valid_lanes
) {
    if constexpr (KmerVectorValidLanesCallback<Callback, E, L, Simd>) {
        callback(pos, kmers, valid_lanes);
    } else {
        static_assert(
            KmerVectorValidLanesCallback<Callback, E, L, Simd>,
            "Callback must be callable as callback(pos, kmers, valid_lanes)."
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

// =================================================================================================
//     Valid Lanes of K-mer Vectors
// =================================================================================================

/**
 * @brief Call `func` with each of the first `valid_count` k-mers of a vector of k-mers, in lane
 * order, as handed out by for_each_kmer_packed_simd().
 *
 * `func` takes single k-mers, as `func(pos, kmer)` or `func(kmer)`, see KmerCallback, where the
 * k-mer in lane `j` starts at `pos + j`.
 */
template <Encoding E, Layout L, SimdArch Simd, typename Func>
    requires KmerCallback<Func, E, L>
FISK_ALWAYS_INLINE
inline void for_each_valid_lane(
    std::size_t pos, KmerVector<E, L, Simd> kmers, std::size_t valid_count, Func&& func
) {
    assert(valid_count <= Simd::lanes);
    std::uint64_t words[Simd::lanes];
    std::memcpy(words, &kmers.value, sizeof(words));
    for (std::size_t j = 0; j < valid_count; ++j) {
        invoke_kmer_callback<E, L>(func, pos + j, words[j]);
    }
}

/**
 * @brief Call `func` with each valid k-mer of a vector of k-mers, in lane order, as handed out by
 * for_each_kmer_packed_checked_simd().
 *
 * As the overload for a count of valid lanes, for the k-mers in the lanes set in `valid_lanes`.
 */
template <Encoding E, Layout L, SimdArch Simd, typename Func>
    requires KmerCallback<Func, E, L>
FISK_ALWAYS_INLINE
inline void for_each_valid_lane(
    std::size_t pos, KmerVector<E, L, Simd> kmers, KmerVectorValidLanes<Simd> valid_lanes,
    Func&& func
) {
    std::uint64_t words[Simd::lanes];
    std::memcpy(words, &kmers.value, sizeof(words));
    std::uint64_t const bits = valid_lanes.bits();

    // All lanes valid, the usual case, as a loop with a fixed count, which the compilers unroll.
    // GCC keeps the loop over the set bits below as a loop, even for a constant mask.
    if (bits == (std::uint64_t{1} << Simd::lanes) - 1) {
        for (std::size_t j = 0; j < Simd::lanes; ++j) {
            invoke_kmer_callback<E, L>(func, pos + j, words[j]);
        }
        return;
    }
    for (std::uint64_t rest = bits; rest != 0; rest &= rest - 1) {
        std::size_t const j = static_cast<std::size_t>(std::countr_zero(rest));
        invoke_kmer_callback<E, L>(func, pos + j, words[j]);
    }
}

/**
 * @brief Store the valid k-mers of a vector of k-mers at `out`, in lane order, and return how many
 * were stored.
 *
 * `valid` is the count of valid lanes or the KmerVectorValidLanes that the vector was handed out
 * with, see for_each_valid_lane(). `out` must have room for `Simd::lanes` k-mers. The positions
 * are not stored: with a count, the stored k-mer `i` starts at `pos + i`, but valid lanes can leave
 * gaps, so for their positions, use for_each_valid_lane() with `func(pos, kmer)` instead.
 */
template <Encoding E, Layout L, SimdArch Simd, typename Valid>
    requires std::same_as<Valid, std::size_t> || std::same_as<Valid, KmerVectorValidLanes<Simd>>
inline std::size_t store_valid_lanes(
    KmerVector<E, L, Simd> kmers, Valid valid, Kmer<E, L>* out
) noexcept {
    std::size_t count = 0;
    for_each_valid_lane(0, kmers, valid, [&](Kmer<E, L> kmer) {
        out[count++] = kmer;
    });
    return count;
}

} // namespace fisk
