#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction Callback Dispatch
// =================================================================================================

/**
 * @brief Invoke a k-mer extraction callback with the k-mer of a raw word, with or without a leading
 * position argument.
 *
 * Accepts either `callback(pos, kmer)` or `callback(kmer)`. This is resolved at compile time, so a
 * callback without `pos` does not pay for it. The word must already be a valid k-mer, see
 * kmer_cast_unchecked().
 */
template <Encoding E, Layout L, typename Callback>
FISK_ALWAYS_INLINE
inline void invoke_kmer_callback(Callback&& callback, std::size_t pos, std::uint64_t raw)
{
    using Value = Kmer<E, L>;
    auto const value = kmer_cast_unchecked<E, L>(raw);
    if constexpr (std::is_invocable_v<Callback, std::size_t, Value const&>) {
        callback(pos, value);
    } else if constexpr (std::is_invocable_v<Callback, Value const&>) {
        callback(value);
    } else {
        static_assert(
            std::is_invocable_v<Callback, Value const&>,
            "Callback must be callable as callback(pos, kmer) or callback(kmer)."
        );
    }
}

/**
 * @brief Invoke a k-mer extraction callback that receives a whole SIMD register of consecutive
 * k-mers, with or without a leading position argument.
 *
 * Accepts either `callback(pos, vec, valid_count)` or `callback(vec, valid_count)`. `pos` is the
 * start position of the k-mer in lane 0; lane `j` holds the k-mer starting at `pos + j`, for
 * `j < valid_count`. As for invoke_kmer_callback(), a callback that ignores position costs nothing.
 */
template <typename Callback, typename Vector>
FISK_ALWAYS_INLINE
inline void invoke_kmer_vector_callback(
    Callback&& callback, std::size_t pos, Vector vec, std::size_t valid_count
) {
    if constexpr (std::is_invocable_v<Callback, std::size_t, Vector, std::size_t>) {
        callback(pos, vec, valid_count);
    } else if constexpr (std::is_invocable_v<Callback, Vector, std::size_t>) {
        callback(vec, valid_count);
    } else {
        static_assert(
            std::is_invocable_v<Callback, Vector, std::size_t>,
            "Callback must be callable as callback(pos, vec, valid_count) or "
            "callback(vec, valid_count)."
        );
    }
}

/**
 * @brief Invoke a spaced k-mer extraction callback with the spaced k-mer of a raw word, with an
 * optional position-only shorthand.
 *
 * Accepts `callback(pos, mask_idx, spaced_kmer)`. When `SingleMask` is true (exactly one mask,
 * known at compile time), the shorthand `callback(pos, spaced_kmer)` is also accepted, as
 * `mask_idx` is always 0 then. The word must already be a valid spaced k-mer, see
 * spaced_kmer_cast_unchecked().
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
    if constexpr (std::is_invocable_v<Callback, std::size_t, std::size_t, Value const&>) {
        callback(pos, mask_idx, value);
    } else if constexpr (SingleMask && std::is_invocable_v<Callback, std::size_t, Value const&>) {
        callback(pos, value);
    } else if constexpr (SingleMask) {
        static_assert(
            std::is_invocable_v<Callback, std::size_t, Value const&>,
            "Callback must be callable as callback(pos, mask_idx, value) or callback(pos, value)."
        );
    } else {
        static_assert(
            std::is_invocable_v<Callback, std::size_t, std::size_t, Value const&>,
            "Callback must be callable as callback(pos, mask_idx, value). The 2-arg "
            "callback(pos, value) shorthand is only available for a single mask (known at "
            "compile time), since with multiple masks the mask index must be explicit."
        );
    }
}

} // namespace fisk
