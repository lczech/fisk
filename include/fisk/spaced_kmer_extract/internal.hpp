#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>

#include "fisk/util/intrinsics.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Extraction, Internal Helpers
// =================================================================================================

/**
 * @brief Check that a span is in [1, 32], and return it.
 */
inline std::size_t check_spaced_kmer_span_(std::size_t span_k)
{
    if (span_k == 0 || span_k > 32) {
        throw std::invalid_argument(
            "Invalid call to spaced k-mer extraction with a mask span not in [1, 32]"
        );
    }
    return span_k;
}

/**
 * @brief Get the span of a non-empty set of masks, checking that it is the same for all of them.
 *
 * Takes the std::span by value, as this function, which may throw, is not always inlined: with a
 * reference, the address of the caller's span would escape into the call, so that the extraction
 * loops would have to reload the span from memory after every store that the compiler cannot see
 * through, such as one in the callback.
 */
template <typename Mask, std::size_t N>
inline std::size_t common_spaced_kmer_span_(std::span<Mask, N> masks)
{
    std::size_t const span_k = masks[0].span_k;
    for (auto const& mask : masks) {
        if (mask.span_k != span_k) {
            throw std::invalid_argument(
                "Invalid call to spaced k-mer extraction with masks of different spans"
            );
        }
    }
    return check_spaced_kmer_span_(span_k);
}

/**
 * @brief Helper function to "iterate" a single mask, and apply a function.
 */
template<typename Mask, typename F>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_mask_(Mask const& mask, F&& f)
{
    f(mask);
}

/**
 * @brief Helper function to iterate over a set of masks of compile-time size, and apply a function.
 *
 * A plain loop with the compile-time trip count N, so that the compiler can unroll it fully where
 * the function is cheap, such as with PEXT, and keep the loop where it is not.
 */
template<typename Mask, std::size_t N, typename F>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_mask_(std::span<Mask, N> masks, F&& f)
{
    for (std::size_t i = 0; i < N; ++i) {
        f(masks[i]);
    }
}

} // namespace fisk
