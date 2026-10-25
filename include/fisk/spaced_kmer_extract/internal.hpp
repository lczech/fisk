#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "fisk/core/intrinsics.hpp"

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
 */
template <typename Masks>
inline std::size_t common_spaced_kmer_span_(Masks const& masks)
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
 * @brief Helper function to iterate over a vector of masks, and apply a function.
 */
template<typename Mask, typename Alloc, typename F>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_mask_(std::vector<Mask, Alloc> const& masks, F&& f)
{
    for (auto const& mask : masks) {
        f(mask);
    }
}

} // namespace fisk
