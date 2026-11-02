#pragma once

#include <algorithm>
#include <cstddef>
#include <string>

// =================================================================================================
//     Spaced Mask Sets
// =================================================================================================

// The largest number of masks in a set that the multi-mask spaced k-mer benchmarks dispatch for,
// see dispatch_static_span(). Set here rather than taken from the library default
// kDefaultMaxMasks, as the mask files in masks/ have more masks per set than that default.
inline constexpr std::size_t kBenchmarkMaxMasks = 10;

// =================================================================================================
//     Span Sweep Masks
// =================================================================================================

// The largest span of the span sweep suites, which time one mask per span from 1 to this.
inline constexpr std::size_t kSpanSweepMaxSpan = 128;

// The mask of the span sweep suites for a span: the first and last position kept, and about two
// thirds of the positions in between, evenly spread, at most 32 positions in total.
inline std::string spaced_mask_for_span(std::size_t span)
{
    std::size_t const weight = std::min<std::size_t>(32, std::max<std::size_t>(
        std::min<std::size_t>(span, 2), (2 * span + 2) / 3
    ));
    std::string mask(span, '0');
    if (weight == 1) {
        mask[0] = '1';
        return mask;
    }
    for (std::size_t i = 0; i < weight; ++i) {
        mask[(i * (span - 1) + (weight - 1) / 2) / (weight - 1)] = '1';
    }
    mask.front() = '1';
    mask.back() = '1';
    return mask;
}
