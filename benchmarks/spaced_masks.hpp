#pragma once

#include <cstddef>

// =================================================================================================
//     Spaced Mask Sets
// =================================================================================================

// The largest number of masks in a set that the multi-mask spaced k-mer benchmarks dispatch for,
// see dispatch_static_span(). Set here rather than taken from the library default
// kDefaultMaxMasks, as the mask files in masks/ have more masks per set than that default.
inline constexpr std::size_t kBenchmarkMaxMasks = 10;
