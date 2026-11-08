#pragma once

#include <cstddef>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "fisk/util/intrinsics.hpp"

namespace fisk {

// =================================================================================================
//     Static Span Dispatch
// =================================================================================================

/**
 * @brief Helper for dispatch_static_span(): call `f` with the span of size `N` if `n == N`, or go
 * on with `N + 1`, up to `MaxN`.
 */
template <typename T, std::size_t N, std::size_t MaxN, typename F>
FISK_ALWAYS_INLINE
inline decltype(auto) dispatch_static_span_(T* data, std::size_t n, F& f)
{
    if constexpr (N < MaxN) {
        if (n != N) {
            return dispatch_static_span_<T, N + 1, MaxN>(data, n, f);
        }
    }
    return f(std::span<T, N>(data, N));
}

/**
 * @brief Throw for a range of `n` elements not in [1, max_n], for dispatch_static_span().
 *
 * Kept out of line and cold, so that building the message adds no code to the callers, into which
 * dispatch_static_span() and the loops that `f` runs are inlined.
 */
FISK_NOINLINE_COLD
inline void throw_invalid_static_span_size_(std::size_t n, std::size_t max_n)
{
    if (n == 0) {
        throw std::invalid_argument("dispatch_static_span(): the range is empty");
    }
    throw std::invalid_argument(
        "dispatch_static_span(): " + std::to_string(n) + " elements, but the compile-time "
        "maximum is " + std::to_string(max_n) + ". Raise the template parameter MaxN of the "
        "call; for sets of spaced k-mer masks, this is usually kDefaultMaxMasks, set by the "
        "CMake option or macro FISK_DEFAULT_MAX_MASKS."
    );
}

/**
 * @brief Call a function with a span of compile-time size over the elements of a contiguous range.
 *
 * The `range` (e.g., a std::vector, a std::array, or a std::span) holds `n` elements, with `n`
 * known at runtime. This calls `f` with a `std::span<T const, N>` over these elements, with
 * `N == n`, so that `f` can use their number as a compile-time constant, for instance to unroll a
 * loop over them. The span refers to the elements of the range; nothing is copied.
 *
 * `f` is instantiated for every `N` in `[1, MaxN]`, and all of these have to return the same type,
 * which is then returned from here.
 *
 * Forced inline, so that `f`, and the loops that it runs, can inline into the caller together with
 * the dispatch, see SpacedKmerExtractor::for_each().
 *
 * @throws std::invalid_argument if `n` is 0 or larger than `MaxN`.
 */
template <std::size_t MaxN, typename Range, typename F>
FISK_ALWAYS_INLINE
inline decltype(auto) dispatch_static_span(Range const& range, F&& f)
{
    static_assert(MaxN >= 1, "dispatch_static_span() needs MaxN >= 1.");
    using T = std::remove_reference_t<decltype(*std::data(range))> const;

    std::size_t const n = std::size(range);
    if (n == 0 || n > MaxN) {
        throw_invalid_static_span_size_(n, MaxN);
    }
    return dispatch_static_span_<T, 1, MaxN>(std::data(range), n, f);
}

} // namespace fisk
