#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "fisk/util/intrinsics.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Extraction, Internal Helpers
// =================================================================================================

// The default largest number of pieces of a mask, see SpacedKmerMask (core/types.hpp), for which
// the extraction loops that take spans above 32 have a copy of their own, with the loop over the
// pieces unrolled. Masks with more pieces run in one loop over the pieces at runtime. Each number
// up to it instantiates the loop once more, so the default is kept small; code that needs more
// passes a larger bound as the template parameter, or raises this default. The CMake cache
// variable of the same name sets it for all code that uses the fisk target alike.
#ifndef FISK_DEFAULT_MAX_COMPILED_PIECES
#define FISK_DEFAULT_MAX_COMPILED_PIECES 2
#endif

inline constexpr std::size_t kDefaultMaxCompiledPieces = FISK_DEFAULT_MAX_COMPILED_PIECES;

/**
 * @brief Throw for a mask span not in [1, max_span], for check_spaced_kmer_span_().
 *
 * Kept out of line and cold, as throw_invalid_kmer_k_() (kmer_extract/internal.hpp), so that
 * building the message adds no code to the extraction loops into which the check is inlined: such
 * code changes how the compilers inline and allocate registers in these loops.
 */
FISK_NOINLINE_COLD
inline void throw_invalid_spaced_kmer_span_(std::size_t span_k, std::size_t max_span)
{
    std::string message =
        "Invalid call to spaced k-mer extraction with a mask span of " +
        std::to_string(span_k) + ", not in [1, " + std::to_string(max_span) + "]";
    if (max_span == 32 && span_k > 32) {
        message += ". The rolling loops take spans up to 32; the chunked loops from ASCII and "
            "the bytewise loops from packed sequences take wider masks.";
    }
    throw std::invalid_argument(message);
}

/**
 * @brief Check that a span is in [1, MaxSpan], and return it.
 *
 * The default, 32, is the largest span of the rolling loops; the others pass the maximum span of
 * their mask type. Always inlined: with the throw out of line, the compilers may otherwise keep
 * the check as a call of its own, which changes how they inline the code around it.
 */
template <std::size_t MaxSpan = 32>
FISK_ALWAYS_INLINE
inline std::size_t check_spaced_kmer_span_(std::size_t span_k)
{
    if (span_k == 0 || span_k > MaxSpan) {
        throw_invalid_spaced_kmer_span_(span_k, MaxSpan);
    }
    return span_k;
}

// The mask type of a single mask, or of a std::span of masks, and their number.
template <typename MaskOrMasks>
struct spaced_kmer_mask_of_
{
    using type = MaskOrMasks;
    static constexpr std::size_t count = 1;
};

template <typename Mask, std::size_t N>
struct spaced_kmer_mask_of_<std::span<Mask, N>>
{
    using type = std::remove_const_t<Mask>;
    static constexpr std::size_t count = N;
};

template <typename MaskOrMasks>
using spaced_kmer_mask_of_t_ = typename spaced_kmer_mask_of_<MaskOrMasks>::type;

/**
 * @brief Throw for masks of different spans, for common_spaced_kmer_span_(), out of line and cold
 * as throw_invalid_spaced_kmer_span_().
 */
FISK_NOINLINE_COLD
inline void throw_spaced_kmer_spans_differ_()
{
    throw std::invalid_argument(
        "Invalid call to spaced k-mer extraction with masks of different spans"
    );
}

/**
 * @brief Get the span of a non-empty set of masks, checking that it is the same for all of them.
 *
 * Always inlined, as check_spaced_kmer_span_(). Takes the std::span by value for where a compiler
 * still keeps it as a call, as MSVC can: with a reference, the address of the caller's span would
 * escape into the call, so that the extraction loops would have to reload the span from memory
 * after every store that the compiler cannot see through, such as one in the callback.
 */
template <std::size_t MaxSpan = 32, typename Mask, std::size_t N>
FISK_ALWAYS_INLINE
inline std::size_t common_spaced_kmer_span_(std::span<Mask, N> masks)
{
    std::size_t const span_k = masks[0].span_k;
    for (auto const& mask : masks) {
        if (mask.span_k != span_k) {
            throw_spaced_kmer_spans_differ_();
        }
    }
    return check_spaced_kmer_span_<MaxSpan>(span_k);
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
