#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/packed.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Extraction from a PackedSequence
// =================================================================================================

// Unlike for_each_spaced_kmer() (kmer_spaced.hpp), which takes ASCII input and checks character
// validity, the functions here read directly from a PackedSequence (seq_pack.hpp), and assume that
// it holds only valid bases. The masks need to have the Layout of the sequence, and the spaced
// k-mers are emitted with its Encoding and Layout.
//
// Each byte of the sequence holds four bases, and a span starts at each of them. Each span is taken
// from the word of the 8 bytes loaded at that byte, plus a ninth byte for spans of 30 to 32, as a
// window whose lowest bits hold the span, as for the rolling window of for_each_spaced_kmer(). The
// masks are hence used as they are. Above the span, the window still holds the neighboring bases.
// They need no clearing, as the bit extraction only takes the bits of the mask, see `bit_ext`
// below.

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// Extract and emit the spaced k-mers of all masks from the `window` of the span starting at `pos`.
template <
    bool SingleMask, Encoding E, Layout L,
    typename MaskOrMasks, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE
inline void emit_spaced_kmers_packed_(
    MaskOrMasks const& masks, std::size_t pos, std::uint64_t window,
    BitExtract& bit_ext, Callback& callback
) {
    std::size_t mask_idx = 0;
    for_each_mask(masks, [&](auto const& mask) {
        auto const word = bit_ext(window, mask);
        assert_fits_width_(word, spaced_kmer_weight(mask));
        invoke_spaced_kmer_callback<E, L, SingleMask>(callback, mask_idx, pos, word);
        ++mask_idx;
    });
}

// -------------------------------------------------------------------------------------------------
//     Narrow Width
// -------------------------------------------------------------------------------------------------

// Spans up to 29 are within the 8 bytes loaded at their starting byte. The span is a runtime value
// here, as the shifts per position do not depend on it.
template <
    bool SingleMask, Encoding E, Layout L,
    typename MaskOrMasks, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_narrow_width_(
    PackedSequence<E, L> const& seq, std::size_t span_k,
    MaskOrMasks const& masks, BitExtract& bit_ext, Callback& callback
) {
    assert(span_k >= 1 && span_k <= 29);

    std::size_t const p_max = seq.length - span_k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes = (seq.length + 3) / 4;
    std::size_t const fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );

    auto const emit_ = [&](std::size_t pos, std::uint64_t window) {
        emit_spaced_kmers_packed_<SingleMask, E, L>(masks, pos, window, bit_ext, callback);
    };
    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t word;
        std::memcpy(&word, &seq.data[b], 8);
        if constexpr (L == Layout::kMSB) {
            // Move the span at position 3 to the lowest bits once, so that the shifts per
            // position are constants, as in for_each_kmer_packed_aligned_narrow_width_().
            word = byte_swap_64(word) >> (58 - 2 * span_k);
            emit_(4 * b + 0, word >> 6);
            emit_(4 * b + 1, word >> 4);
            emit_(4 * b + 2, word >> 2);
            emit_(4 * b + 3, word);
        } else if constexpr (L == Layout::kLSB) {
            emit_(4 * b + 0, word);
            emit_(4 * b + 1, word >> 2);
            emit_(4 * b + 2, word >> 4);
            emit_(4 * b + 3, word >> 6);
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_spaced_kmer_packed_narrow_width_()"
            );
        }
    }
    for_each_kmer_packed_rolling_narrow_width_(seq, span_k, 4 * fast_bytes, emit_);
}

// -------------------------------------------------------------------------------------------------
//     Full Width
// -------------------------------------------------------------------------------------------------

// The window of a span of 30 to 32 starting at base `Local` (0 to 3) of its byte, from the
// `word` loaded at that byte, and the `ninth` byte after it, as described above. Where the span
// reaches into the ninth byte, the bases from it are shifted in at the bottom of the shifted word,
// for kMSB in a way that compilers do not turn into a double shift (shld), which on Intel competes
// with PEXT for the same port.
template <Layout L, unsigned K, unsigned Local>
FISK_ALWAYS_INLINE
inline std::uint64_t spaced_kmer_packed_full_width_window_(std::uint64_t word, std::uint64_t ninth)
{
    constexpr bool in_word = 2 * K + 2 * Local <= 64;
    if constexpr (L == Layout::kMSB) {
        if constexpr (in_word) {
            return word >> (64 - 2 * K - 2 * Local);
        } else {
            constexpr unsigned shift = 2 * K + 2 * Local - 64;
            return (word << shift) | (ninth >> (8 - shift));
        }
    } else if constexpr (L == Layout::kLSB) {
        if constexpr (in_word) {
            return word >> (2 * Local);
        } else {
            return (word >> (2 * Local)) | (ninth << (64 - 2 * Local));
        }
    } else {
        static_assert(
            dependent_false_v<L>, "Unhandled Layout in spaced_kmer_packed_full_width_window_()"
        );
    }
}

// Spans of 30 to 32 can reach into a ninth byte. K is a template parameter, so that it is known at
// compile time which positions need it.
template <
    unsigned K, bool SingleMask, Encoding E, Layout L,
    typename MaskOrMasks, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_full_width_(
    PackedSequence<E, L> const& seq,
    MaskOrMasks const& masks, BitExtract& bit_ext, Callback& callback
) {
    static_assert(K >= 30 && K <= 32, "K must be in [30, 32]");

    std::size_t const p_max = seq.length - K;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes = (seq.length + 3) / 4;
    std::size_t const fast_bytes = (num_bytes >= 9) ? std::min(full_bytes, num_bytes - 9)
                                                     : std::size_t{0};

    auto const emit_ = [&](std::size_t pos, std::uint64_t window) {
        emit_spaced_kmers_packed_<SingleMask, E, L>(masks, pos, window, bit_ext, callback);
    };
    for (std::size_t b = 0; b < fast_bytes; ++b) {
        std::uint64_t word;
        std::memcpy(&word, &seq.data[b], 8);
        if constexpr (L == Layout::kMSB) {
            word = byte_swap_64(word);
        }
        std::uint64_t const ninth = seq.data[b + 8];
        emit_(4 * b + 0, spaced_kmer_packed_full_width_window_<L, K, 0>(word, ninth));
        emit_(4 * b + 1, spaced_kmer_packed_full_width_window_<L, K, 1>(word, ninth));
        emit_(4 * b + 2, spaced_kmer_packed_full_width_window_<L, K, 2>(word, ninth));
        emit_(4 * b + 3, spaced_kmer_packed_full_width_window_<L, K, 3>(word, ninth));
    }
    for_each_kmer_packed_rolling_full_width_(seq, K, 4 * fast_bytes, emit_);
}

// -------------------------------------------------------------------------------------------------
//     Dispatch
// -------------------------------------------------------------------------------------------------

/**
 * @brief Shared implementation behind both for_each_spaced_kmer_packed() overloads below.
 *
 * `SingleMask` is set by each overload, as for for_each_spaced_kmer_(), and `span_k` is the common
 * span of the masks, as checked by the overloads.
 */
template <
    bool SingleMask, Encoding E, Layout L,
    typename MaskOrMasks, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_(
    PackedSequence<E, L> const& seq, std::size_t span_k,
    MaskOrMasks const& masks, BitExtract& bit_ext, Callback& callback
) {
    for_each_mask(masks, [&]([[maybe_unused]] auto const& mask) {
        assert(is_valid_spaced_kmer_mask(mask));
        assert(bit_ext(~std::uint64_t{0}, mask) == bit_ext(spaced_kmer_mask_bits_(mask), mask));
    });
    if (seq.length < span_k) {
        return;
    }

    switch (span_k) {
        case 30: {
            for_each_spaced_kmer_packed_full_width_<30, SingleMask>(seq, masks, bit_ext, callback);
            break;
        }
        case 31: {
            for_each_spaced_kmer_packed_full_width_<31, SingleMask>(seq, masks, bit_ext, callback);
            break;
        }
        case 32: {
            for_each_spaced_kmer_packed_full_width_<32, SingleMask>(seq, masks, bit_ext, callback);
            break;
        }
        default: {
            for_each_spaced_kmer_packed_narrow_width_<SingleMask>(
                seq, span_k, masks, bit_ext, callback
            );
            break;
        }
    }
}

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it, and call a callback on each
 * spaced k-mer, for a single mask.
 *
 * The packed counterpart of for_each_spaced_kmer() for ASCII input. As the sequence holds only
 * valid bases, a spaced k-mer is emitted at every position. It has the Encoding and Layout of the
 * sequence.
 *
 * @tparam Mask     Spaced k-mer mask, see SpacedKmerMask and make_spaced_kmer_mask(), whose Layout
 *                  has to be the one of the sequence.
 * @tparam Callback Callback function called with each extracted spaced k-mer.
 *
 * @param seq       Input sequence.
 * @param mask      Spaced k-mer mask, in the representation that `bit_ext` expects, such as
 *                  BitExtractMask, BitExtractBlockTable, or BitExtractButterflyTable.
 *                  Its span must be in [1, 32].
 * @param bit_ext   Bit extraction functor, called as bit_ext(word, mask). It must ignore all bits
 *                  of `word` outside of the mask, as PEXT and all fisk bit extraction functions do.
 * @param callback  Callback functor, called as callback(pos, mask_idx, spaced_kmer), or as
 *                  callback(pos, spaced_kmer). `pos` is the start position of the spaced k-mer's
 *                  span in `seq`. See invoke_spaced_kmer_callback().
 */
template <Encoding E, Layout L, typename Mask, typename BitExtract, typename Callback>
    requires SpacedKmerMaskType<Mask> && (Mask::layout == L)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed(
    PackedSequence<E, L> const& seq,
    Mask const& mask,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    for_each_spaced_kmer_packed_<true>(
        seq, check_spaced_kmer_span_(mask.span_k), mask, bit_ext, callback
    );
}

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it, and call a callback on each
 * spaced k-mer, across a set of masks.
 *
 * Same as the single-mask overload above, but for a std::vector of masks, which all need to have
 * the same span. At each position, the spaced k-mers of all masks are emitted, in the order of the
 * masks. The callback must be callable as callback(pos, mask_idx, spaced_kmer); the shorthand
 * callback(pos, spaced_kmer) is only available for a single mask.
 */
template <
    Encoding E, Layout L, typename Mask, typename Alloc, typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<Mask> && (Mask::layout == L)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed(
    PackedSequence<E, L> const& seq,
    std::vector<Mask, Alloc> const& masks,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    if (masks.empty()) {
        return;
    }
    for_each_spaced_kmer_packed_<false>(
        seq, common_spaced_kmer_span_(masks), masks, bit_ext, callback
    );
}

} // namespace fisk
