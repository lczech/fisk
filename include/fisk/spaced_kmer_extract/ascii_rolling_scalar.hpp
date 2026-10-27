#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>
#include <span>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/spaced_kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Extraction from ASCII, Rolling, Scalar
// =================================================================================================

/**
 * @brief Shared implementation behind both for_each_spaced_kmer_ascii_checked_rolling_scalar()
 * overloads below.
 *
 * `SingleMask` is set by each overload, mirroring the single mask vs std::span<Mask, N> distinction
 * that for_each_mask_() (internal.hpp) is overloaded on. `L` is the Layout of the masks, and
 * `span_k` their common span, as checked by the overloads.
 */
template<
    bool SingleMask, Layout L,
    typename MaskOrMasks, typename Enc, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_rolling_scalar_(
    std::string_view seq,
    std::size_t const span_k,
    MaskOrMasks const& masks,
    Enc&& enc,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    if (seq.size() < span_k) {
        return;
    }

    // Shorthands for data access
    std::size_t const seq_len = seq.size();
    char const*       data    = seq.data();

    // MSB shifts new bases in at the bottom, and keeps the old ones above the span, as the bit
    // extraction and the validity check below only look at the bits of the mask. Hence, the masks
    // must not have bits above the span, and `bit_ext` must ignore all bits outside of the mask.
    // LSB shifts new bases in at the top of the span instead, and shifts them out below.
    for_each_mask_(masks, [&]([[maybe_unused]] auto const& mask) {
        assert(is_valid_spaced_kmer_mask(mask));
        assert(bit_ext(~std::uint64_t{0}, mask) == bit_ext(spaced_kmer_mask_bits_(mask), mask));
    });
    std::size_t const top_shift = 2 * (span_k - 1);

    // The sequence is scanned as overlapping windows of length `k` (the full span).
    // A rolling 2-bit packed representation of the span is maintained in `kmer_bits`.
    // In parallel, a rolling 2-bit validity word is maintained in `valid_bits`, where
    // each position contributes two bits (11 for valid and 00 for invalid).
    // A spaced k-mer is emitted iff all kept positions are valid, i.e.
    //   (valid_bits & mask) == mask
    // The mask is assumed to to use two-bit coding, and must keep the first and
    // last position of the span. Otherwise, the first k iterations of the loop will
    // trigger premature but invalid k-mer emissions.
    // If masks are needed where the first/last positions are _not_ set, the condition
    // in the loop below needs to have an additional check for `if (i >= k - 1)`.
    std::uint64_t kmer_bits  = 0;
    std::uint64_t valid_bits = 0;
    for( std::size_t i = 0; i < seq_len; ++i ) {
        std::uint8_t const code = static_cast<std::uint8_t>(enc(data[i]));

        // Shift in the base, and 11 for valid or 00 for invalid into the validity word.
        // For invalid bases, the low 2 bits are irrelevant, because validity is tested
        // separately before emission.
        if constexpr (L == Layout::kMSB) {
            kmer_bits = (kmer_bits << 2) | (code & 0x03u);
            valid_bits
                = (valid_bits << 2)
                | (static_cast<std::uint64_t>(code < 4) * 0x03u)
            ;
        } else if constexpr (L == Layout::kLSB) {
            kmer_bits = (kmer_bits >> 2) | (std::uint64_t{code & 0x03u} << top_shift);
            valid_bits
                = (valid_bits >> 2)
                | ((static_cast<std::uint64_t>(code < 4) * 0x03u) << top_shift)
            ;
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_spaced_kmer_ascii_checked_rolling_scalar_()"
            );
        }

        // Start position of the spaced k-mer. This underflows while i < span_k - 1, but is never
        // used then, as no mask can be satisfied before span_k characters have been shifted in,
        // see above.
        std::size_t const pos = i + 1 - span_k;

        // Apply the callback for all masks that are satisfied at this position.
        std::size_t mask_idx = 0;
        for_each_mask_(masks, [&](auto const& mask) {
            if ((valid_bits & mask.mask) == mask.mask) {
                auto const word = bit_ext(kmer_bits, mask);
                assert_fits_width_(word, spaced_kmer_weight(mask.mask));
                invoke_spaced_kmer_callback<
                    std::remove_cvref_t<Enc>::encoding, L, SingleMask
                >(callback, mask_idx, pos, word);
            }
            ++mask_idx;
        });
    }
}

/**
 * @brief Iterate a sequence, extract all valid spaced k-mers from it,
 * and call a callback on each spaced k-mer, for a single mask.
 *
 * A spaced k-mer is emitted iff all kept positions are valid, i.e.
 *
 *     (valid_positions & mask) == mask
 *
 * The spaced k-mers are emitted with the Encoding of the encoder, and the Layout of the mask.
 *
 * @tparam Mask     Spaced k-mer mask, see SpacedKmerMask and make_spaced_kmer_mask().
 * @tparam Enc      Encoder turning characters into two-bit codes, see CharEncoder.
 * @tparam Callback Callback function called with each extracted spaced k-mer.
 *
 * @param seq       Input sequence.
 * @param mask      Spaced k-mer mask, in the representation that `bit_ext` expects, such as
 *                  BitExtractMask, BitExtractBlockShiftMask, or BitExtractButterflyMask.
 *                  Its span must be in [1, 32].
 * @param enc       Encoder functor.
 * @param bit_ext   Bit extraction functor, called as bit_ext(word, mask). It must ignore all bits
 *                  of `word` outside of the mask, as PEXT and all fisk bit extraction functions do.
 * @param callback  Callback functor, called as callback(pos, mask_idx, spaced_kmer), or as
 *                  callback(pos, spaced_kmer) or callback(spaced_kmer). `pos` is the start index
 *                  of the spaced k-mer's span (into `seq`). See invoke_spaced_kmer_callback().
 */
template<typename Mask, typename Enc, typename BitExtract, typename Callback>
    requires SpacedKmerMaskType<Mask> && CharEncoder<std::remove_cvref_t<Enc>>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_rolling_scalar(
    std::string_view seq,
    Mask const& mask,
    Enc&& enc,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    for_each_spaced_kmer_ascii_checked_rolling_scalar_<true, Mask::layout>(
        seq, check_spaced_kmer_span_(mask.span_k), mask, std::forward<Enc>(enc),
        std::forward<BitExtract>(bit_ext), std::forward<Callback>(callback)
    );
}

/**
 * @brief Iterate a sequence, extract all valid spaced k-mers from it,
 * and call a callback on each spaced k-mer, across a set of masks.
 *
 * Same as the single-mask overload above, but for a set of masks, which all need to have the same
 * span, as a std::span of compile-time size, possibly of const masks; see dispatch_static_span()
 * for a std::vector of masks. The callback must be callable as
 * callback(pos, mask_idx, spaced_kmer); the shorthands callback(pos, spaced_kmer) and
 * callback(spaced_kmer) are only available for a single mask. See invoke_spaced_kmer_callback().
 */
template<typename Mask, std::size_t N, typename Enc, typename BitExtract, typename Callback>
    requires SpacedKmerMaskType<std::remove_const_t<Mask>> && (N != std::dynamic_extent)
        && CharEncoder<std::remove_cvref_t<Enc>>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_rolling_scalar(
    std::string_view seq,
    std::span<Mask, N> masks,
    Enc&& enc,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    static_assert(N > 0, "Need at least one mask.");
    for_each_spaced_kmer_ascii_checked_rolling_scalar_<false, Mask::layout>(
        seq, common_spaced_kmer_span_(masks), masks, std::forward<Enc>(enc),
        std::forward<BitExtract>(bit_ext), std::forward<Callback>(callback)
    );
}

} // namespace fisk
