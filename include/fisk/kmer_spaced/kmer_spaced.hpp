#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <string>
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "fisk/kmer_extract/kmer_extract.hpp"
#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/cast.hpp"

namespace fisk {

// =================================================================================================
//     Position Mask
// =================================================================================================

/**
 * @brief Take a spaced k-mer mask string and turn it into a list of the positions of its '1's.
 *
 * This is the position-list counterpart to make_spaced_kmer_mask() below, which instead packs the
 * same mask into a single 2-bit-per-position bitmask.
 */
inline std::vector<std::size_t> spaced_kmer_mask_positions(std::string const& mask)
{
    if( mask.size() == 0 || mask.size() > 32 ) {
        throw std::invalid_argument( "Invalid mask size not in [1,32]" );
    }
    if( mask.front() != '1' || mask.back() != '1' ) {
        throw std::invalid_argument(
            "Invalid spaced k-mer mask: first and last position must be set"
        );
    }
    std::vector<std::size_t> result;
    for( std::size_t i = 0; i < mask.size(); ++i ) {
        if( mask[i] == '0' || mask[i] == '*' ) {
            continue;
        } else if( mask[i] == '1' ) {
            result.push_back(i);
        } else {
            throw std::invalid_argument( "Invalid mask with symbols not in [0,1]" );
        }
    }
    return result;
}

// =================================================================================================
//     Bit Extraction Mask
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Helper Functions
// -------------------------------------------------------------------------------------------------

/**
 * @brief Parse a spaced k-mer mask string into a two-bit mask, with the first position of the span
 * in the highest bits.
 *
 * For instance, mask `1011` becomes `0b11001111`.
 */
inline std::uint64_t parse_spaced_kmer_mask_( std::string const& mask )
{
    if( mask.size() == 0 || mask.size() > 32 ) {
        throw std::invalid_argument( "Invalid spaced k-mer mask size not in [1,32]" );
    }
    if( mask.front() != '1' || mask.back() != '1' ) {
        throw std::invalid_argument(
            "Invalid spaced k-mer mask: first and last position must be set"
        );
    }

    std::uint64_t result = 0;
    for( size_t i = 0; i < mask.size(); ++i ) {
        result <<= 2;
        if( mask[i] == '0' || mask[i] == '*' ) {
            continue;
        } else if( mask[i] == '1' ) {
            result |= 3;
        } else {
            throw std::invalid_argument( "Invalid spaced k-mer mask with symbols not in [0,*,1]" );
        }
    }
    return result;
}

/**
 * @brief Format a two-bit mask of span @p k as a string of 0s and 1s, with the highest bits first.
 */
inline std::string format_spaced_kmer_mask_( std::uint64_t mask, size_t k )
{
    std::string str;
    for( size_t i = 0; i < k; ++i ) {
        if(( mask & 0x3 ) == 0x0 ) {
            str.append("0");
        } else if(( mask & 0x3 ) == 0x3 ) {
            str.append("1");
        } else {
            throw std::invalid_argument(
                "Invalid spaced k-mer mask with entries that are not 00 or 11."
            );
        }
        mask >>= 2;
    }
    if( mask != 0 ) {
        throw std::invalid_argument(
            "Invalid spaced k-mer mask not of size k."
        );
    }
    std::reverse(str.begin(), str.end());
    return str;
}

/**
 * @brief Get the raw two-bit mask of a spaced k-mer mask, in whichever representation it is.
 *
 * `.mask` is the raw mask for BitExtractMask and the tables, and the mask or table of a SIMD
 * kernel.
 */
template <SpacedKmerMaskType Mask>
inline std::uint64_t spaced_kmer_mask_bits_(Mask const& mask) noexcept
{
    if constexpr (std::is_convertible_v<decltype(mask.mask), std::uint64_t>) {
        return mask.mask;
    } else {
        return mask.mask.mask;
    }
}

// Convert the bits of a spaced k-mer mask into another representation `M` via `make`, keeping the
// Layout and span. The bits are already arranged for the Layout, and are hence used as they are.
template <typename M, Layout L, typename Make>
inline SpacedKmerMask<L, M> convert_spaced_kmer_mask_(
    SpacedKmerMask<L, BitExtractMask> const& mask, Make&& make
) {
    SpacedKmerMask<L, M> result;
    static_cast<M&>(result) = make(mask.mask);
    result.span_k = mask.span_k;
    return result;
}

template <typename M, Layout L, typename Make>
inline std::vector<SpacedKmerMask<L, M>> convert_spaced_kmer_masks_(
    std::vector<SpacedKmerMask<L, BitExtractMask>> const& masks, Make&& make
) {
    std::vector<SpacedKmerMask<L, M>> result;
    result.reserve(masks.size());
    for (auto const& mask : masks) {
        result.push_back(convert_spaced_kmer_mask_<M>(mask, make));
    }
    return result;
}

// -------------------------------------------------------------------------------------------------
//     Make Spaced Masks
// -------------------------------------------------------------------------------------------------

/**
 * @brief Create a spaced k-mer mask for extracting spaced k-mers in Layout `L`, from a string.
 *
 * Positions marked `1` are kept, `0` or `*` are skipped. The first and last position must be kept,
 * and the length of the string is the span of the mask, which must be in [1, 32]. Throws otherwise.
 *
 * The first position of the span is in the highest bits of the mask for kMSB, and in the lowest
 * bits for kLSB, matching the rolling window of the extraction in that Layout. As bit extraction
 * keeps the order of the bits, the kept bases then end up in that Layout as well.
 */
template <Layout L = Layout::kMSB>
inline SpacedKmerMask<L, BitExtractMask> make_spaced_kmer_mask(std::string const& mask)
{
    SpacedKmerMask<L, BitExtractMask> result;
    if constexpr (L == Layout::kMSB) {
        static_cast<BitExtractMask&>(result) = BitExtractMask(parse_spaced_kmer_mask_(mask));
    } else if constexpr (L == Layout::kLSB) {
        std::string const mirrored(mask.rbegin(), mask.rend());
        static_cast<BitExtractMask&>(result) = BitExtractMask(parse_spaced_kmer_mask_(mirrored));
    } else {
        static_assert(dependent_false_v<L>, "Unhandled Layout in make_spaced_kmer_mask()");
    }
    result.span_k = mask.size();
    return result;
}

/**
 * @brief Create a set of spaced k-mer masks, see make_spaced_kmer_mask().
 */
template <Layout L = Layout::kMSB>
inline std::vector<SpacedKmerMask<L, BitExtractMask>> make_spaced_kmer_masks(
    std::vector<std::string> const& masks
) {
    std::vector<SpacedKmerMask<L, BitExtractMask>> result;
    result.reserve(masks.size());
    for (auto const& mask : masks) {
        result.push_back(make_spaced_kmer_mask<L>(mask));
    }
    return result;
}

/**
 * @brief Create the butterfly table of a spaced k-mer mask, see make_bit_extract_butterfly_table().
 */
template <Layout L>
inline SpacedKmerMask<L, BitExtractButterflyTable> make_spaced_kmer_butterfly_table(
    SpacedKmerMask<L, BitExtractMask> const& mask
) {
    return convert_spaced_kmer_mask_<BitExtractButterflyTable>(
        mask, make_bit_extract_butterfly_table
    );
}

/**
 * @brief Create the butterfly table of a spaced k-mer mask given as a string, see
 * make_spaced_kmer_mask().
 */
template <Layout L = Layout::kMSB>
inline SpacedKmerMask<L, BitExtractButterflyTable> make_spaced_kmer_butterfly_table(
    std::string const& mask
) {
    return make_spaced_kmer_butterfly_table(make_spaced_kmer_mask<L>(mask));
}

/**
 * @brief Create the butterfly tables of a set of spaced k-mer masks.
 */
template <Layout L>
inline std::vector<SpacedKmerMask<L, BitExtractButterflyTable>> make_spaced_kmer_butterfly_tables(
    std::vector<SpacedKmerMask<L, BitExtractMask>> const& masks
) {
    return convert_spaced_kmer_masks_<BitExtractButterflyTable>(
        masks, make_bit_extract_butterfly_table
    );
}

/**
 * @brief Create the block table of a spaced k-mer mask, see make_bit_extract_block_table().
 */
template <Layout L>
inline SpacedKmerMask<L, BitExtractBlockTable> make_spaced_kmer_block_table(
    SpacedKmerMask<L, BitExtractMask> const& mask
) {
    return convert_spaced_kmer_mask_<BitExtractBlockTable>(mask, make_bit_extract_block_table);
}

/**
 * @brief Create the block table of a spaced k-mer mask given as a string, see
 * make_spaced_kmer_mask().
 */
template <Layout L = Layout::kMSB>
inline SpacedKmerMask<L, BitExtractBlockTable> make_spaced_kmer_block_table(
    std::string const& mask
) {
    return make_spaced_kmer_block_table(make_spaced_kmer_mask<L>(mask));
}

/**
 * @brief Create the block tables of a set of spaced k-mer masks.
 */
template <Layout L>
inline std::vector<SpacedKmerMask<L, BitExtractBlockTable>> make_spaced_kmer_block_tables(
    std::vector<SpacedKmerMask<L, BitExtractMask>> const& masks
) {
    return convert_spaced_kmer_masks_<BitExtractBlockTable>(masks, make_bit_extract_block_table);
}

// -------------------------------------------------------------------------------------------------
//     Convenience Functions
// -------------------------------------------------------------------------------------------------

/**
 * @brief Get the weight of a two-bit encoded spaced k-mer mask, i.e., its number of kept positions.
 *
 * This is the `width` to pass to the Kmer operations for spaced k-mers extracted with this mask.
 * The mask must be valid, see is_valid_spaced_kmer_mask(); every kept position is 0b11.
 */
inline std::size_t spaced_kmer_weight( std::uint64_t const mask )
{
    return static_cast<std::size_t>(std::popcount(mask)) / 2;
}

/**
 * @brief Get the weight of a spaced k-mer mask given as a string of 0, 1 and *, as in
 * make_spaced_kmer_mask().
 */
inline std::size_t spaced_kmer_weight( std::string const& mask )
{
    return spaced_kmer_weight( parse_spaced_kmer_mask_( mask ));
}

/**
 * @brief Get the weight of a spaced k-mer mask, in any of its representations.
 */
template <SpacedKmerMaskType Mask>
inline std::size_t spaced_kmer_weight(Mask const& mask)
{
    return spaced_kmer_weight(spaced_kmer_mask_bits_(mask));
}

/**
 * @brief Check if a spaced k-mer mask is valid.
 *
 * This is the case iff the span @p k is in [1, 32], and the mask is two-bit encoded, i.e., each
 * position is either 00 or 11, with the first and last position kept (11), and no bits set above
 * the span.
 */
inline bool is_valid_spaced_kmer_mask( std::uint64_t const mask, size_t const k )
{
    // Check boundary conditions.
    if( k == 0 || k > 32 ) {
        return false;
    }

    // Check that the first and last positions are kept (11), i.e., the highest and lowest 2-bit
    // lane of the span, in either order depending on the Layout.
    bool const first_kept = ((mask >> (2 * (k - 1))) & 0x3u) == 0x3u;
    bool const last_kept  = (mask & 0x3u) == 0x3u;
    if (!first_kept || !last_kept) {
        return false;
    }

    // Validity check: mask must consists of either 00 or 11 at each position.
    for (std::size_t i = 0; i < k; ++i) {
        const std::uint64_t lane = (mask >> (2 * i)) & 0x3u;
        if (lane != 0x0u && lane != 0x3u) {
            return false;
        }
    }

    // Mask should not have any bits set beyond the 2*k range.
    // For k == 32 the whole uint64_t is within the span; shifting it by 64 is undefined.
    if (k < 32 && (mask >> (2 * k)) != 0) {
        return false;
    }

    return true;
}

/**
 * @brief Check if a spaced k-mer mask is valid for its span, see above.
 */
template <SpacedKmerMaskType Mask>
inline bool is_valid_spaced_kmer_mask(Mask const& mask)
{
    return is_valid_spaced_kmer_mask(spaced_kmer_mask_bits_(mask), mask.span_k);
}

/**
 * @brief Get a spaced k-mer mask as a string of 0s and 1s, in the order of the span, for printing.
 */
template <SpacedKmerMaskType Mask>
inline std::string spaced_kmer_mask_string(Mask const& mask)
{
    std::uint64_t const bits = spaced_kmer_mask_bits_(mask);
    if constexpr (Mask::layout == Layout::kMSB) {
        return format_spaced_kmer_mask_(bits, mask.span_k);
    } else if constexpr (Mask::layout == Layout::kLSB) {
        std::string const mirrored = format_spaced_kmer_mask_(bits, mask.span_k);
        return std::string(mirrored.rbegin(), mirrored.rend());
    } else {
        static_assert(
            dependent_false_v<Mask::layout>, "Unhandled Layout in spaced_kmer_mask_string()"
        );
    }
}

// =================================================================================================
//     Spaced k-mer Encoding
// =================================================================================================

/**
 * @brief Build the spaced k-mer that a mask selects from a window of nucleotides.
 *
 * The spaced counterpart of kmer_encode() for contiguous k-mers, with the same purpose and the same
 * loud failure: it throws on an invalid mask (see spaced_kmer_mask_positions()), on a window whose
 * length differs from the mask's, and on any non-nucleotide at a kept position. Skipped positions
 * are not inspected, as during extraction. The result decodes to the kept bases, with the mask
 * weight as its width.
 */
template <SpacedKmerType K = SpacedKmerAcgtMsb>
[[nodiscard]] inline K spaced_kmer_encode(std::string_view window, std::string const& mask)
{
    auto const positions = spaced_kmer_mask_positions(mask);
    if (window.size() != mask.size()) {
        throw std::invalid_argument(
            "Invalid call to spaced_kmer_encode() with a window length that differs from the mask"
        );
    }

    std::string kept;
    kept.reserve(positions.size());
    for (auto const pos : positions) {
        kept.push_back(window[pos]);
    }
    auto const contiguous = kmer_encode<Kmer<K::encoding, K::layout>>(kept);
    return spaced_kmer_cast_unchecked<K::encoding, K::layout>(kmer_value(contiguous));
}

/**
 * @brief Build the spaced k-mer that a spaced k-mer mask selects from a window of nucleotides.
 *
 * Same as above, but for a mask in any of its representations, see SpacedKmerMask. The Layout is
 * taken from the mask, so that the result is of the same type, and has the same value, as the
 * spaced k-mer that extraction with this mask and an encoder of Encoding `E` emits for the window.
 */
template <Encoding E = Encoding::kACGT, SpacedKmerMaskType Mask>
[[nodiscard]] inline SpacedKmer<E, Mask::layout> spaced_kmer_encode(
    std::string_view window, Mask const& mask
) {
    return spaced_kmer_encode<SpacedKmer<E, Mask::layout>>(window, spaced_kmer_mask_string(mask));
}

/**
 * @brief Get the string representation of a spaced k-mer over the full span of its mask, with the
 * kept bases at their positions, and `-` at the skipped ones.
 *
 * The inverse of spaced_kmer_encode(), up to the skipped positions: for instance, `ACGTA` under the
 * mask `10101` encodes and decodes to `A-G-A`. Use kmer_decode() with the mask weight to get the
 * kept bases only. Throws on an invalid mask, see spaced_kmer_mask_positions().
 */
template <SpacedKmerType K>
[[nodiscard]] inline std::string spaced_kmer_decode(K const& kmer, std::string const& mask)
{
    auto const positions = spaced_kmer_mask_positions(mask);
    auto const kept = kmer_decode(kmer, positions.size());

    std::string result(mask.size(), '-');
    for (std::size_t i = 0; i < positions.size(); ++i) {
        result[positions[i]] = kept[i];
    }
    return result;
}

/**
 * @brief Get the string representation of a spaced k-mer over the full span of its mask, for a
 * mask in any of its representations, see SpacedKmerMask.
 */
template <SpacedKmerType K, SpacedKmerMaskType Mask>
[[nodiscard]] inline std::string spaced_kmer_decode(K const& kmer, Mask const& mask)
{
    return spaced_kmer_decode(kmer, spaced_kmer_mask_string(mask));
}

// =================================================================================================
//     Spaced k-mer Extraction
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
inline void for_each_mask(Mask const& mask, F&& f)
{
    f(mask);
}

/**
 * @brief Helper function to iterate over a vector of masks, and apply a function.
 */
template<typename Mask, typename Alloc, typename F>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_mask(std::vector<Mask, Alloc> const& masks, F&& f)
{
    for (auto const& mask : masks) {
        f(mask);
    }
}

/**
 * @brief Shared implementation behind both for_each_spaced_kmer() overloads below.
 *
 * `SingleMask` is set by each overload, mirroring the single mask vs std::vector<Mask> distinction
 * that for_each_mask() above is overloaded on. `L` is the Layout of the masks, and `span_k` their
 * common span, as checked by the overloads.
 */
template<
    bool SingleMask, Layout L,
    typename MaskOrMasks, typename Enc, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_(
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
    for_each_mask(masks, [&]([[maybe_unused]] auto const& mask) {
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
            static_assert(dependent_false_v<L>, "Unhandled Layout in for_each_spaced_kmer_()");
        }

        // Start position of the spaced k-mer. This underflows while i < span_k - 1, but is never
        // used then, as no mask can be satisfied before span_k characters have been shifted in,
        // see above.
        std::size_t const pos = i + 1 - span_k;

        // Apply the callback for all masks that are satisfied at this position.
        std::size_t mask_idx = 0;
        for_each_mask(masks, [&](auto const& mask) {
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
 *                  BitExtractMask, BitExtractBlockTable, or BitExtractButterflyTable.
 *                  Its span must be in [1, 32].
 * @param enc       Encoder functor.
 * @param bit_ext   Bit extraction functor, called as bit_ext(word, mask). It must ignore all bits
 *                  of `word` outside of the mask, as PEXT and all fisk bit extraction functions do.
 * @param callback  Callback functor, called as callback(pos, mask_idx, spaced_kmer), or as
 *                  callback(pos, spaced_kmer). `pos` is the start index of the spaced k-mer's
 *                  span (into `seq`). See invoke_spaced_kmer_callback().
 */
template<typename Mask, typename Enc, typename BitExtract, typename Callback>
    requires SpacedKmerMaskType<Mask> && CharEncoder<std::remove_cvref_t<Enc>>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer(
    std::string_view seq,
    Mask const& mask,
    Enc&& enc,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    for_each_spaced_kmer_<true, Mask::layout>(
        seq, check_spaced_kmer_span_(mask.span_k), mask, std::forward<Enc>(enc),
        std::forward<BitExtract>(bit_ext), std::forward<Callback>(callback)
    );
}

/**
 * @brief Iterate a sequence, extract all valid spaced k-mers from it,
 * and call a callback on each spaced k-mer, across a set of masks.
 *
 * Same as the single-mask overload above, but for a std::vector of masks, which all need to have
 * the same span. The callback must be callable as callback(pos, mask_idx, spaced_kmer); the
 * shorthand callback(pos, spaced_kmer) is only available for a single mask.
 * See invoke_spaced_kmer_callback().
 */
template<typename Mask, typename Alloc, typename Enc, typename BitExtract, typename Callback>
    requires SpacedKmerMaskType<Mask> && CharEncoder<std::remove_cvref_t<Enc>>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer(
    std::string_view seq,
    std::vector<Mask, Alloc> const& masks,
    Enc&& enc,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    if (masks.empty()) {
        return;
    }
    for_each_spaced_kmer_<false, Mask::layout>(
        seq, common_spaced_kmer_span_(masks), masks, std::forward<Enc>(enc),
        std::forward<BitExtract>(bit_ext), std::forward<Callback>(callback)
    );
}

} // namespace fisk
