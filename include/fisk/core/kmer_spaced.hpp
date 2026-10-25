#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "fisk/core/cast.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/types.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Mask
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

// -------------------------------------------------------------------------------------------------
//     Position Mask
// -------------------------------------------------------------------------------------------------

/**
 * @brief Take a spaced k-mer mask string and turn it into a list of the positions of its '1's.
 *
 * This is the position-list counterpart to make_spaced_kmer_mask() in spaced_kmer_extract/mask.hpp,
 * which instead packs the same mask into a single 2-bit-per-position bitmask.
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
 * @brief Check if a spaced k-mer mask is valid for its span, see above, and if its validity_mask
 * holds the positions that it keeps, see SpacedKmerMask.
 */
template <SpacedKmerMaskType Mask>
inline bool is_valid_spaced_kmer_mask(Mask const& mask)
{
    std::uint64_t const bits = spaced_kmer_mask_bits_(mask);
    if (!is_valid_spaced_kmer_mask(bits, mask.span_k)) {
        return false;
    }

    // The validity mask from the bits, whose first position is in the highest two bits for kMSB,
    // and in the lowest two bits for kLSB.
    std::uint64_t validity_mask = 0;
    for (std::size_t i = 0; i < mask.span_k; ++i) {
        std::size_t const lane = Mask::layout == Layout::kMSB ? mask.span_k - 1 - i : i;
        validity_mask |= ((bits >> (2 * lane)) & 1u) << i;
    }
    return mask.validity_mask == validity_mask;
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
//     Spaced K-mer Encoding
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

} // namespace fisk
