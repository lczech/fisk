#pragma once

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
 * @brief Get the raw two-bit mask of piece @p piece of a spaced k-mer mask, in whichever
 * representation it is.
 *
 * `.mask` is the raw mask for BitExtractMask and the tables, and the mask or table of a SIMD
 * kernel.
 */
template <SpacedKmerMaskType Mask>
inline std::uint64_t spaced_kmer_mask_bits_(Mask const& mask, std::size_t piece = 0) noexcept
{
    auto const& m = mask.mask_pieces[piece];
    if constexpr (std::is_convertible_v<decltype(m.mask), std::uint64_t>) {
        return m.mask;
    } else {
        return m.mask.mask;
    }
}

/**
 * @brief Get the span of piece @p piece of a spaced k-mer mask of span @p span_k, see
 * SpacedKmerMask.
 */
inline constexpr std::size_t spaced_kmer_mask_piece_span_(
    std::size_t span_k, std::size_t piece
) noexcept
{
    return piece + 1 == spaced_kmer_mask_piece_count(span_k)
        ? spaced_kmer_mask_tail_span(span_k) : 28;
}

// -------------------------------------------------------------------------------------------------
//     Position Mask
// -------------------------------------------------------------------------------------------------

/**
 * @brief Take a spaced k-mer mask string and turn it into a list of the positions of its '1's.
 *
 * The first and last position must be kept, and at most 32 positions. The span is not limited
 * here; the masks of the extraction limit it, see SpacedKmerMask.
 *
 * This is the position-list counterpart to make_spaced_kmer_mask() in spaced_kmer_extract/mask.hpp,
 * which instead packs the same mask into a single 2-bit-per-position bitmask.
 */
inline std::vector<std::size_t> spaced_kmer_mask_positions(std::string const& mask)
{
    if( mask.size() == 0 ) {
        throw std::invalid_argument( "Invalid spaced k-mer mask of size 0" );
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
    if( result.size() > 32 ) {
        throw std::invalid_argument( "Invalid spaced k-mer mask with more than 32 kept positions" );
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
    return spaced_kmer_mask_positions( mask ).size();
}

/**
 * @brief Get the weight of a spaced k-mer mask, in any of its representations.
 */
template <SpacedKmerMaskType Mask>
inline std::size_t spaced_kmer_weight(Mask const& mask)
{
    std::size_t weight = 0;
    for (std::size_t j = 0; j < spaced_kmer_mask_piece_count(mask.span_k); ++j) {
        weight += spaced_kmer_weight(spaced_kmer_mask_bits_(mask, j));
    }
    return weight;
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
 * @brief Check if a spaced k-mer mask is valid, see SpacedKmerMask: its span is in [1, MaxSpan],
 * the bits of each of its pieces are 00 or 11 at each position of the piece and 0 above it, with
 * the first and last position of the span kept, its weight is at most 32, its validity pieces hold
 * the positions that it keeps, its shifts place the bits of each piece after those of the pieces
 * before it, and the pieces beyond its piece count are empty.
 */
template <SpacedKmerMaskType Mask>
inline bool is_valid_spaced_kmer_mask(Mask const& mask)
{
    static_assert(
        Mask::layout == Layout::kMSB || Mask::layout == Layout::kLSB,
        "Unhandled Layout in is_valid_spaced_kmer_mask()"
    );
    std::size_t const span_k = mask.span_k;
    if (span_k == 0 || span_k > Mask::max_span) {
        return false;
    }
    std::size_t const count = spaced_kmer_mask_piece_count(span_k);

    // The weight up front, as the shifts for kMSB place each piece by the kept positions after it.
    // With a lane that is neither 00 nor 11, this is off, but the check of its piece below fails.
    std::size_t const total = spaced_kmer_weight(mask);
    if (total > 32) {
        return false;
    }

    // Each piece, as built by make_spaced_kmer_mask() (spaced_kmer_extract/mask.hpp), with a span
    // of 0 for the pieces beyond the count, which then have to be empty.
    std::size_t before = 0;
    for (std::size_t j = 0; j < Mask::max_pieces; ++j) {
        std::uint64_t const bits = spaced_kmer_mask_bits_(mask, j);
        std::size_t const len = j < count ? spaced_kmer_mask_piece_span_(span_k, j) : 0;
        std::uint32_t validity = 0;
        for (std::size_t i = 0; i < len; ++i) {
            std::size_t const lane = Mask::layout == Layout::kMSB ? len - 1 - i : i;
            std::uint64_t const lane_bits = (bits >> (2 * lane)) & 0x3u;
            if (lane_bits != 0x0u && lane_bits != 0x3u) {
                return false;
            }
            validity |= std::uint32_t{lane_bits == 0x3u} << i;
        }

        // For a piece of span 32 the whole uint64_t is within it; shifting it by 64 is undefined.
        if ((len < 32 && (bits >> (2 * len)) != 0) || mask.validity_pieces[j] != validity) {
            return false;
        }
        std::size_t const weight = static_cast<std::size_t>(std::popcount(validity));
        std::size_t const shift = j >= count ? 0 : Mask::layout == Layout::kMSB
            ? 2 * (total - before - weight) : 2 * before;
        if (mask.mask_shifts[j] != shift) {
            return false;
        }
        before += weight;
    }

    // The first and last position of the span, in the order of the span in the validity pieces.
    std::size_t const tail = spaced_kmer_mask_tail_span(span_k);
    return (mask.validity_pieces[0] & 1u) && ((mask.validity_pieces[count - 1] >> (tail - 1)) & 1u);
}

/**
 * @brief Get a spaced k-mer mask as a string of 0s and 1s, in the order of the span, for printing.
 */
template <SpacedKmerMaskType Mask>
inline std::string spaced_kmer_mask_string(Mask const& mask)
{
    static_assert(
        Mask::layout == Layout::kMSB || Mask::layout == Layout::kLSB,
        "Unhandled Layout in spaced_kmer_mask_string()"
    );

    // Each position of each piece, whose first position is in the highest two bits for kMSB, and in
    // the lowest two bits for kLSB.
    std::string result;
    for (std::size_t j = 0; j < spaced_kmer_mask_piece_count(mask.span_k); ++j) {
        std::uint64_t const bits = spaced_kmer_mask_bits_(mask, j);
        std::size_t const len = spaced_kmer_mask_piece_span_(mask.span_k, j);
        for (std::size_t i = 0; i < len; ++i) {
            std::size_t const lane = Mask::layout == Layout::kMSB ? len - 1 - i : i;
            std::uint64_t const lane_bits = (bits >> (2 * lane)) & 0x3u;
            if (lane_bits != 0x0u && lane_bits != 0x3u) {
                throw std::invalid_argument(
                    "Invalid spaced k-mer mask with entries that are not 00 or 11."
                );
            }
            result += lane_bits == 0x3u ? '1' : '0';
        }

        // For a piece of span 32 the whole uint64_t is within it; shifting it by 64 is undefined.
        if (len < 32 && (bits >> (2 * len)) != 0) {
            throw std::invalid_argument("Invalid spaced k-mer mask not of size k.");
        }
    }
    return result;
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
