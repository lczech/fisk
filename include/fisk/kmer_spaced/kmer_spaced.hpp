#pragma once

#include <algorithm>
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

namespace fisk {

// =================================================================================================
//     Position Mask
// =================================================================================================

/**
 * @brief Take a k-mer spaced mask string and turn it into a list of the positions of its '1' bits.
 *
 * This is the position-list counterpart to prepare_spaced_kmer_bit_extract_mask() below, which
 * instead packs the same mask into a single 2-bit-per-position bitmask.
 */
inline std::vector<size_t> prepare_spaced_kmer_position_mask( std::string const& mask )
{
    if( mask.size() == 0 || mask.size() > 32 ) {
        throw std::invalid_argument( "Invalid mask size not in [1,32]" );
    }
    std::vector<size_t> result;
    for( size_t i = 0; i < mask.size(); ++i ) {
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

/**
 * @brief For a set of masks, prepare all their position masks, i.e., the positions of their
 * '1' bits.
 */
inline std::vector<std::vector<size_t>> prepare_spaced_kmer_position_masks(
    std::vector<std::string> const& masks
) {
    std::vector<std::vector<size_t>> result;
    result.reserve( masks.size() );
    for( auto const& mask : masks ) {
        result.push_back( prepare_spaced_kmer_position_mask( mask ));
    }
    return result;
}

// =================================================================================================
//     Bit Extraction
// =================================================================================================

/**
 * @brief Take a k-mer spaced mask string and turn it into a 2bit bit extraction mask.
 *
 * For instance, mask `1011` becomes an uint `0b11001111`.
 */
inline std::uint64_t prepare_spaced_kmer_bit_extract_mask( std::string const& mask )
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
 * @brief Check if a spaced k-mer mask is valid.
 *
 * This is the case iff it is two-bit encoded, i.e., each position is either 00 or 11,
 * and the first and last position are kept (11), fitting with the span @p k.
 */
inline bool is_valid_spaced_kmer_mask( std::uint64_t const mask, size_t const k )
{
    // Check boundary conditions.
    if( k == 0 || k > 32 ) {
        throw std::invalid_argument( "Invalid spaced k-mer mask size not in [1,32]" );
    }

    // Check that the first and last positions are kept (11).
    // Position 0 in the span corresponds to the highest 2-bit lane,
    // position span_k-1 to the lowest 2-bit lane.
    bool const first_kept = ((mask >> (2 * (k - 1))) & 0x3u) == 0x3u;
    bool const last_kept  = (mask & 0x3u) == 0x3u;
    if (!first_kept || !last_kept) {
        return false;
        // throw std::runtime_error(
        //     "Invalid spaced k-mer mask: first and last position must be kept"
        // );
    }

    // Validity check: mask must consists of either 00 or 11 at each position.
    for (std::size_t i = 0; i < k; ++i) {
        const std::uint64_t lane = (mask >> (2 * i)) & 0x3u;
        if (lane != 0x0u && lane != 0x3u) {
            return false;
            // throw std::runtime_error(
            //     "Invalid spaced k-mer mask: each selected position must use both bits"
            // );
        }
    }

    // Mask should not have any bits set beyond the 2*k range.
    if ((mask >> (2 * k)) != 0) {
        return false;
    }

    return true;
}

/**
 * @brief Get the bit extract mask as a string of 0s and 1s, for printing.
 */
inline std::string bit_extract_mask_to_spaced_kmer_mask_string( std::uint64_t mask, size_t k )
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

// =================================================================================================
//     Spaced k-mer Extraction
// =================================================================================================

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
 * @brief Iterate a sequence, extract all valid spaced k-mers from it,
 * and call a callback on each spaced k-mer.
 *
 * A spaced k-mer is emitted iff all kept positions are valid, i.e.
 *
 *     (valid_positions & mask) == mask
 *
 * The mask is assumed to use two-bit encoding, i.e., as 00 and 11 per position, and must have set
 * the first and last position of the span (the beginning and end of spaced k-mer are always kept).
 *
 * @tparam Enc      Encoder turning characters into two-bit codes, see CharEncoder.
 *                  Its Encoding is not yet attached to the spaced k-mers emitted here.
 * @tparam Callback Callback function called with each extracted spaced k-mer.
 *
 * @param seq       Input sequence.
 * @param span_k    Full span length of the spaced seed, must be in [1, 32].
 * @param masks     Two-bit mask over the packed span: kept positions have
 *                  both bits set (0b11), skipped positions have 0b00.
 *                  Can also be a related BitExtract instance such as
 *                  BitExtractBlockTable or BitExtractButterflyTable.
 * @param enc       Encoder functor.
 * @param bit_ext   Bit extraction functor.
 * @param callback  Callback functor, takes the extracted spaced k-mer.
 */
template<typename MaskOrMasks, typename Enc, typename BitExtract, typename Callback>
    requires CharEncoder<std::remove_cvref_t<Enc>>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer(
    std::string_view seq,
    std::size_t const span_k,
    MaskOrMasks const& masks,
    Enc&& enc,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    static_assert(
        std::is_invocable_v<Callback, std::size_t, std::size_t, std::uint64_t>,
        "Callback must be callable as callback(mask_idx, pos, spaced_kmer)."
    );

    // Input boundary checks
    if (span_k == 0 || span_k > 32) {
        throw std::runtime_error(
            "Invalid call to spaced k-mer extraction with k not in [1, 32]"
        );
    }
    if (seq.size() < span_k) {
        return;
    }

    // Shorthands for data access
    std::size_t const seq_len = seq.size();
    char const*       data    = seq.data();

    // Full-span rolling mask: low 2*k bits are used, overhang from shift is removed.
    std::uint64_t const span_mask = (span_k == 32)
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << (2 * span_k)) - 1u);

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

        // Shift in the base. For invalid bases, the low 2 bits are irrelevant,
        // because validity is tested separately before emission.
        kmer_bits = ((kmer_bits << 2) & span_mask) | (code & 0x03u);

        // Shift in 11 for valid, 00 for invalid.
        valid_bits
            = ((valid_bits << 2) & span_mask)
            | (static_cast<std::uint64_t>(code < 4) * 0x03u)
        ;

        // Apply the callback for all masks that are satisfied at this position.
        std::size_t mask_idx = 0;
        for_each_mask(masks, [&](auto const& mask) {
            if ((valid_bits & mask.mask) == mask.mask) {
                callback(mask_idx, i, bit_ext(kmer_bits, mask));
            }
            ++mask_idx;
        });
    }
}

} // namespace fisk
