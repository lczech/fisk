#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/util/static_span.hpp"

namespace fisk {

// =================================================================================================
//     Bit Extraction Mask
// =================================================================================================

// The spaced k-mer masks in the representations of the bit extraction algorithms, for the
// spaced k-mer extraction. The masks as strings and bits, and the spaced k-mers that they select,
// are in core/kmer_spaced.hpp.

// -------------------------------------------------------------------------------------------------
//     Mask Sets
// -------------------------------------------------------------------------------------------------

// The default largest number of masks that a std::vector of masks is dispatched for, as a std::span
// of that compile-time size, see dispatch_static_span() (util/static_span.hpp). Each size up to it
// instantiates the function called on the span once, so the default is kept small; code that needs
// more masks passes a larger bound as the template parameter, or raises this default. The CMake
// cache variable of the same name sets it for all code that uses the fisk target alike.
#ifndef FISK_DEFAULT_MAX_MASKS
#define FISK_DEFAULT_MAX_MASKS 4
#endif

inline constexpr std::size_t kDefaultMaxMasks = FISK_DEFAULT_MAX_MASKS;

// -------------------------------------------------------------------------------------------------
//     Helper Functions
// -------------------------------------------------------------------------------------------------

// Convert the bits of each piece of a spaced k-mer mask into another form `BitExtractForm` via
// `make`, keeping the Layout, span, and the other members. The bits are already arranged for the
// Layout, and are hence used as they are. The pieces beyond the piece count are empty, and stay so.
template <typename BitExtractForm, Layout L, std::size_t MaxSpan, typename Make>
inline SpacedKmerMask<L, BitExtractForm, MaxSpan> convert_spaced_kmer_mask_(
    SpacedKmerMask<L, BitExtractMask, MaxSpan> const& mask, Make&& make
) {
    SpacedKmerMask<L, BitExtractForm, MaxSpan> result;
    for (std::size_t j = 0; j < result.max_pieces; ++j) {
        result.mask_pieces[j] = make(mask.mask_pieces[j].mask);
    }
    result.validity_pieces = mask.validity_pieces;
    result.mask_shifts = mask.mask_shifts;
    result.span_k = mask.span_k;
    return result;
}

template <typename BitExtractForm, Layout L, std::size_t MaxSpan, typename Make>
inline std::vector<SpacedKmerMask<L, BitExtractForm, MaxSpan>> convert_spaced_kmer_masks_(
    std::vector<SpacedKmerMask<L, BitExtractMask, MaxSpan>> const& masks, Make&& make
) {
    std::vector<SpacedKmerMask<L, BitExtractForm, MaxSpan>> result;
    result.reserve(masks.size());
    for (auto const& mask : masks) {
        result.push_back(convert_spaced_kmer_mask_<BitExtractForm>(mask, make));
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
 * at most 32 positions can be kept, and the length of the string is the span of the mask, which
 * must be in [1, MaxSpan]. Throws otherwise.
 *
 * The span is cut into pieces, see SpacedKmerMask. In each piece, the first position is in the
 * highest bits of its mask for kMSB, and in the lowest bits for kLSB, matching the window of the
 * extraction in that Layout. As bit extraction keeps the order of the bits, the kept bases then
 * end up in that Layout as well.
 */
template <Layout L = Layout::kMSB, std::size_t MaxSpan = kDefaultMaxSpan>
inline SpacedKmerMask<L, BitExtractMask, MaxSpan> make_spaced_kmer_mask(std::string const& mask)
{
    static_assert(
        L == Layout::kMSB || L == Layout::kLSB, "Unhandled Layout in make_spaced_kmer_mask()"
    );
    std::size_t const span_k = mask.size();
    if (span_k == 0) {
        throw std::invalid_argument("Invalid spaced k-mer mask of size 0");
    }
    if (span_k > MaxSpan) {
        throw std::invalid_argument(
            "Invalid spaced k-mer mask of span " + std::to_string(span_k) + " above the "
            "compile-time maximum span " + std::to_string(MaxSpan) + ". Raise the template "
            "parameter MaxSpan of the mask, of SpacedKmerMasks and SpacedKmerExtractor, or its "
            "default, the CMake option or macro FISK_DEFAULT_MAX_SPAN."
        );
    }
    if (mask.front() != '1' || mask.back() != '1') {
        throw std::invalid_argument(
            "Invalid spaced k-mer mask: first and last position must be set"
        );
    }

    // The weight up front, as the shifts for kMSB place each piece by the kept positions after it.
    auto const total = static_cast<std::size_t>(std::count(mask.begin(), mask.end(), '1'));
    if (total > 32) {
        throw std::invalid_argument("Invalid spaced k-mer mask with more than 32 kept positions");
    }

    SpacedKmerMask<L, BitExtractMask, MaxSpan> result;
    result.span_k = static_cast<std::uint16_t>(span_k);
    std::size_t before = 0;
    for (std::size_t j = 0; j < spaced_kmer_mask_piece_count(span_k); ++j) {
        std::size_t const len = spaced_kmer_mask_piece_span_(span_k, j);

        // The kept positions of the piece: in its bits with the first position in the highest
        // bits for kMSB and in the lowest bits for kLSB, and in its validity piece in the order of
        // the string for both layouts, see SpacedKmerMask.
        std::uint64_t bits = 0;
        std::size_t weight = 0;
        for (std::size_t i = 0; i < len; ++i) {
            char const c = mask[28 * j + i];
            if (c == '0' || c == '*') {
                continue;
            }
            if (c != '1') {
                throw std::invalid_argument("Invalid mask with symbols not in [0,1]");
            }
            std::size_t const lane = L == Layout::kMSB ? len - 1 - i : i;
            bits |= std::uint64_t{3} << (2 * lane);
            result.validity_pieces[j] |= std::uint32_t{1} << i;
            ++weight;
        }
        result.mask_pieces[j] = BitExtractMask(bits);
        result.mask_shifts[j] = static_cast<std::uint8_t>(
            L == Layout::kMSB ? 2 * (total - before - weight) : 2 * before
        );
        before += weight;
    }
    return result;
}

/**
 * @brief Create a set of spaced k-mer masks, see make_spaced_kmer_mask().
 */
template <Layout L = Layout::kMSB, std::size_t MaxSpan = kDefaultMaxSpan>
inline std::vector<SpacedKmerMask<L, BitExtractMask, MaxSpan>> make_spaced_kmer_masks(
    std::vector<std::string> const& masks
) {
    std::vector<SpacedKmerMask<L, BitExtractMask, MaxSpan>> result;
    result.reserve(masks.size());
    for (auto const& mask : masks) {
        result.push_back(make_spaced_kmer_mask<L, MaxSpan>(mask));
    }
    return result;
}

/**
 * @brief Create the butterfly mask of a spaced k-mer mask, see make_bit_extract_butterfly_mask().
 */
template <Layout L, std::size_t MaxSpan>
inline SpacedKmerMask<L, BitExtractButterflyMask, MaxSpan> make_spaced_kmer_butterfly_mask(
    SpacedKmerMask<L, BitExtractMask, MaxSpan> const& mask
) {
    return convert_spaced_kmer_mask_<BitExtractButterflyMask>(
        mask, make_bit_extract_butterfly_mask
    );
}

/**
 * @brief Create the butterfly mask of a spaced k-mer mask given as a string, see
 * make_spaced_kmer_mask().
 */
template <Layout L = Layout::kMSB, std::size_t MaxSpan = kDefaultMaxSpan>
inline SpacedKmerMask<L, BitExtractButterflyMask, MaxSpan> make_spaced_kmer_butterfly_mask(
    std::string const& mask
) {
    return make_spaced_kmer_butterfly_mask(make_spaced_kmer_mask<L, MaxSpan>(mask));
}

/**
 * @brief Create the butterfly masks of a set of spaced k-mer masks.
 */
template <Layout L, std::size_t MaxSpan>
inline std::vector<SpacedKmerMask<L, BitExtractButterflyMask, MaxSpan>>
make_spaced_kmer_butterfly_masks(
    std::vector<SpacedKmerMask<L, BitExtractMask, MaxSpan>> const& masks
) {
    return convert_spaced_kmer_masks_<BitExtractButterflyMask>(
        masks, make_bit_extract_butterfly_mask
    );
}

/**
 * @brief Create the block shift mask of a spaced k-mer mask, see
 * make_bit_extract_block_shift_mask().
 */
template <Layout L, std::size_t MaxSpan>
inline SpacedKmerMask<L, BitExtractBlockShiftMask, MaxSpan> make_spaced_kmer_block_shift_mask(
    SpacedKmerMask<L, BitExtractMask, MaxSpan> const& mask
) {
    return convert_spaced_kmer_mask_<BitExtractBlockShiftMask>(
        mask, make_bit_extract_block_shift_mask
    );
}

/**
 * @brief Create the block shift mask of a spaced k-mer mask given as a string, see
 * make_spaced_kmer_mask().
 */
template <Layout L = Layout::kMSB, std::size_t MaxSpan = kDefaultMaxSpan>
inline SpacedKmerMask<L, BitExtractBlockShiftMask, MaxSpan> make_spaced_kmer_block_shift_mask(
    std::string const& mask
) {
    return make_spaced_kmer_block_shift_mask(make_spaced_kmer_mask<L, MaxSpan>(mask));
}

/**
 * @brief Create the block shift masks of a set of spaced k-mer masks.
 */
template <Layout L, std::size_t MaxSpan>
inline std::vector<SpacedKmerMask<L, BitExtractBlockShiftMask, MaxSpan>>
make_spaced_kmer_block_shift_masks(
    std::vector<SpacedKmerMask<L, BitExtractMask, MaxSpan>> const& masks
) {
    return convert_spaced_kmer_masks_<BitExtractBlockShiftMask>(
        masks, make_bit_extract_block_shift_mask
    );
}

// -------------------------------------------------------------------------------------------------
//     Make Spaced Kernel Masks
// -------------------------------------------------------------------------------------------------

/**
 * @brief Create a bit extraction kernel of a spaced k-mer mask, such as
 * BitExtractKernelButterfly<SimdAVX2>, for the SIMD spaced k-mer extraction, see
 * ascii_rolling_simd.hpp and packed_bytewise_simd.hpp.
 */
template <typename Kernel, Layout L, std::size_t MaxSpan>
    requires BitExtractKernel<Kernel>
inline SpacedKmerMask<L, Kernel, MaxSpan> make_spaced_kmer_kernel(
    SpacedKmerMask<L, BitExtractMask, MaxSpan> const& mask
) {
    return convert_spaced_kmer_mask_<Kernel>(mask, [](std::uint64_t bits) {
        return Kernel(bits);
    });
}

/**
 * @brief Create a bit extraction kernel of a spaced k-mer mask given as a string, see
 * make_spaced_kmer_mask().
 */
template <typename Kernel, Layout L = Layout::kMSB, std::size_t MaxSpan = kDefaultMaxSpan>
    requires BitExtractKernel<Kernel>
inline SpacedKmerMask<L, Kernel, MaxSpan> make_spaced_kmer_kernel(std::string const& mask)
{
    return make_spaced_kmer_kernel<Kernel>(make_spaced_kmer_mask<L, MaxSpan>(mask));
}

/**
 * @brief Create the bit extraction kernels of a set of spaced k-mer masks, for instance to be
 * handed on to the SIMD extraction as a span of fixed size via dispatch_static_span().
 */
template <typename Kernel, Layout L, std::size_t MaxSpan>
    requires BitExtractKernel<Kernel>
inline std::vector<SpacedKmerMask<L, Kernel, MaxSpan>> make_spaced_kmer_kernels(
    std::vector<SpacedKmerMask<L, BitExtractMask, MaxSpan>> const& masks
) {
    return convert_spaced_kmer_masks_<Kernel>(masks, [](std::uint64_t bits) {
        return Kernel(bits);
    });
}


} // namespace fisk
