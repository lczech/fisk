#pragma once

#include <cstddef>
#include <cstdint>
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

// The largest number of masks that a std::vector of masks is dispatched for, as a std::span of
// that compile-time size, see dispatch_static_span() (util/static_span.hpp). Each size up to this
// one instantiates the function called on the span once. The CMake cache variable of the same name
// sets it for all code that uses the fisk target alike.
#ifndef FISK_MAX_DISPATCHED_MASKS
#define FISK_MAX_DISPATCHED_MASKS 16
#endif

inline constexpr std::size_t kMaxDispatchedMasks = FISK_MAX_DISPATCHED_MASKS;

// -------------------------------------------------------------------------------------------------
//     Helper Functions
// -------------------------------------------------------------------------------------------------

// Convert the bits of a spaced k-mer mask into another representation `M` via `make`, keeping the
// Layout and span. The bits are already arranged for the Layout, and are hence used as they are.
template <typename M, Layout L, typename Make>
inline SpacedKmerMask<L, M> convert_spaced_kmer_mask_(
    SpacedKmerMask<L, BitExtractMask> const& mask, Make&& make
) {
    SpacedKmerMask<L, M> result;
    static_cast<M&>(result) = make(mask.mask);
    result.span_k = mask.span_k;
    result.validity_mask = mask.validity_mask;
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

    // The kept positions in the order of the string, for both layouts, see SpacedKmerMask.
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] == '1') {
            result.validity_mask |= std::uint64_t{1} << i;
        }
    }
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
//     Make Spaced Kernel Masks
// -------------------------------------------------------------------------------------------------

/**
 * @brief Create a bit extraction kernel of a spaced k-mer mask, such as
 * BitExtractKernelButterfly<SimdAVX2>, for the SIMD spaced k-mer extraction, see
 * ascii_rolling_simd.hpp and packed_bytewise_simd.hpp.
 */
template <typename Kernel, Layout L>
    requires BitExtractKernel<Kernel>
inline SpacedKmerMask<L, Kernel> make_spaced_kmer_kernel(
    SpacedKmerMask<L, BitExtractMask> const& mask
) {
    return convert_spaced_kmer_mask_<Kernel>(mask, [](std::uint64_t bits) {
        return Kernel(bits);
    });
}

/**
 * @brief Create a bit extraction kernel of a spaced k-mer mask given as a string, see
 * make_spaced_kmer_mask().
 */
template <typename Kernel, Layout L = Layout::kMSB>
    requires BitExtractKernel<Kernel>
inline SpacedKmerMask<L, Kernel> make_spaced_kmer_kernel(std::string const& mask)
{
    return make_spaced_kmer_kernel<Kernel>(make_spaced_kmer_mask<L>(mask));
}

/**
 * @brief Create the bit extraction kernels of a set of spaced k-mer masks, for instance to be used
 * with BitExtractKernelDispatcher.
 */
template <typename Kernel, Layout L>
    requires BitExtractKernel<Kernel>
inline std::vector<SpacedKmerMask<L, Kernel>> make_spaced_kmer_kernels(
    std::vector<SpacedKmerMask<L, BitExtractMask>> const& masks
) {
    return convert_spaced_kmer_masks_<Kernel>(masks, [](std::uint64_t bits) {
        return Kernel(bits);
    });
}


} // namespace fisk
