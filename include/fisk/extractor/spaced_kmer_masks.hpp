#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/extractor/internal.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Mask Forms
// =================================================================================================

/**
 * @brief The forms of the masks of a SpacedKmerMasks, as a std::tuple of one std::array of
 * `MaxMasks` masks per form: the masks as given, for PEXT, the PEXT kernels, where BMI2 is
 * compiled, the butterfly masks, and the butterfly kernels of the `Simds`, a TypeList.
 */
template <Layout L, std::size_t MaxMasks, typename Simds = CandidateSimdArchs>
struct SpacedKmerMaskForms;

template <Layout L, std::size_t MaxMasks, typename... Simd>
struct SpacedKmerMaskForms<L, MaxMasks, TypeList<Simd...>>
{
    template <typename M>
    using MaskForms = std::array<SpacedKmerMask<L, M>, MaxMasks>;

    using type = std::tuple<
        MaskForms<BitExtractMask>,
        #if defined(FISK_HAS_BMI2)
        MaskForms<BitExtractKernelPext>,
        #endif
        MaskForms<BitExtractButterflyMask>,
        MaskForms<BitExtractKernelButterfly<Simd>>...
    >;
};

// =================================================================================================
//     Spaced K-mer Masks
// =================================================================================================

/**
 * @brief A set of spaced k-mer masks of the same span, prepared in the forms that the algorithms
 * of SpacedKmerExtractor and for_each_spaced_kmer() take.
 *
 * The set is checked and prepared once, on construction, so that the extraction can use it
 * as it is, for as many sequences as needed:
 *
 *     SpacedKmerMasks<Layout::kMSB> const masks({
 *         "1101101101101101101101101101101",
 *         "1011011011011011011011011011011"
 *     });
 *
 * get() returns the masks in one of the forms: BitExtractMask, the masks as given, for PEXT;
 * BitExtractKernelPext, where BMI2 is compiled; BitExtractButterflyMask; and
 * BitExtractKernelButterfly of the SIMD archs of the algorithms. For other forms, see the
 * functions of spaced_kmer_extract/mask.hpp.
 *
 * @tparam L Layout of the spaced k-mers.
 * @tparam MaxMasks The largest number of masks of the set. The extraction compiles its loops for
 *         each number of masks up to it, so that a smaller `MaxMasks`, such as the number of masks
 *         that are used, shortens the compile time.
 */
template <Layout L, std::size_t MaxMasks = kDefaultMaxMasks>
class SpacedKmerMasks
{
    static_assert(MaxMasks >= 1, "SpacedKmerMasks needs MaxMasks >= 1.");

    using Forms = typename SpacedKmerMaskForms<L, MaxMasks>::type;

    template <typename M>
    using MaskForms = typename SpacedKmerMaskForms<L, MaxMasks>::template MaskForms<M>;

public:

    using mask_type = SpacedKmerMask<L, BitExtractMask>;

    static constexpr Layout layout = L;
    static constexpr std::size_t max_masks = MaxMasks;

    // -------------------------------------------------------------------------
    //     Constructors
    // -------------------------------------------------------------------------

    /**
     * @brief Prepare a set of masks, given as strings of `0` and `1`, see make_spaced_kmer_mask().
     *
     * @throws std::invalid_argument if the set is empty, has more than `MaxMasks` masks,
     * a mask is invalid, or the masks differ in span.
     */
    explicit SpacedKmerMasks(std::vector<std::string> const& masks)
        : SpacedKmerMasks(make_spaced_kmer_masks<L>(masks))
    {}

    /**
     * @brief Prepare a set of masks, see make_spaced_kmer_masks().
     */
    explicit SpacedKmerMasks(std::vector<mask_type> const& masks)
        : size_(masks.size())
    {
        check_(masks);
        std::apply([&](auto&... forms) { (prepare_(masks, forms), ...); }, forms_);
    }

    // -------------------------------------------------------------------------
    //     Access
    // -------------------------------------------------------------------------

    /**
     * @brief The number of masks.
     */
    std::size_t size() const noexcept
    {
        return size_;
    }

    /**
     * @brief The masks in the form `M`, such as BitExtractButterflyMask.
     */
    template <typename M>
    std::span<SpacedKmerMask<L, M> const> get() const noexcept
    {
        static_assert(
            has_form_<M>(),
            "SpacedKmerMasks does not hold the masks in this form; make them with the functions "
            "of spaced_kmer_extract/mask.hpp, such as make_spaced_kmer_kernels()."
        );
        return { std::get<MaskForms<M>>(forms_).data(), size_ };
    }

private:

    template <typename M>
    static constexpr bool has_form_()
    {
        return []<typename... F>(std::tuple<F...> const*) {
            return (std::same_as<F, MaskForms<M>> || ...);
        }(static_cast<Forms const*>(nullptr));
    }

    static void check_(std::vector<mask_type> const& masks)
    {
        if (masks.empty()) {
            throw std::invalid_argument("SpacedKmerMasks: no masks given");
        }
        if (masks.size() > MaxMasks) {
            throw std::invalid_argument(
                "SpacedKmerMasks: " + std::to_string(masks.size()) + " masks given, but the "
                "compile-time maximum is " + std::to_string(MaxMasks) + ". Raise the template "
                "parameter MaxMasks of SpacedKmerMasks and SpacedKmerExtractor, or its default, "
                "the CMake option or macro FISK_DEFAULT_MAX_MASKS."
            );
        }
        for (auto const& mask : masks) {
            if (!is_valid_spaced_kmer_mask(mask)) {
                throw std::invalid_argument("SpacedKmerMasks: invalid mask");
            }
            if (mask.span_k != masks.front().span_k) {
                throw std::invalid_argument("SpacedKmerMasks: masks differ in span");
            }
        }
    }

    // Fill the `forms` of one type with the `masks` in that form.
    template <typename M>
    static void prepare_(
        std::vector<mask_type> const& masks, std::array<SpacedKmerMask<L, M>, MaxMasks>& forms
    ) {
        for (std::size_t i = 0; i < masks.size(); ++i) {
            if constexpr (std::same_as<M, BitExtractMask>) {
                forms[i] = masks[i];
            } else if constexpr (std::same_as<M, BitExtractButterflyMask>) {
                forms[i] = make_spaced_kmer_butterfly_mask(masks[i]);
            } else {
                forms[i] = make_spaced_kmer_kernel<M>(masks[i]);
            }
        }
    }

    std::size_t size_;
    Forms forms_;
};

} // namespace fisk
