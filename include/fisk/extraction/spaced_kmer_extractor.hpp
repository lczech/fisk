#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/extraction/estimate.hpp"
#include "fisk/extraction/extractor_plan.hpp"
#include "fisk/extraction/spaced_kmer_estimate.hpp"
#include "fisk/extraction/spaced_kmer_mask_forms.hpp"
#include "fisk/extraction/tune_options.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Extractor Candidates
// =================================================================================================

// The number of nucleotides after the start of the last spaced k-mer of a piece of a sequence that
// it reaches: its span - 1, which all masks of a set share, see ExtractorPlan.
template <Layout L, std::size_t MaxMasks, std::size_t MaxSpan>
inline std::size_t extractor_overlap_(
    SpacedKmerMaskForms<L, MaxMasks, MaxSpan> const& mask_forms
) noexcept
{
    return mask_forms.template get<BitExtractButterflyMask>().front().span_k - 1;
}

// The candidates of a SpacedKmerExtractor, see spaced_kmer_estimate.hpp: PEXT, where compiled, and
// the butterfly bit extraction, in the scalar loops and as kernels of the SIMD loops of the
// `Simds`, a TypeList, see ExtractorSimdArchs, each for up to `MaxMasks` masks; see
// SpacedKmerEstimate for the estimate among them.
template <
    InputFormat I, InputCheck C, Encoding E, Layout L, std::size_t MaxMasks,
    typename Simds = ExtractorSimdArchs
>
struct SpacedKmerExtractorCandidates;

template <
    InputFormat I, InputCheck C, Encoding E, Layout L, std::size_t MaxMasks, typename... Simd
>
struct SpacedKmerExtractorCandidates<I, C, E, L, MaxMasks, TypeList<Simd...>>
{
    using Ascii = TypeList<
        #if defined(FISK_HAS_BMI2)
        SpacedKmerCandidateAsciiChunkedScalar<C, E, L, MaxMasks, BitExtractMask>,
        #endif
        SpacedKmerCandidateAsciiChunkedScalar<C, E, L, MaxMasks, BitExtractButterflyMask>,
        SpacedKmerCandidateAsciiChunkedSimd<C, E, L, MaxMasks, BitExtractKernelButterfly<Simd>>...
    >;

    using Packed = TypeList<
        #if defined(FISK_HAS_BMI2)
        SpacedKmerCandidatePackedBytewiseScalar<C, E, L, MaxMasks, BitExtractMask>,
        SpacedKmerCandidatePackedBytewiseSimd<C, E, L, MaxMasks, BitExtractKernelPext>,
        #endif
        SpacedKmerCandidatePackedBytewiseScalar<C, E, L, MaxMasks, BitExtractButterflyMask>,
        SpacedKmerCandidatePackedBytewiseSimd<C, E, L, MaxMasks, BitExtractKernelButterfly<Simd>>...
    >;

    using type = std::conditional_t<I == InputFormat::kAscii, Ascii, Packed>;
};

// =================================================================================================
//     Spaced K-mer Extractor
// =================================================================================================

/**
 * @brief Extract all spaced k-mers of sequences with the algorithm that is fastest on this
 * hardware.
 *
 * As KmerExtractor, for a set of spaced k-mer masks of the same span instead of a k-mer size:
 *
 *     using Extractor = SpacedKmerExtractor<
 *         InputFormat::kAscii,
 *         InputCheck::kChecked,
 *         Encoding::kACGT,
 *         Layout::kMSB
 *     >;
 *     Extractor ex({
 *         "1101101101101101101101101101101",
 *         "1011011011011011011011011011011"
 *     });
 *     ex.for_each(seq, [&](std::size_t pos, std::size_t mask_idx, SpacedKmerAcgtMsb kmer) {
 *         ...
 *     });
 *
 * As KmerExtractor, the extractor tunes in situ in its first calls of for_each(), see tune(), and
 * is meant to be made once and reused, or copied, for all input.
 *
 * The algorithms are the scalar and SIMD loops of the spaced k-mer extraction, with PEXT, where
 * compiled, and with the butterfly bit extraction.
 *
 * The callback takes single spaced k-mers, as `callback(pos, mask_idx, spaced_kmer)`. It can in
 * addition take the vectors of the SIMD algorithms, as
 * `callback(pos, mask_idx, spaced_kmers, valid_count)` when unchecked, or
 * `callback(pos, mask_idx, spaced_kmers, valid_lanes)` when checked, see SpacedKmerVectorCallback
 * and SpacedKmerVectorValidLanesCallback. The SIMD algorithms then hand it their vectors, and else
 * their spaced k-mers one by one.
 *
 * The spaced k-mers of each mask are handed out in order of their positions; the order between
 * masks differs between the algorithms: the scalar loops hand out the spaced k-mers of all masks
 * at one position before going on to the next position, while the SIMD loops hand out those of
 * one mask for a whole vector of positions, then those of the next mask for the same positions.
 *
 * for_each() is const, and can run on several threads at once; tune(), estimate() and
 * set_algorithm() cannot. ASCII input is packed chunk by chunk into a buffer per thread.
 *
 * The callback is compiled for each algorithm and each number of masks up to `MaxMasks`. A smaller
 * `MaxMasks`, such as the number of masks that are used, hence shortens the compile time.
 *
 * @tparam I InputFormat: ASCII as `std::string_view`, or a PackedSequence of E and L.
 * @tparam C InputCheck: kChecked skips the spaced k-mers with invalid characters at the kept
 *           positions, and takes a PackedSequenceWithValidity for packed input; kUnchecked trusts
 *           the input to be valid.
 * @tparam E Encoding of the spaced k-mers.
 * @tparam L Layout of the spaced k-mers.
 * @tparam MaxMasks The largest number of masks that the extractor takes.
 * @tparam MaxSpan The largest span of the masks that the extractor takes, see SpacedKmerMaskForms.
 */
template <
    InputFormat I, InputCheck C, Encoding E, Layout L,
    std::size_t MaxMasks = kDefaultMaxMasks, std::size_t MaxSpan = kDefaultMaxSpan
>
class SpacedKmerExtractor
{
public:

    using sequence_type = ExtractorSequence<I, C, E, L>;
    using kmer_type = SpacedKmer<E, L>;
    using mask_type = SpacedKmerMask<L, BitExtractMask, MaxSpan>;
    using mask_forms_type = SpacedKmerMaskForms<L, MaxMasks, MaxSpan>;

    static constexpr InputFormat input_format = I;
    static constexpr InputCheck input_check = C;
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t max_masks = MaxMasks;
    static constexpr std::size_t max_span = MaxSpan;

    // -------------------------------------------------------------------------
    //     Constructors
    // -------------------------------------------------------------------------

    /**
     * @brief Set up the extractor for a set of masks, given as strings of `0` and `1`, see
     * make_spaced_kmer_mask(), tuning in situ, see tune().
     *
     * @throws std::invalid_argument if the set is empty, has more than `MaxMasks` masks,
     * a mask is invalid or of a span above `MaxSpan`, or the masks differ in span.
     */
    explicit SpacedKmerExtractor(std::vector<std::string> const& masks)
        : SpacedKmerExtractor(mask_forms_type(masks))
    {}

    /**
     * @brief Set up the extractor for a set of masks, such as a std::vector of them, see
     * make_spaced_kmer_masks().
     */
    explicit SpacedKmerExtractor(std::span<mask_type const> masks)
        : SpacedKmerExtractor(mask_forms_type(masks))
    {}

    /**
     * @brief Set up the extractor for a set of masks prepared in all forms, of the Layout,
     * MaxMasks and MaxSpan of the extractor.
     */
    template <Layout MasksLayout, std::size_t MasksMaxMasks, std::size_t MasksMaxSpan>
    explicit SpacedKmerExtractor(
        SpacedKmerMaskForms<MasksLayout, MasksMaxMasks, MasksMaxSpan> mask_forms
    )
        : plan_(own_mask_forms_(std::move(mask_forms)))
    {}

    /**
     * @brief The masks, in all forms.
     */
    mask_forms_type const& mask_forms() const noexcept
    {
        return plan_.settings();
    }

    // -------------------------------------------------------------------------
    //     Extraction
    // -------------------------------------------------------------------------

    /**
     * @brief Call `callback` with each spaced k-mer of `seq`, with the chosen algorithm.
     */
    template <typename Callback>
        requires SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
    FISK_ALWAYS_INLINE void for_each(sequence_type const& seq, Callback&& callback) const
    {
        plan_.for_each(seq, callback);
    }

    // -------------------------------------------------------------------------
    //     Tuning
    // -------------------------------------------------------------------------

    /**
     * @brief Tune again, in situ, with these options, see KmerExtractor::tune().
     */
    void tune(TuneOptions const& options = {})
    {
        plan_.tune(options);
    }

    /**
     * @brief The name of the chosen algorithm: the name of the spaced k-mer extraction function,
     * followed by the bit extraction, such as
     * "for_each_spaced_kmer_ascii_checked_chunked_scalar_pext" for
     * for_each_spaced_kmer_ascii_checked_chunked_scalar() with bit_extract_pext. While the in-situ
     * tuning is pending, and after estimate(), as for KmerExtractor::algorithm_name().
     */
    std::string algorithm_name() const
    {
        return plan_.algorithm_name();
    }

    /**
     * @brief The names of the algorithms that set_algorithm() accepts in this build on this CPU.
     */
    std::vector<std::string> algorithm_names() const
    {
        return plan_.algorithm_names();
    }

    /**
     * @brief Use the algorithm of this name, see algorithm_name().
     *
     * @throws std::invalid_argument if there is no such algorithm in this build or on this CPU.
     */
    void set_algorithm(std::string_view name)
    {
        plan_.set_algorithm(name);
    }

    /**
     * @brief Use the estimate, without tuning, see KmerExtractor::estimate().
     */
    void estimate()
    {
        plan_.estimate();
    }

    /**
     * @brief Whether the in-situ tuning has not chosen its algorithm yet, see
     * KmerExtractor::tuning_pending().
     */
    bool tuning_pending() const noexcept
    {
        return plan_.tuning_pending();
    }

private:

    // The `mask_forms`, which have to be of the Layout, MaxMasks and MaxSpan of the extractor. The
    // constructor takes others as well, so that they fail here, with a message, instead of leaving
    // no constructor to match.
    template <Layout MasksLayout, std::size_t MasksMaxMasks, std::size_t MasksMaxSpan>
    static mask_forms_type own_mask_forms_(
        SpacedKmerMaskForms<MasksLayout, MasksMaxMasks, MasksMaxSpan> mask_forms
    ) {
        static_assert(
            MasksLayout == L,
            "The masks are for another Layout than the extractor. Make them with the Layout, "
            "MaxMasks and MaxSpan of the extractor, as SpacedKmerMaskForms<L, MaxMasks, MaxSpan>."
        );
        static_assert(
            MasksMaxMasks == MaxMasks,
            "The masks are for another MaxMasks than the extractor. Make them with the Layout, "
            "MaxMasks and MaxSpan of the extractor, as SpacedKmerMaskForms<L, MaxMasks, MaxSpan>."
        );
        static_assert(
            MasksMaxSpan == MaxSpan,
            "The masks are for another MaxSpan than the extractor. Make them with the Layout, "
            "MaxMasks and MaxSpan of the extractor, as SpacedKmerMaskForms<L, MaxMasks, MaxSpan>."
        );
        if constexpr (MasksLayout == L && MasksMaxMasks == MaxMasks && MasksMaxSpan == MaxSpan) {
            return mask_forms;
        } else {
            // Not reached, as the checks above fail to compile.
            throw std::logic_error(
                "SpacedKmerExtractor: masks of another Layout, MaxMasks or MaxSpan"
            );
        }
    }

    using Plan = ExtractorPlan<
        mask_forms_type, sequence_type,
        typename SpacedKmerExtractorCandidates<I, C, E, L, MaxMasks>::type,
        typename SpacedKmerEstimate<I, C, E, L, MaxMasks>::type
    >;
    Plan plan_;
};

} // namespace fisk
