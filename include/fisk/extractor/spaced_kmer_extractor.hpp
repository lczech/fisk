#pragma once

#include <concepts>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/util/cpu_runtime.hpp"
#include "fisk/util/intrinsics.hpp"
#include "fisk/util/static_span.hpp"
#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/extractor/common.hpp"
#include "fisk/extractor/internal.hpp"
#include "fisk/extractor/spaced_kmer_mask_forms.hpp"
#include "fisk/spaced_kmer_extract/ascii_chunked_scalar.hpp"
#include "fisk/spaced_kmer_extract/ascii_chunked_simd.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Extractor Candidates
// =================================================================================================

// The candidate algorithms of SpacedKmerExtractor, see ExtractorPlan for their shape. Each calls
// one of the spaced k-mer extraction functions, with the masks dispatched to a span of fixed size,
// see dispatch_static_span(), and is named after it, followed by the bit extraction.

// -------------------------------------------------------------------------------------------------
//     Bit Extraction
// -------------------------------------------------------------------------------------------------

// The bit extraction of the spaced k-mer masks of the form `BitExtractForm`, a bit extraction mask
// for the scalar loops, or a kernel for the SIMD loops: its name and availability. The masks in
// these forms are in SpacedKmerMaskForms.

template <typename BitExtractForm>
struct is_bit_extract_kernel_butterfly_ : std::false_type {};

template <typename Simd>
struct is_bit_extract_kernel_butterfly_<BitExtractKernelButterfly<Simd>> : std::true_type {};

template <typename BitExtractForm>
inline std::string spaced_kmer_extractor_bit_extract_name_()
{
    if constexpr (is_bit_extract_kernel_butterfly_<BitExtractForm>::value) {
        return "butterfly_" + simd_arch_name_<typename BitExtractForm::simd_arch>();
    } else if constexpr (std::same_as<BitExtractForm, BitExtractButterflyMask>) {
        return "butterfly";
    }
    #if defined(FISK_HAS_BMI2)
    else if constexpr (
        std::same_as<BitExtractForm, BitExtractMask> ||
        std::same_as<BitExtractForm, BitExtractKernelPext>
    ) {
        return "pext";
    }
    #endif
    else {
        static_assert(dependent_false_v<sizeof(BitExtractForm)>, "Unhandled bit extraction");
    }
}

template <typename BitExtractForm>
inline bool spaced_kmer_extractor_bit_extract_available_()
{
    if constexpr (std::same_as<BitExtractForm, BitExtractButterflyMask>) {
        return true;
    } else if constexpr (std::same_as<BitExtractForm, BitExtractMask>) {
        return bmi2_enabled();
    } else if constexpr (std::same_as<typename BitExtractForm::simd_arch, SimdScalar>) {
        // BitExtractKernelPext, the only kernel of the scalar arch here.
        return bmi2_enabled();
    } else {
        return simd_arch_enabled_<typename BitExtractForm::simd_arch>();
    }
}

// Whether the estimate may run the bit extraction of the masks of the form `BitExtractForm`, see
// extractor_candidate_preferred_(): PEXT only where it is likely fast, as it is slow on some CPUs,
// see pext_likely_fast(); tune() measures it wherever it is available.
template <typename BitExtractForm>
inline bool spaced_kmer_extractor_bit_extract_preferred_()
{
    #if defined(FISK_HAS_BMI2)
    if constexpr (
        std::same_as<BitExtractForm, BitExtractMask> ||
        std::same_as<BitExtractForm, BitExtractKernelPext>
    ) {
        return pext_likely_fast();
    }
    #endif
    return spaced_kmer_extractor_bit_extract_available_<BitExtractForm>();
}

// The scalar bit extraction of the masks of the form `BitExtractForm`.
template <typename BitExtractForm>
FISK_ALWAYS_INLINE inline auto spaced_kmer_extractor_bit_extract_()
{
    if constexpr (std::same_as<BitExtractForm, BitExtractButterflyMask>) {
        return bit_extract_butterfly;
    }
    #if defined(FISK_HAS_BMI2)
    else if constexpr (std::same_as<BitExtractForm, BitExtractMask>) {
        return bit_extract_pext;
    }
    #endif
    else {
        static_assert(dependent_false_v<sizeof(BitExtractForm)>, "Unhandled scalar bit extraction");
    }
}

// -------------------------------------------------------------------------------------------------
//     Callbacks
// -------------------------------------------------------------------------------------------------

// The callback that a SIMD candidate hands its vectors to: the vectors as they are, if `callback`
// takes them, else their valid spaced k-mers one by one.
template <typename Callback>
FISK_ALWAYS_INLINE inline auto spaced_kmer_extractor_vector_callback_(Callback& callback)
{
    return [&callback](
        std::size_t pos, std::size_t mask_idx, auto kmers, auto valid
    ) FISK_ALWAYS_INLINE_LAMBDA {
        using Kmers = decltype(kmers);
        using Valid = decltype(valid);
        if constexpr (std::invocable<Callback&, std::size_t, std::size_t, Kmers, Valid>) {
            callback(pos, mask_idx, kmers, valid);
        } else {
            for_each_valid_lane(
                pos, kmers, valid,
                [&](std::size_t lane_pos, auto const& kmer) FISK_ALWAYS_INLINE_LAMBDA {
                    callback(lane_pos, mask_idx, kmer);
                }
            );
        }
    };
}

// Whether `Callback` takes the vectors of spaced k-mers of `Simd`, as handed out with a count of
// valid lanes when unchecked, and with the valid lanes when checked.
template <typename Callback, InputCheck C, Encoding E, Layout L, typename Simd>
inline constexpr bool spaced_kmer_extractor_takes_vectors_ = (C == InputCheck::kChecked)
    ? SpacedKmerVectorValidLanesCallback<Callback&, E, L, Simd, false>
    : SpacedKmerVectorCallback<Callback&, E, L, Simd, false>;

// -------------------------------------------------------------------------------------------------
//     ASCII
// -------------------------------------------------------------------------------------------------

// The run() of each candidate below, and the lambda that it hands to dispatch_static_span(), are
// forced inline, so that the callback does not reach a function out of line, and the whole
// extraction inlines into ExtractorPlan::for_each() (extractor/internal.hpp). A callback that
// reaches a function out of line keeps its state in memory in all of for_each(), for all
// candidates.

template <InputCheck C, Encoding E, Layout L, std::size_t MaxMasks, typename BitExtractForm>
struct SpacedKmerCandidateAsciiChunkedScalar
{
    static std::string name()
    {
        return "for_each_spaced_kmer_ascii_" + input_check_name_(C) + "_chunked_scalar_" +
            spaced_kmer_extractor_bit_extract_name_<BitExtractForm>();
    }
    static bool available()
    {
        return spaced_kmer_extractor_bit_extract_available_<BitExtractForm>();
    }
    static bool preferred()
    {
        return spaced_kmer_extractor_bit_extract_preferred_<BitExtractForm>();
    }

    template <typename Callback>
    static constexpr bool takes_vectors = false;

    template <std::size_t MaxSpan, typename Callback>
    FISK_ALWAYS_INLINE void run(
        SpacedKmerMaskForms<L, MaxMasks, MaxSpan> const& mask_forms, std::string_view seq,
        Callback& callback
    ) const
    {
        ExtractorWordEncoder<E, L> const encoder{};
        auto const bit_extract = spaced_kmer_extractor_bit_extract_<BitExtractForm>();
        auto const& masks = mask_forms.template get<BitExtractForm>();
        dispatch_static_span<MaxMasks>(masks, [&](auto span) FISK_ALWAYS_INLINE_LAMBDA {
            if constexpr (C == InputCheck::kChecked) {
                for_each_spaced_kmer_ascii_checked_chunked_scalar(
                    seq, span, encoder, bit_extract, callback
                );
            } else {
                for_each_spaced_kmer_ascii_unchecked_chunked_scalar(
                    seq, span, encoder, bit_extract, callback
                );
            }
        });
    }
};

template <InputCheck C, Encoding E, Layout L, std::size_t MaxMasks, typename Kernel>
struct SpacedKmerCandidateAsciiChunkedSimd
{
    static std::string name()
    {
        return "for_each_spaced_kmer_ascii_" + input_check_name_(C) + "_chunked_simd_" +
            spaced_kmer_extractor_bit_extract_name_<Kernel>();
    }
    static bool available() { return spaced_kmer_extractor_bit_extract_available_<Kernel>(); }
    static bool preferred() { return spaced_kmer_extractor_bit_extract_preferred_<Kernel>(); }

    template <typename Callback>
    static constexpr bool takes_vectors = spaced_kmer_extractor_takes_vectors_<
        Callback, C, E, L, typename Kernel::simd_arch
    >;

    template <std::size_t MaxSpan, typename Callback>
    FISK_ALWAYS_INLINE void run(
        SpacedKmerMaskForms<L, MaxMasks, MaxSpan> const& mask_forms, std::string_view seq,
        Callback& callback
    ) const
    {
        ExtractorWordEncoder<E, L> const encoder{};
        auto vector_callback = spaced_kmer_extractor_vector_callback_(callback);
        auto const& masks = mask_forms.template get<Kernel>();
        dispatch_static_span<MaxMasks>(masks, [&](auto span) FISK_ALWAYS_INLINE_LAMBDA {
            if constexpr (C == InputCheck::kChecked) {
                for_each_spaced_kmer_ascii_checked_chunked_simd(
                    seq, span, encoder, vector_callback
                );
            } else {
                for_each_spaced_kmer_ascii_unchecked_chunked_simd(
                    seq, span, encoder, vector_callback
                );
            }
        });
    }
};

// -------------------------------------------------------------------------------------------------
//     Packed
// -------------------------------------------------------------------------------------------------

template <InputCheck C, Encoding E, Layout L, std::size_t MaxMasks, typename BitExtractForm>
struct SpacedKmerCandidatePackedBytewiseScalar
{
    static std::string name()
    {
        return "for_each_spaced_kmer_packed_" + input_check_name_(C) + "_bytewise_scalar_" +
            spaced_kmer_extractor_bit_extract_name_<BitExtractForm>();
    }
    static bool available()
    {
        return spaced_kmer_extractor_bit_extract_available_<BitExtractForm>();
    }
    static bool preferred()
    {
        return spaced_kmer_extractor_bit_extract_preferred_<BitExtractForm>();
    }

    template <typename Callback>
    static constexpr bool takes_vectors = false;

    template <std::size_t MaxSpan, typename Seq, typename Callback>
    FISK_ALWAYS_INLINE void run(
        SpacedKmerMaskForms<L, MaxMasks, MaxSpan> const& mask_forms, Seq const& seq,
        Callback& callback
    ) const
    {
        auto const bit_extract = spaced_kmer_extractor_bit_extract_<BitExtractForm>();
        auto const& masks = mask_forms.template get<BitExtractForm>();
        dispatch_static_span<MaxMasks>(masks, [&](auto span) FISK_ALWAYS_INLINE_LAMBDA {
            if constexpr (C == InputCheck::kChecked) {
                for_each_spaced_kmer_packed_checked_bytewise_scalar(
                    seq, span, bit_extract, callback
                );
            } else {
                for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
                    seq, span, bit_extract, callback
                );
            }
        });
    }
};

template <InputCheck C, Encoding E, Layout L, std::size_t MaxMasks, typename Kernel>
struct SpacedKmerCandidatePackedBytewiseSimd
{
    static std::string name()
    {
        return "for_each_spaced_kmer_packed_" + input_check_name_(C) + "_bytewise_simd_" +
            spaced_kmer_extractor_bit_extract_name_<Kernel>();
    }
    static bool available() { return spaced_kmer_extractor_bit_extract_available_<Kernel>(); }
    static bool preferred() { return spaced_kmer_extractor_bit_extract_preferred_<Kernel>(); }

    template <typename Callback>
    static constexpr bool takes_vectors = spaced_kmer_extractor_takes_vectors_<
        Callback, C, E, L, typename Kernel::simd_arch
    >;

    template <std::size_t MaxSpan, typename Seq, typename Callback>
    FISK_ALWAYS_INLINE void run(
        SpacedKmerMaskForms<L, MaxMasks, MaxSpan> const& mask_forms, Seq const& seq,
        Callback& callback
    ) const
    {
        auto vector_callback = spaced_kmer_extractor_vector_callback_(callback);
        auto const& masks = mask_forms.template get<Kernel>();
        dispatch_static_span<MaxMasks>(masks, [&](auto span) FISK_ALWAYS_INLINE_LAMBDA {
            if constexpr (C == InputCheck::kChecked) {
                for_each_spaced_kmer_packed_checked_bytewise_simd(seq, span, vector_callback);
            } else {
                for_each_spaced_kmer_packed_unchecked_bytewise_simd(seq, span, vector_callback);
            }
        });
    }
};

// -------------------------------------------------------------------------------------------------
//     Candidate Lists
// -------------------------------------------------------------------------------------------------

// The candidates of a SpacedKmerExtractor: PEXT, where compiled, and the butterfly bit
// extraction, in the scalar loops and as kernels of the SIMD loops of the `Simds`, a TypeList, each
// for up to `MaxMasks` masks; see SpacedKmerExtractorPreference for the estimate among them.
template <
    InputFormat I, InputCheck C, Encoding E, Layout L, std::size_t MaxMasks,
    typename Simds = CandidateSimdArchs
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

// The estimate of a SpacedKmerExtractor and of for_each_spaced_kmer(), see ExtractorPreference:
// the scalar loop with PEXT, where it is likely fast, else the SIMD loop with the butterfly kernel
// of the `Simds`, a TypeList, which are AVX2, SSE2, or Neon, else the scalar loop with the
// butterfly. These are the fastest for callbacks of single spaced k-mers as well as for those that
// take vectors.
template <
    InputFormat I, InputCheck C, Encoding E, Layout L, std::size_t MaxMasks,
    typename Simds = PreferredSpacedSimdArchs
>
struct SpacedKmerExtractorPreference;

template <
    InputFormat I, InputCheck C, Encoding E, Layout L, std::size_t MaxMasks, typename... Simd
>
struct SpacedKmerExtractorPreference<I, C, E, L, MaxMasks, TypeList<Simd...>>
{
    using Ascii = TypeList<
        #if defined(FISK_HAS_BMI2)
        SpacedKmerCandidateAsciiChunkedScalar<C, E, L, MaxMasks, BitExtractMask>,
        #endif
        SpacedKmerCandidateAsciiChunkedSimd<C, E, L, MaxMasks, BitExtractKernelButterfly<Simd>>...,
        SpacedKmerCandidateAsciiChunkedScalar<C, E, L, MaxMasks, BitExtractButterflyMask>
    >;

    using Packed = TypeList<
        #if defined(FISK_HAS_BMI2)
        SpacedKmerCandidatePackedBytewiseScalar<C, E, L, MaxMasks, BitExtractMask>,
        #endif
        SpacedKmerCandidatePackedBytewiseSimd<
            C, E, L, MaxMasks, BitExtractKernelButterfly<Simd>
        >...,
        SpacedKmerCandidatePackedBytewiseScalar<C, E, L, MaxMasks, BitExtractButterflyMask>
    >;

    using List = std::conditional_t<I == InputFormat::kAscii, Ascii, Packed>;
    using type = ExtractorPreference<List, List>;
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
 *     ex.tune();
 *     ex.for_each(seq, [&](std::size_t pos, std::size_t mask_idx, SpacedKmerAcgtMsb kmer) {
 *         ...
 *     });
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
 * for_each() is const, and can run on several threads at once; tune() and set_algorithm() cannot.
 * ASCII input is packed chunk by chunk into a buffer per thread.
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
     * make_spaced_kmer_mask(), with the estimated algorithm.
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
     * @brief Choose the algorithm, by timing on random sequences, see KmerExtractor::tune().
     *
     * @throws std::runtime_error if two algorithms give different results.
     */
    void tune(TuneOptions const& options = {})
    {
        ExtractorTuneSink sink;
        tune(sink, options);
    }

    /**
     * @brief Choose the algorithm, by timing on a `sample`: one sequence, or a range of them.
     */
    template <typename Sample>
        requires ExtractorSample<Sample, sequence_type>
    void tune(Sample const& sample, TuneOptions const& options = {})
    {
        ExtractorTuneSink sink;
        tune(sample, sink, options);
    }

    /**
     * @brief Choose the algorithm, by timing with `callback` on random sequences.
     *
     * The callback is called with all spaced k-mers of the sequences, `options.repeats` times per
     * algorithm. Its side effects during tuning are up to the caller. All algorithms call the same
     * callback, so that a callback that takes vectors gets single spaced k-mers from some
     * algorithms and vectors from others.
     *
     * Pass the callback as a temporary, such as `tune(MyCallback{})`, or with std::move(), to
     * time the algorithms as for_each() runs them: for_each() inlines into its caller, where the
     * state of a local callback can stay in registers, and tune() then times a local of its own in
     * the same way. A named callback is called as it is, so that it sees all calls, but its state
     * stays in memory while it is timed, so that tune() may choose another algorithm than the
     * one that for_each() runs fastest.
     */
    template <typename Callback>
        requires SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
    void tune(Callback&& callback, TuneOptions const& options = {})
    {
        if (options.level == TuneLevel::kEstimate) {
            plan_.estimate();
        } else {
            plan_.tune(
                make_tune_samples_<I, C, E, L>(options), options, std::forward<Callback>(callback)
            );
        }
    }

    /**
     * @brief Choose the algorithm, by timing with `callback` on a `sample`.
     */
    template <typename Sample, typename Callback>
        requires ExtractorSample<Sample, sequence_type> &&
                 SpacedKmerCallback<std::remove_reference_t<Callback>&, E, L, false>
    void tune(Sample const& sample, Callback&& callback, TuneOptions const& options = {})
    {
        plan_.tune(sample, options, std::forward<Callback>(callback));
    }

    /**
     * @brief The name of the chosen algorithm: the name of the spaced k-mer extraction function,
     * followed by the bit extraction, such as
     * "for_each_spaced_kmer_ascii_checked_chunked_scalar_pext" for
     * for_each_spaced_kmer_ascii_checked_chunked_scalar() with bit_extract_pext.
     */
    std::string algorithm_name() const
    {
        return plan_.template algorithm_name<ExtractorTuneSink>();
    }

    /**
     * @brief The name of the algorithm that for_each() runs with a callback of this type.
     */
    template <typename Callback>
    std::string algorithm_name(Callback&&) const
    {
        return plan_.template algorithm_name<std::remove_reference_t<Callback>>();
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
        typename SpacedKmerExtractorPreference<I, C, E, L, MaxMasks>::type
    >;
    Plan plan_;
};

} // namespace fisk
