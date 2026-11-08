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
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/extractor/common.hpp"
#include "fisk/extractor/internal.hpp"
#include "fisk/kmer_extract/ascii_chunked_scalar.hpp"
#include "fisk/kmer_extract/ascii_chunked_simd.hpp"
#include "fisk/kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/kmer_extract/packed_bytewise_simd.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extractor Candidates
// =================================================================================================

// The candidate algorithms of KmerExtractor, see ExtractorPlan for their shape. Each calls one of
// the k-mer extraction functions, and is named after it, followed by the SIMD arch.

/**
 * @brief The settings of a KmerExtractor, which its candidates run with.
 */
struct KmerExtractorSettings
{
    std::size_t k = 0;
};

// The settings for k-mers of size `k`, for KmerExtractor and for_each_kmer().
inline KmerExtractorSettings make_kmer_extractor_settings_(std::size_t k)
{
    if (k == 0 || k > 32) {
        throw std::invalid_argument(
            "k-mer extraction: k = " + std::to_string(k) + " not in [1, 32]"
        );
    }
    return {k};
}

// The callback that a SIMD candidate hands its vectors to: the vectors as they are, if `callback`
// takes them, else their valid k-mers one by one.
template <typename Callback>
FISK_ALWAYS_INLINE inline auto kmer_extractor_vector_callback_(Callback& callback)
{
    return [&callback](std::size_t pos, auto kmers, auto valid) FISK_ALWAYS_INLINE_LAMBDA {
        if constexpr (std::invocable<Callback&, std::size_t, decltype(kmers), decltype(valid)>) {
            callback(pos, kmers, valid);
        } else {
            for_each_valid_lane(pos, kmers, valid, callback);
        }
    };
}

// Whether `Callback` takes the vectors of k-mers of `Simd`, as handed out with a count of valid
// lanes when unchecked, and with the valid lanes when checked.
template <typename Callback, InputCheck C, Encoding E, Layout L, typename Simd>
inline constexpr bool kmer_extractor_takes_vectors_ = (C == InputCheck::kChecked)
    ? KmerVectorValidLanesCallback<Callback&, E, L, Simd>
    : KmerVectorCallback<Callback&, E, L, Simd>;

// -------------------------------------------------------------------------------------------------
//     ASCII
// -------------------------------------------------------------------------------------------------

template <InputCheck C, Encoding E, Layout L>
struct KmerCandidateAsciiChunkedScalar
{
    static std::string name()
    {
        return "for_each_kmer_ascii_" + input_check_name_(C) + "_chunked_scalar";
    }
    static bool available() { return true; }

    template <typename Callback>
    static constexpr bool takes_vectors = false;

    template <typename Callback>
    FISK_ALWAYS_INLINE void run(
        KmerExtractorSettings const& settings, std::string_view seq, Callback& callback
    ) const
    {
        ExtractorWordEncoder<E, L> const encoder{};
        if constexpr (C == InputCheck::kChecked) {
            for_each_kmer_ascii_checked_chunked_scalar(seq, settings.k, encoder, callback);
        } else {
            for_each_kmer_ascii_unchecked_chunked_scalar(seq, settings.k, encoder, callback);
        }
    }
};

template <InputCheck C, Encoding E, Layout L, typename Simd>
struct KmerCandidateAsciiChunkedSimd
{
    static std::string name()
    {
        return "for_each_kmer_ascii_" + input_check_name_(C) + "_chunked_simd_" +
            simd_arch_name_<Simd>();
    }
    static bool available() { return simd_arch_enabled_<Simd>(); }

    template <typename Callback>
    static constexpr bool takes_vectors = kmer_extractor_takes_vectors_<Callback, C, E, L, Simd>;

    template <typename Callback>
    FISK_ALWAYS_INLINE void run(
        KmerExtractorSettings const& settings, std::string_view seq, Callback& callback
    ) const
    {
        ExtractorWordEncoder<E, L> const encoder{};
        auto vector_callback = kmer_extractor_vector_callback_(callback);
        if constexpr (C == InputCheck::kChecked) {
            for_each_kmer_ascii_checked_chunked_simd<Simd>(
                seq, settings.k, encoder, vector_callback
            );
        } else {
            for_each_kmer_ascii_unchecked_chunked_simd<Simd>(
                seq, settings.k, encoder, vector_callback
            );
        }
    }
};

// -------------------------------------------------------------------------------------------------
//     Packed
// -------------------------------------------------------------------------------------------------

template <InputCheck C, Encoding E, Layout L>
struct KmerCandidatePackedBytewiseScalar
{
    static std::string name()
    {
        return "for_each_kmer_packed_" + input_check_name_(C) + "_bytewise_scalar";
    }
    static bool available() { return true; }

    template <typename Callback>
    static constexpr bool takes_vectors = false;

    template <typename Seq, typename Callback>
    FISK_ALWAYS_INLINE void run(
        KmerExtractorSettings const& settings, Seq const& seq, Callback& callback
    ) const
    {
        if constexpr (C == InputCheck::kChecked) {
            for_each_kmer_packed_checked_bytewise_scalar(seq, settings.k, callback);
        } else {
            for_each_kmer_packed_unchecked_bytewise_scalar(seq, settings.k, callback);
        }
    }
};

template <InputCheck C, Encoding E, Layout L, typename Simd>
struct KmerCandidatePackedBytewiseSimd
{
    static std::string name()
    {
        return "for_each_kmer_packed_" + input_check_name_(C) + "_bytewise_simd_" +
            simd_arch_name_<Simd>();
    }
    static bool available() { return simd_arch_enabled_<Simd>(); }

    template <typename Callback>
    static constexpr bool takes_vectors = kmer_extractor_takes_vectors_<Callback, C, E, L, Simd>;

    template <typename Seq, typename Callback>
    FISK_ALWAYS_INLINE void run(
        KmerExtractorSettings const& settings, Seq const& seq, Callback& callback
    ) const
    {
        auto vector_callback = kmer_extractor_vector_callback_(callback);
        if constexpr (C == InputCheck::kChecked) {
            for_each_kmer_packed_checked_bytewise_simd<Simd>(seq, settings.k, vector_callback);
        } else {
            for_each_kmer_packed_unchecked_bytewise_simd<Simd>(seq, settings.k, vector_callback);
        }
    }
};

// -------------------------------------------------------------------------------------------------
//     Candidate Lists
// -------------------------------------------------------------------------------------------------

// The candidates of a KmerExtractor: the loops that are the fastest on some hardware, in the
// benchmarks, see KmerExtractorPreference for the estimate among them. The SIMD loops are
// those of the `Simds`, a TypeList.
template <
    InputFormat I, InputCheck C, Encoding E, Layout L, typename Simds = CandidateSimdArchs
>
struct KmerExtractorCandidates;

template <InputFormat I, InputCheck C, Encoding E, Layout L, typename... Simd>
struct KmerExtractorCandidates<I, C, E, L, TypeList<Simd...>>
{
    using Ascii = TypeList<
        KmerCandidateAsciiChunkedScalar<C, E, L>,
        KmerCandidateAsciiChunkedSimd<C, E, L, Simd>...
    >;

    using Packed = TypeList<
        KmerCandidatePackedBytewiseScalar<C, E, L>,
        KmerCandidatePackedBytewiseSimd<C, E, L, Simd>...
    >;

    using type = std::conditional_t<I == InputFormat::kAscii, Ascii, Packed>;
};

// The estimate of a KmerExtractor and of for_each_kmer(), see ExtractorPreference: for callbacks of
// single k-mers, the scalar loop, which is as fast as the SIMD loops with their lanes taken out one
// by one, or faster; for callbacks that take vectors, the SIMD loop of the `Simds`, a TypeList,
// which are AVX2 or Neon.
template <
    InputFormat I, InputCheck C, Encoding E, Layout L, typename Simds = PreferredSimdArchs
>
struct KmerExtractorPreference;

template <InputFormat I, InputCheck C, Encoding E, Layout L, typename... Simd>
struct KmerExtractorPreference<I, C, E, L, TypeList<Simd...>>
{
    using AsciiScalar = KmerCandidateAsciiChunkedScalar<C, E, L>;
    using PackedScalar = KmerCandidatePackedBytewiseScalar<C, E, L>;

    using Ascii = ExtractorPreference<
        TypeList<AsciiScalar>,
        TypeList<KmerCandidateAsciiChunkedSimd<C, E, L, Simd>..., AsciiScalar>
    >;

    using Packed = ExtractorPreference<
        TypeList<PackedScalar>,
        TypeList<KmerCandidatePackedBytewiseSimd<C, E, L, Simd>..., PackedScalar>
    >;

    using type = std::conditional_t<I == InputFormat::kAscii, Ascii, Packed>;
};

// =================================================================================================
//     K-mer Extractor
// =================================================================================================

/**
 * @brief Extract all k-mers of sequences with the algorithm that is fastest on this hardware.
 *
 * The extractor is set up once for an InputFormat (ASCII or packed), whether the input is checked
 * for invalid characters (InputCheck), the Encoding and Layout, and the k-mer size `k`. It then
 * chooses one of the k-mer extraction functions, such as
 * for_each_kmer_ascii_checked_chunked_scalar(), and for_each() runs that one:
 *
 *     using namespace fisk;
 *     using Extractor = KmerExtractor<
 *         InputFormat::kAscii,
 *         InputCheck::kChecked,
 *         Encoding::kACGT,
 *         Layout::kMSB
 *     >;
 *     Extractor ex(31);
 *     ex.tune();
 *     ex.for_each(seq, [&](std::size_t pos, KmerAcgtMsb kmer) { ... });
 *
 * The choice is an estimate from what this build and CPU support, or, after tune(), the fastest in
 * a measurement. algorithm_name() names it, and set_algorithm() sets it, for instance to reuse a
 * measurement.
 *
 * The callback takes single k-mers, as `callback(pos, kmer)` or `callback(kmer)`, see
 * KmerCallback. It can in addition take the vectors of k-mers of the SIMD algorithms, see
 * KmerVectorCallback and KmerVectorValidLanesCallback, for instance as a callback_overload, or a
 * struct with several `operator()`. The SIMD algorithms then hand it their vectors, and else their
 * k-mers one by one. The callback is a template parameter of for_each(), so that it can be inlined
 * into the loop. As each algorithm calls it, each callback type compiles all of them.
 *
 * The k-mers are handed out in order of their positions. for_each() is const, and can run on
 * several threads at once; tune() and set_algorithm() cannot. ASCII input is packed chunk by chunk
 * into a buffer per thread.
 *
 * @tparam I InputFormat: ASCII as `std::string_view`, or a PackedSequence of E and L.
 * @tparam C InputCheck: kChecked skips the k-mers with invalid characters, and takes a
 *           PackedSequenceWithValidity for packed input; kUnchecked trusts the input to be valid.
 * @tparam E Encoding of the k-mers.
 * @tparam L Layout of the k-mers.
 */
template <InputFormat I, InputCheck C, Encoding E, Layout L>
class KmerExtractor
{
public:

    using sequence_type = ExtractorSequence<I, C, E, L>;
    using kmer_type = Kmer<E, L>;

    static constexpr InputFormat input_format = I;
    static constexpr InputCheck input_check = C;
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;

    // -------------------------------------------------------------------------
    //     Constructor
    // -------------------------------------------------------------------------

    /**
     * @brief Set up the extractor for k-mers of size `k`, with the estimated algorithm.
     *
     * @throws std::invalid_argument if `k` is not in [1, 32].
     */
    explicit KmerExtractor(std::size_t k)
        : plan_(make_kmer_extractor_settings_(k))
    {}

    /**
     * @brief The k-mer size.
     */
    std::size_t k() const noexcept
    {
        return plan_.settings().k;
    }

    // -------------------------------------------------------------------------
    //     Extraction
    // -------------------------------------------------------------------------

    /**
     * @brief Call `callback` with each k-mer of `seq`, with the chosen algorithm.
     */
    template <typename Callback>
        requires KmerCallback<std::remove_reference_t<Callback>&, E, L>
    void for_each(sequence_type const& seq, Callback&& callback) const
    {
        plan_.for_each(seq, callback);
    }

    // -------------------------------------------------------------------------
    //     Tuning
    // -------------------------------------------------------------------------

    /**
     * @brief Choose the algorithm, by timing on random sequences, see TuneOptions, or by
     * estimate.
     *
     * The timing uses a callback that does nothing with the k-mers; to time with the callback of
     * for_each(), and in particular with its vector forms, use tune(callback).
     *
     * @throws std::runtime_error if two algorithms give different results.
     */
    void tune(TuneOptions const& options = {})
    {
        ExtractorTuneSink sink;
        tune(sink, options);
    }

    /**
     * @brief Choose the algorithm, by timing on a `sample`: one sequence, or a range of them, such
     * as a `std::vector` of reads.
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
     * The callback is called with all k-mers of the sequences, `options.repeats` times per
     * algorithm. Its side effects during tuning are up to the caller. All algorithms call the same
     * callback, so that a callback that takes vectors gets single k-mers from some algorithms and
     * vectors from others.
     */
    template <typename Callback>
        requires KmerCallback<std::remove_reference_t<Callback>&, E, L>
    void tune(Callback&& callback, TuneOptions const& options = {})
    {
        if (options.level == TuneLevel::kEstimate) {
            plan_.estimate();
        } else {
            plan_.tune(make_tune_samples_<I, C, E, L>(options), options, callback);
        }
    }

    /**
     * @brief Choose the algorithm, by timing with `callback` on a `sample`, see the overloads
     * above.
     */
    template <typename Sample, typename Callback>
        requires ExtractorSample<Sample, sequence_type> &&
                 KmerCallback<std::remove_reference_t<Callback>&, E, L>
    void tune(Sample const& sample, Callback&& callback, TuneOptions const& options = {})
    {
        plan_.tune(sample, options, callback);
    }

    /**
     * @brief The name of the chosen algorithm: the name of the k-mer extraction function, followed
     * by the SIMD arch, such as "for_each_kmer_ascii_checked_chunked_simd_avx2" for
     * for_each_kmer_ascii_checked_chunked_simd<SimdAVX2>().
     *
     * Before tune() or set_algorithm(), the estimate depends on whether the callback takes
     * vectors: this names the one for callbacks of single k-mers, see the overload for others.
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

    using Plan = ExtractorPlan<
        KmerExtractorSettings, sequence_type,
        typename KmerExtractorCandidates<I, C, E, L>::type,
        typename KmerExtractorPreference<I, C, E, L>::type
    >;
    Plan plan_;
};

} // namespace fisk
