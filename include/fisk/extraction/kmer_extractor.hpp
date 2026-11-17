#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/extraction/estimate.hpp"
#include "fisk/extraction/extractor_plan.hpp"
#include "fisk/extraction/kmer_estimate.hpp"
#include "fisk/extraction/tune_options.hpp"
#include "fisk/kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extractor Candidates
// =================================================================================================

// The candidates of a KmerExtractor: the loops that are the fastest on some hardware, in the
// benchmarks, see kmer_estimate.hpp, and KmerEstimate for the estimate among them. The SIMD loops
// are those of the `Simds`, a TypeList, see ExtractorSimdArchs.
template <
    InputFormat I, InputCheck C, Encoding E, Layout L, typename Simds = ExtractorSimdArchs
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
 *     for (auto const& seq : sequences) {
 *         ex.for_each(seq, [&](std::size_t pos, KmerAcgtMsb kmer) { ... });
 *     }
 *
 * The extractor tunes in situ: its first calls of for_each() time the algorithms in turns, on
 * their sequences and with their callbacks, and then keep the fastest, see tune() and TuneOptions.
 * Make an extractor once and reuse it, or copies of it, for all input: the first input that it
 * extracts is used for tuning, which continues across the calls of for_each(); long sequences
 * switch to the fastest algorithm within the sequence. An extractor that sees less input, for
 * instance one per short read, spends it on the turns of the algorithms, or, with less input than
 * one turn, runs only the estimate, the choice from what this build and CPU support.
 * algorithm_name() names the choice, set_algorithm() sets it, for instance to reuse a tuning, and
 * estimate() uses the estimate without tuning.
 *
 * A turn is timed from its first call to its last one, so that it includes the code between the
 * calls, the same for all algorithms. Some input can still mislead it: a turn over a long run of
 * `N`, as at the start of many chromosomes, which is fast for some algorithms only; waits between
 * the calls, such as for I/O; and a second extractor that tunes at the same time on the same
 * thread, whose calls then count in the turns of both. Where this matters, tune on a sample first,
 * such as random_acgt() or a few Mb from the middle of a chromosome, with the code that calls
 * for_each(), until tuning_pending() is false, and discard its results; for packed input, pack the
 * sample as the input. With several extractors, tune them one after the other in this way.
 *
 * The callback takes single k-mers, as `callback(pos, kmer)` or `callback(kmer)`, see
 * KmerCallback. It can in addition take the vectors of k-mers of the SIMD algorithms, see
 * KmerVectorCallback and KmerVectorValidLanesCallback, for instance as a callback_overload, or a
 * struct with several `operator()`. The SIMD algorithms then hand it their vectors, and else their
 * k-mers one by one. The callback is a template parameter of for_each(), so that it can be inlined
 * into the loop. As each algorithm calls it, each callback type compiles all of them.
 *
 * The k-mers are handed out in order of their positions. for_each() is const, and can run on
 * several threads at once; tune(), estimate() and set_algorithm() cannot. While it tunes, the
 * first thread that calls for_each() tunes in its calls, and the other threads run the algorithm
 * from before, the estimate for a new extractor; if that thread stops calling for_each() before
 * the tuning is done, the tuning stays pending until tune() starts it again. ASCII input is packed
 * chunk by chunk into a buffer per thread.
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
     * @brief Set up the extractor for k-mers of size `k`, tuning in situ, see tune().
     *
     * @throws std::invalid_argument if `k` is not in [1, 32].
     */
    explicit KmerExtractor(std::size_t k)
        : plan_(checked_k_(k))
    {}

    /**
     * @brief The k-mer size.
     */
    std::size_t k() const noexcept
    {
        return plan_.settings();
    }

    // -------------------------------------------------------------------------
    //     Extraction
    // -------------------------------------------------------------------------

    /**
     * @brief Call `callback` with each k-mer of `seq`, with the chosen algorithm.
     */
    template <typename Callback>
        requires KmerCallback<std::remove_reference_t<Callback>&, E, L>
    FISK_ALWAYS_INLINE void for_each(sequence_type const& seq, Callback&& callback) const
    {
        plan_.for_each(seq, callback);
    }

    // -------------------------------------------------------------------------
    //     Tuning
    // -------------------------------------------------------------------------

    /**
     * @brief Tune again, in situ, with these options: the next calls of for_each() run the
     * algorithms in turns, on their sequences and with their callbacks, until each has run for
     * `options.time_per_algorithm`, and then keep the fastest, see TuneOptions and
     * tuning_pending(). A new extractor tunes in this way with the default options.
     */
    void tune(TuneOptions const& options = {})
    {
        plan_.tune(options);
    }

    /**
     * @brief The name of the chosen algorithm: the name of the k-mer extraction function, followed
     * by the SIMD arch, such as "for_each_kmer_ascii_checked_chunked_simd_avx2" for
     * for_each_kmer_ascii_checked_chunked_simd<SimdAVX2>().
     *
     * While the in-situ tuning is pending, see tuning_pending(), this is the algorithm from before,
     * which other threads run meanwhile: the estimate for a new extractor. After estimate(), it is
     * the estimate.
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
     * @brief Use the estimate, without tuning: the algorithm that this build and CPU support, by a
     * fixed order of preference, the same for all callbacks.
     */
    void estimate()
    {
        plan_.estimate();
    }

    /**
     * @brief Whether the in-situ tuning has not chosen its algorithm yet, see tune(): until then,
     * for_each() runs the algorithm as it was before, in all calls but those that tune.
     */
    bool tuning_pending() const noexcept
    {
        return plan_.tuning_pending();
    }

private:

    // The k-mer size, checked here, so that an invalid one fails when the extractor is made.
    static std::size_t checked_k_(std::size_t k)
    {
        if (k == 0 || k > 32) {
            throw_invalid_kmer_k_(32);
        }
        return k;
    }

    using Plan = ExtractorPlan<
        std::size_t, sequence_type,
        typename KmerExtractorCandidates<I, C, E, L>::type,
        typename KmerEstimate<I, C, E, L>::type
    >;
    Plan plan_;
};

} // namespace fisk
