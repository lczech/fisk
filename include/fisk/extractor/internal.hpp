#pragma once

#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "fisk/util/cpu_runtime.hpp"
#include "fisk/util/intrinsics.hpp"
#include "fisk/util/random.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/types.hpp"
#include "fisk/extractor/common.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/simd/simd.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"

namespace fisk {

// =================================================================================================
//     Extractor Types
// =================================================================================================

/**
 * @brief The input sequence of an extractor: a `std::string_view` for ASCII input, else a
 * PackedSequence, or, to be checked, a PackedSequenceWithValidity.
 */
template <InputFormat I, InputCheck C, Encoding E, Layout L>
using ExtractorSequence = std::conditional_t<
    I == InputFormat::kAscii,
    std::string_view,
    std::conditional_t<
        C == InputCheck::kChecked, PackedSequenceWithValidity<E, L>, PackedSequence<E, L>
    >
>;

/**
 * @brief The word encoder with which the extractors pack ASCII input: WordEncoderDirectDouble with
 * AVX2 or Neon, else with SSE2 (whose double store needs SSSE3), else WordEncoderButterfly.
 */
#if defined(FISK_HAS_AVX2)
template <Encoding E, Layout L>
using ExtractorWordEncoder = WordEncoderDirectDouble<E, L, SimdAVX2>;
#elif defined(FISK_HAS_NEON)
template <Encoding E, Layout L>
using ExtractorWordEncoder = WordEncoderDirectDouble<E, L, SimdNEON>;
#elif defined(FISK_HAS_SSSE3)
template <Encoding E, Layout L>
using ExtractorWordEncoder = WordEncoderDirectDouble<E, L, SimdSSE2>;
#else
template <Encoding E, Layout L>
using ExtractorWordEncoder = WordEncoderButterfly<E, L>;
#endif

// =================================================================================================
//     Candidate Lists
// =================================================================================================

/**
 * @brief A list of types, for the candidate algorithms of the extractors.
 */
template <typename... Ts>
struct TypeList
{};

/**
 * @brief The SIMD archs of the candidates that use SIMD: the best one compiled, and AVX2 next to
 * AVX512, as AVX512 is not faster on all CPUs that have it. SSE2 only when it is the best one.
 */
#if defined(FISK_HAS_AVX512) && defined(FISK_HAS_AVX2)
using CandidateSimdArchs = TypeList<SimdAVX512, SimdAVX2>;
#elif defined(FISK_HAS_AVX512)
using CandidateSimdArchs = TypeList<SimdAVX512>;
#elif defined(FISK_HAS_AVX2)
using CandidateSimdArchs = TypeList<SimdAVX2>;
#elif defined(FISK_HAS_NEON)
using CandidateSimdArchs = TypeList<SimdNEON>;
#elif defined(FISK_HAS_SSE2)
using CandidateSimdArchs = TypeList<SimdSSE2>;
#else
using CandidateSimdArchs = TypeList<>;
#endif

// The name of a SIMD arch in the names of the candidates.
template <SimdArch Simd>
inline std::string simd_arch_name_()
{
    if constexpr (std::same_as<Simd, SimdScalar>) {
        return "scalar";
    }
    #if defined(FISK_HAS_SSE2)
    else if constexpr (std::same_as<Simd, SimdSSE2>) {
        return "sse2";
    }
    #endif
    #if defined(FISK_HAS_AVX2)
    else if constexpr (std::same_as<Simd, SimdAVX2>) {
        return "avx2";
    }
    #endif
    #if defined(FISK_HAS_AVX512)
    else if constexpr (std::same_as<Simd, SimdAVX512>) {
        return "avx512";
    }
    #endif
    #if defined(FISK_HAS_NEON)
    else if constexpr (std::same_as<Simd, SimdNEON>) {
        return "neon";
    }
    #endif
    else {
        static_assert(dependent_false_v<Simd::lanes>, "Unhandled SIMD arch in simd_arch_name_()");
    }
}

// Whether a SIMD arch can be used in this build and on this CPU.
template <SimdArch Simd>
inline bool simd_arch_enabled_()
{
    if constexpr (std::same_as<Simd, SimdScalar>) {
        return true;
    }
    #if defined(FISK_HAS_SSE2)
    else if constexpr (std::same_as<Simd, SimdSSE2>) {
        return sse2_enabled();
    }
    #endif
    #if defined(FISK_HAS_AVX2)
    else if constexpr (std::same_as<Simd, SimdAVX2>) {
        return avx2_enabled();
    }
    #endif
    #if defined(FISK_HAS_AVX512)
    else if constexpr (std::same_as<Simd, SimdAVX512>) {
        return avx512_enabled();
    }
    #endif
    #if defined(FISK_HAS_NEON)
    else if constexpr (std::same_as<Simd, SimdNEON>) {
        return neon_enabled();
    }
    #endif
    else {
        static_assert(
            dependent_false_v<Simd::lanes>, "Unhandled SIMD arch in simd_arch_enabled_()"
        );
    }
}

// =================================================================================================
//     Tuning Data and Callbacks
// =================================================================================================

/**
 * @brief The random sequences that tune() times the candidates on when no sample is given, see
 * TuneOptions: ASCII strings, or these packed.
 */
template <InputFormat I, InputCheck C, Encoding E, Layout L>
inline auto make_tune_samples_(TuneOptions const& options)
{
    // A fixed seed, so that tuning is repeatable.
    Splitmix64 rng(0x5eed);
    std::size_t const total = options.total_length;
    std::size_t const length = options.sequence_length == 0 ? total : options.sequence_length;

    std::vector<std::string> strings;
    for (std::size_t start = 0; start < total; start += length) {
        std::string seq(std::min(length, total - start), 'A');
        for (auto& c : seq) {
            c = "ACGT"[rng.get_uint64() & 3u];
            if (C == InputCheck::kChecked && rng.get_double() < options.invalid_rate) {
                c = 'N';
            }
        }
        strings.push_back(std::move(seq));
    }

    if constexpr (I == InputFormat::kAscii) {
        return strings;
    } else {
        using Packed = ExtractorSequence<I, C, E, L>;
        ExtractorWordEncoder<E, L> const encoder{};
        std::vector<Packed> packed;
        packed.reserve(strings.size());
        for (auto const& seq : strings) {
            if constexpr (C == InputCheck::kChecked) {
                packed.push_back(pack_sequence_checked(seq, encoder));
            } else {
                packed.push_back(pack_sequence_unchecked(seq, encoder));
            }
        }
        return packed;
    }
}

/**
 * @brief The callback that tune() times the candidates with when no callback is given: it takes
 * single k-mers or spaced k-mers, and keeps each from being optimized away, without state that
 * would escape into the loops.
 */
struct ExtractorTuneSink
{
    template <typename K>
    FISK_ALWAYS_INLINE void operator()(std::size_t, K const& kmer) const
    {
        do_not_optimize(kmer_value(kmer));
    }

    template <typename K>
    FISK_ALWAYS_INLINE void operator()(std::size_t, std::size_t, K const& kmer) const
    {
        do_not_optimize(kmer_value(kmer));
    }
};

/**
 * @brief The callback that tune() checks the results of the candidates with: a sum over the hashes
 * of the k-mers or spaced k-mers and their positions and mask indices, independent of their order.
 */
struct ExtractorTuneChecksum
{
    std::uint64_t sum = 0;

    template <typename K>
    void operator()(std::size_t pos, K const& kmer)
    {
        sum += mix64_(kmer_value(kmer) ^ mix64_(pos));
    }

    template <typename K>
    void operator()(std::size_t pos, std::size_t mask_idx, K const& kmer)
    {
        sum += mix64_(kmer_value(kmer) ^ mix64_((static_cast<std::uint64_t>(pos) << 8) ^ mask_idx));
    }
};

// =================================================================================================
//     Extractor Plan
// =================================================================================================

/**
 * @brief The shared logic of KmerExtractor and SpacedKmerExtractor: the candidate algorithms, the
 * choice between them, and running the chosen one.
 *
 * Each candidate is an empty struct with
 *
 *   - `static std::string name()`, its name in the naming scheme of the extraction functions, such
 *     as "chunked_simd_avx2" for for_each_kmer_ascii_checked_chunked_simd<SimdAVX2>(),
 *   - `static bool available()`, whether it can run in this build on this CPU,
 *   - `template <typename Cb> static constexpr bool takes_vectors`, whether it hands vectors to a
 *     callback of type `Cb`, rather than single k-mers,
 *   - `void run(Settings const& settings, Seq const& seq, Cb& callback) const`, the extraction,
 *     with the data that it needs from the `settings`, such as the prepared masks.
 *
 * The chosen candidate is held in a std::variant, which for_each() visits. There are two of them,
 * as the estimate depends on the callback: callbacks that take vectors prefer a candidate that
 * hands out vectors, callbacks of single k-mers one that is fastest for them: `kmer_single_plan_`
 * for callbacks of single k-mers, and `kmer_vector_plan_` for callbacks that take vectors.
 * for_each() picks one by the type of the callback, at compile time, see plan_of_callback_().
 * Either can hold any candidate, scalar or SIMD. Once tune() measured or set_algorithm() set a
 * candidate, both hold that one, for all callbacks.
 */
template <typename Settings, typename Seq, typename Candidates>
class ExtractorPlan;

template <typename Settings, typename Seq, typename... Candidates>
class ExtractorPlan<Settings, Seq, TypeList<Candidates...>>
{
public:

    using Plan = std::variant<Candidates...>;

    /**
     * @brief Prepare the plan for the `settings`, with the estimates as lists of candidate names,
     * in order of preference, for callbacks of single k-mers and of vectors. Names of candidates
     * that are not compiled or not available are skipped.
     */
    ExtractorPlan(
        Settings settings,
        std::vector<std::string> kmer_single_preference,
        std::vector<std::string> kmer_vector_preference
    )
        : settings_(std::move(settings))
        , kmer_single_preference_(std::move(kmer_single_preference))
        , kmer_vector_preference_(std::move(kmer_vector_preference))
        , kmer_single_plan_(make_preferred_(kmer_single_preference_))
        , kmer_vector_plan_(make_preferred_(kmer_vector_preference_))
    {}

    Settings const& settings() const noexcept
    {
        return settings_;
    }

    // -------------------------------------------------------------------------
    //     Run
    // -------------------------------------------------------------------------

    template <typename Callback>
    FISK_ALWAYS_INLINE void for_each(Seq const& seq, Callback& callback) const
    {
        std::visit(
            [&](auto const& candidate) FISK_ALWAYS_INLINE_LAMBDA {
                candidate.run(settings_, seq, callback);
            },
            plan_of_callback_<Callback>()
        );
    }

    // -------------------------------------------------------------------------
    //     Algorithm Names
    // -------------------------------------------------------------------------

    template <typename Callback>
    std::string algorithm_name() const
    {
        return std::visit(
            [](auto const& candidate) {
                return std::remove_cvref_t<decltype(candidate)>::name();
            },
            plan_of_callback_<Callback>()
        );
    }

    std::vector<std::string> algorithm_names() const
    {
        std::vector<std::string> names;
        (
            [&] {
                if (Candidates::available()) {
                    names.push_back(Candidates::name());
                }
            }(),
            ...
        );
        return names;
    }

    void set_algorithm(std::string_view name)
    {
        auto plan = make_named_(name);
        if (!plan) {
            throw std::invalid_argument(
                "set_algorithm(): unknown or unavailable algorithm \"" + std::string(name) + "\""
            );
        }
        kmer_single_plan_ = *plan;
        kmer_vector_plan_ = std::move(*plan);
        estimated_ = false;
    }

    // -------------------------------------------------------------------------
    //     Tune
    // -------------------------------------------------------------------------

    void estimate()
    {
        kmer_single_plan_ = make_preferred_(kmer_single_preference_);
        kmer_vector_plan_ = make_preferred_(kmer_vector_preference_);
        estimated_ = true;
    }

    /**
     * @brief Estimate, or measure on a `sample` with the `callback`, by the level of the `options`.
     * The sample is one sequence, or a range of them, see ExtractorSample.
     */
    template <typename Sample, typename Callback>
        requires ExtractorSample<Sample, Seq>
    void tune(Sample const& sample, TuneOptions const& options, Callback& callback)
    {
        if (options.level == TuneLevel::kEstimate) {
            estimate();
        } else if constexpr (std::convertible_to<Sample const&, Seq const&>) {
            Seq const& seq = sample;
            measure(std::span<Seq const>(&seq, 1), options, callback);
        } else {
            measure(sample, options, callback);
        }
    }

    /**
     * @brief Time each available candidate on the `samples` with the `callback`, and use the
     * fastest. Before that, each candidate runs once with a checksum, which has to match between
     * all of them.
     */
    template <typename Samples, typename Callback>
    void measure(Samples const& samples, TuneOptions const& options, Callback& callback)
    {
        using clock = std::chrono::steady_clock;
        std::optional<std::uint64_t> reference;
        std::string reference_name;
        std::optional<Plan> best;
        auto best_time = clock::duration::max();

        auto const measure_one = [&]<std::size_t I>(std::integral_constant<std::size_t, I>) {
            using Candidate = std::variant_alternative_t<I, Plan>;
            if (!Candidate::available()) {
                return;
            }
            Plan plan(std::in_place_index<I>);
            auto const& candidate = std::get<I>(plan);

            // Checksum, which is also the warmup.
            ExtractorTuneChecksum checksum;
            for (auto const& seq : samples) {
                candidate.run(settings_, seq, checksum);
            }
            if (!reference) {
                reference = checksum.sum;
                reference_name = Candidate::name();
            } else if (*reference != checksum.sum) {
                throw std::runtime_error(
                    "tune(): the results of algorithm \"" + Candidate::name() +
                    "\" differ from those of \"" + reference_name + "\""
                );
            }

            auto candidate_time = clock::duration::max();
            for (std::size_t r = 0; r < std::max<std::size_t>(options.repeats, 1); ++r) {
                auto const start = clock::now();
                for (auto const& seq : samples) {
                    candidate.run(settings_, seq, callback);
                }
                candidate_time = std::min<clock::duration>(candidate_time, clock::now() - start);
            }
            if (candidate_time < best_time) {
                best_time = candidate_time;
                best = std::move(plan);
            }
        };
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (measure_one(std::integral_constant<std::size_t, I>{}), ...);
        }(std::index_sequence_for<Candidates...>{});

        // The scalar candidates are always available.
        if (!best) {
            throw std::runtime_error("tune(): no algorithm available");
        }
        kmer_single_plan_ = *best;
        kmer_vector_plan_ = std::move(*best);
        estimated_ = false;
    }

private:

    // The plan for a callback: while estimated, the one for vectors if any candidate hands vectors
    // to this callback, else the one for single k-mers.
    template <typename Callback>
    FISK_ALWAYS_INLINE Plan const& plan_of_callback_() const
    {
        constexpr bool takes_vectors = (Candidates::template takes_vectors<Callback> || ...);
        if constexpr (takes_vectors) {
            return estimated_ ? kmer_vector_plan_ : kmer_single_plan_;
        } else {
            return kmer_single_plan_;
        }
    }

    // The available candidate of the given name, if any.
    std::optional<Plan> make_named_(std::string_view name) const
    {
        std::optional<Plan> result;
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (
                [&] {
                    using Candidate = std::variant_alternative_t<I, Plan>;
                    if (!result && Candidate::name() == name && Candidate::available()) {
                        result.emplace(std::in_place_index<I>);
                    }
                }(),
                ...
            );
        }(std::index_sequence_for<Candidates...>{});
        return result;
    }

    // The first available candidate of the preferences. Each list ends in a scalar candidate, which
    // is always available.
    Plan make_preferred_(std::vector<std::string> const& preference) const
    {
        for (auto const& name : preference) {
            if (auto plan = make_named_(name)) {
                return std::move(*plan);
            }
        }
        throw std::logic_error("ExtractorPlan: none of the preferred algorithms is available");
    }

    Settings settings_;
    std::vector<std::string> kmer_single_preference_;
    std::vector<std::string> kmer_vector_preference_;
    Plan kmer_single_plan_;
    Plan kmer_vector_plan_;
    bool estimated_ = true;
};

} // namespace fisk
