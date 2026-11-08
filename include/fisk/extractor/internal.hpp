#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
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

/**
 * @brief The SIMD archs of the estimate of the k-mers, see ExtractorPreference: AVX2, also next to
 * AVX512, or Neon, else none, so that the estimate is a scalar loop. The SSE2 loops of the k-mers
 * are not faster than the scalar ones.
 */
#if defined(FISK_HAS_AVX2)
using PreferredSimdArchs = TypeList<SimdAVX2>;
#elif defined(FISK_HAS_NEON)
using PreferredSimdArchs = TypeList<SimdNEON>;
#else
using PreferredSimdArchs = TypeList<>;
#endif

/**
 * @brief The SIMD archs of the estimate of the spaced k-mers: AVX2, also next to AVX512, else SSE2,
 * whose butterfly kernel is faster than the scalar butterfly, else Neon, else none. Neon comes
 * after the x86 archs, as a build has either of them.
 */
#if defined(FISK_HAS_AVX2)
using PreferredSpacedSimdArchs = TypeList<SimdAVX2>;
#elif defined(FISK_HAS_SSE2)
using PreferredSpacedSimdArchs = TypeList<SimdSSE2>;
#elif defined(FISK_HAS_NEON)
using PreferredSpacedSimdArchs = TypeList<SimdNEON>;
#else
using PreferredSpacedSimdArchs = TypeList<>;
#endif

// The input check in the names of the candidates.
inline std::string input_check_name_(InputCheck input_check)
{
    return (input_check == InputCheck::kChecked) ? "checked" : "unchecked";
}

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
//     Extractor Estimate
// =================================================================================================

/**
 * @brief The estimate of an extractor and of the free functions, such as for_each_kmer(): lists of
 * candidates in order of preference, as TypeLists, for callbacks of single k-mers and for callbacks
 * that take vectors. Of the list for a callback, the first candidate that is preferred on this CPU
 * runs, see visit_estimate(). The last one of each list is a scalar one, which always is
 * preferred.
 */
template <typename Single, typename Vectors>
struct ExtractorPreference
{
    // Whether a candidate of the list hands vectors to a callback of type `Callback`.
    template <typename Callback, typename... Candidates>
    static constexpr bool takes_vectors_(TypeList<Candidates...>)
    {
        return (Candidates::template takes_vectors<Callback> || ...);
    }

    // The list for callbacks of type `Callback`.
    template <typename Callback>
    using list = std::conditional_t<takes_vectors_<Callback>(Vectors{}), Vectors, Single>;
};

// Whether the estimate may run `Candidate` on this CPU: its preferred(), where it has one, such as
// the PEXT candidates, which are measured by tune() wherever they are available, but preferred only
// where they are likely fast; else its available().
template <typename Candidate>
inline bool extractor_candidate_preferred_()
{
    if constexpr (requires { Candidate::preferred(); }) {
        return Candidate::preferred();
    } else {
        return Candidate::available();
    }
}

// Call `f` with the first preferred candidate of the list, see visit_estimate().
template <typename Candidate, typename... Rest, typename F>
FISK_ALWAYS_INLINE inline decltype(auto) visit_first_preferred_(F& f)
{
    if constexpr (sizeof...(Rest) == 0) {
        return f(Candidate{});
    } else {
        if (extractor_candidate_preferred_<Candidate>()) {
            return f(Candidate{});
        }
        return visit_first_preferred_<Rest...>(f);
    }
}

template <typename... Candidates, typename F>
FISK_ALWAYS_INLINE inline decltype(auto) visit_first_preferred_(TypeList<Candidates...>, F& f)
{
    return visit_first_preferred_<Candidates...>(f);
}

/**
 * @brief Call `f` with the candidate of the estimate for callbacks of type `Callback`, such as to
 * run it, or to get its name.
 *
 * The `Preference`, an ExtractorPreference, has a list of candidates for the callback type, chosen
 * at compile time. Of this list, the first one that is preferred on this CPU is chosen at runtime,
 * with cached checks of the CPU, which are a few predictable branches.
 */
template <typename Preference, typename Callback, typename F>
FISK_ALWAYS_INLINE inline decltype(auto) visit_estimate(F&& f)
{
    return visit_first_preferred_(typename Preference::template list<Callback>{}, f);
}

// =================================================================================================
//     Extractor Plan
// =================================================================================================

/**
 * @brief The plan of an extractor before tune() and set_algorithm(): the estimate, see
 * visit_estimate().
 */
struct ExtractorEstimate
{};

/**
 * @brief The shared logic of KmerExtractor and SpacedKmerExtractor: the candidate algorithms, the
 * choice between them, and running the chosen one.
 *
 * Each candidate is an empty struct with
 *
 *   - `static std::string name()`, the name of the extraction function that it calls, followed by
 *     its template choices, such as "for_each_kmer_ascii_checked_chunked_simd_avx2" for
 *     for_each_kmer_ascii_checked_chunked_simd<SimdAVX2>(),
 *   - `static bool available()`, whether it can run in this build on this CPU,
 *   - optionally `static bool preferred()`, whether the estimate may run it, see
 *     extractor_candidate_preferred_(),
 *   - `template <typename Cb> static constexpr bool takes_vectors`, whether it hands vectors to a
 *     callback of type `Cb`, rather than single k-mers,
 *   - `void run(Settings const& settings, Seq const& seq, Cb& callback) const`, the extraction,
 *     with the data that it needs from the `settings`, such as the prepared masks.
 *
 * The plan is a std::variant of ExtractorEstimate and the candidates, which for_each() visits. It
 * starts as the estimate, which picks the candidate per callback type, see visit_estimate(): a
 * callback that takes vectors gets a candidate that hands out vectors, a callback of single k-mers
 * one that is fastest for them. Once tune() measured or set_algorithm() set a candidate, the plan
 * holds that one, for all callbacks.
 */
template <typename Settings, typename Seq, typename Candidates, typename Preference>
class ExtractorPlan;

template <typename Settings, typename Seq, typename... Candidates, typename Preference>
class ExtractorPlan<Settings, Seq, TypeList<Candidates...>, Preference>
{
public:

    using Plan = std::variant<ExtractorEstimate, Candidates...>;

    /**
     * @brief Prepare the plan for the `settings`, with the estimate.
     */
    explicit ExtractorPlan(Settings settings)
        : settings_(std::move(settings))
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
        // The candidate to run: the one that the plan holds, or for the estimate, its candidate for
        // this callback type: at compile time, the preference list for the type; at runtime, its
        // first candidate that is preferred on this CPU, see visit_estimate(). So the estimate runs
        // the code of that candidate, instead of a copy of its own.
        std::size_t index = plan_.index();
        if (index == 0) {
            index = visit_estimate<Preference, Callback>(
                [](auto const& preferred) FISK_ALWAYS_INLINE_LAMBDA {
                    return candidate_index_<std::remove_cvref_t<decltype(preferred)>>;
                }
            );
        }

        // At runtime: the jump to that candidate, with one branch per candidate. Not std::visit,
        // which calls the function for an alternative out of line, so that the callback would
        // escape into that call, and its state could not stay in registers in the loops of the
        // candidate.
        // The candidates are empty structs, see above, so each branch runs one of its own.
        [&]<std::size_t... I>(std::index_sequence<I...>) FISK_ALWAYS_INLINE_LAMBDA {
            (void) ((index == I + 1 && (Candidates{}.run(settings_, seq, callback), true)) || ...);
        }(std::index_sequence_for<Candidates...>{});
    }

    // -------------------------------------------------------------------------
    //     Algorithm Names
    // -------------------------------------------------------------------------

    template <typename Callback>
    std::string algorithm_name() const
    {
        return std::visit(
            [](auto const& candidate) {
                using Candidate = std::remove_cvref_t<decltype(candidate)>;
                if constexpr (std::same_as<Candidate, ExtractorEstimate>) {
                    return visit_estimate<Preference, Callback>([](auto const& preferred) {
                        return std::remove_cvref_t<decltype(preferred)>::name();
                    });
                } else {
                    return Candidate::name();
                }
            },
            plan_
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
        bool found = false;
        (
            [&] {
                if (!found && Candidates::name() == name && Candidates::available()) {
                    plan_ = Candidates{};
                    found = true;
                }
            }(),
            ...
        );
        if (!found) {
            throw std::invalid_argument(
                "set_algorithm(): unknown or unavailable algorithm \"" + std::string(name) + "\""
            );
        }
    }

    // -------------------------------------------------------------------------
    //     Tune
    // -------------------------------------------------------------------------

    void estimate()
    {
        plan_ = ExtractorEstimate{};
    }

    /**
     * @brief Estimate, or measure on a `sample` with the `callback`, by the level of the `options`.
     * The sample is one sequence, or a range of them, see ExtractorSample. The callback is
     * forwarded, see measure().
     */
    template <typename Sample, typename Callback>
        requires ExtractorSample<Sample, Seq>
    void tune(Sample const& sample, TuneOptions const& options, Callback&& callback)
    {
        if (options.level == TuneLevel::kEstimate) {
            estimate();
        } else if constexpr (std::convertible_to<Sample const&, Seq const&>) {
            Seq const& seq = sample;
            measure(std::span<Seq const>(&seq, 1), options, std::forward<Callback>(callback));
        } else {
            measure(sample, options, std::forward<Callback>(callback));
        }
    }

    /**
     * @brief Time each available candidate on the `samples` with the `callback`, and use the
     * fastest.
     *
     * Each candidate first runs once with a checksum, which has to match between all of them, and
     * which is also its warmup. Then it repeats its passes over the samples until they take the
     * sample time of the `options`, which sets its number of passes per timed run. The timed runs
     * take turns between the candidates, so that changes of the clock speed of the CPU affect all
     * of them alike; the fastest run per pass of each candidate counts.
     *
     * A callback that is passed as an rvalue, such as a temporary, is moved in here, and each timed
     * run moves it into a local of its own, so that it can stay in registers, as it does in
     * for_each(), which inlines into its caller: so the candidates are timed as they will run. A
     * callback that is passed as an lvalue is called as it is, so that its owner sees all calls.
     */
    template <typename Samples, typename Callback>
    void measure(Samples const& samples, TuneOptions const& options, Callback&& callback)
    {
        using clock = std::chrono::steady_clock;
        constexpr std::size_t n = sizeof...(Candidates);
        auto const sample_time = std::chrono::duration<double>(options.sample_seconds);
        std::optional<std::uint64_t> reference;
        std::string reference_name;

        // Per candidate: its passes per timed run, 0 if it is not available, and its fastest time
        // per pass, in seconds.
        std::array<std::size_t, n> passes{};
        std::array<double, n> pass_time;
        pass_time.fill(std::numeric_limits<double>::infinity());

        auto const run_pass = [&](auto const& candidate, auto& cb) FISK_ALWAYS_INLINE_LAMBDA {
            for (auto const& seq : samples) {
                candidate.run(settings_, seq, cb);
            }
        };

        // The callback of an rvalue, moved in, see above; unused for an lvalue.
        constexpr bool owned = !std::is_lvalue_reference_v<Callback>;
        using Owned = std::conditional_t<
            owned, std::optional<std::remove_cvref_t<Callback>>, std::monostate
        >;
        Owned owned_callback;
        if constexpr (owned) {
            owned_callback.emplace(std::move(callback));
        }

        // One timed run of a candidate: while its passes per run are not set yet, passes until the
        // sample time, at least one, also for samples that take no time, such as empty ones, which
        // then sets them; afterwards, its passes per run, of which the fastest per pass counts.
        // One loop for both, so that the candidate is inlined once for the timing. An owned
        // callback is moved into a local for the run, outside of its timing, and back after it,
        // with its results kept alive, so that they are not optimized away.
        auto const timed_run = [&](auto const& candidate, std::size_t i) {
            auto const run = [&](auto& cb) FISK_ALWAYS_INLINE_LAMBDA {
                bool const calibrate = passes[i] == 0;
                std::size_t p = 0;
                auto const start = clock::now();
                do {
                    run_pass(candidate, cb);
                    ++p;
                } while (calibrate ? clock::now() - start < sample_time : p < passes[i]);
                std::chrono::duration<double> const elapsed = clock::now() - start;
                if (calibrate) {
                    passes[i] = p;
                } else {
                    pass_time[i] = std::min(pass_time[i], elapsed.count() / static_cast<double>(p));
                }
            };
            if constexpr (owned) {
                std::remove_cvref_t<Callback> local = std::move(*owned_callback);
                owned_callback.reset();
                run(local);
                clobber_memory(&local);
                owned_callback.emplace(std::move(local));
            } else {
                run(callback);
            }
        };

        auto const prepare_one = [&](auto const& candidate, std::size_t i) {
            using Candidate = std::remove_cvref_t<decltype(candidate)>;
            if (!Candidate::available()) {
                return;
            }

            ExtractorTuneChecksum checksum;
            run_pass(candidate, checksum);
            if (!reference) {
                reference = checksum.sum;
                reference_name = Candidate::name();
            } else if (*reference != checksum.sum) {
                throw std::runtime_error(
                    "tune(): the results of algorithm \"" + Candidate::name() +
                    "\" differ from those of \"" + reference_name + "\""
                );
            }
            timed_run(candidate, i);
        };

        auto const time_one = [&](auto const& candidate, std::size_t i) {
            if (passes[i] > 0) {
                timed_run(candidate, i);
            }
        };

        // Each fold runs the lambda for each candidate, with its index.
        std::size_t i = 0;
        ((prepare_one(Candidates{}, i), ++i), ...);
        for (std::size_t r = 0; r < std::max<std::size_t>(options.repeats, 1); ++r) {
            i = 0;
            ((time_one(Candidates{}, i), ++i), ...);
        }

        std::optional<Plan> best;
        double best_time = std::numeric_limits<double>::infinity();
        i = 0;
        (
            [&] {
                if (passes[i] > 0 && (!best || pass_time[i] < best_time)) {
                    best_time = pass_time[i];
                    best = Candidates{};
                }
                ++i;
            }(),
            ...
        );

        // The scalar candidates are always available.
        if (!best) {
            throw std::runtime_error("tune(): no algorithm available");
        }
        plan_ = *best;
    }

private:

    // The index of the candidate `C` in the plan: 1 + its position among the candidates, after the
    // estimate.
    template <typename C>
    static constexpr std::size_t candidate_index_ = [] {
        static_assert((std::same_as<C, Candidates> || ...), "Not a candidate of the plan.");
        std::size_t i = 1;
        (void) ((std::same_as<C, Candidates> ? false : (++i, true)) && ...);
        return i;
    }();

    Settings settings_;
    Plan plan_;
};

} // namespace fisk
