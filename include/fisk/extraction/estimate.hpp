#pragma once

#include <concepts>
#include <cstddef>
#include <string>
#include <type_traits>

#include "fisk/util/cpu_runtime.hpp"
#include "fisk/util/intrinsics.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/simd/simd.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"

namespace fisk {

// =================================================================================================
//     Estimate Types
// =================================================================================================

/**
 * @brief The word encoder with which the estimate packs ASCII input: WordEncoderDirectDouble with
 * AVX2 or Neon, else with SSE2 (whose double store needs SSSE3), else WordEncoderButterfly.
 */
#if defined(FISK_HAS_AVX2)
template <Encoding E, Layout L>
using EstimateWordEncoder = WordEncoderDirectDouble<E, L, SimdAVX2>;
#elif defined(FISK_HAS_NEON)
template <Encoding E, Layout L>
using EstimateWordEncoder = WordEncoderDirectDouble<E, L, SimdNEON>;
#elif defined(FISK_HAS_SSSE3)
template <Encoding E, Layout L>
using EstimateWordEncoder = WordEncoderDirectDouble<E, L, SimdSSE2>;
#else
template <Encoding E, Layout L>
using EstimateWordEncoder = WordEncoderButterfly<E, L>;
#endif

// =================================================================================================
//     Candidate Lists
// =================================================================================================

/**
 * @brief A list of types, for the candidate algorithms of the estimate.
 *
 * Each candidate is an empty struct with
 *
 *   - `static std::string name()`, the name of the extraction function that it calls, followed by
 *     its template choices, such as "for_each_kmer_ascii_checked_chunked_simd_avx2" for
 *     for_each_kmer_ascii_checked_chunked_simd<SimdAVX2>(),
 *   - `static bool available()`, whether it can run in this build on this CPU,
 *   - optionally `static bool preferred()`, whether the estimate may run it, see
 *     candidate_preferred_(),
 *   - `void run(Param const& param, Seq const& seq, Cb& callback) const`, the extraction, with
 *     its parameter: the k-mer size `k`, which the extraction function that it calls checks, or
 *     the masks, a SpacedKmerMaskForms, which are checked when they are prepared.
 */
template <typename... Ts>
struct TypeList
{};

/**
 * @brief The SIMD arch of the candidates that use SIMD, as a TypeList: the widest one compiled of
 * AVX2, Neon, and SSE2, or none. A build with an arch runs only on CPUs that have it, as the
 * compiler uses it elsewhere as well, so the narrower archs are not compiled as candidates. The
 * estimates prefer it before the scalar loops, see KmerEstimate and SpacedKmerEstimate.
 *
 * Builds with AVX512 use AVX2 as well. With callbacks of single k-mers, the AVX512 loops are in
 * general not faster than those of AVX2, as each k-mer is handed to the callback on its own. For
 * spaced k-mers in builds with BMI2, the estimate runs PEXT on all CPUs with AVX512, and an AVX512
 * candidate next to it makes GCC keep the state of the callback in memory in the loop with PEXT.
 * Callbacks that take vectors, and spaced k-mers in builds without BMI2, can be faster with the
 * AVX512 functions, called directly, such as for_each_kmer_ascii_checked_chunked_simd<SimdAVX512>().
 */
#if defined(FISK_HAS_AVX2)
using CandidateSimdArchs = TypeList<SimdAVX2>;
#elif defined(FISK_HAS_NEON)
using CandidateSimdArchs = TypeList<SimdNEON>;
#elif defined(FISK_HAS_SSE2)
using CandidateSimdArchs = TypeList<SimdSSE2>;
#else
using CandidateSimdArchs = TypeList<>;
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
//     Estimate
// =================================================================================================

// Whether the estimate may run `Candidate` on this CPU: its preferred(), where it has one, such as
// the PEXT candidates, which are available wherever BMI2 is, but preferred only where PEXT is
// likely fast; else its available().
template <typename Candidate>
inline bool candidate_preferred_()
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
        if (candidate_preferred_<Candidate>()) {
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
 * @brief Call `f` with the candidate of the estimate of the free functions, such as
 * for_each_kmer(), to run it, or to get its name.
 *
 * The `Preference` is a TypeList of candidates in order of preference, for all callbacks, whose
 * last one is a scalar one, which is always preferred. Of this list, the first one that is
 * preferred on this CPU is chosen at runtime, with cached checks of the CPU, which are a few
 * predictable branches.
 */
template <typename Preference, typename F>
FISK_ALWAYS_INLINE inline decltype(auto) visit_estimate(F&& f)
{
    return visit_first_preferred_(Preference{}, f);
}

} // namespace fisk
