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
#include "fisk/extraction/estimate.hpp"
#include "fisk/kmer_extract/ascii_chunked_scalar.hpp"
#include "fisk/kmer_extract/ascii_chunked_simd.hpp"
#include "fisk/kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/kmer_extract/packed_bytewise_simd.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Candidates
// =================================================================================================

// The candidate algorithms of for_each_kmer(), see TypeList for their shape. Each calls one of the
// k-mer extraction functions, and is named after it, followed by the SIMD arch.

/**
 * @brief The settings that the k-mer candidates run with.
 */
struct KmerExtractorSettings
{
    std::size_t k = 0;
};

// The settings for k-mers of size `k`, for for_each_kmer().
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
FISK_ALWAYS_INLINE inline auto kmer_candidate_vector_callback_(Callback& callback)
{
    return [&callback](std::size_t pos, auto kmers, auto valid) FISK_ALWAYS_INLINE_LAMBDA {
        if constexpr (std::invocable<Callback&, std::size_t, decltype(kmers), decltype(valid)>) {
            callback(pos, kmers, valid);
        } else {
            for_each_valid_lane(pos, kmers, valid, callback);
        }
    };
}

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
    FISK_ALWAYS_INLINE void run(
        KmerExtractorSettings const& settings, std::string_view seq, Callback& callback
    ) const
    {
        EstimateWordEncoder<E, L> const encoder{};
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
    FISK_ALWAYS_INLINE void run(
        KmerExtractorSettings const& settings, std::string_view seq, Callback& callback
    ) const
    {
        EstimateWordEncoder<E, L> const encoder{};
        auto vector_callback = kmer_candidate_vector_callback_(callback);
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

    template <typename Seq, typename Callback>
    FISK_ALWAYS_INLINE void run(
        KmerExtractorSettings const& settings, Seq const& seq, Callback& callback
    ) const
    {
        auto vector_callback = kmer_candidate_vector_callback_(callback);
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

// The estimate of for_each_kmer(), see visit_estimate(): the SIMD loop of the first of the
// `Simds`, a TypeList, that the CPU has, in the order of CandidateSimdArchs (AVX512, AVX2, and SSE2
// or Neon), else the scalar loop, for all callbacks.
template <
    InputFormat I, InputCheck C, Encoding E, Layout L, typename Simds = CandidateSimdArchs
>
struct KmerEstimate;

template <InputFormat I, InputCheck C, Encoding E, Layout L, typename... Simd>
struct KmerEstimate<I, C, E, L, TypeList<Simd...>>
{
    using Ascii = TypeList<
        KmerCandidateAsciiChunkedSimd<C, E, L, Simd>..., KmerCandidateAsciiChunkedScalar<C, E, L>
    >;

    using Packed = TypeList<
        KmerCandidatePackedBytewiseSimd<C, E, L, Simd>...,
        KmerCandidatePackedBytewiseScalar<C, E, L>
    >;

    using type = std::conditional_t<I == InputFormat::kAscii, Ascii, Packed>;
};

} // namespace fisk
