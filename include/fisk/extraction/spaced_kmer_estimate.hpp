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
#include "fisk/extraction/estimate.hpp"
#include "fisk/extraction/spaced_kmer_mask_forms.hpp"
#include "fisk/spaced_kmer_extract/ascii_chunked_scalar.hpp"
#include "fisk/spaced_kmer_extract/ascii_chunked_simd.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Candidates
// =================================================================================================

// The candidate algorithms of for_each_spaced_kmer(), see TypeList for their shape. Each calls one
// of the spaced k-mer extraction functions, with the masks dispatched to a span of fixed size,
// see dispatch_static_span(), and is named after it, followed by the bit extraction.

// -------------------------------------------------------------------------------------------------
//     Bit Extraction
// -------------------------------------------------------------------------------------------------

// The bit extraction of the spaced k-mer masks of the form `BitExtractForm`, a bit extraction mask
// for the scalar loops, or a kernel for the SIMD loops: its name and availability. The masks in
// these forms are in SpacedKmerMaskForms.

namespace detail {

template <typename BitExtractForm>
struct is_bit_extract_kernel_butterfly : std::false_type {};

template <typename Simd>
struct is_bit_extract_kernel_butterfly<BitExtractKernelButterfly<Simd>> : std::true_type {};

template <typename BitExtractForm>
inline std::string spaced_kmer_candidate_bit_extract_name()
{
    if constexpr (is_bit_extract_kernel_butterfly<BitExtractForm>::value) {
        return "butterfly_" + simd_arch_name<typename BitExtractForm::simd_arch>();
    } else if constexpr (std::same_as<BitExtractForm, BitExtractButterflyMask>) {
        return "butterfly";
    }
    #if defined(FISK_HAS_BMI2)
    else if constexpr (std::same_as<BitExtractForm, BitExtractMask>) {
        return "pext";
    }
    #endif
    else {
        static_assert(dependent_false_v<sizeof(BitExtractForm)>, "Unhandled bit extraction");
    }
}

template <typename BitExtractForm>
inline bool spaced_kmer_candidate_bit_extract_available()
{
    if constexpr (std::same_as<BitExtractForm, BitExtractButterflyMask>) {
        return true;
    } else if constexpr (std::same_as<BitExtractForm, BitExtractMask>) {
        return bmi2_enabled();
    } else {
        return simd_arch_enabled<typename BitExtractForm::simd_arch>();
    }
}

// Whether the estimate may run the bit extraction of the masks of the form `BitExtractForm`, see
// candidate_preferred(): PEXT only where it is likely fast, as it is slow on some CPUs,
// see pext_likely_fast().
template <typename BitExtractForm>
inline bool spaced_kmer_candidate_bit_extract_preferred()
{
    #if defined(FISK_HAS_BMI2)
    if constexpr (std::same_as<BitExtractForm, BitExtractMask>) {
        return pext_likely_fast();
    }
    #endif
    return spaced_kmer_candidate_bit_extract_available<BitExtractForm>();
}

// The scalar bit extraction of the masks of the form `BitExtractForm`.
template <typename BitExtractForm>
FISK_ALWAYS_INLINE inline auto spaced_kmer_candidate_bit_extract()
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

} // namespace detail

// -------------------------------------------------------------------------------------------------
//     Callbacks
// -------------------------------------------------------------------------------------------------

namespace detail {

// The callback that a SIMD candidate hands its vectors to: the vectors as they are, if `callback`
// takes them, else their valid spaced k-mers one by one.
template <typename Callback>
FISK_ALWAYS_INLINE inline auto spaced_kmer_candidate_vector_callback(Callback& callback)
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

} // namespace detail

// -------------------------------------------------------------------------------------------------
//     ASCII
// -------------------------------------------------------------------------------------------------

// The run() of each candidate below, and the lambda that it hands to dispatch_static_span(), are
// forced inline, so that the callback does not reach a function out of line, and the whole
// extraction inlines into for_each_spaced_kmer() (extraction/for_each.hpp). A callback that reaches
// a function out of line keeps its state in memory in all of the extraction.

namespace detail {

template <InputCheck C, Encoding E, Layout L, std::size_t MaxMasks, typename BitExtractForm>
struct SpacedKmerCandidateAsciiChunkedScalar
{
    static std::string name()
    {
        return "for_each_spaced_kmer_ascii_" + input_check_name(C) + "_chunked_scalar_" +
            spaced_kmer_candidate_bit_extract_name<BitExtractForm>();
    }
    static bool available()
    {
        return spaced_kmer_candidate_bit_extract_available<BitExtractForm>();
    }
    static bool preferred()
    {
        return spaced_kmer_candidate_bit_extract_preferred<BitExtractForm>();
    }

    template <std::size_t MaxSpan, typename Callback>
    FISK_ALWAYS_INLINE void run(
        SpacedKmerMaskForms<L, MaxMasks, MaxSpan> const& mask_forms, std::string_view seq,
        Callback& callback
    ) const
    {
        DefaultWordEncoder<E, L> const encoder{};
        auto const bit_extract = spaced_kmer_candidate_bit_extract<BitExtractForm>();
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
        return "for_each_spaced_kmer_ascii_" + input_check_name(C) + "_chunked_simd_" +
            spaced_kmer_candidate_bit_extract_name<Kernel>();
    }
    static bool available() { return spaced_kmer_candidate_bit_extract_available<Kernel>(); }
    static bool preferred() { return spaced_kmer_candidate_bit_extract_preferred<Kernel>(); }

    template <std::size_t MaxSpan, typename Callback>
    FISK_ALWAYS_INLINE void run(
        SpacedKmerMaskForms<L, MaxMasks, MaxSpan> const& mask_forms, std::string_view seq,
        Callback& callback
    ) const
    {
        DefaultWordEncoder<E, L> const encoder{};
        auto vector_callback = spaced_kmer_candidate_vector_callback(callback);
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

} // namespace detail

// -------------------------------------------------------------------------------------------------
//     Packed
// -------------------------------------------------------------------------------------------------

namespace detail {

template <InputCheck C, Encoding E, Layout L, std::size_t MaxMasks, typename BitExtractForm>
struct SpacedKmerCandidatePackedBytewiseScalar
{
    static std::string name()
    {
        return "for_each_spaced_kmer_packed_" + input_check_name(C) + "_bytewise_scalar_" +
            spaced_kmer_candidate_bit_extract_name<BitExtractForm>();
    }
    static bool available()
    {
        return spaced_kmer_candidate_bit_extract_available<BitExtractForm>();
    }
    static bool preferred()
    {
        return spaced_kmer_candidate_bit_extract_preferred<BitExtractForm>();
    }

    template <std::size_t MaxSpan, typename Seq, typename Callback>
    FISK_ALWAYS_INLINE void run(
        SpacedKmerMaskForms<L, MaxMasks, MaxSpan> const& mask_forms, Seq const& seq,
        Callback& callback
    ) const
    {
        auto const bit_extract = spaced_kmer_candidate_bit_extract<BitExtractForm>();
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
        return "for_each_spaced_kmer_packed_" + input_check_name(C) + "_bytewise_simd_" +
            spaced_kmer_candidate_bit_extract_name<Kernel>();
    }
    static bool available() { return spaced_kmer_candidate_bit_extract_available<Kernel>(); }
    static bool preferred() { return spaced_kmer_candidate_bit_extract_preferred<Kernel>(); }

    template <std::size_t MaxSpan, typename Seq, typename Callback>
    FISK_ALWAYS_INLINE void run(
        SpacedKmerMaskForms<L, MaxMasks, MaxSpan> const& mask_forms, Seq const& seq,
        Callback& callback
    ) const
    {
        auto vector_callback = spaced_kmer_candidate_vector_callback(callback);
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

} // namespace detail

// -------------------------------------------------------------------------------------------------
//     Candidate Lists
// -------------------------------------------------------------------------------------------------

namespace detail {

// The estimate of for_each_spaced_kmer(), see visit_estimate(): the scalar loop with PEXT, where
// it is likely fast, else the SIMD loop with the butterfly kernel of the arch of the `Simds`, a
// TypeList, see CandidateSimdArchs, where the CPU has it, else the scalar loop with the butterfly,
// for all callbacks.
template <
    InputFormat I, InputCheck C, Encoding E, Layout L, std::size_t MaxMasks,
    typename Simds = CandidateSimdArchs
>
struct SpacedKmerEstimate;

template <
    InputFormat I, InputCheck C, Encoding E, Layout L, std::size_t MaxMasks, typename... Simd
>
struct SpacedKmerEstimate<I, C, E, L, MaxMasks, TypeList<Simd...>>
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

    using type = std::conditional_t<I == InputFormat::kAscii, Ascii, Packed>;
};

} // namespace detail

} // namespace fisk
