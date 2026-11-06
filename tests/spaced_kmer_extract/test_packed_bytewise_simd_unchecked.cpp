#include "spaced_kmer_extract/packed_simd_common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Extraction
// =================================================================================================

// Checks one kernel type for one set of masks, through the array overload, and, for a single mask,
// also through the single-kernel overload and the shorthand without mask_idx, against the oracle.
template <typename Kernel, Encoding E, Layout L, std::size_t N>
static void check_kernel(
    std::array<std::string, N> const& mask_strings, PackedSequence<E, L> const& packed,
    std::vector<SpacedEvent> const& expected
) {
    using Simd = typename Kernel::simd_arch;
    auto const kernels = make_kernels<Kernel, L>(mask_strings);
    auto const extract_array = [&](auto const& callback) {
        for_each_spaced_kmer_packed_unchecked_bytewise_simd(packed, std::span{kernels}, callback);
    };
    check_spaced_events(collect_vector_events<Simd, E, L>(N, extract_array), expected);

    if constexpr (N == 1) {
        auto const extract_single = [&](auto const& callback) {
            for_each_spaced_kmer_packed_unchecked_bytewise_simd(packed, kernels[0], callback);
        };
        check_spaced_events(collect_vector_events<Simd, E, L>(1, extract_single), expected);
        check_spaced_events(
            collect_vector_events<Simd, E, L, true>(1, extract_single), expected
        );
        check_spaced_events(
            collect_vector_events<Simd, E, L, true>(1, extract_array), expected
        );
    }
}

// Checks the butterfly and block shift kernels of every arch of the build, and the PEXT kernel, for
// one set of masks.
template <Encoding E, Layout L, std::size_t N>
static void check_all_kernels(std::array<std::string, N> const& mask_strings)
{
    WordEncoderButterfly<E, L> const encoder;
    for (auto const& seq : packed_test_sequences()) {
        auto const packed = pack_sequence_unchecked(seq, encoder);
        auto const expected = oracle_spaced_events<E, L>(seq, mask_strings);
        auto const check_arch = [&]<typename Simd>() {
            check_kernel<BitExtractKernelButterfly<Simd>>(mask_strings, packed, expected);
            check_kernel<BitExtractKernelBlockShift<Simd>>(mask_strings, packed, expected);
        };
        check_arch.template operator()<SimdScalar>();
        #if defined(FISK_HAS_SSE2)
        check_arch.template operator()<SimdSSE2>();
        #endif
        #if defined(FISK_HAS_AVX2)
        check_arch.template operator()<SimdAVX2>();
        #endif
        #if defined(FISK_HAS_AVX512)
        check_arch.template operator()<SimdAVX512>();
        #endif
        #if defined(FISK_HAS_NEON)
        check_arch.template operator()<SimdNEON>();
        #endif
        #if defined(FISK_HAS_BMI2)
        check_kernel<BitExtractKernelPext>(mask_strings, packed, expected);
        #endif
    }
}

// For every span in [1, 32], a single random mask, and a set of three random masks, plus masks
// that keep all positions, or only the first and the last.
template <Encoding E, Layout L>
static void check_extraction()
{
    Splitmix64 rng(9009);
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_all_kernels<E, L>(random_masks<1>(rng, span_k));
        check_all_kernels<E, L>(random_masks<3>(rng, span_k));
    }
    check_all_kernels<E, L, 1>({ std::string(32, '1') });
    check_all_kernels<E, L, 2>(
        { std::string(29, '1'), std::string("1").append(27, '0').append("1") }
    );
    check_all_kernels<E, L, 1>({ std::string("1").append(30, '0').append("1") });
}

// =================================================================================================
//     Error Contract
// =================================================================================================

// The kernels need to have the Layout of the sequence.
template <typename Seq, typename KernelMask>
concept CanExtractPackedSimd = requires(Seq const& seq, KernelMask const& kernel) {
    for_each_spaced_kmer_packed_unchecked_bytewise_simd(seq, kernel, NoopSpacedVectorCallback{});
};

static_assert(CanExtractPackedSimd<
    PackedSequence<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelButterfly<SimdScalar>>
>);
static_assert(CanExtractPackedSimd<
    PackedSequence<Encoding::kACTG, Layout::kLSB>,
    SpacedKmerMask<Layout::kLSB, BitExtractKernelBlockShift<SimdScalar>>
>);
static_assert(!CanExtractPackedSimd<
    PackedSequence<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kLSB, BitExtractKernelButterfly<SimdScalar>>
>);
static_assert(!CanExtractPackedSimd<
    PackedSequence<Encoding::kACGT, Layout::kMSB>, BitExtractKernelButterfly<SimdScalar>
>);

#if defined(FISK_HAS_BMI2)

static_assert(CanExtractPackedSimd<
    PackedSequence<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelPext>
>);

#endif

// The same for the checked extraction, which also needs the validity bits.
template <typename Seq, typename KernelMask>
concept CanExtractPackedCheckedSimd = requires(Seq const& seq, KernelMask const& kernel) {
    for_each_spaced_kmer_packed_checked_bytewise_simd(seq, kernel, NoopSpacedValidLanesCallback{});
};

static_assert(CanExtractPackedCheckedSimd<
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelButterfly<SimdScalar>>
>);
static_assert(!CanExtractPackedCheckedSimd<
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kLSB, BitExtractKernelButterfly<SimdScalar>>
>);
static_assert(!CanExtractPackedCheckedSimd<
    PackedSequence<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelButterfly<SimdScalar>>
>);

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SpacedKmerExtractPackedBytewiseSimd, UncheckedAcgtMsb)
{
    check_extraction<Encoding::kACGT, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, UncheckedAcgtLsb)
{
    check_extraction<Encoding::kACGT, Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, UncheckedActgMsb)
{
    check_extraction<Encoding::kACTG, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, UncheckedActgLsb)
{
    check_extraction<Encoding::kACTG, Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, ErrorContract)
{
    check_simd_error_contract<Layout::kMSB, kNarrowMaxSpan>();
    check_simd_error_contract<Layout::kLSB, kNarrowMaxSpan>();
}
