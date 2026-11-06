#include "spaced_kmer_extract/packed_simd_common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Extraction
// =================================================================================================

// As check_kernel(), for the checked extraction, whose vectors come with their valid lanes.
template <typename Kernel, Encoding E, Layout L, std::size_t N>
static void check_checked_kernel(
    std::array<std::string, N> const& mask_strings, PackedSequenceWithValidity<E, L> const& packed,
    std::vector<SpacedEvent> const& expected
) {
    using Simd = typename Kernel::simd_arch;
    std::size_t const span_k = mask_strings.front().size();
    std::size_t const positions = packed.length() < span_k ? 0 : packed.length() - span_k + 1;
    auto const kernels = make_kernels<Kernel, L>(mask_strings);
    auto const collect = [&]<bool Shorthand>(std::size_t mask_count, auto const& extract) {
        return collect_valid_lanes_events<Simd, E, L, Shorthand, true>(
            mask_count, positions, extract
        );
    };
    auto const extract_array = [&](auto const& callback) {
        for_each_spaced_kmer_packed_checked_bytewise_simd(packed, std::span{kernels}, callback);
    };
    check_spaced_events(collect.template operator()<false>(N, extract_array), expected);

    if constexpr (N == 1) {
        auto const extract_single = [&](auto const& callback) {
            for_each_spaced_kmer_packed_checked_bytewise_simd(packed, kernels[0], callback);
        };
        check_spaced_events(collect.template operator()<false>(1, extract_single), expected);
        check_spaced_events(collect.template operator()<true>(1, extract_single), expected);
        check_spaced_events(collect.template operator()<true>(1, extract_array), expected);
    }
}

// As check_all_kernels(), for the checked extraction, on sequences with invalid characters.
template <Encoding E, Layout L, std::size_t N>
static void check_all_checked_kernels(std::array<std::string, N> const& mask_strings)
{
    WordEncoderButterfly<E, L> const encoder;
    for (auto const& seq : packed_checked_test_sequences()) {
        auto const packed = pack_sequence_checked(seq, encoder);
        auto const expected = oracle_spaced_events<E, L>(seq, mask_strings);
        auto const check_arch = [&]<typename Simd>() {
            check_checked_kernel<BitExtractKernelButterfly<Simd>>(mask_strings, packed, expected);
            check_checked_kernel<BitExtractKernelBlockShift<Simd>>(mask_strings, packed, expected);
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
        check_checked_kernel<BitExtractKernelPext>(mask_strings, packed, expected);
        #endif
    }
}

// As check_extraction(), for the checked extraction.
template <Encoding E, Layout L>
static void check_checked_extraction()
{
    Splitmix64 rng(9229);
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_all_checked_kernels<E, L>(random_masks<1>(rng, span_k));
        check_all_checked_kernels<E, L>(random_masks<3>(rng, span_k));
    }
    check_all_checked_kernels<E, L, 1>({ std::string(32, '1') });
    check_all_checked_kernels<E, L, 1>({ std::string("1").append(30, '0').append("1") });
}

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SpacedKmerExtractPackedBytewiseSimd, CheckedAcgtMsb)
{
    check_checked_extraction<Encoding::kACGT, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, CheckedAcgtLsb)
{
    check_checked_extraction<Encoding::kACGT, Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, CheckedActgMsb)
{
    check_checked_extraction<Encoding::kACTG, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, CheckedActgLsb)
{
    check_checked_extraction<Encoding::kACTG, Layout::kLSB>();
}
