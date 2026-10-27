#include <array>
#include <cstddef>
#include <stdexcept>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/core/types.hpp"
#include "fisk/spaced_kmer_extract/ascii_rolling_simd.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "simd_lanes.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

template <std::size_t N>
static std::array<std::string, N> generated_masks()
{
    std::array<std::string, N> masks{};
    for (std::size_t m = 0; m < N; ++m) {
        std::string mask(9, '0');
        mask.front() = '1';
        mask.back() = '1';
        for (std::size_t i = 1; i + 1 < mask.size(); ++i) {
            if (((i + m) % 3) != 0) {
                mask[i] = '1';
            }
        }
        masks[m] = std::move(mask);
    }
    if constexpr (N >= 3) {
        masks[N - 1] = masks[0];
    }
    return masks;
}

template <std::size_t N>
static std::vector<std::string> to_vector(std::array<std::string, N> const& masks)
{
    return std::vector<std::string>(masks.begin(), masks.end());
}

// =================================================================================================
//     SIMD Extraction
// =================================================================================================

template <typename Kernel, Layout L, std::size_t N>
static std::array<SpacedKmerMask<L, Kernel>, N> make_kernels(
    std::array<std::string, N> const& masks
) {
    std::array<SpacedKmerMask<L, Kernel>, N> kernels{};
    for (std::size_t i = 0; i < N; ++i) {
        kernels[i] = make_spaced_kmer_kernel<Kernel, L>(masks[i]);
    }
    return kernels;
}

// The number of positions of spaced k-mers of span `span_k` in `seq`.
static std::size_t spaced_kmer_positions(std::string const& seq, std::size_t span_k)
{
    return seq.size() < span_k ? 0 : seq.size() - span_k + 1;
}

// Checks the SIMD extraction for one set of masks, for every sequence in the corpus, through the
// array overload, and, for a single mask, also through the single-kernel overload and the shorthand
// without mask_idx.
template <typename Kernel, Layout L, std::size_t N, typename Enc>
static void check_simd_set(std::array<std::string, N> const& mask_array, Enc const& enc)
{
    using Simd = typename Kernel::simd_arch;
    constexpr Encoding E = Enc::encoding;
    std::vector<std::string> const masks = to_vector(mask_array);
    auto const kernels = make_kernels<Kernel, L>(mask_array);
    std::size_t const span_k = masks.front().size();

    for (auto const& seq : sequence_corpus(span_k, Kernel::lanes)) {
        auto const expected = oracle_spaced_events<E, L>(seq, masks);
        std::size_t const positions = spaced_kmer_positions(seq, span_k);
        auto const extract_array = [&](auto const& callback) {
            for_each_spaced_kmer_ascii_checked_rolling_simd(seq, std::span{kernels}, enc, callback);
        };
        check_spaced_events(
            collect_valid_lanes_events<Simd, E, L>(N, positions, extract_array), expected
        );

        if constexpr (N == 1) {
            auto const extract_single = [&](auto const& callback) {
                for_each_spaced_kmer_ascii_checked_rolling_simd(seq, kernels[0], enc, callback);
            };
            check_spaced_events(
                collect_valid_lanes_events<Simd, E, L>(1, positions, extract_single), expected
            );
            check_spaced_events(
                collect_valid_lanes_events<Simd, E, L, true>(1, positions, extract_single),
                expected
            );
            check_spaced_events(
                collect_valid_lanes_events<Simd, E, L, true>(1, positions, extract_array),
                expected
            );
        }
    }
}

template <typename Kernel, Layout L>
static void check_simd_kernel_in_layout()
{
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_simd_set<Kernel, L>(
            std::array<std::string, 1>{mask_for_span(span_k)},
            CharEncoderTable<Encoding::kACGT>{}
        );
    }
    check_simd_set<Kernel, L>(generated_masks<3>(), CharEncoderTable<Encoding::kACGT>{});
}

template <typename Kernel>
static void check_simd_kernel()
{
    check_simd_kernel_in_layout<Kernel, Layout::kMSB>();
    check_simd_kernel_in_layout<Kernel, Layout::kLSB>();
}

template <typename Kernel, Layout L>
static void check_simd_dispatch_in_layout(std::vector<std::string> const& masks)
{
    auto const kernels = make_spaced_kmer_kernels<Kernel>(make_spaced_kmer_masks<L>(masks));
    std::size_t const span_k = masks.front().size();

    for (auto const& seq : sequence_corpus(span_k, Kernel::lanes)) {
        auto const expected = oracle_spaced_events<Encoding::kACGT, L>(seq, masks);
        auto const extract = [&](auto const& callback) {
            dispatch_static_span<kMaxDispatchedMasks>(kernels, [&](auto kernel_span) {
                for_each_spaced_kmer_ascii_checked_rolling_simd(
                    seq, kernel_span, CharEncoderTable<Encoding::kACGT>{}, callback
                );
            });
        };
        check_spaced_events(
            collect_valid_lanes_events<typename Kernel::simd_arch, Encoding::kACGT, L>(
                masks.size(), spaced_kmer_positions(seq, span_k), extract
            ),
            expected
        );
    }
}

template <typename Kernel>
static void check_simd_dispatch(std::vector<std::string> const& masks)
{
    check_simd_dispatch_in_layout<Kernel, Layout::kMSB>(masks);
    check_simd_dispatch_in_layout<Kernel, Layout::kLSB>(masks);
}

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SpacedKmerExtractAsciiRollingSimd, ButterflyScalar)
{
    check_simd_kernel<BitExtractKernelButterfly<SimdScalar>>();
}

TEST(SpacedKmerExtractAsciiRollingSimd, BlockScalar)
{
    check_simd_kernel<BitExtractKernelBlockShift<SimdScalar>>();
}

#if defined(FISK_HAS_SSE2)

TEST(SpacedKmerExtractAsciiRollingSimd, ButterflySse2)
{
    check_simd_kernel<BitExtractKernelButterfly<SimdSSE2>>();
}

TEST(SpacedKmerExtractAsciiRollingSimd, BlockSse2)
{
    check_simd_kernel<BitExtractKernelBlockShift<SimdSSE2>>();
}

#endif

#if defined(FISK_HAS_AVX2)

TEST(SpacedKmerExtractAsciiRollingSimd, ButterflyAvx2)
{
    check_simd_kernel<BitExtractKernelButterfly<SimdAVX2>>();
}

TEST(SpacedKmerExtractAsciiRollingSimd, BlockAvx2)
{
    check_simd_kernel<BitExtractKernelBlockShift<SimdAVX2>>();
}

#endif

#if defined(FISK_HAS_AVX512)

TEST(SpacedKmerExtractAsciiRollingSimd, ButterflyAvx512)
{
    check_simd_kernel<BitExtractKernelButterfly<SimdAVX512>>();
}

TEST(SpacedKmerExtractAsciiRollingSimd, BlockAvx512)
{
    check_simd_kernel<BitExtractKernelBlockShift<SimdAVX512>>();
}

#endif

#if defined(FISK_HAS_NEON)

TEST(SpacedKmerExtractAsciiRollingSimd, ButterflyNeon)
{
    check_simd_kernel<BitExtractKernelButterfly<SimdNEON>>();
}

TEST(SpacedKmerExtractAsciiRollingSimd, BlockNeon)
{
    check_simd_kernel<BitExtractKernelBlockShift<SimdNEON>>();
}

#endif

#if defined(FISK_HAS_BMI2)

TEST(SpacedKmerExtractAsciiRollingSimd, Pext)
{
    check_simd_kernel<BitExtractKernelPEXT>();
}

#endif

// The span is set by make_spaced_kmer_kernel(), so an invalid one is only possible by setting it by
// hand, and still has to be rejected.
TEST(SpacedKmerExtractAsciiRollingSimd, InvalidSpan)
{
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        auto kernel = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>("1");
        kernel.span_k = k;
        for_each_spaced_kmer_ascii_checked_rolling_simd(
            seq, kernel, CharEncoderTable<Encoding::kACGT>{}, [](std::size_t, auto, auto) {}
        );
    });
}

// All kernels of one extraction share one rolling window, and hence need the same span.
TEST(SpacedKmerExtractAsciiRollingSimd, MixedSpans)
{
    std::string const seq = "ACGTACGTACGT";

    using Kernel = SpacedKmerMask<Layout::kMSB, BitExtractKernelButterfly<SimdScalar>>;
    std::array<Kernel, 2> const kernels{{
        make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>("101"),
        make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>("10001")
    }};
    EXPECT_THROW(
        for_each_spaced_kmer_ascii_checked_rolling_simd(
            seq, std::span{kernels}, CharEncoderTable<Encoding::kACGT>{},
            [](std::size_t, std::size_t, auto, auto) {}
        ),
        std::invalid_argument
    );
}

TEST(SpacedKmerExtractAsciiRollingSimd, DispatchMaskCounts)
{
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_simd_dispatch<BitExtractKernelButterfly<SimdScalar>>({mask_for_span(span_k)});
    }
    check_simd_dispatch<BitExtractKernelButterfly<SimdScalar>>(to_vector(generated_masks<1>()));
    check_simd_dispatch<BitExtractKernelButterfly<SimdScalar>>(to_vector(generated_masks<2>()));
    check_simd_dispatch<BitExtractKernelButterfly<SimdScalar>>(to_vector(generated_masks<3>()));
    check_simd_dispatch<BitExtractKernelButterfly<SimdScalar>>(to_vector(generated_masks<9>()));
    check_simd_dispatch<BitExtractKernelButterfly<SimdScalar>>(to_vector(generated_masks<16>()));

    #if defined(FISK_HAS_AVX512)
    // Many masks on eight lanes.
    check_simd_dispatch<BitExtractKernelButterfly<SimdAVX512>>(to_vector(generated_masks<9>()));
    check_simd_dispatch<BitExtractKernelButterfly<SimdAVX512>>(to_vector(generated_masks<16>()));
    #endif
}

// One representative SIMD path uses ACTG too. Scalar tests above cover every scalar extractor
// under both encodings; this verifies the generic SIMD outer loop forwards its encoder unchanged.
TEST(SpacedKmerExtractAsciiRollingSimd, ActgForwarding)
{
    std::array<std::string, 3> const masks{"10101", "11101", "10001"};
    check_simd_set<BitExtractKernelButterfly<SimdScalar>, Layout::kMSB>(
        masks, CharEncoderTable<Encoding::kACTG>{}
    );
    check_simd_set<BitExtractKernelButterfly<SimdScalar>, Layout::kLSB>(
        masks, CharEncoderTable<Encoding::kACTG>{}
    );
}
