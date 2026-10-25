#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "bit_extract/common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     SIMD Kernel Tests
// =================================================================================================

// All simd.hpp kernels share the same structural interface (a `simd_vector` type, a `lanes`
// constant, static load()/store()/from_values(), a bit_extract(simd_vector) SIMD path, a
// bit_extract_scalar() scalar fallback path, and construction from a single uint64_t mask), so
// one generic helper covers all of them. Checks all paths against the same bit_extract_oracle()
// used above, for every mask in the shared edge-case + random mask set.
template <typename KernelType>
static void check_kernel()
{
    std::vector<std::uint64_t> masks = edge_case_masks();
    for (auto const m : random_masks()) {
        masks.push_back(m);
    }

    std::vector<std::uint64_t> values = {
        0x0000000000000000ULL,
        0xFFFFFFFFFFFFFFFFULL,
    };
    for (auto const v : random_values()) {
        values.push_back(v);
    }

    constexpr std::size_t lanes = KernelType::lanes;

    for (auto const mask : masks) {
        KernelType const kernel(mask);

        // Scalar fallback path: every value checked directly.
        for (auto const value : values) {
            EXPECT_EQ(kernel.bit_extract_scalar(value), bit_extract_oracle(value, mask));
        }

        // SIMD-lane path: values processed in chunks of `lanes`, wrapping around to fill the
        // last chunk if `values.size()` is not a multiple of `lanes`.
        alignas(64) std::array<std::uint64_t, lanes> in_buf{};
        alignas(64) std::array<std::uint64_t, lanes> out_buf{};
        for (std::size_t chunk_start = 0; chunk_start < values.size(); chunk_start += lanes) {
            for (std::size_t i = 0; i < lanes; ++i) {
                in_buf[i] = values[(chunk_start + i) % values.size()];
            }
            auto const vec = KernelType::load(in_buf.data());
            auto const result = kernel.bit_extract(vec);
            KernelType::store(result, out_buf.data());
            for (std::size_t i = 0; i < lanes; ++i) {
                EXPECT_EQ(out_buf[i], bit_extract_oracle(in_buf[i], mask));
            }

            // Same via from_values(), which builds the vector from lane values.
            KernelType::store(kernel.bit_extract(KernelType::from_values(in_buf)), out_buf.data());
            for (std::size_t i = 0; i < lanes; ++i) {
                EXPECT_EQ(out_buf[i], bit_extract_oracle(in_buf[i], mask));
            }

            // Same via the static overload that takes the table instead, for kernels that have it.
            if constexpr (requires { KernelType::bit_extract(vec, kernel.mask); }) {
                KernelType::store(KernelType::bit_extract(vec, kernel.mask), out_buf.data());
                for (std::size_t i = 0; i < lanes; ++i) {
                    EXPECT_EQ(out_buf[i], bit_extract_oracle(in_buf[i], mask));
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
//     Butterfly Kernels
// -------------------------------------------------------------------------------------------------

TEST(BitExtractSimd, KernelButterflyScalar)
{
    check_kernel<BitExtractKernelButterfly<SimdScalar>>();
}

#ifdef FISK_HAS_SSE2

TEST(BitExtractSimd, KernelButterflySSE2)
{
    check_kernel<BitExtractKernelButterfly<SimdSSE2>>();
}

#endif

#ifdef FISK_HAS_AVX2

TEST(BitExtractSimd, KernelButterflyAVX2)
{
    check_kernel<BitExtractKernelButterfly<SimdAVX2>>();
}

#endif

#ifdef FISK_HAS_AVX512

TEST(BitExtractSimd, KernelButterflyAVX512)
{
    check_kernel<BitExtractKernelButterfly<SimdAVX512>>();
}

#endif

#ifdef FISK_HAS_NEON

TEST(BitExtractSimd, KernelButterflyNEON)
{
    check_kernel<BitExtractKernelButterfly<SimdNEON>>();
}

#endif

// -------------------------------------------------------------------------------------------------
//     Block Kernels
// -------------------------------------------------------------------------------------------------

// Each ISA tier is swept over UF = 1, 8 (the library default), and 32 (the low and high
// extremes plus the default of the or_reduce_ compile-time balanced-tree reduction).
TEST(BitExtractSimd, KernelBlockScalar)
{
    check_kernel<BitExtractKernelBlock<SimdScalar, 1>>();
    check_kernel<BitExtractKernelBlock<SimdScalar, 8>>();
    check_kernel<BitExtractKernelBlock<SimdScalar, 32>>();
}

#ifdef FISK_HAS_SSE2

TEST(BitExtractSimd, KernelBlockSSE2)
{
    check_kernel<BitExtractKernelBlock<SimdSSE2, 1>>();
    check_kernel<BitExtractKernelBlock<SimdSSE2, 8>>();
    check_kernel<BitExtractKernelBlock<SimdSSE2, 32>>();
}

#endif

#ifdef FISK_HAS_AVX2

TEST(BitExtractSimd, KernelBlockAVX2)
{
    check_kernel<BitExtractKernelBlock<SimdAVX2, 1>>();
    check_kernel<BitExtractKernelBlock<SimdAVX2, 8>>();
    check_kernel<BitExtractKernelBlock<SimdAVX2, 32>>();
}

#endif

#ifdef FISK_HAS_AVX512

TEST(BitExtractSimd, KernelBlockAVX512)
{
    check_kernel<BitExtractKernelBlock<SimdAVX512, 1>>();
    check_kernel<BitExtractKernelBlock<SimdAVX512, 8>>();
    check_kernel<BitExtractKernelBlock<SimdAVX512, 32>>();
}

#endif

#ifdef FISK_HAS_NEON

TEST(BitExtractSimd, KernelBlockNEON)
{
    check_kernel<BitExtractKernelBlock<SimdNEON, 1>>();
    check_kernel<BitExtractKernelBlock<SimdNEON, 8>>();
    check_kernel<BitExtractKernelBlock<SimdNEON, 32>>();
}

#endif

// -------------------------------------------------------------------------------------------------
//     PEXT Kernel
// -------------------------------------------------------------------------------------------------

#ifdef FISK_HAS_BMI2

TEST(BitExtractSimd, KernelPext)
{
    check_kernel<BitExtractKernelPEXT>();
}

#endif

// =================================================================================================
//     Dispatcher Tests
// =================================================================================================

// BitExtractKernelDispatcher is independent of which Kernel type is plugged in,
// so a single kernel type is enough here.
using DispatcherTestKernel = BitExtractKernelButterfly<SimdScalar>;

TEST(BitExtractSimd, DispatcherRoundTrip)
{
    // Checks that dispatcher.run()'s array slot i actually corresponds to input mask i.

    std::vector<std::uint64_t> const sample_values = {
        0x0000000000000000ULL,
        0xFFFFFFFFFFFFFFFFULL,
        random_values()[0],
        random_values()[1],
        random_values()[2],
    };

    // Every count of masks is a separate kernel array size in the dispatcher.
    for (std::size_t n = 1; n <= 16; ++n) {
        std::vector<std::uint64_t> masks;
        std::vector<DispatcherTestKernel> kernels;
        for (std::size_t i = 0; i < n; ++i) {
            masks.push_back(random_masks()[i]);
            kernels.push_back(DispatcherTestKernel(masks.back()));
        }

        auto const check_dispatcher = [&](auto const& dispatcher) {
            EXPECT_EQ(dispatcher.size(), n);
            dispatcher.run([&](auto const& kernels_arr) {
                EXPECT_EQ(kernels_arr.size(), n);
                for (std::size_t i = 0; i < n; ++i) {
                    for (auto const value : sample_values) {
                        EXPECT_EQ(
                            kernels_arr[i].bit_extract(value),
                            bit_extract_oracle(value, masks[i])
                        );
                    }
                }
            });
        };

        check_dispatcher(BitExtractKernelDispatcher<DispatcherTestKernel>(masks));
        check_dispatcher(BitExtractKernelDispatcher<DispatcherTestKernel>(kernels));
    }
}

TEST(BitExtractSimd, DispatcherInvalidMaskCount)
{
    // N=0 and N>16 must be rejected by both constructors;
    // N=1 and N=16 (the inclusive boundary values) must succeed.

    std::vector<std::uint64_t> const empty_masks;
    std::vector<std::uint64_t> const one_mask = {random_masks()[0]};
    std::vector<std::uint64_t> sixteen_masks;
    for (std::size_t i = 0; i < 16; ++i) {
        sixteen_masks.push_back(random_masks()[i]);
    }
    std::vector<std::uint64_t> seventeen_masks = sixteen_masks;
    seventeen_masks.push_back(random_masks()[16]);

    std::vector<DispatcherTestKernel> empty_kernels;
    std::vector<DispatcherTestKernel> one_kernel = {DispatcherTestKernel(one_mask[0])};
    std::vector<DispatcherTestKernel> sixteen_kernels;
    for (auto const m : sixteen_masks) {
        sixteen_kernels.push_back(DispatcherTestKernel(m));
    }
    std::vector<DispatcherTestKernel> seventeen_kernels;
    for (auto const m : seventeen_masks) {
        seventeen_kernels.push_back(DispatcherTestKernel(m));
    }

    // N=0 and N=17: both constructors must reject, for both masks and pre-built kernels.
    EXPECT_ANY_THROW(BitExtractKernelDispatcher<DispatcherTestKernel>(empty_masks));
    EXPECT_ANY_THROW(BitExtractKernelDispatcher<DispatcherTestKernel>(seventeen_masks));
    EXPECT_ANY_THROW(BitExtractKernelDispatcher<DispatcherTestKernel>(empty_kernels));
    EXPECT_ANY_THROW(BitExtractKernelDispatcher<DispatcherTestKernel>(seventeen_kernels));

    // N=1 and N=16: both constructors must succeed, for both masks and pre-built kernels.
    EXPECT_EQ(
        BitExtractKernelDispatcher<DispatcherTestKernel>(one_mask).size(), std::size_t{1}
    );
    EXPECT_EQ(
        BitExtractKernelDispatcher<DispatcherTestKernel>(sixteen_masks).size(), std::size_t{16}
    );
    EXPECT_EQ(
        BitExtractKernelDispatcher<DispatcherTestKernel>(one_kernel).size(), std::size_t{1}
    );
    EXPECT_EQ(
        BitExtractKernelDispatcher<DispatcherTestKernel>(sixteen_kernels).size(), std::size_t{16}
    );
}
