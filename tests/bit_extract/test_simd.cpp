#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
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
TEST(BitExtractSimd, KernelBlockShiftScalar)
{
    check_kernel<BitExtractKernelBlockShift<SimdScalar, 1>>();
    check_kernel<BitExtractKernelBlockShift<SimdScalar, 8>>();
    check_kernel<BitExtractKernelBlockShift<SimdScalar, 32>>();
}

#ifdef FISK_HAS_SSE2

TEST(BitExtractSimd, KernelBlockShiftSSE2)
{
    check_kernel<BitExtractKernelBlockShift<SimdSSE2, 1>>();
    check_kernel<BitExtractKernelBlockShift<SimdSSE2, 8>>();
    check_kernel<BitExtractKernelBlockShift<SimdSSE2, 32>>();
}

#endif

#ifdef FISK_HAS_AVX2

TEST(BitExtractSimd, KernelBlockShiftAVX2)
{
    check_kernel<BitExtractKernelBlockShift<SimdAVX2, 1>>();
    check_kernel<BitExtractKernelBlockShift<SimdAVX2, 8>>();
    check_kernel<BitExtractKernelBlockShift<SimdAVX2, 32>>();
}

#endif

#ifdef FISK_HAS_AVX512

TEST(BitExtractSimd, KernelBlockShiftAVX512)
{
    check_kernel<BitExtractKernelBlockShift<SimdAVX512, 1>>();
    check_kernel<BitExtractKernelBlockShift<SimdAVX512, 8>>();
    check_kernel<BitExtractKernelBlockShift<SimdAVX512, 32>>();
}

#endif

#ifdef FISK_HAS_NEON

TEST(BitExtractSimd, KernelBlockShiftNEON)
{
    check_kernel<BitExtractKernelBlockShift<SimdNEON, 1>>();
    check_kernel<BitExtractKernelBlockShift<SimdNEON, 8>>();
    check_kernel<BitExtractKernelBlockShift<SimdNEON, 32>>();
}

#endif

// -------------------------------------------------------------------------------------------------
//     PEXT Kernel
// -------------------------------------------------------------------------------------------------

#ifdef FISK_HAS_BMI2

TEST(BitExtractSimd, KernelPext)
{
    check_kernel<BitExtractKernelPext>();
}

#endif
