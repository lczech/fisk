#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/random.hpp"
#include "testing.hpp"

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Ground truth for bit extraction (PEXT semantics), written independently:
// Walk the mask from bit 0 up, and for every set mask bit, append the corresponding bit of `x`
// to the output. Every algorithm is checked against this for the same (x, mask).
static std::uint64_t bit_extract_oracle(std::uint64_t x, std::uint64_t mask)
{
    std::uint64_t out_val = 0;
    std::uint64_t out_pos = 0;
    for (unsigned bit = 0; bit < 64; ++bit) {
        std::uint64_t const bit_mask = std::uint64_t{1} << bit;
        if (mask & bit_mask) {
            if (x & bit_mask) {
                out_val |= (std::uint64_t{1} << out_pos);
            }
            ++out_pos;
        }
    }
    return out_val;
}

static std::vector<std::uint64_t> const& edge_case_masks()
{
    static std::vector<std::uint64_t> const masks = {
        0x0000000000000000ULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x0000000000000001ULL, // single bit, position 0
        0x0000000080000000ULL, // single bit, position 31
        0x8000000000000000ULL, // single bit, position 63
        0x5555555555555555ULL, // alternating bits: 32 runs of length 1, the block table's max
        0x00000000FFFF0000ULL, // one contiguous run
        0xF0F0F0F0F0F0F0F0ULL, // several contiguous runs of length 4
    };
    return masks;
}

static std::vector<std::uint64_t> const& random_masks()
{
    static std::vector<std::uint64_t> const masks = [] {
        std::vector<std::uint64_t> m;
        Splitmix64 rng(1001);
        for (int i = 0; i < 50; ++i) {
            m.push_back(rng.get_uint64());
        }
        return m;
    }();
    return masks;
}

static std::vector<std::uint64_t> const& random_values()
{
    static std::vector<std::uint64_t> const values = [] {
        std::vector<std::uint64_t> v;
        Splitmix64 rng(2002);
        for (int i = 0; i < 100; ++i) {
            v.push_back(rng.get_uint64());
        }
        return v;
    }();
    return values;
}

// Run `func(value, mask)` against bit_extract_oracle() for the full shared mask x value set.
template <typename Func>
static void check_against_oracle(Func&& func)
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

    for (auto const mask : masks) {
        for (auto const value : values) {
            EXPECT_EQ(func(value, mask), bit_extract_oracle(value, mask));
        }
    }
}

// =================================================================================================
//     Bit Extraction Tests
// =================================================================================================

TEST(BitExtract, Bitloop)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        return bit_extract_bitloop(x, BitExtractMask(mask));
    });
}

TEST(BitExtract, Split32)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        return bit_extract_split32(x, BitExtractMask(mask));
    });
}

TEST(BitExtract, ByteTable)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        return bit_extract_byte_table(x, BitExtractMask(mask));
    });
}

TEST(BitExtract, BlockTable)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        auto const table = bit_extract_block_table_preprocess(mask);
        return bit_extract_block_table(x, table);
    });
}

// bit_extract_block_table_unrolled<UF>() is the same algorithm at different compile-time unroll
// factors; run every supported UF against the same oracle rather than treating them as separate
// tests, since they are not distinct algorithms.
template <std::size_t UF>
static void check_block_table_unrolled()
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        auto const table = bit_extract_block_table_preprocess(mask);
        return bit_extract_block_table_unrolled<UF>(x, table);
    });
}

TEST(BitExtract, BlockTableUnrolled)
{
    check_block_table_unrolled<1>();
    check_block_table_unrolled<2>();
    check_block_table_unrolled<4>();
    check_block_table_unrolled<8>();
    check_block_table_unrolled<16>();
    check_block_table_unrolled<32>();
}

TEST(BitExtract, ButterflyTable)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        auto const table = bit_extract_butterfly_table_preprocess(mask);
        return bit_extract_butterfly_table(x, table);
    });
}

#ifdef FISK_HAS_BMI2
TEST(BitExtract, Pext)
{
    check_against_oracle([](std::uint64_t x, std::uint64_t mask) {
        return bit_extract_pext(x, BitExtractMask(mask));
    });
}
#endif

// =================================================================================================
//     SIMD Kernel Tests
// =================================================================================================

// All simd.hpp kernels share the same structural interface (a `simd_vector` type, a `lanes`
// constant, static load()/store(), a bit_extract(simd_vector) SIMD path, a bit_extract(uint64_t)
// scalar fallback path, and construction from a single uint64_t mask), so one generic helper
// covers all of them. Checks both paths against the same bit_extract_oracle() used above, for
// every mask in the shared edge-case + random mask set.
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
            EXPECT_EQ(kernel.bit_extract(value), bit_extract_oracle(value, mask));
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
        }
    }
}

// -----------------------------------------------------------------------------
//     Butterfly Kernels
// -----------------------------------------------------------------------------

TEST(BitExtractSimd, KernelButterflyScalar)
{
    check_kernel<BitExtractKernelButterflyScalar>();
}

#ifdef FISK_HAS_SSE2
TEST(BitExtractSimd, KernelButterflySSE2)
{
    check_kernel<BitExtractKernelButterflySSE2>();
}
#endif

#ifdef FISK_HAS_AVX2
TEST(BitExtractSimd, KernelButterflyAVX2)
{
    check_kernel<BitExtractKernelButterflyAVX2>();
}
#endif

#ifdef FISK_HAS_AVX512
TEST(BitExtractSimd, KernelButterflyAVX512)
{
    check_kernel<BitExtractKernelButterflyAVX512>();
}
#endif

#ifdef FISK_HAS_NEON
TEST(BitExtractSimd, KernelButterflyNEON)
{
    check_kernel<BitExtractKernelButterflyNEON>();
}
#endif

// -----------------------------------------------------------------------------
//     Block Kernels
// -----------------------------------------------------------------------------

// Each ISA tier is swept over UF = 1, 8 (the library default), and 32 (the low and high
// extremes plus the default of the or_reduce_ compile-time balanced-tree reduction).

TEST(BitExtractSimd, KernelBlockScalar)
{
    check_kernel<BitExtractKernelBlockScalar<1>>();
    check_kernel<BitExtractKernelBlockScalar<8>>();
    check_kernel<BitExtractKernelBlockScalar<32>>();
}

#ifdef FISK_HAS_SSE2
TEST(BitExtractSimd, KernelBlockSSE2)
{
    check_kernel<BitExtractKernelBlockSSE2<1>>();
    check_kernel<BitExtractKernelBlockSSE2<8>>();
    check_kernel<BitExtractKernelBlockSSE2<32>>();
}
#endif

#ifdef FISK_HAS_AVX2
TEST(BitExtractSimd, KernelBlockAVX2)
{
    check_kernel<BitExtractKernelBlockAVX2<1>>();
    check_kernel<BitExtractKernelBlockAVX2<8>>();
    check_kernel<BitExtractKernelBlockAVX2<32>>();
}
#endif

#ifdef FISK_HAS_AVX512
TEST(BitExtractSimd, KernelBlockAVX512)
{
    check_kernel<BitExtractKernelBlockAVX512<1>>();
    check_kernel<BitExtractKernelBlockAVX512<8>>();
    check_kernel<BitExtractKernelBlockAVX512<32>>();
}
#endif

#ifdef FISK_HAS_NEON
TEST(BitExtractSimd, KernelBlockNEON)
{
    check_kernel<BitExtractKernelBlockNEON<1>>();
    check_kernel<BitExtractKernelBlockNEON<8>>();
    check_kernel<BitExtractKernelBlockNEON<32>>();
}
#endif

// -----------------------------------------------------------------------------
//     PEXT Kernel
// -----------------------------------------------------------------------------

#ifdef FISK_HAS_BMI2
TEST(BitExtractSimd, KernelPext)
{
    check_kernel<BitExtractKernelPEXT<1>>();
    check_kernel<BitExtractKernelPEXT<2>>();
    check_kernel<BitExtractKernelPEXT<4>>();
    check_kernel<BitExtractKernelPEXT<8>>();
}
#endif
