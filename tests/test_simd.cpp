#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/random.hpp"
#include "fisk/core/simd.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Concepts
// =================================================================================================

static_assert(SimdArch<SimdScalar>);
static_assert(BitExtractKernel<BitExtractKernelButterfly<SimdScalar>>);
static_assert(BitExtractKernel<BitExtractKernelBlock<SimdScalar>>);

#if defined(FISK_HAS_SSE2)
static_assert(SimdArch<SimdSSE2>);
static_assert(BitExtractKernel<BitExtractKernelButterfly<SimdSSE2>>);
static_assert(BitExtractKernel<BitExtractKernelBlock<SimdSSE2>>);
#endif

#if defined(FISK_HAS_AVX2)
static_assert(SimdArch<SimdAVX2>);
static_assert(BitExtractKernel<BitExtractKernelButterfly<SimdAVX2>>);
static_assert(BitExtractKernel<BitExtractKernelBlock<SimdAVX2>>);
#endif

#if defined(FISK_HAS_AVX512)
static_assert(SimdArch<SimdAVX512>);
static_assert(BitExtractKernel<BitExtractKernelButterfly<SimdAVX512>>);
static_assert(BitExtractKernel<BitExtractKernelBlock<SimdAVX512>>);
#endif

#if defined(FISK_HAS_NEON)
static_assert(SimdArch<SimdNEON>);
static_assert(BitExtractKernel<BitExtractKernelButterfly<SimdNEON>>);
static_assert(BitExtractKernel<BitExtractKernelBlock<SimdNEON>>);
#endif

#if defined(FISK_HAS_BMI2)
static_assert(BitExtractKernel<BitExtractKernelPEXT<1>>);
static_assert(BitExtractKernel<BitExtractKernelPEXT<8>>);
#endif

// =================================================================================================
//     Helpers
// =================================================================================================

template <SimdArch Simd>
using Lanes = std::array<std::uint64_t, Simd::lanes>;

template <SimdArch Simd>
static Lanes<Simd> to_lanes(typename Simd::simd_vector v)
{
    alignas(64) Lanes<Simd> out{};
    Simd::store_aligned(v, out.data());
    return out;
}

template <SimdArch Simd>
static typename Simd::simd_vector from_lanes(Lanes<Simd> const& in)
{
    alignas(64) Lanes<Simd> buf = in;
    return Simd::load_aligned(buf.data());
}

// Random lane values, with all-zero and all-one lanes mixed in for the edge cases.
template <SimdArch Simd>
static std::vector<Lanes<Simd>> random_lanes(std::uint64_t seed)
{
    Splitmix64 rng(seed);
    std::vector<Lanes<Simd>> result;
    for (std::size_t r = 0; r < 64; ++r) {
        Lanes<Simd> lanes{};
        for (std::size_t i = 0; i < Simd::lanes; ++i) {
            lanes[i] = rng.get_uint64();
        }
        result.push_back(lanes);
    }
    result.push_back(Lanes<Simd>{});
    Lanes<Simd> ones{};
    ones.fill(~std::uint64_t{0});
    result.push_back(ones);
    return result;
}

// =================================================================================================
//     Checks
// =================================================================================================

template <SimdArch Simd>
static void check_simd_memory(Lanes<Simd> const& in)
{
    EXPECT_TRUE(to_lanes<Simd>(from_lanes<Simd>(in)) == in);

    // Unaligned load from an odd offset into a byte buffer.
    alignas(64) unsigned char raw[sizeof(Lanes<Simd>) + 8]{};
    std::memcpy(raw + 3, in.data(), sizeof(Lanes<Simd>));
    auto const v = Simd::load_unaligned(reinterpret_cast<std::uint64_t const*>(raw + 3));
    EXPECT_TRUE(to_lanes<Simd>(v) == in);
}

template <SimdArch Simd>
static void check_simd_bitwise(Lanes<Simd> const& a, Lanes<Simd> const& b)
{
    auto const va = from_lanes<Simd>(a);
    auto const vb = from_lanes<Simd>(b);
    auto const out_and = to_lanes<Simd>(Simd::bitwise_and(va, vb));
    auto const out_or  = to_lanes<Simd>(Simd::bitwise_or(va, vb));
    auto const out_xor = to_lanes<Simd>(Simd::bitwise_xor(va, vb));
    for (std::size_t i = 0; i < Simd::lanes; ++i) {
        EXPECT_EQ(out_and[i], a[i] & b[i]);
        EXPECT_EQ(out_or[i],  a[i] | b[i]);
        EXPECT_EQ(out_xor[i], a[i] ^ b[i]);
    }
}

// Shift by each immediate in [1, 63], via a compile-time loop over the shift amounts.
template <SimdArch Simd, int... Ss>
static void check_simd_shift_immediate(Lanes<Simd> const& in, std::integer_sequence<int, Ss...>)
{
    auto const v = from_lanes<Simd>(in);
    ([&] {
        constexpr int S = Ss + 1;
        auto const out = to_lanes<Simd>(Simd::template shift_right<S>(v));
        for (std::size_t i = 0; i < Simd::lanes; ++i) {
            EXPECT_EQ(out[i], in[i] >> S);
        }
    }(), ...);
}

template <SimdArch Simd>
static void check_simd_shifts(Lanes<Simd> const& in)
{
    check_simd_shift_immediate<Simd>(in, std::make_integer_sequence<int, 63>{});

    auto const v = from_lanes<Simd>(in);
    for (std::uint64_t n = 0; n < 64; ++n) {
        auto const out = to_lanes<Simd>(Simd::shift_right(v, Simd::make_shift_count(n)));
        for (std::size_t i = 0; i < Simd::lanes; ++i) {
            EXPECT_EQ(out[i], in[i] >> n);
        }
    }
}

template <SimdArch Simd, bool ByteSwap>
static void check_simd_store_truncated(Lanes<Simd> const& in)
{
    // Guard bytes after the output, to check that nothing is written past it.
    constexpr std::size_t n_bytes = 2 * Simd::lanes;
    constexpr unsigned char guard = 0xAB;
    unsigned char out[n_bytes + 16];
    std::memset(out, guard, sizeof(out));
    Simd::template store_truncated_u16<ByteSwap>(
        from_lanes<Simd>(in), reinterpret_cast<char*>(out)
    );

    for (std::size_t i = 0; i < Simd::lanes; ++i) {
        std::uint16_t expected = static_cast<std::uint16_t>(in[i]);
        if constexpr (ByteSwap) {
            expected = byte_swap_16(expected);
        }
        std::uint16_t actual;
        std::memcpy(&actual, out + 2 * i, sizeof(actual));
        EXPECT_EQ(actual, expected);
    }
    for (std::size_t i = n_bytes; i < sizeof(out); ++i) {
        EXPECT_EQ(out[i], guard);
    }
}

template <SimdArch Simd>
static void check_simd_arch()
{
    Lanes<Simd> zeros{};
    EXPECT_TRUE(to_lanes<Simd>(Simd::zero()) == zeros);

    auto const lanes = random_lanes<Simd>(42);
    for (std::size_t r = 0; r < lanes.size(); ++r) {
        auto const& in = lanes[r];

        Lanes<Simd> broadcast{};
        broadcast.fill(in[0]);
        EXPECT_TRUE(to_lanes<Simd>(Simd::broadcast(in[0])) == broadcast);

        check_simd_memory<Simd>(in);
        check_simd_bitwise<Simd>(in, lanes[(r + 1) % lanes.size()]);
        check_simd_shifts<Simd>(in);

        auto const swapped = to_lanes<Simd>(Simd::byte_swap_lanes(from_lanes<Simd>(in)));
        for (std::size_t i = 0; i < Simd::lanes; ++i) {
            EXPECT_EQ(swapped[i], byte_swap_64(in[i]));
        }

        check_simd_store_truncated<Simd, false>(in);
        check_simd_store_truncated<Simd, true>(in);
    }
}

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SimdArch, Scalar)
{
    check_simd_arch<SimdScalar>();
}

#if defined(FISK_HAS_SSE2)
TEST(SimdArch, SSE2)
{
    check_simd_arch<SimdSSE2>();
}
#endif

#if defined(FISK_HAS_AVX2)
TEST(SimdArch, AVX2)
{
    check_simd_arch<SimdAVX2>();
}
#endif

#if defined(FISK_HAS_AVX512)
TEST(SimdArch, AVX512)
{
    check_simd_arch<SimdAVX512>();
}
#endif

#if defined(FISK_HAS_NEON)
TEST(SimdArch, NEON)
{
    check_simd_arch<SimdNEON>();
}
#endif
