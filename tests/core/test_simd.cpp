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
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/types.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Concepts
// =================================================================================================

static_assert(BitExtractKernel<BitExtractKernelButterfly<SimdScalar>>);
static_assert(BitExtractKernel<BitExtractKernelBlock<SimdScalar>>);

#if defined(FISK_HAS_SSE2)

static_assert(BitExtractKernel<BitExtractKernelButterfly<SimdSSE2>>);
static_assert(BitExtractKernel<BitExtractKernelBlock<SimdSSE2>>);

#endif

#if defined(FISK_HAS_AVX2)

static_assert(BitExtractKernel<BitExtractKernelButterfly<SimdAVX2>>);
static_assert(BitExtractKernel<BitExtractKernelBlock<SimdAVX2>>);

#endif

#if defined(FISK_HAS_AVX512)

static_assert(BitExtractKernel<BitExtractKernelButterfly<SimdAVX512>>);
static_assert(BitExtractKernel<BitExtractKernelBlock<SimdAVX512>>);

#endif

#if defined(FISK_HAS_NEON)

static_assert(BitExtractKernel<BitExtractKernelButterfly<SimdNEON>>);
static_assert(BitExtractKernel<BitExtractKernelBlock<SimdNEON>>);

#endif

#if defined(FISK_HAS_BMI2)

static_assert(BitExtractKernel<BitExtractKernelPEXT>);

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
    EXPECT_TRUE(to_lanes<Simd>(Simd::from_values(in)) == in);

    // Unaligned load from an odd offset into a byte buffer.
    alignas(64) unsigned char raw[sizeof(Lanes<Simd>) + 8]{};
    std::memcpy(raw + 3, in.data(), sizeof(Lanes<Simd>));
    auto const v = Simd::load_unaligned(raw + 3);
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

// Expected output of store_two_bit_codes() for the 2-bit codes in the bytes of the given lanes.
template <SimdArch Simd>
static std::array<unsigned char, 2 * Simd::lanes> packed_two_bit_codes(
    Lanes<Simd> const& codes, bool msb_first
) {
    std::array<unsigned char, 2 * Simd::lanes> packed{};
    for (std::size_t i = 0; i < 8 * Simd::lanes; ++i) {
        unsigned const code = (codes[i / 8] >> (8 * (i % 8))) & 0x3;
        unsigned const shift = static_cast<unsigned>(msb_first ? 6 - 2 * (i % 4) : 2 * (i % 4));
        packed[i / 4] = static_cast<unsigned char>(packed[i / 4] | (code << shift));
    }
    return packed;
}

// Lanes with a 2-bit code in each byte, as store_two_bit_codes() expects them.
template <SimdArch Simd>
static Lanes<Simd> two_bit_codes(Lanes<Simd> lanes)
{
    for (auto& lane : lanes) {
        lane &= 0x0303030303030303ULL;
    }
    return lanes;
}

template <SimdArch Simd, bool MsbFirst>
static void check_simd_store_two_bit_codes(Lanes<Simd> const& in)
{
    // Guard bytes after the output, to check that nothing is written past it.
    constexpr std::size_t n_bytes = 2 * Simd::lanes;
    constexpr unsigned char guard = 0xAB;
    unsigned char out[n_bytes + 16];
    std::memset(out, guard, sizeof(out));
    auto const codes = two_bit_codes<Simd>(in);
    Simd::template store_two_bit_codes<MsbFirst>(
        from_lanes<Simd>(codes), reinterpret_cast<char*>(out)
    );

    auto const expected = packed_two_bit_codes<Simd>(codes, MsbFirst);
    for (std::size_t i = 0; i < n_bytes; ++i) {
        EXPECT_EQ(out[i], expected[i]);
    }
    for (std::size_t i = n_bytes; i < sizeof(out); ++i) {
        EXPECT_EQ(out[i], guard);
    }
}

// Shift by counts that differ between the lanes, starting at each count in [0, 63] for lane 0.
template <SimdArchWithLaneShifts Simd>
static void check_simd_lane_shifts(Lanes<Simd> const& in)
{
    auto const v = from_lanes<Simd>(in);
    for (std::uint64_t n = 0; n < 64; ++n) {
        Lanes<Simd> counts;
        for (std::size_t i = 0; i < Simd::lanes; ++i) {
            counts[i] = (n + 7 * i) % 64;
        }
        auto const c = from_lanes<Simd>(counts);
        auto const out_right = to_lanes<Simd>(Simd::shift_right_lanes(v, c));
        auto const out_left  = to_lanes<Simd>(Simd::shift_left_lanes(v, c));
        for (std::size_t i = 0; i < Simd::lanes; ++i) {
            EXPECT_EQ(out_right[i], in[i] >> counts[i]);
            EXPECT_EQ(out_left[i],  in[i] << counts[i]);
        }
    }
}

template <SimdArchWithDoubleStore Simd, bool MsbFirst>
static void check_simd_store_two_bit_codes_double(Lanes<Simd> const& in0, Lanes<Simd> const& in1)
{
    // Guard bytes after the output, to check that nothing is written past it.
    constexpr std::size_t n_bytes = 4 * Simd::lanes;
    constexpr unsigned char guard = 0xAB;
    unsigned char out[n_bytes + 16];
    std::memset(out, guard, sizeof(out));
    auto const codes0 = two_bit_codes<Simd>(in0);
    auto const codes1 = two_bit_codes<Simd>(in1);
    Simd::template store_two_bit_codes<MsbFirst>(
        from_lanes<Simd>(codes0), from_lanes<Simd>(codes1), reinterpret_cast<char*>(out)
    );

    auto const expected0 = packed_two_bit_codes<Simd>(codes0, MsbFirst);
    auto const expected1 = packed_two_bit_codes<Simd>(codes1, MsbFirst);
    for (std::size_t i = 0; i < n_bytes / 2; ++i) {
        EXPECT_EQ(out[i], expected0[i]);
        EXPECT_EQ(out[n_bytes / 2 + i], expected1[i]);
    }
    for (std::size_t i = n_bytes; i < sizeof(out); ++i) {
        EXPECT_EQ(out[i], guard);
    }
}

template <SimdArchWithByteShuffle Simd>
static void check_simd_lookup_low_nibbles(Lanes<Simd> const& in)
{
    static constexpr std::array<std::uint8_t, 16> table = {
        0x9E, 0x37, 0x79, 0xB9, 0x7F, 0x4A, 0x7C, 0x15,
        0xF3, 0x9C, 0xC0, 0x60, 0x5C, 0xED, 0xC8, 0x34
    };
    auto const out = to_lanes<Simd>(Simd::lookup_low_nibbles(from_lanes<Simd>(in), table));
    for (std::size_t i = 0; i < 8 * Simd::lanes; ++i) {
        std::uint8_t const byte   = static_cast<std::uint8_t>(in[i / 8] >> (8 * (i % 8)));
        std::uint8_t const actual = static_cast<std::uint8_t>(out[i / 8] >> (8 * (i % 8)));

        // Bytes of 128 and above give zero on the x86 archs, and are looked up as well on Neon.
        if (byte < 128 || actual != 0) {
            EXPECT_EQ(actual, table[byte & 0x0F]);
        }
    }
}

// The lanes of `mask`, where the arch has it as a vector, and otherwise the bits of the mask
// register. Told apart by size, as GCC warns about vector types as template arguments, such as of
// std::is_same_v.
template <SimdArch Simd>
static void check_simd_lane_mask(typename Simd::lane_mask mask, std::uint64_t expected_bits)
{
    if constexpr (sizeof(typename Simd::lane_mask) == sizeof(typename Simd::simd_vector)) {
        auto const out = to_lanes<Simd>(mask);
        for (std::size_t i = 0; i < Simd::lanes; ++i) {
            EXPECT_EQ(out[i], ((expected_bits >> i) & 1) ? ~std::uint64_t{0} : 0);
        }
    } else {
        EXPECT_EQ(static_cast<std::uint64_t>(mask), expected_bits);
    }
}

// Differ from `a` by each byte value at each byte, alone or next to unequal bytes.
template <SimdArch Simd>
static void check_simd_compare_eq_bytes(Lanes<Simd> const& a)
{
    constexpr std::size_t bytes = 8 * Simd::lanes;
    for (std::size_t at = 0; at < bytes; ++at) {
        for (unsigned diff = 0; diff < 256; ++diff) {
            for (std::uint64_t const others : {0x00ULL, 0x01ULL, 0x80ULL}) {
                Lanes<Simd> b = a;
                std::uint64_t expected = 0;
                for (std::size_t j = 0; j < bytes; ++j) {
                    std::uint64_t const d = j == at ? diff : (j % 2 ? others : 0);
                    b[j / 8] ^= d << (8 * (j % 8));
                    expected |= std::uint64_t{d == 0} << j;
                }
                EXPECT_EQ(
                    Simd::compare_eq_bytes_to_bits(from_lanes<Simd>(a), from_lanes<Simd>(b)),
                    expected
                );
            }
        }
    }
}

template <SimdArch Simd>
static void check_simd_arch()
{
    Lanes<Simd> zeros{};
    EXPECT_TRUE(to_lanes<Simd>(Simd::zero()) == zeros);
    for (std::uint64_t bits = 0; bits < (std::uint64_t{1} << Simd::lanes); ++bits) {
        check_simd_lane_mask<Simd>(Simd::lane_mask_from_bits(bits), bits);
        check_simd_lane_mask<Simd>(SimdVectorValidLanes<Simd>{bits}.lane_mask(), bits);
    }

    // A few random inputs, and the all-zero and all-one ones at the end.
    auto const byte_inputs = random_lanes<Simd>(7);
    for (std::size_t const r : {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
        check_simd_compare_eq_bytes<Simd>(byte_inputs[r]);
    }
    check_simd_compare_eq_bytes<Simd>(byte_inputs[byte_inputs.size() - 2]);
    check_simd_compare_eq_bytes<Simd>(byte_inputs[byte_inputs.size() - 1]);

    auto const lanes = random_lanes<Simd>(42);
    for (std::size_t r = 0; r < lanes.size(); ++r) {
        auto const& in = lanes[r];

        Lanes<Simd> broadcast{};
        broadcast.fill(in[0]);
        EXPECT_TRUE(to_lanes<Simd>(Simd::broadcast(in[0])) == broadcast);

        check_simd_memory<Simd>(in);
        check_simd_bitwise<Simd>(in, lanes[(r + 1) % lanes.size()]);
        check_simd_shifts<Simd>(in);
        if constexpr (SimdArchWithLaneShifts<Simd>) {
            check_simd_lane_shifts<Simd>(in);
        }

        auto const swapped = to_lanes<Simd>(Simd::byte_swap_lanes(from_lanes<Simd>(in)));
        for (std::size_t i = 0; i < Simd::lanes; ++i) {
            EXPECT_EQ(swapped[i], byte_swap_64(in[i]));
        }

        check_simd_store_truncated<Simd, false>(in);
        check_simd_store_truncated<Simd, true>(in);
        check_simd_store_two_bit_codes<Simd, false>(in);
        check_simd_store_two_bit_codes<Simd, true>(in);

        if constexpr (SimdArchWithByteShuffle<Simd>) {
            check_simd_lookup_low_nibbles<Simd>(in);
        }
        if constexpr (SimdArchWithDoubleStore<Simd>) {
            auto const& other = lanes[(r + 1) % lanes.size()];
            check_simd_store_two_bit_codes_double<Simd, false>(in, other);
            check_simd_store_two_bit_codes_double<Simd, true>(in, other);
        }
    }
}

// =================================================================================================
//     Tests
// =================================================================================================

TEST(CoreSimd, Scalar)
{
    check_simd_arch<SimdScalar>();
}

#if defined(FISK_HAS_SSE2)

TEST(CoreSimd, SSE2)
{
    check_simd_arch<SimdSSE2>();
}

// The operations that SimdSSE2 has in place of the shifts per lane.
TEST(CoreSimd, SSE2LaneHelpers)
{
    using Simd = SimdSSE2;
    auto const lanes = random_lanes<Simd>(42);
    for (std::size_t r = 0; r < lanes.size(); ++r) {
        auto const& a = lanes[r];
        auto const& b = lanes[(r + 1) % lanes.size()];
        auto const va = from_lanes<Simd>(a);
        auto const vb = from_lanes<Simd>(b);

        EXPECT_TRUE(to_lanes<Simd>(Simd::interleave_low(va, vb))  == (Lanes<Simd>{a[0], b[0]}));
        EXPECT_TRUE(to_lanes<Simd>(Simd::interleave_high(va, vb)) == (Lanes<Simd>{a[1], b[1]}));

        for (std::uint64_t n = 0; n < 64; ++n) {
            auto const out = to_lanes<Simd>(Simd::shift_left(va, Simd::make_shift_count(n)));
            for (std::size_t i = 0; i < Simd::lanes; ++i) {
                EXPECT_EQ(out[i], a[i] << n);
            }
        }
    }
}

// SimdSSE2 picks one of each of these per build, so that the test above only covers those.
TEST(CoreSimd, SSE2ByteHelpers)
{
    using Simd = SimdSSE2;
    auto const check_swap = [](auto const& swap, Lanes<Simd> const& in) {
        auto const out = to_lanes<Simd>(swap(from_lanes<Simd>(in)));
        for (std::size_t i = 0; i < Simd::lanes; ++i) {
            EXPECT_EQ(out[i], byte_swap_64(in[i]));
        }
    };
    auto const check_store = [](auto const& store, bool byte_swap, Lanes<Simd> const& in) {
        char out[2 * Simd::lanes];
        store(from_lanes<Simd>(in), out);
        for (std::size_t i = 0; i < Simd::lanes; ++i) {
            std::uint16_t const low = static_cast<std::uint16_t>(in[i]);
            std::uint16_t actual;
            std::memcpy(&actual, out + 2 * i, sizeof(actual));
            EXPECT_EQ(actual, byte_swap ? byte_swap_16(low) : low);
        }
    };

    auto const check_codes = [](auto const& store, bool msb_first, Lanes<Simd> const& in) {
        auto const codes = two_bit_codes<Simd>(in);
        char out[2 * Simd::lanes];
        store(from_lanes<Simd>(codes), out);
        auto const expected = packed_two_bit_codes<Simd>(codes, msb_first);
        for (std::size_t i = 0; i < sizeof(out); ++i) {
            EXPECT_EQ(static_cast<unsigned char>(out[i]), expected[i]);
        }
    };

    for (auto const& in : random_lanes<Simd>(42)) {
        check_swap(Simd::byte_swap_lanes_sse2_, in);
        check_store(Simd::store_truncated_u16_sse2_<false>, false, in);
        check_store(Simd::store_truncated_u16_sse2_<true>, true, in);
        check_codes(Simd::store_two_bit_codes_sse2_<false>, false, in);
        check_codes(Simd::store_two_bit_codes_sse2_<true>, true, in);

        #if defined(FISK_HAS_SSSE3)
            check_swap(Simd::byte_swap_lanes_ssse3_, in);
            check_store(Simd::store_truncated_u16_ssse3_<false>, false, in);
            check_store(Simd::store_truncated_u16_ssse3_<true>, true, in);
            check_codes(Simd::store_two_bit_codes_ssse3_<false>, false, in);
            check_codes(Simd::store_two_bit_codes_ssse3_<true>, true, in);
        #endif
    }
}

#endif

#if defined(FISK_HAS_AVX2)

TEST(CoreSimd, AVX2)
{
    check_simd_arch<SimdAVX2>();
}

#endif

#if defined(FISK_HAS_AVX512)

TEST(CoreSimd, AVX512)
{
    check_simd_arch<SimdAVX512>();
}

#endif

#if defined(FISK_HAS_NEON)

TEST(CoreSimd, NEON)
{
    check_simd_arch<SimdNEON>();
}

#endif
