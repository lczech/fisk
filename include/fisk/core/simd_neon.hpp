#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/simd.hpp"

namespace fisk {

// =================================================================================================
//     NEON
// =================================================================================================

#if defined(FISK_HAS_NEON)

/**
 * @brief ARM NEON SIMD arch, with two lanes.
 */
struct SimdNEON
{
    using simd_vector = uint64x2_t;
    using shift_count = int64x2_t;
    using lane_mask = uint64x2_t;
    static constexpr std::size_t lanes = 2;

    // -------------------------------------------
    //     Memory
    // -------------------------------------------

    static simd_vector load_aligned(std::uint64_t const* a) noexcept
    {
        return vld1q_u64(a);
    }

    static simd_vector load_unaligned(void const* a) noexcept
    {
        // Byte loads have no alignment requirement, unlike the 64-bit lane loads above.
        return vreinterpretq_u64_u8(vld1q_u8(static_cast<std::uint8_t const*>(a)));
    }

    static void store_aligned(simd_vector v, std::uint64_t* out) noexcept
    {
        vst1q_u64(out, v);
    }

    // -------------------------------------------
    //     Construction
    // -------------------------------------------

    static simd_vector from_values(std::array<std::uint64_t, lanes> values) noexcept
    {
        return vcombine_u64(
            vcreate_u64(values[0]),
            vcreate_u64(values[1])
        );
    }

    static simd_vector broadcast(std::uint64_t value) noexcept
    {
        return vdupq_n_u64(value);
    }

    static simd_vector zero() noexcept
    {
        return vdupq_n_u64(0);
    }

    // -------------------------------------------
    //     Bitwise
    // -------------------------------------------

    static simd_vector bitwise_and(simd_vector a, simd_vector b) noexcept
    {
        return vandq_u64(a, b);
    }

    static simd_vector bitwise_or(simd_vector a, simd_vector b) noexcept
    {
        return vorrq_u64(a, b);
    }

    static simd_vector bitwise_xor(simd_vector a, simd_vector b) noexcept
    {
        return veorq_u64(a, b);
    }

    // -------------------------------------------
    //     Shifts
    // -------------------------------------------

    // vshrq_n_u64() needs a compile time constant shift in [1, 64].
    template <int S>
    static simd_vector shift_right(simd_vector v) noexcept
    {
        return vshrq_n_u64(v, S);
    }

    // vshlq_u64() shifts left for positive counts, and right for negative ones.
    static shift_count make_shift_count(std::uint64_t n) noexcept
    {
        return vdupq_n_s64(-static_cast<std::int64_t>(n));
    }

    static simd_vector shift_right(simd_vector v, shift_count count) noexcept
    {
        return vshlq_u64(v, count);
    }

    // Shift each lane by its own count, which needs to be below 64. As above, vshlq_u64() shifts
    // right for negative counts.
    static simd_vector shift_right_lanes(simd_vector v, simd_vector counts) noexcept
    {
        return vshlq_u64(v, vnegq_s64(vreinterpretq_s64_u64(counts)));
    }

    static simd_vector shift_left_lanes(simd_vector v, simd_vector counts) noexcept
    {
        return vshlq_u64(v, vreinterpretq_s64_u64(counts));
    }

    // -------------------------------------------
    //     Compare
    // -------------------------------------------

    // Lane j keeps bit j and compares it to itself.
    static lane_mask lane_mask_from_bits(std::uint64_t bits) noexcept
    {
        uint64x2_t const select = vcombine_u64(vcreate_u64(1), vcreate_u64(2));
        return vceqq_u64(vandq_u64(vdupq_n_u64(bits), select), select);
    }

    // Keep bit j % 8 of each equal byte j, and add up each half of eight bytes, with three
    // pairwise additions that leave the sums of the halves in the two lowest bytes.
    static std::uint64_t compare_eq_bytes_to_bits(simd_vector a, simd_vector b) noexcept
    {
        uint8x16_t const equal = vceqq_u8(vreinterpretq_u8_u64(a), vreinterpretq_u8_u64(b));
        uint8x16_t bits = vandq_u8(
            equal, vreinterpretq_u8_u64(vdupq_n_u64(0x8040201008040201ULL))
        );
        bits = vpaddq_u8(bits, bits);
        bits = vpaddq_u8(bits, bits);
        bits = vpaddq_u8(bits, bits);
        return vgetq_lane_u16(vreinterpretq_u16_u8(bits), 0);
    }

    // -------------------------------------------
    //     Bytes
    // -------------------------------------------

    static simd_vector byte_swap_lanes(simd_vector v) noexcept
    {
        return vreinterpretq_u64_u8(vrev64q_u8(vreinterpretq_u8_u64(v)));
    }

    // For each byte, look up its low four bits in the table. The table lookup of NEON indexes by
    // the whole byte, so that the other bits are masked out first.
    static simd_vector lookup_low_nibbles(
        simd_vector v, std::array<std::uint8_t, 16> const& table
    ) noexcept
    {
        uint8x16_t const nibbles = vandq_u8(vreinterpretq_u8_u64(v), vdupq_n_u8(0x0F));
        return vreinterpretq_u64_u8(vqtbl1q_u8(vld1q_u8(table.data()), nibbles));
    }

    // Store the low 16 bits of each lane contiguously.
    template <bool ByteSwap>
    static void store_truncated_u16(simd_vector v, char* out) noexcept
    {
        uint16x8_t low = vandq_u16(vreinterpretq_u16_u64(v), vdupq_n_u16(0xFFFF));
        if constexpr (ByteSwap) {
            low = vorrq_u16(vshlq_n_u16(low, 8), vshrq_n_u16(low, 8));
        }
        std::uint32_t const packed =
            static_cast<std::uint32_t>(vget_lane_u16(vget_low_u16(low), 0))
            | (static_cast<std::uint32_t>(vget_lane_u16(vget_high_u16(low), 0)) << 16);
        std::memcpy(out, &packed, sizeof(packed));
    }

    // Each byte holds a 2-bit code. Store them packed, four codes per byte, see SimdScalar. Two
    // shifts with accumulation or insertion combine first the two codes of each 16-bit element,
    // and then the two halves of each 32-bit element, into one packed byte per element. That
    // byte is its lowest one, or its third one for MsbFirst, and the narrowing to 16 bits keeps
    // it in the low byte.
    template <bool MsbFirst>
    static void store_two_bit_codes(simd_vector v, char* out) noexcept
    {
        uint32x4_t const quads = two_bit_code_quads_<MsbFirst>(v);
        uint16x4_t words;
        if constexpr (MsbFirst) {
            words = vshrn_n_u32(quads, 16);
        } else {
            words = vmovn_u32(quads);
        }

        // The packed bytes are the low bytes of the four 16-bit elements.
        uint8x8_t const bytes = vreinterpret_u8_u16(words);
        std::uint32_t const packed = vget_lane_u32(vreinterpret_u32_u8(vuzp1_u8(bytes, bytes)), 0);
        std::memcpy(out, &packed, sizeof(packed));
    }

    // Same as store_two_bit_codes() for one vector, but for two at once. Narrowing both vectors
    // into one, and then to their low bytes, stores all eight packed bytes directly, instead of
    // moving the four of each vector through a general-purpose register.
    template <bool MsbFirst>
    static void store_two_bit_codes(simd_vector v0, simd_vector v1, char* out) noexcept
    {
        uint32x4_t const quads0 = two_bit_code_quads_<MsbFirst>(v0);
        uint32x4_t const quads1 = two_bit_code_quads_<MsbFirst>(v1);
        uint16x8_t words;
        if constexpr (MsbFirst) {
            words = vshrn_high_n_u32(vshrn_n_u32(quads0, 16), quads1, 16);
        } else {
            words = vmovn_high_u32(vmovn_u32(quads0), quads1);
        }
        vst1_u8(reinterpret_cast<std::uint8_t*>(out), vmovn_u16(words));
    }

    // The packed byte of each 32-bit element, see store_two_bit_codes().
    template <bool MsbFirst>
    static uint32x4_t two_bit_code_quads_(simd_vector v) noexcept
    {
        uint16x8_t const codes = vreinterpretq_u16_u64(v);
        if constexpr (MsbFirst) {
            uint32x4_t const halves = vreinterpretq_u32_u16(
                vsraq_n_u16(vshlq_n_u16(codes, 2), codes, 8)
            );
            return vsliq_n_u32(halves, halves, 20);
        } else {
            uint32x4_t const halves = vreinterpretq_u32_u16(vsraq_n_u16(codes, codes, 6));
            return vsraq_n_u32(halves, halves, 12);
        }
    }
};

static_assert(SimdArch<SimdNEON>);
static_assert(SimdArchWithLaneShifts<SimdNEON>);
static_assert(SimdArchWithByteShuffle<SimdNEON>);
static_assert(SimdArchWithDoubleStore<SimdNEON>);

#endif // FISK_HAS_NEON

} // namespace fisk
