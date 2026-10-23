#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/simd.hpp"

namespace fisk {

// =================================================================================================
//     Scalar
// =================================================================================================

/**
 * @brief Scalar SIMD arch, with a single lane in a plain `std::uint64_t`.
 */
struct SimdScalar
{
    using simd_vector = std::uint64_t;
    using shift_count = std::uint8_t;
    using lane_mask = std::uint64_t;
    static constexpr std::size_t lanes = 1;

    // -------------------------------------------
    //     Memory
    // -------------------------------------------

    static simd_vector load_aligned(std::uint64_t const* a) noexcept
    {
        return a[0];
    }

    static simd_vector load_unaligned(void const* a) noexcept
    {
        simd_vector v;
        std::memcpy(&v, a, sizeof(v));
        return v;
    }

    static void store_aligned(simd_vector v, std::uint64_t* out) noexcept
    {
        out[0] = v;
    }

    // -------------------------------------------
    //     Construction
    // -------------------------------------------

    static simd_vector from_values(std::array<std::uint64_t, lanes> values) noexcept
    {
        return values[0];
    }

    static simd_vector broadcast(std::uint64_t value) noexcept
    {
        return value;
    }

    static simd_vector zero() noexcept
    {
        return 0;
    }

    // -------------------------------------------
    //     Bitwise
    // -------------------------------------------

    static simd_vector bitwise_and(simd_vector a, simd_vector b) noexcept
    {
        return a & b;
    }

    static simd_vector bitwise_or(simd_vector a, simd_vector b) noexcept
    {
        return a | b;
    }

    static simd_vector bitwise_xor(simd_vector a, simd_vector b) noexcept
    {
        return a ^ b;
    }

    // -------------------------------------------
    //     Shifts
    // -------------------------------------------

    template <int S>
    static simd_vector shift_right(simd_vector v) noexcept
    {
        return v >> S;
    }

    static shift_count make_shift_count(std::uint64_t n) noexcept
    {
        return static_cast<shift_count>(n);
    }

    static simd_vector shift_right(simd_vector v, shift_count count) noexcept
    {
        return v >> count;
    }

    // Shift each lane by its own count, which needs to be below 64. With a single lane, these are
    // plain shifts.
    static simd_vector shift_right_lanes(simd_vector v, simd_vector counts) noexcept
    {
        return v >> counts;
    }

    static simd_vector shift_left_lanes(simd_vector v, simd_vector counts) noexcept
    {
        return v << counts;
    }

    // -------------------------------------------
    //     Compare
    // -------------------------------------------

    static lane_mask lane_mask_from_bits(std::uint64_t bits) noexcept
    {
        return std::uint64_t{0} - (bits & 1);
    }

    // Bit j set iff byte j of a and b are equal. The test of each byte for zero adds within its
    // low seven bits only, so that no carry crosses into the next byte, which keeps it exact. The
    // multiplication then gathers the eight high bits into the lowest byte.
    static std::uint64_t compare_eq_bytes_to_bits(simd_vector a, simd_vector b) noexcept
    {
        constexpr std::uint64_t low7 = 0x7F7F7F7F7F7F7F7FULL;
        std::uint64_t const x = a ^ b;
        std::uint64_t const zero_high = ~(((x & low7) + low7) | x | low7);
        return ((zero_high >> 7) * 0x0102040810204080ULL) >> 56;
    }

    // -------------------------------------------
    //     Bytes
    // -------------------------------------------

    static simd_vector byte_swap_lanes(simd_vector v) noexcept
    {
        return byte_swap_64(v);
    }

    template <bool ByteSwap>
    static void store_truncated_u16(simd_vector v, char* out) noexcept
    {
        std::uint16_t v16 = static_cast<std::uint16_t>(v);
        if constexpr (ByteSwap) {
            v16 = byte_swap_16(v16);
        }
        std::memcpy(out, &v16, sizeof(v16));
    }

    // Each byte of the vector holds a 2-bit code. Store them packed, four codes per byte, with
    // the first code of each four in the lowest bits of its byte, or in the highest for MsbFirst.
    template <bool MsbFirst>
    static void store_two_bit_codes(simd_vector v, char* out) noexcept
    {
        if constexpr (MsbFirst) {
            v = byte_swap_64(v);
        }
        std::uint16_t v16 = static_cast<std::uint16_t>(pack_two_bit_codes_(v));
        if constexpr (MsbFirst) {
            v16 = byte_swap_16(v16);
        }
        std::memcpy(out, &v16, sizeof(v16));
    }

    // Gather the 2-bit codes of the eight bytes into the low 16 bits, with the first code lowest.
    // Combining each two codes leaves four 4-bit values, 16 bits apart. The multiplication then
    // adds up shifted copies of them, such that they end up next to each other in bits 36 to 51.
    static std::uint64_t pack_two_bit_codes_(std::uint64_t v) noexcept
    {
        std::uint64_t const pairs = (v | (v >> 6)) & 0x000F000F000F000FULL;
        return ((pairs * 0x0000001001001001ULL) >> 36) & 0xFFFF;
    }
};

static_assert(SimdArch<SimdScalar>);
static_assert(SimdArchWithLaneShifts<SimdScalar>);

} // namespace fisk
