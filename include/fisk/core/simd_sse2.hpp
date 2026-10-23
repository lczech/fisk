#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/simd.hpp"

namespace fisk {

// =================================================================================================
//     SSE2
// =================================================================================================

#if defined(FISK_HAS_SSE2)

/**
 * @brief SSE2 SIMD arch, with two lanes.
 *
 * Uses the byte shuffle of SSSE3 where the build enables it.
 */
struct SimdSSE2
{
    using simd_vector = __m128i;
    using shift_count = __m128i;
    using lane_mask = __m128i;
    static constexpr std::size_t lanes = 2;

    // -------------------------------------------
    //     Memory
    // -------------------------------------------

    static simd_vector load_aligned(std::uint64_t const* a) noexcept
    {
        return _mm_load_si128(reinterpret_cast<__m128i const*>(a));
    }

    static simd_vector load_unaligned(void const* a) noexcept
    {
        return _mm_loadu_si128(static_cast<__m128i const*>(a));
    }

    static void store_aligned(simd_vector v, std::uint64_t* out) noexcept
    {
        _mm_store_si128(reinterpret_cast<__m128i*>(out), v);
    }

    // -------------------------------------------
    //     Construction
    // -------------------------------------------

    static simd_vector from_values(std::array<std::uint64_t, lanes> values) noexcept
    {
        return _mm_set_epi64x(
            static_cast<long long>(values[1]),
            static_cast<long long>(values[0])
        );
    }

    static simd_vector broadcast(std::uint64_t value) noexcept
    {
        static_assert(sizeof(long long) >= sizeof(std::uint64_t));
        return _mm_set1_epi64x(static_cast<long long>(value));
    }

    static simd_vector zero() noexcept
    {
        return _mm_setzero_si128();
    }

    // -------------------------------------------
    //     Bitwise
    // -------------------------------------------

    static simd_vector bitwise_and(simd_vector a, simd_vector b) noexcept
    {
        return _mm_and_si128(a, b);
    }

    static simd_vector bitwise_or(simd_vector a, simd_vector b) noexcept
    {
        return _mm_or_si128(a, b);
    }

    static simd_vector bitwise_xor(simd_vector a, simd_vector b) noexcept
    {
        return _mm_xor_si128(a, b);
    }

    // -------------------------------------------
    //     Shifts
    // -------------------------------------------

    template <int S>
    static simd_vector shift_right(simd_vector v) noexcept
    {
        return _mm_srli_epi64(v, S);
    }

    // The shift count is taken from the low 64 bits of the vector (x86 convention).
    static shift_count make_shift_count(std::uint64_t n) noexcept
    {
        return _mm_set_epi64x(0, static_cast<long long>(n));
    }

    static simd_vector shift_right(simd_vector v, shift_count count) noexcept
    {
        return _mm_srl_epi64(v, count);
    }

    static simd_vector shift_left(simd_vector v, shift_count count) noexcept
    {
        return _mm_sll_epi64(v, count);
    }

    // -------------------------------------------
    //     Lanes
    // -------------------------------------------

    // SSE2 can not shift each lane by its own count, see SimdArchWithLaneShifts. Lanes that need
    // different shifts are instead taken from two vectors that are each shifted as a whole:
    // lane 0 of `a` and of `b`, or lane 1 of `a` and of `b`.
    static simd_vector interleave_low(simd_vector a, simd_vector b) noexcept
    {
        return _mm_unpacklo_epi64(a, b);
    }

    static simd_vector interleave_high(simd_vector a, simd_vector b) noexcept
    {
        return _mm_unpackhi_epi64(a, b);
    }

    // -------------------------------------------
    //     Compare
    // -------------------------------------------

    // Both 32-bit halves of lane j keep bit j and compare it to itself, as SSE2 has no 64-bit
    // compare.
    static lane_mask lane_mask_from_bits(std::uint64_t bits) noexcept
    {
        __m128i const select = _mm_set_epi32(2, 2, 1, 1);
        __m128i const kept = _mm_and_si128(_mm_set1_epi32(static_cast<int>(bits)), select);
        return _mm_cmpeq_epi32(kept, select);
    }

    static std::uint64_t compare_eq_bytes_to_bits(simd_vector a, simd_vector b) noexcept
    {
        return static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi8(a, b)));
    }

    // -------------------------------------------
    //     Bytes
    // -------------------------------------------

    static simd_vector byte_swap_lanes(simd_vector v) noexcept
    {
        #if defined(FISK_HAS_SSSE3)
            return byte_swap_lanes_ssse3_(v);
        #else
            return byte_swap_lanes_sse2_(v);
        #endif
    }

    // Store the low 16 bits of each lane contiguously.
    template <bool ByteSwap>
    static void store_truncated_u16(simd_vector v, char* out) noexcept
    {
        #if defined(FISK_HAS_SSSE3)
            store_truncated_u16_ssse3_<ByteSwap>(v, out);
        #else
            store_truncated_u16_sse2_<ByteSwap>(v, out);
        #endif
    }

    // Each byte holds a 2-bit code. Store them packed, four codes per byte, see SimdScalar.
    template <bool MsbFirst>
    static void store_two_bit_codes(simd_vector v, char* out) noexcept
    {
        #if defined(FISK_HAS_SSSE3)
            store_two_bit_codes_ssse3_<MsbFirst>(v, out);
        #else
            store_two_bit_codes_sse2_<MsbFirst>(v, out);
        #endif
    }

    // SSE2 has no byte shuffle, so this swaps the two bytes of each 16-bit word, and then reverses
    // the order of the four words of each lane.
    static simd_vector byte_swap_lanes_sse2_(simd_vector v) noexcept
    {
        v = _mm_or_si128(_mm_slli_epi16(v, 8), _mm_srli_epi16(v, 8));
        v = _mm_shufflelo_epi16(v, _MM_SHUFFLE(0, 1, 2, 3));
        return _mm_shufflehi_epi16(v, _MM_SHUFFLE(0, 1, 2, 3));
    }

    // SSE2 has no byte shuffle, but a mask, 16-bit byte swap, and byte shift are sufficient here.
    template <bool ByteSwap>
    static void store_truncated_u16_sse2_(simd_vector v, char* out) noexcept
    {
        __m128i low = _mm_and_si128(v, _mm_set1_epi64x(0xFFFF));
        if constexpr (ByteSwap) {
            low = _mm_or_si128(_mm_slli_epi16(low, 8), _mm_srli_epi16(low, 8));
        }
        low = _mm_or_si128(low, _mm_srli_si128(low, 6));
        std::uint32_t const packed = static_cast<std::uint32_t>(_mm_cvtsi128_si32(low));
        std::memcpy(out, &packed, sizeof(packed));
    }

    // SSE2 has no multiply-add of bytes. Instead, multiplying each 16-bit element moves its two
    // codes next to each other into its high byte: into the low half of that byte for the first
    // element of each pair, and into the high half for the second, so that adding the two gives
    // the packed byte. The two multipliers differing also keeps the compilers from replacing the
    // multiplication by several shifts.
    template <bool MsbFirst>
    static void store_two_bit_codes_sse2_(simd_vector v, char* out) noexcept
    {
        __m128i const factors = _mm_set1_epi32(MsbFirst ? 0x04014010 : 0x10400104);
        __m128i const halves = _mm_srli_epi16(_mm_mullo_epi16(v, factors), 8);
        __m128i const quads = _mm_madd_epi16(halves, _mm_set1_epi16(1));

        // Each 32-bit element now holds one packed byte.
        __m128i const words = _mm_packs_epi32(quads, quads);
        std::uint32_t const packed = static_cast<std::uint32_t>(
            _mm_cvtsi128_si32(_mm_packus_epi16(words, words))
        );
        std::memcpy(out, &packed, sizeof(packed));
    }

    #if defined(FISK_HAS_SSSE3)

    static simd_vector byte_swap_lanes_ssse3_(simd_vector v) noexcept
    {
        return _mm_shuffle_epi8(
            v, _mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8)
        );
    }

    // For each byte, look up its low four bits in the table. Bytes of 128 and above give zero.
    static simd_vector lookup_low_nibbles(
        simd_vector v, std::array<std::uint8_t, 16> const& table
    ) noexcept
    {
        return _mm_shuffle_epi8(
            _mm_loadu_si128(reinterpret_cast<__m128i const*>(table.data())), v
        );
    }

    template <bool ByteSwap>
    static void store_truncated_u16_ssse3_(simd_vector v, char* out) noexcept
    {
        __m128i const pack_control = [] {
            if constexpr (ByteSwap) {
                return _mm_setr_epi8(
                    1, 0, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
                );
            } else {
                return _mm_setr_epi8(
                    0, 1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
                );
            }
        }();
        std::uint32_t const packed = static_cast<std::uint32_t>(
            _mm_cvtsi128_si32(_mm_shuffle_epi8(v, pack_control))
        );
        std::memcpy(out, &packed, sizeof(packed));
    }

    // Two multiply-adds weigh the four codes of each 32-bit element by 1, 4, 16, and 64 (or the
    // reverse for MsbFirst) and add them up, into one packed byte per element.
    template <bool MsbFirst>
    static void store_two_bit_codes_ssse3_(simd_vector v, char* out) noexcept
    {
        __m128i const weights = _mm_set1_epi32(MsbFirst ? 0x01041040 : 0x40100401);
        __m128i const quads = _mm_madd_epi16(_mm_maddubs_epi16(v, weights), _mm_set1_epi16(1));
        __m128i const pack_control = _mm_setr_epi8(
            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        );
        std::uint32_t const packed = static_cast<std::uint32_t>(
            _mm_cvtsi128_si32(_mm_shuffle_epi8(quads, pack_control))
        );
        std::memcpy(out, &packed, sizeof(packed));
    }

    // Same as store_two_bit_codes() for one vector, but for two at once. Narrowing the 16-bit
    // elements of both vectors together saves the shuffle that collects the packed bytes.
    template <bool MsbFirst>
    static void store_two_bit_codes(simd_vector v0, simd_vector v1, char* out) noexcept
    {
        __m128i const weights = _mm_set1_epi32(MsbFirst ? 0x01041040 : 0x40100401);
        __m128i const halves = _mm_packus_epi16(
            _mm_maddubs_epi16(v0, weights), _mm_maddubs_epi16(v1, weights)
        );
        __m128i const words = _mm_maddubs_epi16(halves, _mm_set1_epi8(1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out), _mm_packus_epi16(words, words));
    }

    #endif // FISK_HAS_SSSE3
};

static_assert(SimdArch<SimdSSE2>);
#if defined(FISK_HAS_SSSE3)
static_assert(SimdArchWithByteShuffle<SimdSSE2>);
static_assert(SimdArchWithDoubleStore<SimdSSE2>);
#endif

#endif // FISK_HAS_SSE2

} // namespace fisk
