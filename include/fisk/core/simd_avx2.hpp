#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/simd.hpp"

namespace fisk {

// =================================================================================================
//     AVX2
// =================================================================================================

#if defined(FISK_HAS_AVX2)

/**
 * @brief AVX2 SIMD arch, with four lanes.
 */
struct SimdAVX2
{
    using simd_vector = __m256i;
    using shift_count = __m128i;
    using lane_mask = __m256i;
    static constexpr std::size_t lanes = 4;

    // -------------------------------------------
    //     Memory
    // -------------------------------------------

    static simd_vector load_aligned(std::uint64_t const* a) noexcept
    {
        return _mm256_load_si256(reinterpret_cast<__m256i const*>(a));
    }

    static simd_vector load_unaligned(void const* a) noexcept
    {
        return _mm256_loadu_si256(static_cast<__m256i const*>(a));
    }

    static void store_aligned(simd_vector v, std::uint64_t* out) noexcept
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(out), v);
    }

    // -------------------------------------------
    //     Construction
    // -------------------------------------------

    static simd_vector from_values(std::array<std::uint64_t, lanes> values) noexcept
    {
        return _mm256_set_epi64x(
            static_cast<long long>(values[3]), static_cast<long long>(values[2]),
            static_cast<long long>(values[1]), static_cast<long long>(values[0])
        );
    }

    static simd_vector broadcast(std::uint64_t value) noexcept
    {
        static_assert(sizeof(long long) >= sizeof(std::uint64_t));
        return _mm256_set1_epi64x(static_cast<long long>(value));
    }

    static simd_vector zero() noexcept
    {
        return _mm256_setzero_si256();
    }

    // -------------------------------------------
    //     Bitwise
    // -------------------------------------------

    static simd_vector bitwise_and(simd_vector a, simd_vector b) noexcept
    {
        return _mm256_and_si256(a, b);
    }

    static simd_vector bitwise_or(simd_vector a, simd_vector b) noexcept
    {
        return _mm256_or_si256(a, b);
    }

    static simd_vector bitwise_xor(simd_vector a, simd_vector b) noexcept
    {
        return _mm256_xor_si256(a, b);
    }

    // -------------------------------------------
    //     Shifts
    // -------------------------------------------

    template <int S>
    static simd_vector shift_right(simd_vector v) noexcept
    {
        return _mm256_srli_epi64(v, S);
    }

    // The shift count is taken from the low 64 bits of the vector (x86 convention).
    static shift_count make_shift_count(std::uint64_t n) noexcept
    {
        return _mm_set_epi64x(0, static_cast<long long>(n));
    }

    static simd_vector shift_right(simd_vector v, shift_count count) noexcept
    {
        return _mm256_srl_epi64(v, count);
    }

    // Shift each lane by its own count, which needs to be below 64.
    static simd_vector shift_right_lanes(simd_vector v, simd_vector counts) noexcept
    {
        return _mm256_srlv_epi64(v, counts);
    }

    static simd_vector shift_left_lanes(simd_vector v, simd_vector counts) noexcept
    {
        return _mm256_sllv_epi64(v, counts);
    }

    // -------------------------------------------
    //     Compare
    // -------------------------------------------

    static lane_mask compare_eq_lanes(simd_vector a, simd_vector b) noexcept
    {
        return _mm256_cmpeq_epi64(a, b);
    }

    static lane_mask lane_mask_all() noexcept
    {
        return _mm256_set1_epi64x(-1);
    }

    static std::uint64_t lane_mask_to_bits(lane_mask m) noexcept
    {
        return static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(m)));
    }

    // Lane j keeps bit j and compares it to itself.
    static lane_mask lane_mask_from_bits(std::uint64_t bits) noexcept
    {
        __m256i const select = _mm256_set_epi64x(8, 4, 2, 1);
        __m256i const kept = _mm256_and_si256(
            _mm256_set1_epi64x(static_cast<long long>(bits)), select
        );
        return _mm256_cmpeq_epi64(kept, select);
    }

    static std::uint64_t compare_eq_bytes_to_bits(simd_vector a, simd_vector b) noexcept
    {
        return static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b)));
    }

    // -------------------------------------------
    //     Bytes
    // -------------------------------------------

    // _mm256_shuffle_epi8 shuffles within each 128-bit half independently (no cross-half
    // indexing), so the 8-byte-reversal pattern is written once per half and repeated.
    static simd_vector byte_swap_lanes(simd_vector v) noexcept
    {
        static constexpr std::uint8_t pattern[32] = {
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8
        };
        simd_vector const ctrl = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(pattern));
        return _mm256_shuffle_epi8(v, ctrl);
    }

    // For each byte, look up its low four bits in the table. Bytes of 128 and above give zero.
    static simd_vector lookup_low_nibbles(
        simd_vector v, std::array<std::uint8_t, 16> const& table
    ) noexcept
    {
        return _mm256_shuffle_epi8(
            _mm256_broadcastsi128_si256(
                _mm_loadu_si128(reinterpret_cast<__m128i const*>(table.data()))
            ),
            v
        );
    }

    // Store the low 16 bits of each lane contiguously. AVX2 can select both low words per 128-bit
    // half in one byte shuffle, avoiding a lane spill.
    template <bool ByteSwap>
    static void store_truncated_u16(simd_vector v, char* out) noexcept
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
        __m128i const lo = _mm_shuffle_epi8(_mm256_castsi256_si128(v), pack_control);
        __m128i const hi = _mm_shuffle_epi8(_mm256_extracti128_si256(v, 1), pack_control);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out), _mm_unpacklo_epi32(lo, hi));
    }

    // Each byte holds a 2-bit code. Store them packed, four codes per byte, see SimdScalar. Two
    // multiply-adds weigh the four codes of each 32-bit element by 1, 4, 16, and 64 (or the
    // reverse for MsbFirst) and add them up, into one packed byte per element.
    template <bool MsbFirst>
    static void store_two_bit_codes(simd_vector v, char* out) noexcept
    {
        __m256i const weights = _mm256_set1_epi32(MsbFirst ? 0x01041040 : 0x40100401);
        __m256i const quads = _mm256_madd_epi16(
            _mm256_maddubs_epi16(v, weights), _mm256_set1_epi16(1)
        );
        __m256i const pack_control = _mm256_setr_epi8(
            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        );
        __m256i const bytes = _mm256_shuffle_epi8(quads, pack_control);
        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(out),
            _mm_unpacklo_epi32(_mm256_castsi256_si128(bytes), _mm256_extracti128_si256(bytes, 1))
        );
    }

    // Same as store_two_bit_codes() for one vector, but for two at once. Narrowing the 16-bit
    // elements of both vectors together saves most of the steps that collect the packed bytes
    // from the 128-bit halves. It leaves the four bytes of each half of each vector in separate
    // 32-bit elements, which the final permutation brings into order.
    template <bool MsbFirst>
    static void store_two_bit_codes(simd_vector v0, simd_vector v1, char* out) noexcept
    {
        __m256i const weights = _mm256_set1_epi32(MsbFirst ? 0x01041040 : 0x40100401);
        __m256i const halves = _mm256_packus_epi16(
            _mm256_maddubs_epi16(v0, weights), _mm256_maddubs_epi16(v1, weights)
        );
        __m256i const words = _mm256_maddubs_epi16(halves, _mm256_set1_epi8(1));
        __m256i const bytes = _mm256_packus_epi16(words, words);
        __m256i const order = _mm256_setr_epi32(0, 4, 1, 5, 0, 0, 0, 0);
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(out),
            _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(bytes, order))
        );
    }
};

static_assert(SimdArch<SimdAVX2>);
static_assert(SimdArchWithLaneShifts<SimdAVX2>);
static_assert(SimdArchWithByteShuffle<SimdAVX2>);
static_assert(SimdArchWithDoubleStore<SimdAVX2>);

#endif // FISK_HAS_AVX2

} // namespace fisk
