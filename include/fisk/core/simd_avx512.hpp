#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/simd.hpp"

namespace fisk {

// =================================================================================================
//     AVX512
// =================================================================================================

#if defined(FISK_HAS_AVX512)

/**
 * @brief AVX512 SIMD arch, with eight lanes. Needs AVX512F and AVX512BW.
 */
struct SimdAVX512
{
    using simd_vector = __m512i;
    using shift_count = __m128i;
    using lane_mask = __mmask8;
    static constexpr std::size_t lanes = 8;

    // -------------------------------------------
    //     Memory
    // -------------------------------------------

    static simd_vector load_aligned(std::uint64_t const* a) noexcept
    {
        return _mm512_load_si512(reinterpret_cast<__m512i const*>(a));
    }

    static simd_vector load_unaligned(std::uint64_t const* a) noexcept
    {
        return _mm512_loadu_si512(reinterpret_cast<__m512i const*>(a));
    }

    static void store_aligned(simd_vector v, std::uint64_t* out) noexcept
    {
        _mm512_store_si512(reinterpret_cast<__m512i*>(out), v);
    }

    // -------------------------------------------
    //     Construction
    // -------------------------------------------

    static simd_vector from_values(std::array<std::uint64_t, lanes> values) noexcept
    {
        return _mm512_set_epi64(
            static_cast<long long>(values[7]), static_cast<long long>(values[6]),
            static_cast<long long>(values[5]), static_cast<long long>(values[4]),
            static_cast<long long>(values[3]), static_cast<long long>(values[2]),
            static_cast<long long>(values[1]), static_cast<long long>(values[0])
        );
    }

    static simd_vector broadcast(std::uint64_t value) noexcept
    {
        static_assert(sizeof(long long) >= sizeof(std::uint64_t));
        return _mm512_set1_epi64(static_cast<long long>(value));
    }

    static simd_vector zero() noexcept
    {
        return _mm512_setzero_si512();
    }

    // -------------------------------------------
    //     Bitwise
    // -------------------------------------------

    static simd_vector bitwise_and(simd_vector a, simd_vector b) noexcept
    {
        return _mm512_and_si512(a, b);
    }

    static simd_vector bitwise_or(simd_vector a, simd_vector b) noexcept
    {
        return _mm512_or_si512(a, b);
    }

    static simd_vector bitwise_xor(simd_vector a, simd_vector b) noexcept
    {
        return _mm512_xor_si512(a, b);
    }

    // -------------------------------------------
    //     Shifts
    // -------------------------------------------

    // The shift is a template parameter: GCC's -O0 macro form of _mm512_srli_epi64 warns on a
    // runtime unsigned shift (-Wsign-conversion).
    template <int S>
    static simd_vector shift_right(simd_vector v) noexcept
    {
        return _mm512_srli_epi64(v, static_cast<unsigned int>(S));
    }

    // The shift count is taken from the low 64 bits of the vector (x86 convention).
    static shift_count make_shift_count(std::uint64_t n) noexcept
    {
        return _mm_set_epi64x(0, static_cast<long long>(n));
    }

    static simd_vector shift_right(simd_vector v, shift_count count) noexcept
    {
        return _mm512_srl_epi64(v, count);
    }

    // Shift each lane by its own count, which needs to be below 64.
    static simd_vector shift_right_lanes(simd_vector v, simd_vector counts) noexcept
    {
        return _mm512_srlv_epi64(v, counts);
    }

    static simd_vector shift_left_lanes(simd_vector v, simd_vector counts) noexcept
    {
        return _mm512_sllv_epi64(v, counts);
    }

    // -------------------------------------------
    //     Compare
    // -------------------------------------------

    // The lane mask is the mask register of the compare, with bit j for lane j already.
    static lane_mask compare_eq_lanes(simd_vector a, simd_vector b) noexcept
    {
        return _mm512_cmpeq_epi64_mask(a, b);
    }

    static lane_mask lane_mask_all() noexcept
    {
        return static_cast<lane_mask>(0xFF);
    }

    static std::uint64_t lane_mask_to_bits(lane_mask m) noexcept
    {
        return m;
    }

    static std::uint64_t compare_eq_bytes_to_bits(simd_vector a, simd_vector b) noexcept
    {
        return _mm512_cmpeq_epi8_mask(a, b);
    }

    // -------------------------------------------
    //     Bytes
    // -------------------------------------------

    // _mm512_shuffle_epi8 (AVX512BW) shuffles within each 128-bit quarter independently (no
    // cross-quarter indexing), so the 8-byte-reversal pattern is written once and repeated 4x.
    static simd_vector byte_swap_lanes(simd_vector v) noexcept
    {
        static constexpr std::uint8_t pattern[64] = {
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8
        };
        simd_vector const ctrl = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(pattern));
        return _mm512_shuffle_epi8(v, ctrl);
    }

    // For each byte, look up its low four bits in the table. Bytes of 128 and above give zero.
    static simd_vector lookup_low_nibbles(
        simd_vector v, std::array<std::uint8_t, 16> const& table
    ) noexcept
    {
        return _mm512_shuffle_epi8(
            _mm512_broadcast_i32x4(
                _mm_loadu_si128(reinterpret_cast<__m128i const*>(table.data()))
            ),
            v
        );
    }

    // Store the low 16 bits of each lane contiguously.
    template <bool ByteSwap>
    static void store_truncated_u16(simd_vector v, char* out) noexcept
    {
        // Shuffle each 128-bit quarter to expose its two low words, then permute those words
        // across quarters. Both instructions are AVX512BW, so this stays within the project's
        // baseline rather than requiring AVX512VBMI2's byte-compress instruction.
        static constexpr std::uint8_t plain_shuffle_pattern[64] = {
            0, 1, 8, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            0, 1, 8, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            0, 1, 8, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            0, 1, 8, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
        };
        static constexpr std::uint8_t swap_shuffle_pattern[64] = {
            1, 0, 9, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            1, 0, 9, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            1, 0, 9, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
            1, 0, 9, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
        };
        static constexpr std::uint16_t word_indices[32] = {
            0, 1, 8, 9, 16, 17, 24, 25, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };
        std::uint8_t const* const shuffle_pattern
            = ByteSwap ? swap_shuffle_pattern : plain_shuffle_pattern
        ;
        __m512i const shuffled = _mm512_shuffle_epi8(
            v, _mm512_loadu_si512(reinterpret_cast<__m512i const*>(shuffle_pattern))
        );
        __m512i const compacted = _mm512_permutexvar_epi16(
            _mm512_loadu_si512(reinterpret_cast<__m512i const*>(word_indices)), shuffled
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out), _mm512_castsi512_si128(compacted));
    }

    // Each byte holds a 2-bit code. Store them packed, four codes per byte, see SimdScalar. Two
    // multiply-adds weigh the four codes of each 32-bit element by 1, 4, 16, and 64 (or the
    // reverse for MsbFirst) and add them up, into one packed byte per element.
    template <bool MsbFirst>
    static void store_two_bit_codes(simd_vector v, char* out) noexcept
    {
        __m512i const weights = _mm512_set1_epi32(MsbFirst ? 0x01041040 : 0x40100401);
        __m512i const quads = _mm512_madd_epi16(
            _mm512_maddubs_epi16(v, weights), _mm512_set1_epi16(1)
        );
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out), _mm512_cvtepi32_epi8(quads));
    }

    // Same as store_two_bit_codes() for one vector, but for two at once. Narrowing the 16-bit
    // elements of both vectors together saves one of the two narrowing steps. It leaves the four
    // bytes of each 128-bit lane of each vector in separate 32-bit elements, which the final
    // permutation brings into order.
    template <bool MsbFirst>
    static void store_two_bit_codes(simd_vector v0, simd_vector v1, char* out) noexcept
    {
        __m512i const weights = _mm512_set1_epi32(MsbFirst ? 0x01041040 : 0x40100401);
        __m512i const halves = _mm512_packus_epi16(
            _mm512_maddubs_epi16(v0, weights), _mm512_maddubs_epi16(v1, weights)
        );
        __m512i const words = _mm512_maddubs_epi16(halves, _mm512_set1_epi8(1));
        __m256i const bytes = _mm512_cvtepi16_epi8(words);
        __m256i const order = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(out), _mm256_permutevar8x32_epi32(bytes, order)
        );
    }
};

static_assert(SimdArch<SimdAVX512>);
static_assert(SimdArchWithLaneShifts<SimdAVX512>);
static_assert(SimdArchWithByteShuffle<SimdAVX512>);
static_assert(SimdArchWithDoubleStore<SimdAVX512>);

#endif // FISK_HAS_AVX512

} // namespace fisk
