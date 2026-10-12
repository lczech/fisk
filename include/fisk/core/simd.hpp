#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "fisk/core/intrinsics.hpp"

namespace fisk {

// =================================================================================================
//     SIMD Arch Concept
// =================================================================================================

// A SIMD arch is a stateless set of vector operations for one instruction set, operating on
// vectors of 64-bit lanes. They are the building blocks of the kernels, such as the bit extraction
// kernels in bit_extract/simd.hpp, which implement an algorithm for a given arch. SimdScalar is
// the trivial arch with a single lane, for the scalar variants of these kernels.
//
// Aligned loads and stores expect their memory to be aligned to the vector size. To build a vector
// from lane values that are held in registers instead, use from_values(): going through memory
// with several scalar stores and one vector load stalls the CPU on store forwarding. Conversely,
// for data in memory, a vector load is a single instruction, and preferable to from_values().

/**
 * @brief Concept for a SIMD arch, such as SimdAVX2.
 */
template <typename Simd>
concept SimdArch = requires(
    typename Simd::simd_vector const v,
    typename Simd::shift_count const count,
    std::uint64_t const* in,
    std::uint64_t* out,
    char* bytes,
    std::uint64_t value,
    std::array<std::uint64_t, Simd::lanes> values
) {
    { Simd::lanes } -> std::convertible_to<std::size_t>;
    Simd::load_aligned(in);
    Simd::load_unaligned(in);
    Simd::store_aligned(v, out);
    Simd::from_values(values);
    Simd::broadcast(value);
    Simd::zero();
    Simd::bitwise_and(v, v);
    Simd::bitwise_or(v, v);
    Simd::bitwise_xor(v, v);
    Simd::template shift_right<1>(v);
    Simd::make_shift_count(value);
    Simd::shift_right(v, count);
    Simd::byte_swap_lanes(v);
    Simd::template store_truncated_u16<false>(v, bytes);
    Simd::template store_truncated_u16<true>(v, bytes);
};

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
    static constexpr std::size_t lanes = 1;

    // -------------------------------------------
    //     Memory
    // -------------------------------------------

    static simd_vector load_aligned(std::uint64_t const* a) noexcept
    {
        return a[0];
    }

    static simd_vector load_unaligned(std::uint64_t const* a) noexcept
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
};

// =================================================================================================
//     SSE2
// =================================================================================================

#if defined(FISK_HAS_SSE2)

/**
 * @brief SSE2 SIMD arch, with two lanes.
 */
struct SimdSSE2
{
    using simd_vector = __m128i;
    using shift_count = __m128i;
    static constexpr std::size_t lanes = 2;

    // -------------------------------------------
    //     Memory
    // -------------------------------------------

    static simd_vector load_aligned(std::uint64_t const* a) noexcept
    {
        return _mm_load_si128(reinterpret_cast<__m128i const*>(a));
    }

    static simd_vector load_unaligned(std::uint64_t const* a) noexcept
    {
        return _mm_loadu_si128(reinterpret_cast<__m128i const*>(a));
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

    // -------------------------------------------
    //     Bytes
    // -------------------------------------------

    // SSE2 has no byte shuffle, so this swaps the 32-bit, 16-bit, and 8-bit halves in turn.
    static simd_vector byte_swap_lanes(simd_vector v) noexcept
    {
        simd_vector const m32 = broadcast(0x00000000FFFFFFFFULL);
        v = _mm_or_si128(
            _mm_slli_epi64(_mm_and_si128(v, m32), 32), _mm_srli_epi64(_mm_andnot_si128(m32, v), 32)
        );
        simd_vector const m16 = broadcast(0x0000FFFF0000FFFFULL);
        v = _mm_or_si128(
            _mm_slli_epi64(_mm_and_si128(v, m16), 16), _mm_srli_epi64(_mm_andnot_si128(m16, v), 16)
        );
        simd_vector const m8 = broadcast(0x00FF00FF00FF00FFULL);
        v = _mm_or_si128(
            _mm_slli_epi64(_mm_and_si128(v, m8), 8), _mm_srli_epi64(_mm_andnot_si128(m8, v), 8)
        );
        return v;
    }

    // Store the low 16 bits of each lane contiguously. SSE2 has no byte shuffle, but a mask,
    // 16-bit byte swap, and byte shift are sufficient here.
    template <bool ByteSwap>
    static void store_truncated_u16(simd_vector v, char* out) noexcept
    {
        __m128i low = _mm_and_si128(v, _mm_set1_epi64x(0xFFFF));
        if constexpr (ByteSwap) {
            low = _mm_or_si128(_mm_slli_epi16(low, 8), _mm_srli_epi16(low, 8));
        }
        low = _mm_or_si128(low, _mm_srli_si128(low, 6));
        std::uint32_t const packed = static_cast<std::uint32_t>(_mm_cvtsi128_si32(low));
        std::memcpy(out, &packed, sizeof(packed));
    }
};

#endif // FISK_HAS_SSE2

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
    static constexpr std::size_t lanes = 4;

    // -------------------------------------------
    //     Memory
    // -------------------------------------------

    static simd_vector load_aligned(std::uint64_t const* a) noexcept
    {
        return _mm256_load_si256(reinterpret_cast<__m256i const*>(a));
    }

    static simd_vector load_unaligned(std::uint64_t const* a) noexcept
    {
        return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(a));
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
};

#endif // FISK_HAS_AVX2

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
};

#endif // FISK_HAS_AVX512

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
    static constexpr std::size_t lanes = 2;

    // -------------------------------------------
    //     Memory
    // -------------------------------------------

    static simd_vector load_aligned(std::uint64_t const* a) noexcept
    {
        return vld1q_u64(a);
    }

    static simd_vector load_unaligned(std::uint64_t const* a) noexcept
    {
        return vld1q_u64(a);
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

    // -------------------------------------------
    //     Bytes
    // -------------------------------------------

    static simd_vector byte_swap_lanes(simd_vector v) noexcept
    {
        return vreinterpretq_u64_u8(vrev64q_u8(vreinterpretq_u8_u64(v)));
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
};

#endif // FISK_HAS_NEON

} // namespace fisk
