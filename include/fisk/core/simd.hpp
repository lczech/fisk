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
//
// Some operations only exist for some of the archs, and hence are not part of the SimdArch
// concept. Each of them has its own concept that refines SimdArch, to select an implementation
// by. The archs with a byte shuffle offer lookup_low_nibbles(), which SimdScalar and plain SSE2
// lack. The overload of store_two_bit_codes() that takes two vectors exists for the archs where
// storing two vectors together is faster than one after the other.
// All archs but SSE2 can shift each lane by its own count, with shift_right_lanes() and
// shift_left_lanes(). SimdSSE2 has interleave_low() and interleave_high() instead, to pick lanes
// from vectors that were shifted as a whole.

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
    Simd::template store_two_bit_codes<false>(v, bytes);
    Simd::template store_two_bit_codes<true>(v, bytes);
};

/**
 * @brief Concept for a SIMD arch with a byte shuffle instruction.
 *
 * These archs offer lookup_low_nibbles(), which the concept checks for, and their
 * byte_swap_lanes() is a single instruction.
 */
template <typename Simd>
concept SimdArchWithByteShuffle = SimdArch<Simd> && requires(
    typename Simd::simd_vector const v,
    std::array<std::uint8_t, 16> const& table
) {
    Simd::lookup_low_nibbles(v, table);
};

/**
 * @brief Concept for a SIMD arch that can store the 2-bit codes of two vectors at once, see the
 * overload of store_two_bit_codes() for two vectors.
 */
template <typename Simd>
concept SimdArchWithDoubleStore = SimdArch<Simd> && requires(
    typename Simd::simd_vector const v,
    char* bytes
) {
    Simd::template store_two_bit_codes<false>(v, v, bytes);
    Simd::template store_two_bit_codes<true>(v, v, bytes);
};

/**
 * @brief Concept for a SIMD arch that can shift each lane by its own count, see
 * shift_right_lanes() and shift_left_lanes().
 */
template <typename Simd>
concept SimdArchWithLaneShifts = SimdArch<Simd> && requires(typename Simd::simd_vector const v) {
    Simd::shift_right_lanes(v, v);
    Simd::shift_left_lanes(v, v);
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
    // byte is its lowest one, or its third one for MsbFirst.
    template <bool MsbFirst>
    static void store_two_bit_codes(simd_vector v, char* out) noexcept
    {
        uint16x8_t const codes = vreinterpretq_u16_u64(v);
        uint16x4_t words;
        if constexpr (MsbFirst) {
            uint32x4_t const halves = vreinterpretq_u32_u16(
                vsraq_n_u16(vshlq_n_u16(codes, 2), codes, 8)
            );
            words = vshrn_n_u32(vsliq_n_u32(halves, halves, 20), 16);
        } else {
            uint32x4_t const halves = vreinterpretq_u32_u16(vsraq_n_u16(codes, codes, 6));
            words = vmovn_u32(vsraq_n_u32(halves, halves, 12));
        }

        // The packed bytes are the low bytes of the four 16-bit elements.
        uint8x8_t const bytes = vreinterpret_u8_u16(words);
        std::uint32_t const packed = vget_lane_u32(vreinterpret_u32_u8(vuzp1_u8(bytes, bytes)), 0);
        std::memcpy(out, &packed, sizeof(packed));
    }
};

#endif // FISK_HAS_NEON

} // namespace fisk
