#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>

namespace fisk {

// =================================================================================================
//     SIMD Arch Concept
// =================================================================================================

// A SIMD arch is a stateless set of vector operations for one instruction set, operating on
// vectors of 64-bit lanes. They are the building blocks of the kernels, such as the bit extraction
// kernels in bit_extract/simd.hpp, which implement an algorithm for a given arch. SimdScalar is
// the trivial arch with a single lane, for the scalar variants of these kernels.
//
// Aligned loads and stores expect their memory to be aligned to the vector size, and take it as
// 64-bit words. Unaligned loads take any address, as raw memory, so that callers never need to form
// a misaligned pointer to a 64-bit word, whose value is unspecified. To build a vector
// from lane values that are held in registers instead, use from_values(): going through memory
// with several scalar stores and one vector load stalls the CPU on store forwarding. Conversely,
// for data in memory, a vector load is a single instruction, and preferable to from_values().
//
// Some operations only exist for some of the archs, and hence are not part of the SimdArch
// concept. Each of them has its own concept that refines SimdArch, to select an implementation
// by. The archs with a byte shuffle offer lookup_low_nibbles(), which SimdScalar and plain SSE2
// lack. The overload of store_two_bit_codes() that takes two vectors exists for the archs where
// storing two vectors together takes fewer instructions than one after the other.
// All archs but SSE2 can shift each lane by its own count, with shift_right_lanes() and
// shift_left_lanes(). SimdSSE2 has interleave_low() and interleave_high() instead, to pick lanes
// from vectors that were shifted as a whole.
//
// Comparing the lanes of two vectors gives a `lane_mask`, in the form native to the arch: a vector
// with all bits set in the equal lanes and zero in the others, or the mask register of AVX512.
// lane_mask_to_bits() turns it into bit `j` for lane `j`, as compare_eq_bytes_to_bits() does
// for the bytes of two vectors, and lane_mask_from_bits() turns such bits back into a lane mask.

/**
 * @brief Concept for a SIMD arch, such as SimdAVX2.
 */
template <typename Simd>
concept SimdArch = requires(
    typename Simd::simd_vector const v,
    typename Simd::shift_count const count,
    std::uint64_t const* in,
    void const* raw,
    std::uint64_t* out,
    char* bytes,
    std::uint64_t value,
    std::array<std::uint64_t, Simd::lanes> values,
    typename Simd::lane_mask const mask
) {
    { Simd::lanes } -> std::convertible_to<std::size_t>;
    Simd::load_aligned(in);
    Simd::load_unaligned(raw);
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
    { Simd::compare_eq_lanes(v, v) } -> std::same_as<typename Simd::lane_mask>;
    { Simd::lane_mask_all() } -> std::same_as<typename Simd::lane_mask>;
    { Simd::lane_mask_to_bits(mask) } -> std::same_as<std::uint64_t>;
    { Simd::lane_mask_from_bits(value) } -> std::same_as<typename Simd::lane_mask>;
    { Simd::compare_eq_bytes_to_bits(v, v) } -> std::same_as<std::uint64_t>;
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

} // namespace fisk
