#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/types.hpp"

// Adapted from https://github.com/Daniel-Liu-c0deb0t/cute-nucleotides/blob/master/src/n_to_bits.rs,
// encoding functions, translated from Rust.
// Source snapshot SHA-256: f423b0e00d2fe902bfb6f4734b578857a0867489a9784b7087e7b2b68f67e5f4
// Daniel Liu, cute-nucleotides (2020); the movemask kernel includes the improvement by aqrit.
// Adaptations: LUT initialized for every byte, as consistent tails; output into the bytes of a
// fisk::PackedSequence instead of 64-bit words, writing only the bytes of the tail word that it
// holds; alias-safe loads and stores.
//
// Their encoding, A=00, C=01, T=10, G=11 with the first base in the low bits, is fisk's ACTG
// encoding with LSB layout, so that the kernels are compared in that group. The table maps all of
// their algorithms to fisk. Only the two that differ from fisk's word encoders are ported here:
//
//   cute-nucleotides Method fisk ----------------------- ------------------------------------------
//   ----------------------------- n_to_bits_lut Scalar table, OR into 64-bit words
//   WordEncoderDirect<SimdScalar>;
//                                                                        tails of the ported kernels
//   n_to_bits_pext           Four PEXT on the words of a 32-byte vector  WordEncoderPext<SimdAVX2>
//   n_to_bits_shift          Mask, two shift-OR rounds, byte shuffle     WordEncoderFold
//   n_to_bits_mul            Mask, vpmulld as bit shuffle, byte shuffle  Ported: encode_mul();
//                                                                        same idea as
//                                                                        WordEncoderDirect, which
//                                                                        multiplies with vpmaddubsw
//   n_to_bits_movemask vpermq, byte unpacks, two vpmovmskb Ported: encode_movemask() bits_to_n_*
//   Decoding packed bases to ASCII None, not ported n_to_bits2_*, bits_to_n2_* Five-letter alphabet
//   with N, 7 bits per None, not ported: fisk keeps
//                            3 bases                                     validity bits instead
//
// MIT License
// Copyright (c) 2020 Daniel Liu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

namespace cute_nucleotides {

// =================================================================================================
//     Lookup Table Encoding
// =================================================================================================

/**
 * @brief Codes for every byte, extending the original entries for ACGTU and acgtu.
 */
inline constexpr std::array<std::uint8_t, 256> byte_lut = [] {
    std::array<std::uint8_t, 256> table{};
    for (std::size_t i = 0; i < table.size(); ++i) {
        table[i] = static_cast<std::uint8_t>((i >> 1) & 3);
    }
    return table;
}();

/**
 * @brief Encode the fewer than 32 bases after the last full vector, as the original does via its
 * LUT loop, into the one word that they fill.
 * @pre `seq.size() < 32`, and output has room for ceil(seq.size() / 4) bytes.
 */
inline void encode_lut_tail(std::string_view seq, std::uint8_t* out) noexcept
{
    auto const* in = reinterpret_cast<unsigned char const*>(seq.data());
    std::uint64_t word = 0;
    for (std::size_t i = 0; i < seq.size(); ++i) {
        word |= static_cast<std::uint64_t>(byte_lut[in[i]]) << ((i & 31) << 1);
    }
    std::memcpy(out, &word, (seq.size() + 3) / 4);
}

// =================================================================================================
//     Vector Encoding
// =================================================================================================

#if defined(FISK_HAS_AVX2)

/**
 * @brief Encode 32 bases per vector via multiplication by a constant that moves the code bits of
 * each 32-bit element into its top byte.
 * @pre Output has room for ceil(seq.size() / 4) bytes.
 */
inline void encode_mul(std::string_view seq, std::uint8_t* out) noexcept
{
    std::size_t const end_idx = seq.size() >> 5;

    __m256i const ascii_mask = _mm256_set1_epi8(0b00000110);
    __m256i const mul_mask = [] {
        std::uint32_t m = 0;
        // m |= 1 << (length - input byte offset + output bit offset - 1 LSB to ignore);
        m |= 1u << (32 -  8 + 0 - 1);
        m |= 1u << (32 - 16 + 2 - 1);
        m |= 1u << (32 - 24 + 4 - 1);
        m |= 1u << (32 - 32 + 6 - 1);
        return _mm256_set1_epi32(static_cast<int>(m));
    }();
    __m256i const shuffle_mask = _mm256_set_epi32(-1, -1, -1, 0x0F0B0703, -1, -1, -1, 0x0F0B0703);

    // The original alternates between two aligned buffers to read the 64-bit words back from.
    alignas(32) std::uint64_t arr[2][4];

    for (std::size_t i = 0; i < end_idx; ++i) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(seq.data() + (i << 5)));

        // mask out unimportant bits
        v = _mm256_and_si256(v, ascii_mask);

        // multiply to pack left exactly 4 nucleotides (8 bits)
        v = _mm256_mullo_epi32(v, mul_mask);

        // extract last 8 bits of every 32-bit integer
        std::size_t const arr_idx = i & 1;
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(arr[arr_idx]), _mm256_shuffle_epi8(v, shuffle_mask)
        );

        // combine first 32 bits from both lanes
        std::uint64_t const res = arr[arr_idx][0] | (arr[arr_idx][2] << 32);
        std::memcpy(out + (i << 3), &res, sizeof(res));
    }

    encode_lut_tail(seq.substr(end_idx << 5), out + (end_idx << 3));
}

/**
 * @brief Encode 32 bases per vector by gathering both bits of each code with `vpmovmskb`, after
 * interleaving them bytewise.
 * @pre Output has room for ceil(seq.size() / 4) bytes.
 */
inline void encode_movemask(std::string_view seq, std::uint8_t* out) noexcept
{
    std::size_t const end_idx = seq.size() >> 5;

    for (std::size_t i = 0; i < end_idx; ++i) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(seq.data() + (i << 5)));

        // permute because unpacks works on the low/high 64 bits in each lane
        v = _mm256_permute4x64_epi64(v, 0b11011000);

        // shift each group of two bits for each nucleotide to the end of each byte
        __m256i const lo = _mm256_slli_epi64(v, 6);
        __m256i const hi = _mm256_slli_epi64(v, 5);

        // interleave bytes then extract the bit at the end of each byte
        __m256i const a = _mm256_unpackhi_epi8(lo, hi);
        __m256i const b = _mm256_unpacklo_epi8(lo, hi);

        // zero extend after movemask
        auto const a_bits = static_cast<std::uint64_t>(
            static_cast<std::uint32_t>(_mm256_movemask_epi8(a))
        );
        auto const b_bits = static_cast<std::uint64_t>(
            static_cast<std::uint32_t>(_mm256_movemask_epi8(b))
        );

        std::uint64_t const res = (a_bits << 32) | b_bits;
        std::memcpy(out + (i << 3), &res, sizeof(res));
    }

    encode_lut_tail(seq.substr(end_idx << 5), out + (end_idx << 3));
}

#endif // FISK_HAS_AVX2

// =================================================================================================
//     Packed Sequence Adapter
// =================================================================================================

/**
 * @brief Reuse fisk output storage around a reference kernel, without changing its loop.
 */
template <typename Kernel>
FISK_ALWAYS_INLINE
inline void pack_sequence_unchecked(
    std::string_view seq,
    Kernel kernel,
    fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
) {
    out.resize(seq.size());
    kernel(seq, out.codes().data());
}

} // namespace cute_nucleotides
