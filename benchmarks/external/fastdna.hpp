#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/types.hpp"

// Adapted from https://github.com/eitanf/fastdna/blob/main/fastdna.cc, encoding functions.
// Source snapshot SHA-256: 1c740fc71d769e71bd8b94263189b9abe2bb2dbc623c994c5718909970a316f4
// Corontzos and Frachtenberg, Direct-Coding DNA With Multilevel Parallelism (2024).
// Adaptations: initialized LUT, consistent tails, alias-safe loads/stores, portable unroll hints.
// Unroll factors 1 and 2 reproduce their configurations; 4 is an additional comparison.
//
// MIT License
// Copyright (c) 2023 Eitan Frachtenberg
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

namespace fastdna {

// =================================================================================================
//     Lookup Table Encoding
// =================================================================================================

/**
 * @brief ACTG codes for every byte, extending the original four initialized entries.
 */
inline constexpr std::array<std::uint8_t, 256> bit_map = [] {
    std::array<std::uint8_t, 256> table{};
    for (std::size_t i = 0; i < table.size(); ++i) {
        table[i] = static_cast<std::uint8_t>((i >> 1) & 3);
    }
    return table;
}();

/**
 * @brief Encode four bases per byte via the paper's LUT loop, with a consistent partial byte.
 * @pre Output has room for ceil(seq.size() / 4) bytes.
 */
template <fisk::Layout L>
inline void encode_lut(std::string_view seq, std::uint8_t* out) noexcept
{
    static_assert(L == fisk::Layout::kMSB || L == fisk::Layout::kLSB);
    auto const* in = reinterpret_cast<unsigned char const*>(seq.data());
    std::size_t i = 0;
    for (; seq.size() - i >= 4; i += 4) {
        if constexpr (L == fisk::Layout::kMSB) {
            *out = bit_map[in[i]];
            *out = static_cast<std::uint8_t>((*out << 2) | bit_map[in[i + 1]]);
            *out = static_cast<std::uint8_t>((*out << 2) | bit_map[in[i + 2]]);
            *out = static_cast<std::uint8_t>((*out << 2) | bit_map[in[i + 3]]);
        } else {
            *out = bit_map[in[i]];
            *out = static_cast<std::uint8_t>(*out | (bit_map[in[i + 1]] << 2));
            *out = static_cast<std::uint8_t>(*out | (bit_map[in[i + 2]] << 4));
            *out = static_cast<std::uint8_t>(*out | (bit_map[in[i + 3]] << 6));
        }
        ++out;
    }

    // Original tails reverse partial bytes; unordered tails also change layout.
    if (i < seq.size()) {
        *out = 0;
        for (std::size_t j = 0; i + j < seq.size(); ++j) {
            std::size_t const shift = L == fisk::Layout::kMSB ? 6 - 2 * j : 2 * j;
            *out = static_cast<std::uint8_t>(*out | (bit_map[in[i + j]] << shift));
        }
    }
}

// =================================================================================================
//     PEXT Encoding
// =================================================================================================

#if defined(FISK_HAS_BMI2)

#define FASTDNA_STR(x) #x
#if defined(__clang__)
    #define FASTDNA_UNROLL(n) _Pragma(FASTDNA_STR(clang loop unroll_count(n)))
#elif defined(__GNUC__)
    #define FASTDNA_UNROLL(n) _Pragma(FASTDNA_STR(GCC unroll n))
#else
    #define FASTDNA_UNROLL(n)
#endif

/**
 * @brief Encode eight bases per PEXT, preserving the original loop and unroll boundary.
 * @pre Output has room for ceil(seq.size() / 4) bytes.
 */
template <fisk::Layout L, int Unroll = 1>
inline void encode_pext(std::string_view seq, std::uint8_t* out) noexcept
{
    static_assert(L == fisk::Layout::kMSB || L == fisk::Layout::kLSB);
    static_assert(Unroll == 1 || Unroll == 2 || Unroll == 4);
    constexpr std::uint64_t mask = 0x0606060606060606ULL;
    constexpr std::size_t block_size = 8 * Unroll;
    std::size_t const limit = (seq.size() / block_size) * block_size;

    auto const step = [&](std::size_t i) {
        std::uint64_t word;
        std::memcpy(&word, seq.data() + i, sizeof(word));
        if constexpr (L == fisk::Layout::kMSB) {
            auto const encoded = _pext_u64(fisk::byte_swap_64(word), mask);
            *out++ = static_cast<std::uint8_t>(encoded >> 8);
            *out++ = static_cast<std::uint8_t>(encoded);
        } else {
            auto const encoded = static_cast<std::uint16_t>(_pext_u64(word, mask));
            std::memcpy(out, &encoded, sizeof(encoded));
            out += sizeof(encoded);
        }
    };

    // GCC < 14 rejects template parameters in unroll pragmas, so dispatch to literal factors.
    if constexpr (Unroll == 1) {
        FASTDNA_UNROLL(1)
        for (std::size_t i = 0; i < limit; i += 8) {
            step(i);
        }
    } else if constexpr (Unroll == 2) {
        FASTDNA_UNROLL(2)
        for (std::size_t i = 0; i < limit; i += 8) {
            step(i);
        }
    } else {
        FASTDNA_UNROLL(4)
        for (std::size_t i = 0; i < limit; i += 8) {
            step(i);
        }
    }
    encode_lut<L>(seq.substr(limit), out);
}

#undef FASTDNA_UNROLL
#undef FASTDNA_STR

#endif // FISK_HAS_BMI2

// =================================================================================================
//     Packed Sequence Adapter
// =================================================================================================

/**
 * @brief Reuse fisk output storage around a reference kernel, without changing its loop.
 */
template <fisk::Layout L, bool UsePext = false, int Unroll = 1>
FISK_ALWAYS_INLINE
inline void pack_sequence(
    std::string_view seq, fisk::PackedSequence<fisk::Encoding::kACTG, L>& out
) {
    out.resize(seq.size());
    if constexpr (UsePext) {
        #if defined(FISK_HAS_BMI2)
        encode_pext<L, Unroll>(seq, out.codes().data());
        #else
        static_assert(!UsePext, "The fastdna PEXT reference requires BMI2");
        #endif
    } else {
        encode_lut<L>(seq, out.codes().data());
    }
}

} // namespace fastdna
