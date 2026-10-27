#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>

#include "fisk/core/char_encoder.hpp"
#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_rolling_scalar.hpp"
#include "fisk/kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from ASCII, Rolling, SWAR
// =================================================================================================

/**
 * @brief Encode 8 ASCII characters into 2-bit nucleotide codes of Encoding `E`, using SWAR (SIMD
 * within a register) on a single 64-bit word.
 *
 * Returns an 8-bit mask with bit i set iff lane i is valid A/C/G/T (case-insensitive). As in
 * encode_32_nts_avx2_(), invalid lanes are not zeroed.
 */
template <Encoding E>
inline std::uint32_t encode_8_nts_swar_(char const* data, std::uint8_t* codes) noexcept
{
    std::uint64_t x;
    std::memcpy(&x, data, sizeof(x)); // unaligned-safe; usually optimized to one load

    // ASCII trick of ascii_bits_to_code_(), per byte. This works for both upper/lowercase ACGT
    // because the relevant lower bits are identical for upper and lower case. The final mask
    // also drops the bits shifted in from the neighboring byte.
    std::uint64_t enc;
    if constexpr (E == Encoding::kACGT) {
        enc = ((x >> 1) ^ (x >> 2)) & 0x0303030303030303ull;
    } else if constexpr (E == Encoding::kACTG) {
        enc = (x >> 1) & 0x0303030303030303ull;
    } else {
        static_assert(dependent_false_v<E>, "Unhandled Encoding in encode_8_nts_swar_()");
    }

    // Little-endian, so that byte i of the word, and hence codes[i], is character i.
    std::memcpy(codes, &enc, sizeof(enc));

    // ASCII uppercase: clears the lowercase bit, so a/c/g/t become A/C/G/T.
    x &= 0xDFDFDFDFDFDFDFDFull;

    // Compare against 'A', 'C', 'G', 'T' in parallel,
    // getting 0x80 in each byte where there's a match.
    // Exact zero-byte test, as no carry crosses a byte (all addends < 0x80). The common
    // `(z - lo) & ~z & hi` is not exact: a zero byte's borrow also flags a following 0x01.
    auto byte_eq = [](std::uint64_t v, std::uint64_t c) noexcept {
        constexpr std::uint64_t low7 = 0x7F7F7F7F7F7F7F7Full;
        std::uint64_t const z = v ^ c;                  // zero byte where equal
        return ~(((z & low7) + low7) | z | low7);       // high bit set in equal bytes
    };
    std::uint64_t const m =
        byte_eq(x, 0x4141414141414141ull) |   // A
        byte_eq(x, 0x4343434343434343ull) |   // C
        byte_eq(x, 0x4747474747474747ull) |   // G
        byte_eq(x, 0x5454545454545454ull);    // T

    // Compress byte high bits 7,15,...,63 into bits 0..7.
    return static_cast<std::uint32_t>((m * 0x0002040810204081ull) >> 56);
}

/**
 * @brief Iterate over all valid k-mers in a sequence, with 8-character blocks encoded via SWAR.
 *
 * Portable reference implementation of the block-wise structure of
 * for_each_kmer_ascii_checked_rolling_avx2(), with 8-character blocks encoded via SWAR (SIMD within
 * a register) on plain 64-bit words, so that it needs no SIMD instructions. Kept for testing and
 * benchmarking: on valid input it is roughly on par with
 * for_each_kmer_ascii_checked_rolling_scalar(), and behind it with frequent invalid characters, so
 * regular callers should use that, or for_each_kmer_ascii_checked_rolling_avx2() where AVX2 is
 * available.
 *
 * E and L, and the callback, are as for for_each_kmer_ascii_checked_rolling_avx2().
 */
template<Encoding E = Encoding::kACGT, Layout L = Layout::kMSB, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_checked_rolling_swar(
    std::string_view seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.size() < k) {
        return;
    }

    std::size_t const seq_len = seq.size();
    char const* data = seq.data();

    // Rolling constants for MSB and LSB, see for_each_kmer_ascii_checked_rolling_scalar().
    std::uint64_t const mask = (k == 32)
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << (2 * k)) - 1u);
    unsigned const top_shift = 2 * (static_cast<unsigned>(k) - 1);

    std::uint64_t kmer = 0;
    std::size_t valid = 0;
    std::size_t i = 0;

    alignas(8) std::array<std::uint8_t, 8> codes{};
    for (; i + 8 <= seq_len; i += 8) {
        std::uint32_t const valid_bits = encode_8_nts_swar_<E>(data + i, codes.data());
        for_each_kmer_ascii_checked_rolling_block_<E, L, 8>(
            codes.data(), valid_bits, i, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
        );
    }

    // Scalar tail.
    for_each_kmer_ascii_checked_rolling_tail_<E, L>(
        data, i, seq_len, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
    );
}

} // namespace fisk
