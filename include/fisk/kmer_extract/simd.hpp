#pragma once

#include <array>
#include <algorithm>
#include <string>
#include <string_view>
#include <type_traits>
#include <cstdint>
#include <cstring>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"

namespace fisk {

// =================================================================================================
//     Shared Helper Functions
// =================================================================================================

/**
 * @brief Increment a counter of consecutive valid characters, or reset it to zero, without a
 * branch.
 *
 * Some compilers turn `keep ? count + 1 : 0` into a branch, which mispredicts on scattered invalid
 * characters. Instead, `count + 1` is masked with all zeros or all ones, which relies on the
 * modular arithmetic of unsigned types that are not promoted to int.
 */
template <typename T>
FISK_ALWAYS_INLINE
constexpr T increment_or_reset_(T count, bool keep) noexcept
{
    static_assert(
        std::is_unsigned_v<T> && sizeof(T) >= sizeof(unsigned),
        "increment_or_reset_() needs an unsigned type that is not promoted to int"
    );
    return (count + 1) & (T{0} - static_cast<T>(keep));
}

/**
 * @brief Consume a contiguous run of valid two-bit codes.
 *
 * `base` is the index into the original sequence of `codes[0]`, used to report the start position
 * of each emitted k-mer to the callback. `mask` and `top_shift` are as in for_each_kmer_rolling(),
 * for Layout::kMSB and Layout::kLSB, respectively. For MSB, `mask` is only applied to the emitted
 * k-mers, see window_bits_().
 */
template<Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE
inline void for_each_kmer_in_valid_run_(
    std::uint8_t const* codes,
    std::size_t len,
    std::size_t base,
    std::size_t k,
    std::uint64_t mask,
    unsigned top_shift,
    std::uint64_t& kmer,
    std::size_t& valid,
    Func&& func
) {
    // The codes of a valid run are all within 0-3, so they need no masking.
    std::size_t idx = 0;

    // Warm-up phase until we have accumulated k valid characters.
    if (valid < k) {
        std::size_t const need = k - valid;
        std::size_t const warm = std::min(len, need);

        for (; idx < warm; ++idx) {
            if constexpr (L == Layout::kMSB) {
                kmer = (kmer << 2) | codes[idx];
            } else if constexpr (L == Layout::kLSB) {
                kmer = (kmer >> 2) | (std::uint64_t{codes[idx]} << top_shift);
            } else {
                static_assert(
                    dependent_false_v<L>, "Unhandled Layout in for_each_kmer_in_valid_run_()"
                );
            }
        }

        valid += warm;

        // Still not enough valid characters to emit anything.
        if (valid < k) {
            return;
        }

        // The last character processed above, at index base + idx - 1, just completed the first
        // valid k-mer, which hence starts at base + idx - k.
        assert_fits_width_(window_bits_<L>(kmer, mask), k);
        invoke_kmer_callback<E, L>(func, base + idx - k, window_bits_<L>(kmer, mask));

        // Saturate valid so we do not keep incrementing a counter that is only used
        // as a threshold predicate.
        valid = k;
    }

    // Fully warm: every additional valid code yields one k-mer.
    for (; idx < len; ++idx) {
        if constexpr (L == Layout::kMSB) {
            kmer = (kmer << 2) | codes[idx];
        } else if constexpr (L == Layout::kLSB) {
            kmer = (kmer >> 2) | (std::uint64_t{codes[idx]} << top_shift);
        } else {
            static_assert(
                dependent_false_v<L>, "Unhandled Layout in for_each_kmer_in_valid_run_()"
            );
        }
        assert_fits_width_(window_bits_<L>(kmer, mask), k);
        invoke_kmer_callback<E, L>(func, base + idx + 1 - k, window_bits_<L>(kmer, mask));
    }
}

/**
 * @brief Iterate the k-mers ending in one block of `Width` characters, given their codes and their
 * validity.
 *
 * Bit j of `valid_bits` is set iff `codes[j]` is a valid code. Fully valid blocks are handed to
 * for_each_kmer_in_valid_run_(). Other blocks are rolled in one character at a time, with invalid
 * characters resetting the window; this is branch-free apart from the emission, as splitting the
 * block into valid runs instead would mispredict on scattered invalid characters. Codes of invalid
 * characters are hence rolled into the k-mer as well, and must be within 0-3, too.
 * `base` is the index into the original sequence of `codes[0]`. The other parameters are as for
 * for_each_kmer_in_valid_run_().
 */
template<Encoding E, Layout L, unsigned Width, typename Func>
FISK_ALWAYS_INLINE
inline void for_each_kmer_in_block_(
    std::uint8_t const* codes,
    std::uint32_t valid_bits,
    std::size_t base,
    std::size_t k,
    std::uint64_t mask,
    unsigned top_shift,
    std::uint64_t& kmer,
    std::size_t& valid,
    Func&& func
) {
    static_assert(Width >= 1 && Width <= 32, "for_each_kmer_in_block_() supports 1 to 32 lanes");
    constexpr std::uint32_t all_valid = (Width == 32)
        ? 0xFFFFFFFFu
        : ((std::uint32_t{1} << Width) - 1u);

    // Fast path: whole block valid.
    if (valid_bits == all_valid) {
        for_each_kmer_in_valid_run_<E, L>(
            codes, Width, base, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
        );
        return;
    }

    // Slow path: block contains invalid characters.
    for (unsigned j = 0; j < Width; ++j) {
        if constexpr (L == Layout::kMSB) {
            kmer = (kmer << 2) | codes[j];
        } else if constexpr (L == Layout::kLSB) {
            kmer = (kmer >> 2) | (std::uint64_t{codes[j]} << top_shift);
        } else {
            static_assert(dependent_false_v<L>, "Unhandled Layout in for_each_kmer_in_block_()");
        }

        valid = increment_or_reset_(valid, ((valid_bits >> j) & 1u) != 0);

        if (valid >= k) {
            assert_fits_width_(window_bits_<L>(kmer, mask), k);
            invoke_kmer_callback<E, L>(func, base + j + 1 - k, window_bits_<L>(kmer, mask));
        }
    }
}

/**
 * @brief Iterate the k-mers ending in `data[begin, end)` one character at a time, continuing from
 * the state that a block-wise main loop left behind.
 *
 * The same loop as in for_each_kmer_rolling(), with the encoder of for_each_kmer(). Used as the
 * scalar tail of the block-wise extractors, and as their full fallback where they have no blocks.
 * The other parameters are as for for_each_kmer_in_valid_run_().
 */
template<Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE
inline void for_each_kmer_scalar_tail_(
    char const* data,
    std::size_t begin,
    std::size_t end,
    std::size_t k,
    std::uint64_t mask,
    unsigned top_shift,
    std::uint64_t& kmer,
    std::size_t& valid,
    Func&& func
) {
    CharEncoderTable<E> const encoder;
    for (std::size_t i = begin; i < end; ++i) {
        std::uint8_t const code = encoder(data[i]);

        if constexpr (L == Layout::kMSB) {
            kmer = (kmer << 2) | (code & 0x03u);
        } else if constexpr (L == Layout::kLSB) {
            kmer = (kmer >> 2) | (std::uint64_t{code & 0x03u} << top_shift);
        } else {
            static_assert(dependent_false_v<L>, "Unhandled Layout in for_each_kmer_scalar_tail_()");
        }
        valid = (code < 4) ? (valid + 1) : 0;

        if (valid >= k) {
            assert_fits_width_(window_bits_<L>(kmer, mask), k);
            invoke_kmer_callback<E, L>(func, i + 1 - k, window_bits_<L>(kmer, mask));
        }
    }
}

// =================================================================================================
//     K-mer Extraction SIMD AVX2
// =================================================================================================

// AVX2 vectorizes the encoding and validity check of 32 characters at a time; the k-mers themselves
// are still rolled one character at a time, as each one only differs from the previous one in its
// first and last character, so there is not much room for vectorizing that part.

#if defined(FISK_HAS_AVX2)

/**
 * @brief Encode 32 ASCII characters into 2-bit nucleotide codes of Encoding `E`.
 *
 * Returns a 32-bit mask with bit i set iff lane i is valid A/C/G/T (case-insensitive). Invalid
 * lanes are not zeroed: their codes are arbitrary values within 0-3, which is all that
 * for_each_kmer_in_block_() needs.
 */
template <Encoding E>
inline std::uint32_t encode_32_nts_avx2_(char const* data, std::uint8_t* codes) noexcept
{
    __m256i const in    = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(data));
    __m256i const lower = _mm256_or_si256(in, _mm256_set1_epi8(0x20));

    // Validity: compare against a/c/g/t after ASCII lowercasing.
    __m256i const is_a = _mm256_cmpeq_epi8(lower, _mm256_set1_epi8('a'));
    __m256i const is_c = _mm256_cmpeq_epi8(lower, _mm256_set1_epi8('c'));
    __m256i const is_g = _mm256_cmpeq_epi8(lower, _mm256_set1_epi8('g'));
    __m256i const is_t = _mm256_cmpeq_epi8(lower, _mm256_set1_epi8('t'));

    __m256i const valid_vec = _mm256_or_si256(
        _mm256_or_si256(is_a, is_c),
        _mm256_or_si256(is_g, is_t)
    );

    // Same ASCII bit trick as ascii_bits_to_code_(), but vectorized.
    // Shift within 16-bit lanes, then mask back to byte-local bits.
    __m256i const enc = [&] {
        if constexpr (E == Encoding::kACGT) {
            __m256i const s1 = _mm256_and_si256(
                _mm256_srli_epi16(lower, 1),
                _mm256_set1_epi8(0x7f)
            );
            __m256i const s2 = _mm256_and_si256(
                _mm256_srli_epi16(lower, 2),
                _mm256_set1_epi8(0x3f)
            );
            return _mm256_and_si256(
                _mm256_xor_si256(s1, s2),
                _mm256_set1_epi8(0x03)
            );
        } else if constexpr (E == Encoding::kACTG) {
            // The final mask also drops the bit shifted in from the neighboring byte.
            return _mm256_and_si256(
                _mm256_srli_epi16(lower, 1),
                _mm256_set1_epi8(0x03)
            );
        } else {
            static_assert(dependent_false_v<E>, "Unhandled Encoding in encode_32_nts_avx2_()");
        }
    }();

    _mm256_storeu_si256(reinterpret_cast<__m256i*>(codes), enc);

    return static_cast<std::uint32_t>(_mm256_movemask_epi8(valid_vec));
}

#endif

/**
 * @brief SIMD-assisted iteration over all valid k-mers in a sequence.
 *
 * This AVX2 version vectorizes nucleotide encoding and validity testing in blocks
 * of 32 characters, then rolls the k-mers over each block, with a tight inner loop for fully valid
 * blocks, see for_each_kmer_in_block_().
 *
 * Invalid characters reset the valid-window state and suppress any overlapping k-mers.
 *
 * The callback receives a Kmer<E, L>, ACGT/MSB by default, as for for_each_kmer():
 *
 *     for_each_kmer_simd(seq, k, func);                                 // Kmer<kACGT, kMSB>
 *     for_each_kmer_simd<Encoding::kACTG, Layout::kLSB>(seq, k, func);  // Kmer<kACTG, kLSB>
 *
 * Func may be called either as `func(pos, kmer)` or as `func(kmer)`, see for_each_kmer_rolling().
 */
template<Encoding E = Encoding::kACGT, Layout L = Layout::kMSB, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_simd(std::string_view seq, std::size_t k, Func&& func)
{
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.size() < k) {
        return;
    }

    std::size_t const seq_len = seq.size();
    char const* data = seq.data();

    // Rolling constants for MSB and LSB, see for_each_kmer_rolling().
    std::uint64_t const mask = (k == 32)
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << (2 * k)) - 1u);
    unsigned const top_shift = 2 * (static_cast<unsigned>(k) - 1);

    std::uint64_t kmer = 0;
    std::size_t valid = 0;
    std::size_t i = 0;

    #if defined(FISK_HAS_AVX2)

    alignas(32) std::array<std::uint8_t, 32> codes{};

    for (; i + 32 <= seq_len; i += 32) {
        std::uint32_t const valid_bits = encode_32_nts_avx2_<E>(data + i, codes.data());
        for_each_kmer_in_block_<E, L, 32>(
            codes.data(), valid_bits, i, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
        );
    }

    #endif

    // Scalar tail, and full fallback if AVX2 is unavailable.
    for_each_kmer_scalar_tail_<E, L>(
        data, i, seq_len, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
    );
}

// =================================================================================================
//     K-mer Extraction SWAR
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
 * Portable reference implementation of the block-wise structure of for_each_kmer_simd(), with
 * 8-character blocks encoded via SWAR (SIMD within a register) on plain 64-bit words, so that it
 * needs no SIMD instructions. Kept for testing and benchmarking: on valid input it is roughly on
 * par with for_each_kmer(), and behind it with frequent invalid characters, so regular callers
 * should use for_each_kmer(), or for_each_kmer_simd() where AVX2 is available.
 *
 * E and L, and the callback, are as for for_each_kmer_simd().
 */
template<Encoding E = Encoding::kACGT, Layout L = Layout::kMSB, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_swar(std::string_view seq, std::size_t k, Func&& func)
{
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.size() < k) {
        return;
    }

    std::size_t const seq_len = seq.size();
    char const* data = seq.data();

    // Rolling constants for MSB and LSB, see for_each_kmer_rolling().
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
        for_each_kmer_in_block_<E, L, 8>(
            codes.data(), valid_bits, i, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
        );
    }

    // Scalar tail.
    for_each_kmer_scalar_tail_<E, L>(
        data, i, seq_len, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
    );
}

} // namespace fisk
