#pragma once

#include <array>
#include <algorithm>
#include <string>
#include <string_view>
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
//     K-mer Extraction SIMD AVX2
// =================================================================================================

// AVX2 vectorizes the encoding and validity check of 32 characters at a time; the k-mers themselves
// are still rolled one character at a time, as each one only differs from the previous one in its
// first and last character, so there is not much room for vectorizing that part.

#if defined(FISK_HAS_AVX2)

/**
 * @brief Encode 32 ASCII characters into 2-bit nucleotide codes of Encoding `E`.
 *
 * Returns a 32-bit mask with bit i set iff lane i is valid A/C/G/T (case-insensitive).
 * Invalid lanes are written as zero into `codes`.
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

    // Zero out invalid lanes so the consumer does not need to care.
    __m256i const enc_masked = _mm256_and_si256(enc, valid_vec);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(codes), enc_masked);

    return static_cast<std::uint32_t>(_mm256_movemask_epi8(valid_vec));
}

#endif

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
 * @brief Count trailing zeros in a nonzero 32-bit value.
 *
 * Unlike std::countr_zero(), undefined for zero, which spares the zero check that the former
 * needs on targets without BMI1 (TZCNT).
 */
inline unsigned ctz32_(std::uint32_t x) noexcept
{
    #if defined(_MSC_VER)
        unsigned long idx = 0;
        _BitScanForward(&idx, x);
        return static_cast<unsigned>(idx);
    #else
        return static_cast<unsigned>(__builtin_ctz(x));
    #endif
}

/**
 * @brief SIMD-assisted iteration over all valid k-mers in a sequence.
 *
 * This AVX2 version vectorizes nucleotide encoding and validity testing in blocks
 * of 32 characters, then consumes contiguous valid runs with a tight inner loop.
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
        std::uint32_t valid_bits = encode_32_nts_avx2_<E>(data + i, codes.data());

        // Fast path: whole block valid.
        if (valid_bits == 0xFFFFFFFFu) {
            for_each_kmer_in_valid_run_<E, L>(
                codes.data(), 32, i, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
            );
            continue;
        }

        std::size_t j = 0;
        while (j < 32) {
            // Skip invalid region.
            std::uint32_t tail = valid_bits >> j;
            if (tail == 0) {
                valid = 0;
                break;
            }

            if ((tail & 1u) == 0u) {
                unsigned const skip = ctz32_(tail);
                j += skip;
                valid = 0;
                if (j >= 32) {
                    break;
                }
            }

            // Consume contiguous valid run starting at j.
            tail = valid_bits >> j;

            // Number of leading 1-bits in tail, computed as ctz(~tail).
            // This works because the right shift fills upper bits with zero.
            unsigned const run_len = ctz32_(~tail);
            for_each_kmer_in_valid_run_<E, L>(
                codes.data() + j, run_len, i + j, k, mask, top_shift, kmer, valid,
                std::forward<Func>(func)
            );
            j += run_len;
        }
    }

    #endif

    // Scalar tail, and full fallback if AVX2 is unavailable: the same loop as in
    // for_each_kmer_rolling(), with the encoder of for_each_kmer(), continuing from the state that
    // the blocks above left behind.
    CharEncoderTable<E> const encoder;
    for (; i < seq_len; ++i) {
        std::uint8_t const code = encoder(data[i]);

        if constexpr (L == Layout::kMSB) {
            kmer = (kmer << 2) | (code & 0x03u);
        } else if constexpr (L == Layout::kLSB) {
            kmer = (kmer >> 2) | (std::uint64_t{code & 0x03u} << top_shift);
        } else {
            static_assert(dependent_false_v<L>, "Unhandled Layout in for_each_kmer_simd()");
        }
        valid = (code < 4) ? (valid + 1) : 0;

        if (valid >= k) {
            assert_fits_width_(window_bits_<L>(kmer, mask), k);
            invoke_kmer_callback<E, L>(func, i + 1 - k, window_bits_<L>(kmer, mask));
        }
    }
}

// =================================================================================================
//     K-mer Extraction SWAR
// =================================================================================================

// Alternative implementation, using only 64 bit scalar operations to do SIMD within a register.
// It is not as fast as the AVX above, or just the lookup table... But kept here for reference.
// E and L, and the callback, are as for for_each_kmer_simd().

template<Encoding E = Encoding::kACGT, Layout L = Layout::kMSB, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_swar(
    std::string_view seq,
    std::size_t k,
    Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.size() < k) {
        return;
    }

    // Helper function to run the ASCII encoding exploit on a single character,
    // returning the 2-bit code and validity in one go.
    // Note that they are not bit packed yet here, and in separate bytes,
    // to avoid unnecessary bit manipulation in the hot loop. We will pack them later.
    auto encode8_ascii_ = [](char const* p) noexcept -> std::uint64_t
    {
        std::uint64_t x;
        std::memcpy(&x, p, sizeof(x)); // unaligned-safe; usually optimized to one load

        // ASCII trick of ascii_bits_to_code_(), per byte. This works for both upper/lowercase ACGT
        // because the relevant lower bits are identical for upper and lower case. The final mask
        // also drops the bits shifted in from the neighboring byte.
        if constexpr (E == Encoding::kACGT) {
            return ((x >> 1) ^ (x >> 2)) & 0x0303030303030303ull;
        } else if constexpr (E == Encoding::kACTG) {
            return (x >> 1) & 0x0303030303030303ull;
        } else {
            static_assert(
                dependent_false_v<E>, "Unhandled Encoding in for_each_kmer_swar()"
            );
        }
    };

    // The above function does not perform a validity check, so we need to do that separately.
    // We can do it in the same pass, and pack the result into a byte.
    auto valid_acgt8_mask = [](char const* p) noexcept -> std::uint8_t
    {
        std::uint64_t x;
        std::memcpy(&x, p, sizeof(x)); // unaligned-safe; optimized to one load

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
        std::uint64_t m =
            byte_eq(x, 0x4141414141414141ull) |   // A
            byte_eq(x, 0x4343434343434343ull) |   // C
            byte_eq(x, 0x4747474747474747ull) |   // G
            byte_eq(x, 0x5454545454545454ull);    // T

        // Compress byte high bits 7,15,...,63 into bits 0..7.
        std::uint8_t r = static_cast<std::uint8_t>((m * 0x0002040810204081ull) >> 56);

        return r;
    };

    // Shorthands
    char const* const data    = seq.data();
    std::size_t const seq_len = seq.size();
    unsigned const kk = static_cast<unsigned>(k);

    // Masks, and the LSB counterpart, see for_each_kmer_rolling(). As in for_each_kmer_simd(), both
    // masks are applied when checking and emitting rather than at every step, which keeps them off
    // the dependency chains through `kmer` and `valid`.
    std::uint64_t const kmer_mask = (kk == 32)
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << (2u * kk)) - 1u);
    unsigned const top_shift = 2u * (kk - 1u);

    std::uint64_t const valid_mask = (std::uint64_t{1} << kk) - 1u;

    // State during iteration.
    std::uint64_t kmer  = 0;
    std::uint64_t valid = 0;
    std::size_t seen = 0;

    // Process one of the input bytes, as encoded above.
    // `code` is always within 0-3 here, also for invalid characters, so it needs no masking.
    auto step = [&](std::uint64_t code, std::uint8_t is_valid) {
        if constexpr (L == Layout::kMSB) {
            kmer = (kmer << 2u) | std::uint64_t{code};
        } else if constexpr (L == Layout::kLSB) {
            kmer = (kmer >> 2u) | (std::uint64_t{code} << top_shift);
        } else {
            static_assert(dependent_false_v<L>, "Unhandled Layout in for_each_kmer_swar()");
        }
        valid = (valid << 1u) | std::uint64_t{is_valid};

        // `seen` counts every character processed so far, so the k-mer ending here starts at
        // seen - kk.
        ++seen;
        if (seen >= kk && (valid & valid_mask) == valid_mask) {
            assert_fits_width_(window_bits_<L>(kmer, kmer_mask), kk);
            invoke_kmer_callback<E, L>(func, seen - kk, window_bits_<L>(kmer, kmer_mask));
        }
    };

    // Full 8-character blocks, unrolling the encoded data from the helper functions,
    // and emitting 8 k-mers per iteration.
    std::size_t i = 0;
    for (; i + 8 <= seq_len; i += 8) {
        std::uint64_t const enc = encode8_ascii_(data + i);
        std::uint8_t  const val = valid_acgt8_mask(data + i);

        step( (enc >>  0) & 0x03u, (val >> 0) & 0x01u );
        step( (enc >>  8) & 0x03u, (val >> 1) & 0x01u );
        step( (enc >> 16) & 0x03u, (val >> 2) & 0x01u );
        step( (enc >> 24) & 0x03u, (val >> 3) & 0x01u );
        step( (enc >> 32) & 0x03u, (val >> 4) & 0x01u );
        step( (enc >> 40) & 0x03u, (val >> 5) & 0x01u );
        step( (enc >> 48) & 0x03u, (val >> 6) & 0x01u );
        step( (enc >> 56) & 0x03u, (val >> 7) & 0x01u );
    }

    // Scalar tail, with the encoder of for_each_kmer().
    CharEncoderTable<E> const encoder;
    for (; i < seq_len; ++i) {
        std::uint8_t const code = encoder(data[i]);

        step(
            static_cast<std::uint64_t>(code & 0x03u),
            static_cast<std::uint8_t>(code < 4u)
        );
    }
}

} // namespace fisk
