#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

#include "fisk/core/char_encoder.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_rolling_scalar.hpp"
#include "fisk/kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from ASCII, Rolling, AVX2
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
 * for_each_kmer_ascii_checked_rolling_block_() needs.
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
 * @brief AVX2-assisted iteration over all valid k-mers in a sequence.
 *
 * Vectorizes nucleotide encoding and validity testing with AVX2 in blocks of 32 characters, then
 * rolls the k-mers over each block, with a tight inner loop for fully valid blocks, see
 * for_each_kmer_ascii_checked_rolling_block_(). This is a fixed AVX2 loop, not generic over the
 * SIMD archs of core/simd.hpp. Without AVX2, all characters take the scalar path, see
 * for_each_kmer_ascii_checked_rolling_tail_().
 *
 * Invalid characters reset the valid-window state and suppress any overlapping k-mers.
 *
 * The callback receives a Kmer<E, L>, ACGT/MSB by default:
 *
 *     // Kmer<kACGT, kMSB>
 *     for_each_kmer_ascii_checked_rolling_avx2(seq, k, func);
 *     // Kmer<kACTG, kLSB>
 *     for_each_kmer_ascii_checked_rolling_avx2<Encoding::kACTG, Layout::kLSB>(seq, k, func);
 *
 * Func may be called either as `func(pos, kmer)` or as `func(kmer)`, see
 * for_each_kmer_ascii_checked_rolling_scalar().
 */
template<Encoding E = Encoding::kACGT, Layout L = Layout::kMSB, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_checked_rolling_avx2(
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

    #if defined(FISK_HAS_AVX2)

    alignas(32) std::array<std::uint8_t, 32> codes{};

    for (; i + 32 <= seq_len; i += 32) {
        std::uint32_t const valid_bits = encode_32_nts_avx2_<E>(data + i, codes.data());
        for_each_kmer_ascii_checked_rolling_block_<E, L, 32>(
            codes.data(), valid_bits, i, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
        );
    }

    #endif

    // Scalar tail, and full fallback if AVX2 is unavailable.
    for_each_kmer_ascii_checked_rolling_tail_<E, L>(
        data, i, seq_len, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
    );
}

} // namespace fisk
