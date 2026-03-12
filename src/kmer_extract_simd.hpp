#pragma once

#include <array>
#include <algorithm>
#include <string>
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

#include "bit_extract_simd.hpp"
#include "seq_enc.hpp"

// =================================================================================================
//     K-mer Extraction SIMD
// =================================================================================================

// We here only test k-mer extraction with AVX2, to see how much gain we get.
// It is somewhat faster, giving 0.8ns instead of 1.0ns per k-mer.
// The spaced k-mer extraction also uses SIMD for acceleration, but currently needs
// to process each char individually. Might be worth it to optimize this further,
// but left as future work for now.
// The fundamental issue is that k-mers depend on each other, each one only differing
// in the first and last character, so there is not much room for vectorization.

#if defined(FISK_HAS_AVX2)

/**
 * @brief Encode 32 ASCII characters into 2-bit nucleotide codes.
 *
 * Returns a 32-bit mask with bit i set iff lane i is valid A/C/G/T (case-insensitive).
 * Invalid lanes are written as zero into `codes`.
 */
inline std::uint32_t encode_32_nts_avx2(char const* data, std::uint8_t* codes) noexcept
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

    // Same ASCII bit trick as char_to_nt_ascii(), but vectorized.
    // Shift within 16-bit lanes, then mask back to byte-local bits.
    __m256i const s1 = _mm256_and_si256(
        _mm256_srli_epi16(lower, 1),
        _mm256_set1_epi8(0x7f)
    );
    __m256i const s2 = _mm256_and_si256(
        _mm256_srli_epi16(lower, 2),
        _mm256_set1_epi8(0x3f)
    );
    __m256i const enc = _mm256_and_si256(
        _mm256_xor_si256(s1, s2),
        _mm256_set1_epi8(0x03)
    );

    // Zero out invalid lanes so the consumer does not need to care.
    __m256i const enc_masked = _mm256_and_si256(enc, valid_vec);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(codes), enc_masked);

    return static_cast<std::uint32_t>(_mm256_movemask_epi8(valid_vec));
}

#endif

/**
 * @brief Consume a contiguous run of valid two-bit codes.
 */
template<typename Func>
inline void consume_valid_run(
    std::uint8_t const* codes,
    std::size_t len,
    std::size_t k,
    std::uint64_t mask,
    std::uint64_t& kmer,
    std::size_t& valid,
    Func&& func
) {
    std::size_t idx = 0;

    // Warm-up phase until we have accumulated k valid characters.
    if (valid < k) {
        std::size_t const need = k - valid;
        std::size_t const warm = std::min(len, need);

        for (; idx < warm; ++idx) {
            kmer = ((kmer << 2) & mask) | codes[idx];
        }

        valid += warm;

        // Still not enough valid characters to emit anything.
        if (valid < k) {
            return;
        }

        // The last character processed above just completed the first valid k-mer.
        func(kmer);

        // Saturate valid so we do not keep incrementing a counter that is only used
        // as a threshold predicate.
        valid = k;
    }

    // Fully warm: every additional valid code yields one k-mer.
    for (; idx < len; ++idx) {
        kmer = ((kmer << 2) & mask) | codes[idx];
        func(kmer);
    }
}

/**
 * @brief Count trailing zeros in a nonzero 32-bit value.
 */
inline unsigned ctz32(std::uint32_t x) noexcept
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
 */
template<typename Func>
inline void for_each_kmer_simd(std::string_view seq, std::size_t k, Func&& func)
{
    if (k == 0 || k > 32) {
        throw std::runtime_error(
            "Invalid call to k-mer extraction with k not in [1, 32]"
        );
    }
    if (seq.size() < k) {
        return;
    }

    std::size_t const seq_len = seq.size();
    char const* data = seq.data();

    std::uint64_t const mask = (k == 32)
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << (2 * k)) - 1u);

    std::uint64_t kmer = 0;
    std::size_t valid = 0;
    std::size_t i = 0;

    #if defined(FISK_HAS_AVX2)

    alignas(32) std::array<std::uint8_t, 32> codes{};

    for (; i + 32 <= seq_len; i += 32) {
        std::uint32_t valid_bits = encode_32_nts_avx2(data + i, codes.data());

        // Fast path: whole block valid.
        if (valid_bits == 0xFFFFFFFFu) {
            consume_valid_run(
                codes.data(), 32, k, mask, kmer, valid, std::forward<Func>(func)
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
                unsigned const skip = ctz32(tail);
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
            unsigned const run_len = ctz32(~tail);
            consume_valid_run(
                codes.data() + j, run_len, k, mask, kmer, valid, std::forward<Func>(func)
            );
            j += run_len;
        }
    }

    #endif

    // Scalar tail, and full fallback if AVX2 is unavailable.
    for (; i < seq_len; ++i) {
        std::uint8_t const code = char_to_nt_ascii(data[i]);

        if (code < 4) {
            kmer = ((kmer << 2) & mask) | (code & 0x03u);
            valid = (valid < k) ? (valid + 1) : k;

            if (valid >= k) {
                func(kmer);
            }
        } else {
            valid = 0;
        }
    }
}

// =================================================================================================
//     XOR Hashing
// =================================================================================================

inline std::uint64_t compute_kmer_hash_simd(
    std::string_view seq, std::size_t k
) {
    // Simple wrapper around the main loop function which also keeps track of a "hash"
    // by xor-ing all k-mers, just as a validity check that all implementations give the same.
    std::uint64_t hash = 0;

    for_each_kmer_simd(
        std::string_view(seq),
        k,
        [&](std::uint64_t kmer_word) {
            hash ^= kmer_word;
        }
    );

    return hash;
}
