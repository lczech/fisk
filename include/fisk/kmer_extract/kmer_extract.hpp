#pragma once

#include <string>
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

#include "fisk/core/seq_enc.hpp"

// =================================================================================================
//     K-mer Extraction
// =================================================================================================

/**
 * @brief Throw for an invalid k-mer extraction precondition.
 *
 * The extracted k-mer length must satisfy `1 <= k <= k_max`.
 * This helper is kept separate from the hot loop so the performance-critical extractor
 * can remain branch-free and avoid the runtime overhead of an inlined exception path.
 */
[[gnu::noinline, gnu::cold]]
inline void throw_invalid_kmer_k_(std::size_t k_max)
{
    // Keep the validity check out of the hot path: throwing here avoids a cold branch/landing pad
    // in otherwise branch-free extractors, which could prevent their inlining.
    throw std::runtime_error(
        "Invalid call to k-mer extraction with k not in [1, " + std::to_string(k_max) + "]"
    );
}

/**
 * @brief Iterate a sequence, extract all valid k-mers from it (using bit shifts),
 * and call a callback function on each k-mer.
 *
 * The encoder `enc` must return:
 *   - 0,1,2,3 for valid A/C/G/T-like symbols
 *   - any value >= 4 for invalid symbols
 *
 * Any k-mer overlapping an invalid symbol is skipped.
 *
 * This is the generic, encoder-parameterized building block behind for_each_kmer() below; most
 * callers should use that instead. Call this directly only to plug in a specific encoder or
 * ordering, e.g. for benchmarking different encoders against each other.
 *
 * @tparam Enc  Encoding function to turn characters into two-bit encoding.
 * @tparam Func Callback function to be called for each valid k-mer.
 */
template<typename Enc, typename Func>
inline void for_each_kmer_rolling(
    std::string_view seq, std::size_t k, Enc&& enc, Func&& func
) {
    // Iterate all k-mers of a sequence, encoded as 2-bit packed uint64_t.
    // For a sequence of length n and k <= 32, this function visits
    // (n - k + 1) overlapping k-mers. Each k-mer is encoded into the
    // lowest 2*k bits of a uint64_t, using the two-bit encoding provided by `enc`.

    // Boundary checks
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.size() < k) {
        return;
    }

    // Shorthands for data access
    std::size_t const seq_len = seq.size();
    char const*       data    = seq.data();

    // Mask to keep only the lowest 2*k bits.
    // This works for all k in [1, 32].
    std::uint64_t const mask = (k == 32)
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << (2 * k)) - 1u);

    std::uint64_t kmer = 0;
    std::size_t valid = 0;
    for( std::size_t i = 0; i < seq_len; ++i ) {
        std::uint8_t const code = enc(data[i]);

        // Always shift in the low 2 bits. For invalid symbols this value is ignored,
        // because all overlapping k-mers will be skipped until the bad position
        // has slid out of the window.
        kmer = ((kmer << 2) & mask) | (code & 0x03u);
        valid = (code < 4) ? (valid + 1) : 0;

        // We can emit once we have seen at least k characters, and the current
        // k-mer window does not overlap the most recent invalid character.
        if( valid >= k ) {
            func(kmer);
        }
    }
}

/**
 * @brief Iterate a sequence, extract all k-mers from it (using re-extraction each time),
 * and call a callback function on each k-mer.
 *
 * This is the same as for_each_kmer_rolling(), but re-extract the k-mer each time from the input
 * characters. This is of course slower, but apparently used in practice. We hence implement
 * it here for benchmarking.
 */
template<typename Enc, typename Func>
inline void for_each_kmer_reextract(
    std::string_view seq, std::size_t k, Enc&& enc, Func&& func
) {
    // Same as above, but each k-mer is extracted separately.
    // Not efficient, and worse for larger k. Meant only for benchmarking.

    // Boundary checks
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.size() < k) {
        return;
    }

    // Shorthands for data access
    std::size_t const seq_len = seq.size();
    char const*       data    = seq.data();

    // Slide the window over the sequence.
    std::size_t const stop = seq_len - k;
    for (std::size_t i = 0; i <= stop; ++i) {
        std::uint64_t kmer = 0;
        bool valid = true;

        for (std::size_t x = 0; x < k; ++x) {
            std::uint8_t const code = static_cast<std::uint8_t>(enc(data[i + x]));
            valid &= (code < 4);
            kmer = (kmer << 2) | (code & 0x3u);
        }

        if (valid) {
            func(kmer);
        }
    }
}

/**
 * @brief Iterate a sequence, extract all valid k-mers from it, and call a callback function on
 * each k-mer.
 *
 * Convenience entry point for callers who do not need to choose an encoder or extraction
 * technique themselves: fixes the encoder to the ACGT lookup table (char_to_nt_table_acgt(), see
 * core/seq_enc.hpp) and forwards to for_each_kmer_rolling(), the fastest of the extraction
 * techniques offered here. Call for_each_kmer_rolling() directly instead to plug in a different
 * encoder or ordering.
 *
 * @tparam Func Callback function to be called for each valid k-mer.
 */
template<typename Func>
inline void for_each_kmer(std::string_view seq, std::size_t k, Func&& func)
{
    for_each_kmer_rolling(seq, k, char_to_nt_table_acgt, func);
}

// =================================================================================================
//     k-mer to string
// =================================================================================================

/**
 * @brief Get the string representation of a k-mer, as a sequence of `ACGT` characters.
 */
inline std::string decode_kmer_2bit( std::uint64_t kmer, std::size_t k )
{
    static const char lut[4] = {'A','C','G','T'};

    std::string s;
    s.resize(k);

    for (std::size_t i = 0; i < k; ++i) {
        std::size_t shift = 2 * (k - 1 - i);
        std::uint64_t code = (kmer >> shift) & 0x3ULL;
        s[i] = lut[code];
    }

    return s;
}
