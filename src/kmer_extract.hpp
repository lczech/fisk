#pragma once

#include <string>
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

#include "seq_enc.hpp"

// =================================================================================================
//     K-mer Extraction
// =================================================================================================

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
 * @tparam Enc  Encoding function to turn characters into two-bit encoding.
 * @tparam Func Callback function to be called for each valid k-mer.
 */
template<typename Enc, typename Func>
inline void for_each_kmer(
    std::string_view seq, std::size_t k, Enc&& enc, Func&& func
) {
    // Iterate all k-mers of a sequence, encoded as 2-bit packed uint64_t.
    // For a sequence of length n and k <= 32, this function visits
    // (n - k + 1) overlapping k-mers. Each k-mer is encoded into the
    // lowest 2*k bits of a uint64_t, using the two-bit encoding provided by `enc`.

    // Boundary checks
    if (k == 0 || k > 32) {
        throw std::runtime_error(
            "Invalid call to k-mer extraction with k not in [1, 32]"
        );
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
 * This is the same as for_each_kmer(), but re-extract the k-mer each time from the input
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
        throw std::runtime_error(
            "Invalid call to k-mer extraction with k not in [1, 32]"
        );
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

// =================================================================================================
//     XOR Hashing
// =================================================================================================

/**
 * @brief Simple "hashing" of k-mers by computing the xor of all their two bit encodings.
 *
 * This is just for benchmarking, to ensure that the values are actually used (and thus the
 * compuation cannot be omitted by the compiler), as well as to ensure consistent results
 * between different implementations.
 */
template<typename Enc>
inline std::uint64_t compute_kmer_hash(
    std::string_view seq, std::size_t k, Enc&& enc
) {
    // Simple wrapper around the main loop function which also keeps track of a "hash"
    // by xor-ing all k-mers, just as a validity check that all implementations give the same.
    std::uint64_t hash = 0;

    for_each_kmer(
        std::string_view(seq),
        k,
        enc,
        [&](std::uint64_t kmer_word) {
            // Simple order-independent checksum.
            // All implementations must use the same aggregation so sinks match.
            hash ^= kmer_word;
        }
    );

    return hash;
}

/**
 * @brief Simple "hashing" of k-mers by computing the xor of all their two bit encodings.
 *
 * This is similar to compute_kmer_hash(), but re-extracts the whole k-mer in each step.
 * This is computationally wasteful compred to bit shifts, thus only used for benchmarking.
 */
template<typename Enc>
inline std::uint64_t compute_kmer_hash_reextract(
    std::string_view seq, std::size_t k, Enc&& enc
) {
    // Simple wrapper around the main loop function whcih also keeps track of a "hash"
    // by xor-ing all k-mers, just as a validity check that all implementations give the same.
    std::uint64_t hash = 0;

    for_each_kmer_reextract(
        std::string_view(seq),
        k,
        enc,
        [&](std::uint64_t kmer_word) {
            // Simple order-independent checksum.
            // All implementations must use the same aggregation so sinks match.
            hash ^= kmer_word;
        }
    );

    return hash;
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
