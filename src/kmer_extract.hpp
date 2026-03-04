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
 * @brief Iterate a sequence, extract all k-mers from it (using bit shifts),
 * and call a callback function on each k-mer.
 *
 * @tparam Enc  Encoding function to turn characters into two bit encoding.
 * @tparam Func Callback function to be called for each k-mer in the iteration.
 */
template<typename Enc, typename Func>
inline void for_each_kmer_2bit(
    std::string_view seq, std::size_t k, Enc&& enc, Func&& func
) {
    // Iterate all k-mers of a sequence, encoded as 2-bit packed uint64_t.
    // For a sequence of length n and k <= 32, this function visits
    // (n - k + 1) overlapping k-mers. Each k-mer is encoded into the
    // lowest 2*k bits of a uint64_t, using the two-bit encoding provided by `enc`.

    // Boundary checks
    if (k == 0 || k > 32 ) {
        throw std::runtime_error( "Invalid call to k-mer extraction with k not in [1, 32]" );
    }
    if( seq.size() < k ) {
        return;
    }

    const std::size_t n    = seq.size();
    const char*       data = seq.data();

    // Mask to keep only the lowest 2*k bits.
    // This works for all k in [1, 32].
    const std::uint64_t mask = (k == 32)
        ? ~std::uint64_t{0}                         // all 64 bits
        : ((std::uint64_t{1} << (2 * k)) - 1u);     // lower 2*k bits set

    // Build the first k-mer.
    std::uint64_t kmer = 0;
    for (std::size_t i = 0; i < k; ++i) {
        kmer = (kmer << 2) | enc(data[i]);
    }
    func(kmer);

    // Slide the window over the sequence.
    const std::size_t stop = n - k;
    for (std::size_t i = 1; i <= stop; ++i) {
        // Drop the highest 2 bits by masking, then append the new base.
        kmer = ((kmer << 2) & mask) | enc(data[i + k - 1]);
        func(kmer);
    }
}

/**
 * @brief Iterate a sequence, extract all k-mers from it (using re-extraction each time),
 * and call a callback function on each k-mer.
 *
 * This is the same as for_each_kmer_2bit(), but re-extract the k-mer each time from the input
 * characters. This is of course slower, but apparently used in practice. We hence implement
 * it here for benchmarking.
 */
template<typename Enc, typename Func>
inline void for_each_kmer_2bit_reextract(
    std::string_view seq, std::size_t k, Enc&& enc, Func&& func
) {
    // Same as above, but each kmer is extracted separately. Not efficient, and worse for larger k.

    // Boundary checks
    if (k == 0 || k > 32 ) {
        throw std::runtime_error( "Invalid call to k-mer extraction with k not in [1, 32]" );
    }
    if( seq.size() < k ) {
        return;
    }

    const std::size_t n    = seq.size();
    const char*       data = seq.data();

    // Slide the window over the sequence.
    const std::size_t stop = n - k;
    for (std::size_t i = 0; i <= stop; ++i) {
        std::uint64_t kmer = 0;
        for (std::size_t x = 0; x < k; ++x) {
            kmer = (kmer << 2) | enc(data[i+x]);
        }
        func(kmer);
    }
}

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
