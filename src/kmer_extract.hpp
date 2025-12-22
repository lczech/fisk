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

template<typename Enc, typename Func>
inline void for_each_kmer_2bit(std::string_view seq, std::size_t k, Enc&& enc, Func&& func)
{
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

template<typename Enc, typename Func>
inline void for_each_kmer_2bit_reextract(std::string_view seq, std::size_t k, Enc&& enc, Func&& func)
{
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
