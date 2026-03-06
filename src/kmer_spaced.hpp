#pragma once

#include <algorithm>
#include <string>
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "kmer_extract.hpp"
#include "bit_extract.hpp"
#include "seq_enc.hpp"

// =================================================================================================
//     Comin
// =================================================================================================

// The below is a re-implementation of parts of https://github.com/CominLab/MISSH
// where we losely follow their code, in order to get a baseline for comparison.

inline std::vector<size_t> comin_prepare_mask( std::string const& mask )
{
    if( mask.size() == 0 || mask.size() > 32 ) {
        throw std::invalid_argument( "Invalid mask size not in [1,32]" );
    }
    std::vector<size_t> result;
    for( size_t i = 0; i < mask.size(); ++i ) {
        if( mask[i] == '0' ) {
            continue;
        } else if( mask[i] == '1' ) {
            result.push_back(i);
        } else {
            throw std::invalid_argument( "Invalid mask with symbols not in [0,1]" );
        }
    }
    return result;
}

inline std::vector<std::vector<size_t>> comin_prepare_masks( std::vector<std::string> const& masks )
{
    std::vector<std::vector<size_t>> result;
    result.resize( masks.size() );
    for( auto const& mask : masks ) {
        result.push_back( comin_prepare_mask( mask ));
    }
    return result;
}

inline std::uint64_t comin_compute_spaced_kmer(
    std::string_view seq, std::vector<size_t> const& mask, size_t start_pos
) {
    // Compute a single spaced kmer at the given position
    std::uint64_t result = 0;
    for( size_t i = 0; i < mask.size(); ++i ) {
        // Comin et al use a switch statement for the encoding
        auto const c = static_cast<std::uint64_t>(
            char_to_nt_switch_throw( seq[start_pos + mask[i]] )
        );

        // The original code builds the kmer backwards, with the last base at the highest bits.
        // result |= (c << (2 * i));

        // We instead keep it in order, so that sorting of kmers etc works as expected.
        // The speed of this is not significantly different from the above, in our tests.
        result <<= 2;
        result |= c;
    }
    return result;
}

inline std::uint64_t comin_compute_spaced_kmer_improved(
    std::string_view seq, std::vector<size_t> const& mask, size_t start_pos
) {
    // Compute a single spaced kmer at the given position
    std::uint64_t result = 0;
    for( auto p : mask ) {
        auto const c = static_cast<std::uint64_t>( char_to_nt_table_throw( seq[start_pos + p] ));
        result <<= 2;
        result |= c;
    }
    return result;
}

template<typename Mask, typename Comp>
inline std::uint64_t compute_spaced_kmer_hash_comin(
    std::string const& seq, size_t k, Mask const& mask, Comp&& comp
) {
    // Compute all spaced kmers across the sequence, and xor their hashes, for our checking.
    std::uint64_t hash = 0;

    // Slide the window over the sequence.
    const std::size_t stop = seq.size() - k;
    for (std::size_t i = 0; i <= stop; ++i) {
        hash ^= comp( std::string_view(seq), mask, i );
    }
    return hash;
}

// =================================================================================================
//     Bit Extraction
// =================================================================================================

/**
 * @brief Take a k-mer spaced mask string and turn it into a 2bit bit extraction mask.
 *
 * For instance, mask `1011` becomes an uint `0b11001111`.
 */
inline std::uint64_t prepare_spaced_kmer_bit_extract_mask( std::string const& mask )
{
    if( mask.size() == 0 || mask.size() > 32 ) {
        throw std::invalid_argument( "Invalid spaced k-mer mask size not in [1,32]" );
    }
    std::uint64_t result = 0;
    for( size_t i = 0; i < mask.size(); ++i ) {
        result <<= 2;
        if( mask[i] == '0' || mask[i] == '*' ) {
            continue;
        } else if( mask[i] == '1' ) {
            result |= 3;
        } else {
            throw std::invalid_argument( "Invalid spaced k-mer mask with symbols not in [0,1]" );
        }
    }
    return result;
}

/**
 * @brief Get the bit extract mask as a string of 0s and 1s, for printing.
 */
inline std::string bit_extract_mask_to_spaced_kmer_mask_string( std::uint64_t mask, size_t k )
{
    std::string str;
    for( size_t i = 0; i < k; ++i ) {
        if(( mask & 0x3 ) == 0x0 ) {
            str.append("0");
        } else if(( mask & 0x3 ) == 0x3 ) {
            str.append("1");
        } else {
            throw std::invalid_argument(
                "Invalid spaced k-mer mask with entries that are not 00 or 11."
            );
        }
        mask >>= 2;
    }
    if( mask != 0 ) {
        throw std::invalid_argument(
            "Invalid spaced k-mer mask not of size k."
        );
    }
    std::reverse(str.begin(), str.end());
    return str;
}

/**
 * @brief Compute a simple "hash" of a sequence by xoring all spaced k-mers in the sequence.
 *
 * This is just for benchmarking, to ensure that the values are actually used (and thus the
 * compuation cannot be omitted by the compiler), as well as to ensure consistent results
 * between different implementations.
 */
template<typename Mask, typename Enc, typename BitExtract>
inline std::uint64_t compute_spaced_kmer_hash(
    std::string const& seq, size_t const k, Mask const& mask, Enc&& enc, BitExtract&& bit_ext
) {
    // Compute all spaced kmers across the sequence, and xor their hashes, for our checking.
    std::uint64_t hash = 0;

    for_each_kmer_2bit(
        std::string_view(seq),
        k,
        enc,
        [&](std::uint64_t kmer_word) {
            // Extract the spaced k-mer, and combine it into the hash.
            hash ^= bit_ext( kmer_word, mask );
        }
    );
    return hash;
}

/**
 * @brief Compute a simple "hash" of a sequence by xoring all spaced k-mers in the sequence,
 * across a set of masks.
 *
 * This is just for benchmarking, to ensure that the values are actually used (and thus the
 * compuation cannot be omitted by the compiler), as well as to ensure consistent results
 * between different implementations.
 */
template<typename Mask, typename Enc, typename BitExtract>
inline std::uint64_t compute_spaced_kmer_hash(
    std::string const& seq, size_t const k, std::vector<Mask> const& masks,
    Enc&& enc, BitExtract&& bit_ext
) {
    // Compute all spaced kmers across the sequence, and xor their hashes, for our checking.
    std::uint64_t hash = 0;

    for_each_kmer_2bit(
        std::string_view(seq),
        k,
        enc,
        [&](std::uint64_t kmer_word) {
            // Extract the spaced k-mer, and combine it into the hash.
            for( auto const& mask : masks ) {
                hash ^= bit_ext( kmer_word, mask );
            }
        }
    );
    return hash;
}
