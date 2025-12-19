#pragma once

#include <string>
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "kmer_extract.hpp"
#include "pext.hpp"
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
    std::vector<size_t> const& mask, std::string_view seq, size_t start_pos
) {
    // Compute a single spaced kmer at the given position
    std::uint64_t result = 0;
    for( size_t i = 0; i < mask.size(); ++i ) {
        // Comin et al use a switch statement for the encoding
        auto const c = static_cast<std::uint64_t>( char_to_nt_switch( seq[start_pos + mask[i]] ));

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
    std::vector<size_t> const& mask, std::string_view seq, size_t start_pos
) {
    // Compute a single spaced kmer at the given position
    std::uint64_t result = 0;
    for( auto p : mask ) {
        auto const c = static_cast<std::uint64_t>( char_to_nt_table( seq[start_pos + p] ));
        result <<= 2;
        result |= c;
    }
    return result;
}

template<typename Mask, typename Comp>
inline std::uint64_t comin_compute_sequence_hash(
    size_t k, Mask const& mask, std::string const& seq, Comp&& comp
) {
    // Compute all spaced kmers across the sequence, and xor their hashes, for our checking.
    std::uint64_t hash = 0;

    // Slide the window over the sequence.
    const std::size_t stop = seq.size() - k;
    for (std::size_t i = 0; i <= stop; ++i) {
        hash ^= comp( mask, std::string_view(seq), i );
    }
    return hash;
}

// =================================================================================================
//     CLARK
// =================================================================================================

// ------------------------------------------------------------------------
//     Original
// ------------------------------------------------------------------------

// The code in this section is adapted from https://github.com/rouni001/CLARK,
// https://github.com/rouni001/CLARK/blob/42fb5734b9443e141f18cf225b62841d2551817d/src/kmersConversion.cc#L148-L223
// which is under the GNU General Public License (GPL) v3 license,
// and hence differs from the rest of the license of this project.

inline void clark_getSpacedSeedOPTSS95s2(const uint64_t& _kmerR,  uint64_t& _skmerR)
{
    // 1111011101110010111001011011111
    // 1111*111*111**1*111**1*11*11111
    _skmerR = _kmerR >> 54; // Add 1111
    _skmerR <<= 6 ;
    _skmerR ^= (_kmerR & 0xFFFFFFFFFFFFFUL)>> 46; // Add 111
    _skmerR <<= 6 ;
    _skmerR ^= (_kmerR & 0xFFFFFFFFFFFUL) >> 38; // Add 111
    _skmerR <<= 2 ;
    _skmerR ^= (_kmerR & 0x3FFFFFFFFUL) >> 32; // Add 1
    _skmerR <<= 6 ;
    _skmerR ^= (_kmerR & 0x3FFFFFFFUL) >> 24; // Add 111
    _skmerR <<= 2 ;
    _skmerR ^= (_kmerR & 0xFFFFFUL) >> 18; // Add 1
    _skmerR <<= 4 ;
    _skmerR ^= (_kmerR & 0xFFFFUL) >> 12; // Add 11
    _skmerR <<= 10 ;
    _skmerR ^= (_kmerR & 0x3FFUL); // Add 11111
    return;
}

inline void clark_getSpacedSeedT38570(const uint64_t& _kmerR, uint64_t& _skmerR)
{
    // 1111101011100101101110011011111
    // 11111*1*111**1*11*111**11*11111
    _skmerR = _kmerR >> 52; // 26, Add 1
    _skmerR <<= 2 ;
    _skmerR ^= (_kmerR & 0x3FFFFFFFFFFFFUL)>> 48; // Add 1
    _skmerR <<= 6 ;
    _skmerR ^= (_kmerR & 0x3FFFFFFFFFFFUL)>> 40; // Add 1
    _skmerR <<= 2 ;
    _skmerR ^= (_kmerR & 0xFFFFFFFFFUL)>> 34; // Add 1
    _skmerR <<= 4 ;
    _skmerR ^= (_kmerR & 0xFFFFFFFFUL)>> 28; // Add 1
    _skmerR <<= 6 ;
    _skmerR ^= (_kmerR & 0x3FFFFFFUL)>> 20; // Add 1
    _skmerR <<= 4 ;
    _skmerR ^= (_kmerR & 0xFFFFUL)>> 12; // Add 1
    _skmerR <<= 10;
    _skmerR ^= (_kmerR & 0x3FFUL); // Add 1
    return;
}

inline void clark_getSpacedSeedT58570(const uint64_t& _kmerR,  uint64_t& _skmerR)
{
    // 1111101001110101101100111011111
    // 11111*1**111*1*11*11**111*11111
    _skmerR = _kmerR >> 52; // 26, Add 1
    _skmerR <<= 2 ;
    _skmerR ^= (_kmerR & 0x3FFFFFFFFFFFFUL)>> 48; // Add 1
    _skmerR <<= 6 ;
    _skmerR ^= (_kmerR & 0xFFFFFFFFFFFUL)>> 38; // Add 1
    _skmerR <<= 2 ;
    _skmerR ^= (_kmerR & 0xFFFFFFFFFUL)>> 34; // Add 1
    _skmerR <<= 4 ;
    _skmerR ^= (_kmerR & 0xFFFFFFFFUL)>> 28; // Add 1
    _skmerR <<= 4 ;
    _skmerR ^= (_kmerR & 0x3FFFFFFUL)>> 22; // Add 1
    _skmerR <<= 6 ;
    _skmerR ^= (_kmerR & 0x3FFFFUL)>> 12; // Add 1
    _skmerR <<= 10;
    _skmerR ^= (_kmerR & 0x3FFUL); // Add 1
    return;
}

inline void clark_getSpacedSeed(
    const std::string& _setting, const uint64_t& _kmerR, uint64_t& _skmerR
) {
    // For whatever reason, the conditions were not in the same order
    // as the functions that are called below, and neither in alphabetical order.
    // We reordered, for clarity.

    if (_setting == "T295"){
        return clark_getSpacedSeedOPTSS95s2(_kmerR, _skmerR);
        }
    if (_setting == "T38570"){
        return clark_getSpacedSeedT38570(_kmerR, _skmerR);
    }
    if (_setting == "T58570"){
        return clark_getSpacedSeedT58570(_kmerR, _skmerR);
    }
    std::cerr << "Failed to find the mask for the spaced seed requested."<< std::endl;
    exit(1);
}

// ------------------------------------------------------------------------
//     Improved
// ------------------------------------------------------------------------

inline void clark_getSpacedSeedOPTSS95s2_improved(const uint64_t& _kmerR,  uint64_t& _skmerR)
{
    // Mask:
    // 1111011101110010111001011011111
    // 1111*111*111**1*111**1*11*11111

    // Expand by doubling every digit, then construct hex values for them.
    // 0011'1111'1100'1111'1100'1111'1100'0011'0011'1111'0000'1100'1111'0011'1111'1111
    //    3    F    C    F    C    F    C    3    3    F    0    C    F    3    F    F
    // 0x3FCFCFC33F0CF3FFULL

    // We can now decompose the kmer into blocks, using the above as block masks,
    // shifted to the correct positions already. Better processor pipelining.
    _skmerR =
          ( _kmerR & 0x00000000000003FFULL)
        | ((_kmerR & 0x000000000000F000ULL) >> 2)
        | ((_kmerR & 0x00000000000C0000ULL) >> 4)
        | ((_kmerR & 0x000000003F000000ULL) >> 8)
        | ((_kmerR & 0x0000000300000000ULL) >> 10)
        | ((_kmerR & 0x00000FC000000000ULL) >> 14)
        | ((_kmerR & 0x000FC00000000000ULL) >> 16)
        | ((_kmerR & 0x3FC0000000000000ULL) >> 18);
}

inline void clark_getSpacedSeedT38570_improved(const uint64_t& _kmerR, uint64_t& _skmerR)
{
    // Mask:
    // 1111101011100101101110011011111
    // 11111*1*111**1*11*111**11*11111

    // Expand by doubling every digit, then construct hex values for them.
    // 0011'1111'1111'0011'0011'1111'0000'1100'1111'0011'1111'0000'1111'0011'1111'1111
    //    3    F    F    3    3    F    0    C    F    3    F    0    F    3    F    F
    // 0x3FF33F0CF3F0F3FFULL

    // We can now decompose the kmer into blocks, using the above as block masks,
    // shifted to the correct positions already. Better processor pipelining.
    _skmerR =
          ( _kmerR & 0x00000000000003FFULL)
        | ((_kmerR & 0x000000000000F000ULL) >> 2)
        | ((_kmerR & 0x0000000003F00000ULL) >> 6)
        | ((_kmerR & 0x00000000F0000000ULL) >> 8)
        | ((_kmerR & 0x0000000C00000000ULL) >> 10)
        | ((_kmerR & 0x00003F0000000000ULL) >> 14)
        | ((_kmerR & 0x0003000000000000ULL) >> 16)
        | ((_kmerR & 0x3FF0000000000000ULL) >> 18);
}

inline void clark_getSpacedSeedT58570_improved(const uint64_t& _kmerR,  uint64_t& _skmerR)
{
    // Mask:
    // 1111101001110101101100111011111
    // 11111*1**111*1*11*11**111*11111

    // 0011'1111'1111'0011'0000'1111'1100'1100'1111'0011'1100'0011'1111'0011'1111'1111
    //    3    F    F    3    0    F    C    C    F    3    C    3    F    3    F    F
    // 0x3FF30FCCF3C3F3FFULL

    // We can now decompose the kmer into blocks, using the above as block masks,
    // shifted to the correct positions already. Better processor pipelining.
    _skmerR =
          ( _kmerR & 0x00000000000003FFULL)
        | ((_kmerR & 0x000000000003F000ULL) >> 2)
        | ((_kmerR & 0x0000000003C00000ULL) >> 6)
        | ((_kmerR & 0x00000000F0000000ULL) >> 8)
        | ((_kmerR & 0x0000000C00000000ULL) >> 10)
        | ((_kmerR & 0x00000FC000000000ULL) >> 12)
        | ((_kmerR & 0x0003000000000000ULL) >> 16)
        | ((_kmerR & 0x3FF0000000000000ULL) >> 18);
}

inline void clark_getSpacedSeed_improved(
    size_t _setting, const uint64_t& _kmerR, uint64_t& _skmerR
) {
    // We eliminate the expensive string comparison from computing each spaced k-mer.
    // The switch is still expensive
    switch( _setting ) {
        case 0: return clark_getSpacedSeedOPTSS95s2_improved(_kmerR, _skmerR);
        case 1: return clark_getSpacedSeedT38570_improved(_kmerR, _skmerR);
        case 2: return clark_getSpacedSeedT58570_improved(_kmerR, _skmerR);
        default: {
            std::cerr << "Failed to find the mask for the spaced seed requested."<< std::endl;
            exit(1);
        }
    }
}

// ------------------------------------------------------------------------
//     Wrapper for sequence
// ------------------------------------------------------------------------

template<typename Mask, typename Clark>
inline std::uint64_t clark_compute_sequence_hash(
    size_t k, Mask const& mask, std::string const& seq, Clark&& clark
) {
    // Compute all spaced kmers across the sequence, and xor their hashes, for our checking.
    std::uint64_t hash = 0;

    for_each_kmer_2bit(
        std::string_view(seq),
        k,
        char_to_nt_table,
        [&](std::uint64_t kmer_word) {
            // Extract the spaced k-mer, and combine it into the hash.
            // Clark does this via output parameter...
            std::uint64_t result = 0;
            clark( mask, kmer_word, result );
            hash ^= result;
        }
    );
    return hash;
}

// =================================================================================================
//     PEXT
// =================================================================================================

inline std::uint64_t pext_prepare_mask( std::string const& mask )
{
    if( mask.size() == 0 || mask.size() > 32 ) {
        throw std::invalid_argument( "Invalid mask size not in [1,32]" );
    }
    std::uint64_t result = 0;
    for( size_t i = 0; i < mask.size(); ++i ) {
        result <<= 2;
        if( mask[i] == '0' ) {
            continue;
        } else if( mask[i] == '1' ) {
            result |= 3;
        } else {
            throw std::invalid_argument( "Invalid mask with symbols not in [0,1]" );
        }
    }
    return result;
}

template<typename Mask, typename Enc, typename Pext>
inline std::uint64_t pext_compute_sequence_hash(
    size_t k, Mask const& mask, std::string const& seq, Enc&& enc, Pext&& pext
) {
    // Compute all spaced kmers across the sequence, and xor their hashes, for our checking.
    std::uint64_t hash = 0;

    for_each_kmer_2bit(
        std::string_view(seq),
        k,
        enc,
        [&](std::uint64_t kmer_word) {
            // Extract the spaced k-mer, and combine it into the hash.
            hash ^= pext( kmer_word, mask );
        }
    );
    return hash;
}
