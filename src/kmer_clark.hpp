#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <string>
#include <vector>

#include "kmer_extract.hpp"
#include "kmer_spaced.hpp"
#include "bit_extract.hpp"
#include "seq_enc.hpp"

// =================================================================================================
//     CLARK Original
// =================================================================================================

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
    // In each iteration in clark, all three masks are applied. But each time, the kmer is first
    // fully extracted from scratch, again and again. Furthermore, mask selection is done via
    // string comparison, which is slow-ish in itself (despite short string optimization)
    // and induces branch predictor misses each time, as we are rotating through all three masks.

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

template<typename Enc>
inline bool clark_isFwdValid(
    std::string_view seq,
    std::size_t i,
    std::array<bool, 32> const& m_mask,
    Enc&& enc,
    std::uint64_t& out_kmer
) {
    // A slower version of the kmer extract, where each kmer is extracted fully from scratch.
    // Also, adding some conditions that clark uses, and repeated calls to enc.
    out_kmer = 0;
    std::size_t const k = 31;
    if( seq.size() < i + k ) {
        return false;
    }

    // Extract the kmer at the current positions.
    size_t c = 0;  // fill state of the kmer
    size_t ic = i; // progression in the sequence, starting at current pos
    while(true) {
        if( c == k ) {
            // done with the kmer
            return true;
        }
        if( ic == seq.size() && c < k ) {
            // reached end of data. won't happen, as the outer loop with not run
            // in that case. but added here for clark completeness
            return false;
        }
        if( enc(seq[ic]) == -10 ) {
            // special condition for new lines, never happening here with our input
            ++ic;
            continue;
        }
        if( enc(seq[ic]) < 0 ) {
            // special condition for other invalids, never happening here with our input
            return false;
        }
        if( enc(seq[ic]) != 4 ) {
            // the good case: add enc to the kmer
            out_kmer <<= 2;
            out_kmer ^= static_cast<uint64_t>(enc(seq[ic]));
            ic++;
            c++;
            continue;
        }
        if( !m_mask[c] ) {
            // This is a weird case: It triggers for positions that are 0 in the mask,
            // which will be filtered out when extracting the spaced k-mer later anyway.
            // So it seems it might not be needed. But for `N` bases for instance, it should
            // still move the k-mer forward by left shifting, so that the base is not forgotten.
            // Not sure what is going on here, and it seems to work, so let's keep it.
            out_kmer <<= 2;
            out_kmer ^= 0;
            ic++;
            c++;
            continue;
        }
        // if( enc(seq[ic]) == 4 && m_mask[c]) {
        //     throw std::runtime_error(
        //         "Infinite-loop case hit: ambiguous base at required mask position"
        //     );
        // }
        return false;
    }
    return false;
}

template<typename Enc>
inline bool clark_querySpacedElement(
    std::string_view seq,
    std::size_t i,
    std::string const& mask,
    std::array<bool, 32> const& m_mask,
    Enc&& enc,
    std::uint64_t& out_kmer
) {
    std::uint64_t kmer;
    if (!clark_isFwdValid(seq, i, m_mask, enc, kmer)) {
        return false;
    }
    clark_getSpacedSeed(mask, kmer, out_kmer);
    return true;
}

inline std::array<int, 256> clark_get_m_table()
{
    std::array<int, 256> m_table{};
    m_table.fill(-1);

    // Clark uses the inverse of our encoding; change it to fit.
    m_table['A']  = 0; m_table['C'] = 1; m_table['G'] = 2; m_table['T'] = 3; m_table['U'] = 3;
    m_table['a']  = 0; m_table['c'] = 1; m_table['g'] = 2; m_table['t'] = 3; m_table['u'] = 3;
    m_table['\n'] = -10;
    m_table['n']  = 4; m_table['N'] = 4;
    m_table['M']  = 4; m_table['R'] = 4; m_table['W'] = 4; m_table['V'] = 4; m_table['D'] = 4;
    m_table['K']  = 4; m_table['Y'] = 4; m_table['S'] = 4; m_table['H'] = 4; m_table['B'] = 4;
    m_table['m']  = 4; m_table['r'] = 4; m_table['w'] = 4; m_table['v'] = 4; m_table['d'] = 4;
    m_table['k']  = 4; m_table['y'] = 4; m_table['s'] = 4; m_table['h'] = 4; m_table['b'] = 4;

    // m_table['A']  = 3; m_table['C'] = 2; m_table['G'] = 1; m_table['T'] = 0; m_table['U'] = 0;
    // m_table['a']  = 3; m_table['c'] = 2; m_table['g'] = 1; m_table['t'] = 0; m_table['u'] = 0;
    // m_table['\n'] = -10;
    // m_table['n']  = 4; m_table['N'] = 4;
    // m_table['M']  = 4; m_table['R'] = 4; m_table['W'] = 4; m_table['V'] = 4; m_table['D'] = 4;
    // m_table['K']  = 4; m_table['Y'] = 4; m_table['S'] = 4; m_table['H'] = 4; m_table['B'] = 4;
    // m_table['m']  = 4; m_table['r'] = 4; m_table['w'] = 4; m_table['v'] = 4; m_table['d'] = 4;
    // m_table['k']  = 4; m_table['y'] = 4; m_table['s'] = 4; m_table['h'] = 4; m_table['b'] = 4;
    return m_table;
}

inline std::array<bool, 32> clark_get_m_mask( std::string const& _name )
{
    std::array<bool, 32> m_mask{};

    // Get the correct mask string for the given name.
    std::string _mask;
    if (_name =="T295") {
        _mask="1111*111*111**1*111**1*11*11111";
    }
    if (_name =="T38570") {
        _mask="11111*1*111**1*11*111**11*11111";
    }
    if (_name =="T58570") {
        _mask="11111*1**111*1*11*11**111*11111";
    }
    if( _mask.empty() ) {
        throw std::runtime_error( "Invalid mask name: " + _name );
    }

    // Build the bool array for the mask
    for(size_t t = 0; t < _mask.size(); t++) {
        if( _mask[t] == '0' || _mask[t] == '*' ) {
            m_mask[t] = false;
        } else if( _mask[t] == '1' ) {
            m_mask[t] = true;
        } else {
            throw std::invalid_argument( "Invalid mask symbol" );
        }
    }

    return m_mask;
}

inline std::vector<std::array<bool, 32>> clark_get_m_masks(
    std::vector<std::string> const& masks
) {
    std::vector<std::array<bool, 32>> m_masks;
    for( auto const& mask : masks ) {
        m_masks.push_back( clark_get_m_mask(mask) );
    }
    return m_masks;
}

inline std::uint64_t clark_getObjectsDataComputeFull(
    std::string_view seq,
    std::vector<std::string> const& masks
) {
    // Iterate the sequence, processing every kmer at every position, for each of the masks.
    // This is really inefficient in clark, as for each mask, we extract the full kmer again.
    // The original runs all of this twice then, forward and reverse complement.
    // But that seems to be just exactly double the work, so we do only half here for simplicity,
    // and match that in our improved version. Maybe there would be a way to improve on that
    // aspect as well, but that's for later.

    // Our computation here is again the simple xor of all found spaced kmers, as a check.
    std::uint64_t hash = 0;

    // Boundary checks
    size_t const k = 31;
    if (k == 0 || k > 32 ) {
        throw std::runtime_error( "Invalid call to k-mer extraction with k not in [1, 32]" );
    }
    if( seq.size() < k ) {
        return 0;
    }

    // Prepare the ascii to value table that clark uses
    static const std::array<int, 256> m_table = clark_get_m_table();
    auto clark_enc = [&](char c)
    {
        return m_table[static_cast<unsigned char>(c)];
    };

    // Prepare the m_masks, in which masked positions in the spaced k-mer are already left out
    // when building the original k-mer. Not sure why, because those are spaced out anyway.
    // Static, so that they are not recomputed on every call - but thus fixed for the masks.
    // This is thus only for our benchmarking, where we always use the same masks.
    // In Clark, these are stored within the spacedKmer class, achieving more flexibility.
    static const auto m_masks = clark_get_m_masks( masks );

    // Slide the kmer window over the sequence, getting all kmers.
    const std::size_t n    = seq.size();
    const std::size_t stop = n - k;
    for( std::size_t i = 0; i <= stop; ++i) {
        for( size_t m = 0; m < masks.size(); ++m ) {
            std::uint64_t kmer;
            if (clark_querySpacedElement( seq, i, masks[m], m_masks[m], clark_enc, kmer )) {
                hash += kmer;
            }
        }
    }
    return hash;
}

// =================================================================================================
//     Improved
// =================================================================================================

inline std::uint64_t clark_getSpacedSeedOPTSS95s2_improved(const uint64_t& _kmerR)
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
    return
          ( _kmerR & 0x00000000000003FFULL)
        | ((_kmerR & 0x000000000000F000ULL) >> 2)
        | ((_kmerR & 0x00000000000C0000ULL) >> 4)
        | ((_kmerR & 0x000000003F000000ULL) >> 8)
        | ((_kmerR & 0x0000000300000000ULL) >> 10)
        | ((_kmerR & 0x00000FC000000000ULL) >> 14)
        | ((_kmerR & 0x000FC00000000000ULL) >> 16)
        | ((_kmerR & 0x3FC0000000000000ULL) >> 18);
}

inline std::uint64_t clark_getSpacedSeedT38570_improved(const uint64_t& _kmerR)
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
    return
          ( _kmerR & 0x00000000000003FFULL)
        | ((_kmerR & 0x000000000000F000ULL) >> 2)
        | ((_kmerR & 0x0000000003F00000ULL) >> 6)
        | ((_kmerR & 0x00000000F0000000ULL) >> 8)
        | ((_kmerR & 0x0000000C00000000ULL) >> 10)
        | ((_kmerR & 0x00003F0000000000ULL) >> 14)
        | ((_kmerR & 0x0003000000000000ULL) >> 16)
        | ((_kmerR & 0x3FF0000000000000ULL) >> 18);
}

inline std::uint64_t clark_getSpacedSeedT58570_improved(const uint64_t& _kmerR)
{
    // Mask:
    // 1111101001110101101100111011111
    // 11111*1**111*1*11*11**111*11111

    // 0011'1111'1111'0011'0000'1111'1100'1100'1111'0011'1100'0011'1111'0011'1111'1111
    //    3    F    F    3    0    F    C    C    F    3    C    3    F    3    F    F
    // 0x3FF30FCCF3C3F3FFULL

    // We can now decompose the kmer into blocks, using the above as block masks,
    // shifted to the correct positions already. Better processor pipelining.
    return
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
    uint64_t _kmerR, uint64_t& r1, uint64_t& r2, uint64_t& r3
) {
    // We eliminate the expensive string comparison from computing each spaced k-mer.
    // Instead, we call all three masks at once, as we need all of them anyway.
    // This function is only workable for sequences without non-ACGT charactters though.
    r1 = clark_getSpacedSeedOPTSS95s2_improved(_kmerR);
    r2 = clark_getSpacedSeedT38570_improved(_kmerR);
    r3 = clark_getSpacedSeedT58570_improved(_kmerR);
}

// ------------------------------------------------------------------------
//     Wrapper for sequence
// ------------------------------------------------------------------------

inline std::uint64_t clark_improved(
    std::string const& seq,
    std::vector<BitExtractMask> const& masks
) {
    // Compute all spaced kmers across the sequence, and xor their hashes, for our checking.
    std::uint64_t hash = 0;
    size_t const k = 31;
    if (seq.size() < k) {
        return 0;
    }

    // Helpful constants
    std::uint64_t const span_mask = ((std::uint64_t{1} << (2 * k)) - 1u);
    std::size_t const seq_len = seq.size();
    char const* data = seq.data();

    // Iterate all k-mers.
    // Clark seems to ignore invalid bases during the extraction, and just pretent they are A.
    // So that's what we are doing here as well. Our table gives value 4 = 0x100, which gets
    // reduced to 0x00.
    std::uint64_t kmer_word  = 0;
    std::uint64_t valid_bits = 0;
    for( std::size_t i = 0; i < seq_len; ++i ) {
        std::uint8_t const code = static_cast<std::uint8_t>(char_to_nt_table(data[i]));

        // Shift in the base. For invalid bases, the low 2 bits are irrelevant,
        // because validity is tested separately before emission.
        kmer_word = ((kmer_word << 2) & span_mask) | (code & 0x03u);

        // Shift in 11 for valid, 00 for invalid.
        valid_bits
            = ((valid_bits << 2) & span_mask)
            | (static_cast<std::uint64_t>(code < 4) * 0x03u)
        ;

        // Apply the callback for all masks that are satisfied at this position.
        // There are only three hard-coded masks, so we manually unroll.
        if ((valid_bits & masks[0].mask) == masks[0].mask) {
            hash += clark_getSpacedSeedOPTSS95s2_improved(kmer_word);
        }
        if ((valid_bits & masks[1].mask) == masks[1].mask) {
            hash += clark_getSpacedSeedT38570_improved(kmer_word);
        }
        if ((valid_bits & masks[2].mask) == masks[2].mask) {
            hash += clark_getSpacedSeedT58570_improved(kmer_word);
        }

        // Simple approach that adds every kmer to the hash.
        // if( i < 31 - 1 ) {
        //     continue;
        // }
        // uint64_t v1, v2, v3;
        // clark_getSpacedSeed_improved( kmer_word, v1, v2, v3 );
        // hash ^= v1 ^ v2 ^ v3;
    }
    return hash;
}
