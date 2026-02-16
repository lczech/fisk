#pragma once

#include <string>
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

// =================================================================================================
//     Character Encoding
// =================================================================================================

// Encode an uppercase nucleotide (A/C/G/T) into 2 bits.
//
// Encoding convention:
//   A -> 0b00
//   C -> 0b01
//   G -> 0b10
//   T -> 0b11
//
// The caller is expected to only pass A/C/G/T.
// We provide and benchmark different variants of this function here.

// -----------------------------------------------------------------------------
//     ifs
// -----------------------------------------------------------------------------

// Simple if statements, as used in MISSH.
inline constexpr std::uint8_t char_to_nt_ifs(char ch)
{
	if(ch == 'A')
		return 0;
	if(ch == 'C')
		return 1;
	if(ch == 'G')
		return 2;
	if(ch == 'T')
		return 3;
	return 4; //ERROR CODE
}

// -----------------------------------------------------------------------------
//     switch
// -----------------------------------------------------------------------------

// Simply switch statement, often used.
inline constexpr std::uint8_t char_to_nt_switch(char c)
{
    switch (c) {
        case 'A': return 0u;
        case 'C': return 1u;
        case 'G': return 2u;
        case 'T': return 3u;
        default:
            // return 4u;
            throw std::runtime_error(
                "Handling of non-ACGT characters not supported in this simple benchmark"
            );
    }
}

// -----------------------------------------------------------------------------
//     table
// -----------------------------------------------------------------------------

// Another typical implementation: ascii char lookup table.
// Adapted from Heng Li:
// https://github.com/lh3/minimap2/blob/e2542e6425a40adaa710e07ae5e6188c91f8728c/sketch.c#L9

constexpr std::uint8_t seq_nt4_table[256] = {
	0, 1, 2, 3,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  3, 3, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};

inline constexpr std::uint8_t char_to_nt_table(char c)
{
    auto const r = seq_nt4_table[static_cast<std::uint8_t>(c)];
    if( r == 4 ) {
        throw std::runtime_error(
            "Handling of non-ACGT characters not supported in this simple benchmark"
        );
    }
    return r;
}

// -----------------------------------------------------------------------------
//     ascii
// -----------------------------------------------------------------------------

inline constexpr std::uint8_t char_to_nt_ascii(char c)
{
    // The check below is the fastest in our tests;
    // faster than using toupper() to avoid the extra case checks.
    if(
        ( c != 'A' ) && ( c != 'C' ) && ( c != 'G' ) && ( c != 'T' ) &&
        ( c != 'a' ) && ( c != 'c' ) && ( c != 'g' ) && ( c != 't' )
    ) {
        throw std::runtime_error(
            "Handling of non-ACGT characters not supported in this simple benchmark"
        );
        // return 4;
    }

    auto const u = static_cast<std::uint8_t>(c);
    return ((u >> 1) ^ (u >> 2)) & 3;
}

inline constexpr std::uint8_t char_to_nt_ascii_unchecked(char c)
{
    auto const u = static_cast<std::uint8_t>(c);
    return ((u >> 1) ^ (u >> 2)) & 3;
}

// =================================================================================================
//     Sequence Encoding
// =================================================================================================

// Scan a sequence and encode each character, combining them to get a final "hash".
// Not a good one, but enough to check that all the above functions give the same result,
// and sufficient to force the compiler to actually run the encoding.
template <typename EncodeFunc>
inline std::uint64_t sequence_encode(std::string_view seq, EncodeFunc&& encode)
{
    std::uint64_t h = 0;
    for (char c : seq) {
        h += encode(c);
    }
    return h;
}
