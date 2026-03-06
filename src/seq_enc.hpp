#pragma once

#include <array>
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

/**
 * @brief Get the two-bit encoding of a char, using a series of `if` statements, non-throwing.
 */
inline constexpr std::uint8_t char_to_nt_ifs_nothrow(char ch) noexcept
{
    // Make char lower case. The std implementation is locale dependend,
    // and very slow; we hence assume ASCII and simply set the lower case bit.
    // ch = static_cast<char>(std::tolower(ch));
    ch = (ch | 0x20);

    // Return the correct two-bit code
    if(ch == 'a') return 0u;
    if(ch == 'c') return 1u;
    if(ch == 'g') return 2u;
    if(ch == 't') return 3u;
    return 4;

    // Longer variant without lower case but more checks.
    // Same speed in our tests.
    // if(ch == 'A' || ch == 'a') return 0u;
    // if(ch == 'C' || ch == 'c') return 1u;
    // if(ch == 'G' || ch == 'g') return 2u;
    // if(ch == 'T' || ch == 't') return 3u;
}

/**
 * @brief Get the two-bit encoding of a char, using a series of `if` statements,
 * throwing an exception if the char is not in `ACGT`.
 */
inline constexpr std::uint8_t char_to_nt_ifs_throw(char ch)
{
    // Same as above
    ch = (ch | 0x20);
    if(ch == 'a') return 0u;
    if(ch == 'c') return 1u;
    if(ch == 'g') return 2u;
    if(ch == 't') return 3u;
	throw std::runtime_error(
        "Handling of non-ACGT characters not supported in this simple benchmark"
    );
}

// -----------------------------------------------------------------------------
//     switch
// -----------------------------------------------------------------------------

/**
 * @brief Get the two-bit encoding of a char, using switch statement, non-throwing.
 */
inline constexpr std::uint8_t char_to_nt_switch_nothrow(char ch) noexcept
{
    switch( ch ) {
        case 'A': case 'a': return 0u;
        case 'C': case 'c': return 1u;
        case 'G': case 'g': return 2u;
        case 'T': case 't': return 3u;
        default:
            return 4u;
    }
}

/**
 * @brief Get the two-bit encoding of a char, using switch statement,
 * throwing an exception if the char is not in `ACGT`.
 */
inline constexpr std::uint8_t char_to_nt_switch_throw(char ch)
{
    switch( ch ) {
        case 'A': case 'a': return 0u;
        case 'C': case 'c': return 1u;
        case 'G': case 'g': return 2u;
        case 'T': case 't': return 3u;
        default:
            throw std::runtime_error(
                "Handling of non-ACGT characters not supported in this simple benchmark"
            );
    }
}

// -----------------------------------------------------------------------------
//     ascii
// -----------------------------------------------------------------------------

/**
 * @brief Get the two-bit encoding of a char, using bit twiddling to utilize a coincidence
 * in ASCII code, non-throwing.
 */
inline constexpr std::uint8_t char_to_nt_ascii_nothrow(char c) noexcept
{
    // We here exploit the ASCII code of the characters.
    //
    // The lower halves of each character in ASCII are:
    // A 0001
    // C 0011
    // G 0111
    // T 0100
    //   -^^-
    //
    // These have a pattern in the middle bits (marked) that we use; doing a single right shift
    // puts those into the two rightmost bits of the result. The first of them (the left one)
    // is already what we want (A=C=0 and G=T=1), but the other (the right one) is not
    // (A=T=0 and C=G=1, but we want A=G=0 and C=T=1 for that bit). We xor with the other bit
    // to get our result, as that has a 1 for the G and the T, and gives us the encoding that
    // we want. Luckily, the fourth bit is always zero here, so that it does not mess this up.
    // This works for upper and lower case, as the case bit is in the higher four bits,
    // which are ignored here anyway.

    auto const u = static_cast<std::uint8_t>(c);
    return ((u >> 1) ^ (u >> 2)) & 0x03u;
}

/**
 * @brief Get the two-bit encoding of a char, using bit twiddling to utilize a coincidence
 * in ASCII code, throwing an exception if the char is not in `ACGT`.
 */
inline constexpr std::uint8_t char_to_nt_ascii_throw(char c)
{
    // // The check below is the fastest in our tests;
    // // faster than using toupper() to avoid the extra case checks.
    // if(
    //     ( c != 'A' ) && ( c != 'C' ) && ( c != 'G' ) && ( c != 'T' ) &&
    //     ( c != 'a' ) && ( c != 'c' ) && ( c != 'g' ) && ( c != 't' )
    // ) {
    //     throw std::runtime_error(
    //         "Handling of non-ACGT characters not supported in this simple benchmark"
    //     );
    //     // return 4;
    // }

    // auto const u = static_cast<std::uint8_t>(c);
    // return ((u >> 1) ^ (u >> 2)) & 3;


    // We need ASCII for the following to work. Probably fine, but doesn't hurt to check.
    static_assert( static_cast<int>('A') == 0x41, "Non-ASCII char set" );
    static_assert( static_cast<int>('C') == 0x43, "Non-ASCII char set" );
    static_assert( static_cast<int>('G') == 0x47, "Non-ASCII char set" );
    static_assert( static_cast<int>('T') == 0x54, "Non-ASCII char set" );
    static_assert( static_cast<int>('a') == 0x61, "Non-ASCII char set" );
    static_assert( static_cast<int>('c') == 0x63, "Non-ASCII char set" );
    static_assert( static_cast<int>('g') == 0x67, "Non-ASCII char set" );
    static_assert( static_cast<int>('t') == 0x74, "Non-ASCII char set" );

    // Fold to lowercase: 'A'..'Z' -> 'a'..'z', ASCII only.
    std::uint8_t const u = static_cast<std::uint8_t>(c) | 0x20u;

    // Extract the relevant bits to get two-bit code.
    std::uint8_t const e = ((u >> 1) ^ (u >> 2)) & 0x03u;

    // Use a bitset validator to check for correct char;
    // should be faster than actual character comparisons.
    // a & 31 = 1
    // c & 31 = 3
    // g & 31 = 7
    // t & 31 = 20
    constexpr std::uint32_t valid_mask = (1u << 1) | (1u << 3) | (1u << 7) | (1u << 20);
    std::uint32_t const is_valid = (valid_mask >> (u & 31u)) & 1u;
    // return is_valid ? e : 4;
    return is_valid
        ? e
        : throw std::runtime_error(
            "Handling of non-ACGT characters not supported in this simple benchmark"
        )
    ;
}

// -----------------------------------------------------------------------------
//     table
// -----------------------------------------------------------------------------

// Another typical implementation: ascii char lookup table.
// The table is hardcoded here, to allow static constexpr inlining.

/**
 * @brief Lookup table for ASCII to two-bit encoding of nucleotides.
 *
 * See SEQ_NT4_INVALID for the magic constant holding the "invalid" value for all ASCII chars
 * that are not `ACGT` or `acgt`.
 */
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

/**
 * @brief Magic value for the positions in seq_nt4_table that are not `ACGT` or `acgt`.
 */
constexpr std::uint8_t SEQ_NT4_INVALID = 4;

/**
 * @brief Get the two-bit encoding of a char, using a lookup table, non-throwing.
 */
inline constexpr std::uint8_t char_to_nt_table_throw(char c)
{
    auto const r = seq_nt4_table[static_cast<std::uint8_t>(c)];
    if( r == SEQ_NT4_INVALID ) {
        throw std::runtime_error(
            "Handling of non-ACGT characters not supported in this simple benchmark"
        );
    }
    return r;
}

/**
 * @brief Get the two-bit encoding of a char, using a lookup table,
 * throwing an exception if the char is not in `ACGT`.
 */
inline constexpr std::uint8_t char_to_nt_table_nothrow(char c) noexcept
{
    return seq_nt4_table[static_cast<std::uint8_t>(c)];
}

/**
 * @brief Generate the lookup table for two-bit encoding.
 *
 * Helper function to generate the table, instead of hardcoding it.
 * However, static constexpr variables are C++23, which means our simple hard coded table
 * is likely better suited for compiler optimizations. We are thus not using this function here,
 * and just offer it for completeness.
 */
inline std::array<std::uint8_t,256> const& get_seq_nt4_table()
{
    static const std::array<std::uint8_t,256> seq_nt4_table_ = []{
        std::array<std::uint8_t,256> t{};
        for (auto& x: t) {
            // x = 0xFF;
            x = SEQ_NT4_INVALID;
        }
        t[static_cast<unsigned char>('A')] = 0;
        t[static_cast<unsigned char>('C')] = 1;
        t[static_cast<unsigned char>('G')] = 2;
        t[static_cast<unsigned char>('T')] = 3;
        t[static_cast<unsigned char>('a')] = 0;
        t[static_cast<unsigned char>('c')] = 1;
        t[static_cast<unsigned char>('g')] = 2;
        t[static_cast<unsigned char>('t')] = 3;
        return t;
    }();
    return seq_nt4_table_;
}

/**
 * @brief Get the two-bit encoding of a char, using a lookup table, non-throwing.
 *
 * This helper struct encapsulates the lookup table and provides a simple interface
 * for encoding nucleotide characters. The encoding returns 0..3 for A,C,G,T (and their lower case
 * equivalents); returns `INVALID_NT = 4` for all other characters.
 */
struct NucleotideEncoder
{
    // Constants. The table is initialized in the translation unit `seq_enc.cpp`
    static constexpr std::uint8_t INVALID_NT = 4;
    static const std::array<std::uint8_t, 256> table;

    /**
     * @brief Generate lookup table for typical nucleotide two-bit encoding.
     */
    static constexpr std::array<std::uint8_t, 256> make_table()
    {
        std::array<std::uint8_t, 256> t{};
        for( auto& x : t ) {
            x = INVALID_NT;
        }
        t[static_cast<unsigned char>('A')] = 0;
        t[static_cast<unsigned char>('C')] = 1;
        t[static_cast<unsigned char>('G')] = 2;
        t[static_cast<unsigned char>('T')] = 3;
        t[static_cast<unsigned char>('a')] = 0;
        t[static_cast<unsigned char>('c')] = 1;
        t[static_cast<unsigned char>('g')] = 2;
        t[static_cast<unsigned char>('t')] = 3;
        return t;
    }

    /**
     * @brief Get the two-bit encoding for a char.
     */
    static inline constexpr std::uint8_t encode(char c) noexcept
    {
        return table[static_cast<std::uint8_t>(c)];
    }
};

// =================================================================================================
//     Sequence Encoding
// =================================================================================================

/**
 * @brief Scan a sequence and encode each character, combining them to get a final "hash".
 *
 * The hash obtained here is not a good one, as it is simply the sum of all two-bit encodings
 * of the characters. But it is enough to check that all the above functions give the same result,
 * and sufficient to force the compiler to actually run the encoding.
 */
template <typename EncodeFunc>
inline std::uint64_t sequence_encode(std::string_view seq, EncodeFunc&& encode)
{
    std::uint64_t h = 0;
    for (char c : seq) {
        h += encode(c);
    }
    return h;
}
