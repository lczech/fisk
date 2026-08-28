#pragma once

#include <array>
#include <string>
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <cassert>
#include <iostream>
#include <vector>

// =================================================================================================
//     Character Encoding
// =================================================================================================

// Encode an uppercase nucleotide (A/C/G/T) into 2 bits. We offer two different orderings:
//
// ACGT (alphabetical):
//   A -> 0b00
//   C -> 0b01
//   G -> 0b10
//   T -> 0b11
//
//   Complementary bases (A<->T, C<->G) are bitwise complements of each other under this ordering
//   (XOR with 0b11), which makes reverse-complement handling cheap.
//
// ACTG:
//   A -> 0b00
//   C -> 0b01
//   T -> 0b10
//   G -> 0b11
//
//   This is the ordering that falls out directly of bits 1-2 of the raw ASCII code, with no
//   further transformation, making it cheaper to compute than ACGT (see char_to_nt_ascii_actg()
//   below). It does not have ACGT's cheap-reverse-complement property, so use ACGT instead
//   whenever that property is needed. Only the lookup table and ascii-trick variants are offered
//   for ACTG, as those are the two performant implementations; the `ifs`/`switch` variants are
//   only offered for ACGT.
//
// We provide and benchmark different variants of these functions here.

// -----------------------------------------------------------------------------
//     ifs (ACGT)
// -----------------------------------------------------------------------------

/**
 * @brief Get the two-bit ACGT encoding of a char, using a series of `if` statements, non-throwing.
 */
inline constexpr std::uint8_t char_to_nt_ifs_acgt(char ch) noexcept
{
    // We need ASCII for the following to work.
    static_assert( static_cast<int>('A') == 0x41, "Non-ASCII char set" );
    static_assert( static_cast<int>('C') == 0x43, "Non-ASCII char set" );
    static_assert( static_cast<int>('G') == 0x47, "Non-ASCII char set" );
    static_assert( static_cast<int>('T') == 0x54, "Non-ASCII char set" );
    static_assert( static_cast<int>('a') == 0x61, "Non-ASCII char set" );
    static_assert( static_cast<int>('c') == 0x63, "Non-ASCII char set" );
    static_assert( static_cast<int>('g') == 0x67, "Non-ASCII char set" );
    static_assert( static_cast<int>('t') == 0x74, "Non-ASCII char set" );

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

// -----------------------------------------------------------------------------
//     switch (ACGT)
// -----------------------------------------------------------------------------

/**
 * @brief Get the two-bit ACGT encoding of a char, using switch statement, non-throwing.
 */
inline constexpr std::uint8_t char_to_nt_switch_acgt(char ch) noexcept
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

// -----------------------------------------------------------------------------
//     ascii (ACGT)
// -----------------------------------------------------------------------------

/**
 * @brief Get the two-bit ACGT encoding of a char, using bit twiddling to utilize a coincidence
 * in ASCII code, returning invalid code `4` if the char is not in `ACGT`.
 */
inline constexpr std::uint8_t char_to_nt_ascii_acgt(char c) noexcept
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
    //
    // This is the ACGT convention. The XOR step here is what makes this the more expensive of
    // the two conventions; see char_to_nt_ascii_actg() below for the ACTG convention, which skips
    // it, at the cost of no longer having ACGT's cheap-reverse-complement property.

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
    std::uint8_t const value = static_cast<std::uint8_t>(c) | 0x20u;

    // Extract the relevant bits to get two-bit code.
    std::uint8_t const encoding = ((value >> 1) ^ (value >> 2)) & 0x03u;

    // Use a bitset validator to check for correct char;
    // should be faster than actual character comparisons.
    // a & 31 = 1
    // c & 31 = 3
    // g & 31 = 7
    // t & 31 = 20
    std::uint32_t constexpr valid_mask  = (1u << 1) | (1u << 3) | (1u << 7) | (1u << 20);
    std::uint32_t const     low5_valid  = (valid_mask >> (value & 31u)) & 1u;
    std::uint32_t const     ascii_lower = ((value & 0xE0u) == 0x60u);
    std::uint32_t const     is_valid    = low5_valid & ascii_lower;
    return is_valid ? encoding : 4;

    // Alternative implementation with simple checks. Somewhat slower due to all the comparisons.
    // if(
    //     ( c != 'A' ) && ( c != 'C' ) && ( c != 'G' ) && ( c != 'T' ) &&
    //     ( c != 'a' ) && ( c != 'c' ) && ( c != 'g' ) && ( c != 't' )
    // ) {
    //     return 4;
    // }

    // auto const u = static_cast<std::uint8_t>(c);
    // return ((u >> 1) ^ (u >> 2)) & 3;
}

/**
 * @brief Get the two-bit ACGT encoding of a char, using bit twiddling to utilize a coincidence
 * in ASCII code, without char validity check. Only use when it is clear that the input is
 * in `ACGT`.
 */
inline constexpr std::uint8_t char_to_nt_ascii_unchecked_acgt(char c) noexcept
{
    // Same logic as above.
    auto const u = static_cast<std::uint8_t>(c);
    return ((u >> 1) ^ (u >> 2)) & 0x03u;
}

/**
 * @brief Verify the char_to_nt_ascii_acgt() function for all ASCII values.
 */
inline void test_char_to_nt_ascii_acgt()
{
    auto expected = [](unsigned char c) -> std::uint8_t {
        switch (c) {
            case 'A': case 'a': return 0;
            case 'C': case 'c': return 1;
            case 'G': case 'g': return 2;
            case 'T': case 't': return 3;
            default:            return 4;
        }
    };

    for (int i = 0; i < 256; ++i) {
        unsigned char c = static_cast<unsigned char>(i);

        std::uint8_t const got = char_to_nt_ascii_acgt(static_cast<char>(c));
        std::uint8_t const exp = expected(c);

        if (got != exp) {
            std::cerr
                << "Mismatch for byte 0x"
                << std::hex << i
                << " ('" << (std::isprint(c) ? char(c) : '?') << "')"
                << " expected=" << std::dec << int(exp)
                << " got=" << int(got) << "\n";

            assert(false);
        }
    }
    // std::cout << "char_to_nt_ascii_acgt(): all 256 values verified\n";
}

// -----------------------------------------------------------------------------
//     ascii (ACTG)
// -----------------------------------------------------------------------------

/**
 * @brief Get the two-bit ACTG encoding of a char, using bit twiddling to utilize a coincidence
 * in ASCII code, returning invalid code `4` if the char is not in `ACGT`.
 *
 * Unlike char_to_nt_ascii_acgt(), this extracts bits 1-2 of the (lowercase-folded) ASCII code
 * directly, with no further transformation, giving A=0, C=1, T=2, G=3 (ACTG order). This is
 * cheaper than the ACGT variant, at the cost of not having ACGT's cheap-reverse-complement
 * property.
 *
 * Note though that this variant still performs a validity check, which is more expensive than
 * the encoding itself. For full speed, and if it is clear that the input is in `ACGT`, use
 * char_to_nt_ascii_unchecked_actg() instead.
 */
inline constexpr std::uint8_t char_to_nt_ascii_actg(char c) noexcept
{
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
    std::uint8_t const value = static_cast<std::uint8_t>(c) | 0x20u;

    // Extract bits 1-2 directly (no XOR needed), giving A=0, C=1, T=2, G=3.
    std::uint8_t const encoding = (value >> 1) & 0x03u;

    // Same validity check as the ACGT variant; independent of which bits encode the value.
    std::uint32_t constexpr valid_mask  = (1u << 1) | (1u << 3) | (1u << 7) | (1u << 20);
    std::uint32_t const     low5_valid  = (valid_mask >> (value & 31u)) & 1u;
    std::uint32_t const     ascii_lower = ((value & 0xE0u) == 0x60u);
    std::uint32_t const     is_valid    = low5_valid & ascii_lower;
    return is_valid ? encoding : 4;
}

/**
 * @brief Get the two-bit ACTG encoding of a char, using bit twiddling to utilize a coincidence
 * in ASCII code, without char validity check. Only use when it is clear that the input is
 * in `ACGT`. It is the fastest of all the char_to_nt_ascii_*() variants.
 */
inline constexpr std::uint8_t char_to_nt_ascii_unchecked_actg(char c) noexcept
{
    // Same logic as above.
    auto const u = static_cast<std::uint8_t>(c);
    return (u >> 1) & 0x03u;
}

/**
 * @brief Verify the char_to_nt_ascii_actg() function for all ASCII values.
 */
inline void test_char_to_nt_ascii_actg()
{
    auto expected = [](unsigned char c) -> std::uint8_t {
        switch (c) {
            case 'A': case 'a': return 0;
            case 'C': case 'c': return 1;
            case 'T': case 't': return 2;
            case 'G': case 'g': return 3;
            default:            return 4;
        }
    };

    for (int i = 0; i < 256; ++i) {
        unsigned char c = static_cast<unsigned char>(i);

        std::uint8_t const got = char_to_nt_ascii_actg(static_cast<char>(c));
        std::uint8_t const exp = expected(c);

        if (got != exp) {
            std::cerr
                << "Mismatch for byte 0x"
                << std::hex << i
                << " ('" << (std::isprint(c) ? char(c) : '?') << "')"
                << " expected=" << std::dec << int(exp)
                << " got=" << int(got) << "\n";

            assert(false);
        }
    }
    // std::cout << "char_to_nt_ascii_actg(): all 256 values verified\n";
}

// -----------------------------------------------------------------------------
//     table (ACGT)
// -----------------------------------------------------------------------------

// Another typical implementation: ascii char lookup table.
// The table is hardcoded here, to allow static constexpr inlining.

/**
 * @brief Magic value for the positions in seq_nt4_table_acgt/seq_nt4_table_actg that are not
 * `ACGT` or `acgt`. Shared between both conventions, as "invalid" (4) does not depend on which
 * of the two orderings is otherwise in use.
 */
constexpr std::uint8_t SEQ_NT4_INVALID = 4;

/**
 * @brief Lookup table for ASCII to two-bit ACGT encoding of nucleotides.
 *
 * See SEQ_NT4_INVALID for the magic constant holding the "invalid" value for all ASCII chars
 * that are not `ACGT` or `acgt`. The original table from Heng Li uses 0,1,2,3 as the first
 * four entries, probably to make the table idempotent, but we explitly disallow this
 * here to avoid accidental misuse. Also maps 'U'/'u' to the same value as 'T'/'t'.
 */
inline constexpr std::uint8_t seq_nt4_table_acgt[256] = {
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
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
 * @brief Get the two-bit ACGT encoding of a char (values 0-3 for `ACGT`), using a lookup table,
 * and returning `SEQ_NT4_INVALID = 4` if the char is not in `ACGT`.
 */
inline constexpr std::uint8_t char_to_nt_table_acgt(char c) noexcept
{
    return seq_nt4_table_acgt[static_cast<std::uint8_t>(c)];
}

/**
 * @brief Generate the lookup table for two-bit ACGT encoding.
 *
 * Helper function to generate the table, instead of hardcoding it.
 * However, static constexpr variables are C++23, which means our simple hard coded table
 * is likely better suited for compiler optimizations. We are thus not using this function here,
 * and just offer it for completeness.
 */
inline std::array<std::uint8_t,256> const& get_seq_nt4_table_acgt()
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
 * @brief Get the two-bit ACGT encoding of a char, using a lookup table, non-throwing.
 *
 * This helper struct encapsulates the lookup table and provides a simple interface
 * for encoding nucleotide characters. The encoding returns 0..3 for A,C,G,T (and their lower case
 * equivalents); returns `INVALID_NT = 4` for all other characters.
 */
struct NucleotideEncoderAcgt
{
    // Constants. The table is initialized in the translation unit `seq_enc.cpp`
    static constexpr std::uint8_t INVALID_NT = 4;
    static inline constexpr std::array<std::uint8_t, 256> table = []
    {
        // Generate lookup table for typical nucleotide two-bit encoding.
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
    }();

    /**
     * @brief Get the two-bit encoding for a char.
     */
    static inline constexpr std::uint8_t encode(char c) noexcept
    {
        return table[static_cast<std::uint8_t>(c)];
    }
};

// -----------------------------------------------------------------------------
//     table (ACTG)
// -----------------------------------------------------------------------------

/**
 * @brief Lookup table for ASCII to two-bit ACTG encoding of nucleotides.
 *
 * Same structure as seq_nt4_table_acgt, but with A=0, C=1, T=2, G=3, including the same
 * 'U'/'u' as 'T'/'t' synonym handling.
 */
inline constexpr std::uint8_t seq_nt4_table_actg[256] = {
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 0, 4, 1,  4, 4, 4, 3,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  2, 2, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 0, 4, 1,  4, 4, 4, 3,  4, 4, 4, 4,  4, 4, 4, 4,
	4, 4, 4, 4,  2, 2, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
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
 * @brief Get the two-bit ACTG encoding of a char (values 0,1,2,3 for A,C,T,G), using a lookup
 * table, and returning `SEQ_NT4_INVALID = 4` if the char is not in `ACGT`.
 */
inline constexpr std::uint8_t char_to_nt_table_actg(char c) noexcept
{
    return seq_nt4_table_actg[static_cast<std::uint8_t>(c)];
}

/**
 * @brief Get the two-bit ACTG encoding of a char, using a lookup table, non-throwing.
 *
 * Same as NucleotideEncoderAcgt, but with A=0, C=1, T=2, G=3. Does not handle 'U'/'u', matching
 * NucleotideEncoderAcgt's own behavior.
 */
struct NucleotideEncoderActg
{
    static constexpr std::uint8_t INVALID_NT = 4;
    static inline constexpr std::array<std::uint8_t, 256> table = []
    {
        std::array<std::uint8_t, 256> t{};
        for( auto& x : t ) {
            x = INVALID_NT;
        }
        t[static_cast<unsigned char>('A')] = 0;
        t[static_cast<unsigned char>('C')] = 1;
        t[static_cast<unsigned char>('T')] = 2;
        t[static_cast<unsigned char>('G')] = 3;
        t[static_cast<unsigned char>('a')] = 0;
        t[static_cast<unsigned char>('c')] = 1;
        t[static_cast<unsigned char>('t')] = 2;
        t[static_cast<unsigned char>('g')] = 3;
        return t;
    }();

    static inline constexpr std::uint8_t encode(char c) noexcept
    {
        return table[static_cast<std::uint8_t>(c)];
    }
};

// =================================================================================================
//     Two-Bit Packed Sequence
// =================================================================================================

/**
 * @brief Bit order convention for how bases are packed within each word of a TwoBitSequence.
 *
 * `Msb`: the first (leftmost) base of a word occupies the most significant bits, matching the
 * rolling k-mer convention `kmer = (kmer << 2) | code`, and hence preserving lexicographic string
 * order as integer order. This is the same convention widely used by k-mer tools
 * (e.g. KMC packs k-mers "leftmost symbol first") for exactly this reason, useful for canonical
 * k-mer/minimizer selection.
 *
 * `Lsb`: the first base occupies the least significant bits. This is the order that falls out
 * directly of a little-endian PEXT-based batch encode with no extra transformation, and is
 * therefore cheaper to produce with PEXT, at the cost of not preserving lexicographic order.
 *
 * Deliberately tracked as a template parameter on TwoBitSequence rather than left to convention,
 * so that mixing up the two orderings between producer and consumer is a compile error rather
 * than a silent error.
 */
enum class BitOrder
{
    Msb,
    Lsb
};

/**
 * @brief A whole sequence, encoded into a densely packed two-bit-per-base representation.
 *
 * `data` holds `ceil(length / 32)` words of 32 bases each, plus one trailing all-zero sentinel
 * word past the real content, so that any window read spanning up to two words is guaranteed to
 * have valid bits to read even at the very end of the sequence. Bases at or beyond `length` are
 * unspecified padding and must not be accessed by callers.
 *
 * See BitOrder for what the `Order` template parameter means.
 */
template <BitOrder Order>
struct TwoBitSequence
{
    static constexpr BitOrder order = Order;

    std::vector<std::uint64_t> data;
    std::size_t length = 0;
};
