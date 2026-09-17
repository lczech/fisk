#pragma once

#include <array>
#include <concepts>
#include <string>
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

#include "fisk/core/platform.hpp"
#include "fisk/core/types.hpp"

namespace fisk {

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
//   further transformation, making it cheaper to compute than ACGT (see CharEncoderAscii below).
//   It does not sort k-mers in lexicographic order.
//
// Each encoder below is a stateless functor templated on the Encoding it produces, so that the
// convention is selected by template type tag at compile time, so that the encoder can hand its
// convention on to its consumer. What distinguishes them from each other is only the
// technique used to get there; we provide several so that they can be benchmarked against
// each other. All of them return 0-3 for a valid nucleotide and a value >= kInvalidNucleotide
// otherwise, for both upper and lower case.

// Every encoder below relies on the ASCII code points of the nucleotide characters, whether
// through the lookup tables or the bit tricks. Assert once here that the host character set
// really is ASCII, rather than repeating the same check inside each of them.
static_assert( static_cast<int>('A') == 0x41, "Non-ASCII char set" );
static_assert( static_cast<int>('C') == 0x43, "Non-ASCII char set" );
static_assert( static_cast<int>('G') == 0x47, "Non-ASCII char set" );
static_assert( static_cast<int>('T') == 0x54, "Non-ASCII char set" );
static_assert( static_cast<int>('a') == 0x61, "Non-ASCII char set" );
static_assert( static_cast<int>('c') == 0x63, "Non-ASCII char set" );
static_assert( static_cast<int>('g') == 0x67, "Non-ASCII char set" );
static_assert( static_cast<int>('t') == 0x74, "Non-ASCII char set" );

// -----------------------------------------------------------------------------
//     Shared
// -----------------------------------------------------------------------------

/**
 * @brief Code returned for any character that is not a nucleotide.
 *
 * Shared across all encoders and both encodings: "not a nucleotide" does not depend on which of
 * the two orderings is otherwise in use.
 */
constexpr std::uint8_t kInvalidNucleotide = 4;

/**
 * @brief Concept for anything usable as a per-character nucleotide encoder.
 *
 * Requires the encoder to state which Encoding its codes are in, so that consumers can tag what
 * they build from those codes rather than assuming a convention.
 */
template <typename T>
concept CharEncoder = requires(T encoder, char c) {
    { T::encoding } -> std::convertible_to<Encoding>;
    { encoder(c) } -> std::convertible_to<std::uint8_t>;
};

// -----------------------------------------------------------------------------
//     ifs
// -----------------------------------------------------------------------------

/**
 * @brief Get the two-bit encoding of a char, using a series of `if` statements, non-throwing.
 *
 * Kept for comparison against the faster techniques below rather than for production use.
 */
template <Encoding E>
struct CharEncoderIfs
{
    static constexpr Encoding encoding = E;

    inline constexpr std::uint8_t operator()(char ch) const noexcept
    {
        // Make char lower case. The std implementation is locale dependend,
        // and very slow; we hence assume ASCII and simply set the lower case bit,
        // which is fine, as we already statically asserted that we are working with ASCII.
        // ch = static_cast<char>(std::tolower(ch));
        ch = (ch | 0x20);

        // Return the correct two-bit code
        if constexpr (E == Encoding::kACGT) {
            if (ch == 'a') return 0u;
            if (ch == 'c') return 1u;
            if (ch == 'g') return 2u;
            if (ch == 't') return 3u;
        } else if constexpr (E == Encoding::kACTG) {
            if (ch == 'a') return 0u;
            if (ch == 'c') return 1u;
            if (ch == 't') return 2u;
            if (ch == 'g') return 3u;
        } else {
            static_assert(dependent_false_v<E>, "Unhandled Encoding in CharEncoderIfs");
        }
        return kInvalidNucleotide;

        // Longer variant without lower case but more checks.
        // Same speed in our tests.
        // if(ch == 'A' || ch == 'a') return 0u;
        // ...
    }
};

// -----------------------------------------------------------------------------
//     switch
// -----------------------------------------------------------------------------

/**
 * @brief Get the two-bit encoding of a char, using a switch statement, non-throwing.
 *
 * Kept for comparison against the faster techniques below rather than for production use.
 */
template <Encoding E>
struct CharEncoderSwitch
{
    static constexpr Encoding encoding = E;

    inline constexpr std::uint8_t operator()(char ch) const noexcept
    {
        if constexpr (E == Encoding::kACGT) {
            switch( ch ) {
                case 'A': case 'a': return 0u;
                case 'C': case 'c': return 1u;
                case 'G': case 'g': return 2u;
                case 'T': case 't': return 3u;
                default:            return kInvalidNucleotide;
            }
        } else if constexpr (E == Encoding::kACTG) {
            switch( ch ) {
                case 'A': case 'a': return 0u;
                case 'C': case 'c': return 1u;
                case 'T': case 't': return 2u;
                case 'G': case 'g': return 3u;
                default:            return kInvalidNucleotide;
            }
        } else {
            static_assert(dependent_false_v<E>, "Unhandled Encoding in CharEncoderSwitch");
            return kInvalidNucleotide;
        }
    }
};

// -----------------------------------------------------------------------------
//     ascii
// -----------------------------------------------------------------------------

/**
 * @brief Extract the two-bit code of a nucleotide from its ASCII byte, without validity check.
 *
 * We here exploit the ASCII code of the characters.
 *
 * The lower halves of each character in ASCII are:
 * A 0001
 * C 0011
 * G 0111
 * T 0100
 *   -^^-
 *
 * These have a pattern in the middle bits (marked) that we use; doing a single right shift
 * puts those into the two rightmost bits of the result. That already is the ACTG ordering,
 * with no further work: A=0, C=1, T=2, G=3.
 *
 * For ACGT we need one more step. The first of the two bits (the left one) is already what we
 * want (A=C=0 and G=T=1), but the other (the right one) is not (A=T=0 and C=G=1, but we want
 * A=G=0 and C=T=1 for that bit). We xor with the other bit to get our result, as that has a 1
 * for the G and the T, and gives us the encoding that we want. Luckily, the fourth bit is always
 * zero here, so that it does not mess this up. That XOR is what makes ACGT the more expensive of
 * the two orderings here, in exchange for its cheap-reverse-complement property.
 *
 * This works for upper and lower case, as the case bit is in the higher four bits, which are
 * ignored here anyway; callers may hence pass the raw or the case-folded byte.
 */
template <Encoding E>
inline constexpr std::uint8_t ascii_bits_to_code_(std::uint8_t value) noexcept
{
    if constexpr (E == Encoding::kACGT) {
        return static_cast<std::uint8_t>(((value >> 1) ^ (value >> 2)) & 0x03u);
    } else if constexpr (E == Encoding::kACTG) {
        return static_cast<std::uint8_t>((value >> 1) & 0x03u);
    } else {
        static_assert(dependent_false_v<E>, "Unhandled Encoding in ascii_bits_to_code_()");
        return 0;
    }
}

/**
 * @brief Get the two-bit encoding of a char, using bit twiddling to utilize a coincidence in the
 * ASCII code, returning kInvalidNucleotide if the char is not in `ACGT`.
 *
 * See ascii_bits_to_code_() for how the bit trick works. Note that the validity check here is
 * more expensive than the encoding itself; use CharEncoderAsciiUnchecked instead where the input is
 * known to be valid.
 */
template <Encoding E>
struct CharEncoderAscii
{
    static constexpr Encoding encoding = E;

    inline constexpr std::uint8_t operator()(char c) const noexcept
    {
        // Fold to lowercase: 'A'..'Z' -> 'a'..'z', ASCII only.
        std::uint8_t const value = static_cast<std::uint8_t>(c) | 0x20u;

        // Extract the relevant bits to get two-bit code.
        std::uint8_t const encoded = ascii_bits_to_code_<E>(value);

        // Use a bitset validator to check for correct char;
        // should be faster than actual character comparisons. Independent of the encoding,
        // as which characters are valid does not depend on which codes they map to.
        // a & 31 = 1
        // c & 31 = 3
        // g & 31 = 7
        // t & 31 = 20
        std::uint32_t constexpr valid_mask = (1u << 1) | (1u << 3) | (1u << 7) | (1u << 20);
        std::uint32_t const     low5_valid = ((valid_mask >> (value & 31u)) & 1u);
        std::uint32_t const     hig3_valid = ((value & 0xE0u) == 0x60u);
        std::uint32_t const     is_valid   = low5_valid & hig3_valid;
        return is_valid ? encoded : kInvalidNucleotide;
    }
};

/**
 * @brief Get the two-bit encoding of a char via the ASCII exploit, skipping the validity check.
 *
 * See CharEncoderAscii for details. Only use when it is clear that the input is in `ACGT`; invalid
 * characters silently produce one of the four valid codes rather than kInvalidNucleotide. This is
 * the fastest of the encoders offered here.
 */
template <Encoding E>
struct CharEncoderAsciiUnchecked
{
    static constexpr Encoding encoding = E;

    inline constexpr std::uint8_t operator()(char c) const noexcept
    {
        // No case fold needed: the bits the trick reads are unaffected by the case bit.
        return ascii_bits_to_code_<E>(static_cast<std::uint8_t>(c));
    }
};

// -----------------------------------------------------------------------------
//     table
// -----------------------------------------------------------------------------

/**
 * @brief Build the ASCII lookup table for an encoding.
 *
 * Generated rather than spelled out as 256 literals (as it is in the implementations of Heng Li),
 * so that the whole difference between the two encodings is the four lines below rather than two
 * blocks of numbers to compare by eye. Being constexpr, this is evaluated at compile time and
 * produces exactly the same constant data a hand-written table would.
 *
 * Deliberately ACGT-only, matching every other encoder here: 'U'/'u' (as in RNA input) is treated
 * as invalid rather than aliased to 'T'/'t'. The original table from Heng Li uses 0,1,2,3 as the
 * first four entries, probably to make the table idempotent, but we explicitly disallow this here
 * to avoid accidental misuse.
 */
template <Encoding E>
inline constexpr std::array<std::uint8_t, 256> make_nucleotide_table_() noexcept
{
    std::array<std::uint8_t, 256> table{};
    for( auto& entry : table ) {
        entry = kInvalidNucleotide;
    }

    auto set_char_ = [&table]( char upper, char lower, std::uint8_t code )
    {
        table[static_cast<unsigned char>(upper)] = code;
        table[static_cast<unsigned char>(lower)] = code;
    };

    set_char_('A', 'a', 0);
    set_char_('C', 'c', 1);
    if constexpr (E == Encoding::kACGT) {
        set_char_('G', 'g', 2);
        set_char_('T', 't', 3);
    } else if constexpr (E == Encoding::kACTG) {
        set_char_('T', 't', 2);
        set_char_('G', 'g', 3);
    } else {
        static_assert(dependent_false_v<E>, "Unhandled Encoding in make_nucleotide_table_()");
    }
    return table;
}

/**
 * @brief Get the two-bit encoding of a char, using an ASCII lookup table, non-throwing.
 *
 * The most predictable of the encoders here: unlike the ASCII bit trick, its cost does not depend
 * on the encoding at all, since the ordering is baked into the table's contents rather than into
 * the work done per character. It can however usually not be vectorized by the compiler, so for
 * some algorithm and compiler, it might be slower than the ASCII bit trick.
 */
template <Encoding E>
struct CharEncoderTable
{
    static constexpr Encoding encoding = E;

    static inline constexpr std::array<std::uint8_t, 256> table = make_nucleotide_table_<E>();

    inline constexpr std::uint8_t operator()(char c) const noexcept
    {
        return table[static_cast<std::uint8_t>(c)];
    }
};

// -----------------------------------------------------------------------------
//     Concept checks
// -----------------------------------------------------------------------------

// Every encoder offered here has to satisfy the concept that the k-mer extraction loops require,
// under both encodings; asserted here rather than left to the first call site that tries.
static_assert(CharEncoder<CharEncoderIfs<Encoding::kACGT>>);
static_assert(CharEncoder<CharEncoderIfs<Encoding::kACTG>>);
static_assert(CharEncoder<CharEncoderSwitch<Encoding::kACGT>>);
static_assert(CharEncoder<CharEncoderSwitch<Encoding::kACTG>>);
static_assert(CharEncoder<CharEncoderAscii<Encoding::kACGT>>);
static_assert(CharEncoder<CharEncoderAscii<Encoding::kACTG>>);
static_assert(CharEncoder<CharEncoderAsciiUnchecked<Encoding::kACGT>>);
static_assert(CharEncoder<CharEncoderAsciiUnchecked<Encoding::kACTG>>);
static_assert(CharEncoder<CharEncoderTable<Encoding::kACGT>>);
static_assert(CharEncoder<CharEncoderTable<Encoding::kACTG>>);

// A bare function carries no Encoding, and so must not satisfy it.
static_assert(!CharEncoder<std::uint8_t(*)(char)>);

} // namespace fisk
