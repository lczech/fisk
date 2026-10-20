#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "fisk/core/cast.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/types.hpp"

namespace fisk {

// =================================================================================================
//     Convention Dispatch
// =================================================================================================

// The operations further down branch on Encoding and Layout in several places. Rather than
// repeating an `if constexpr` chain in each of them, the branching is centralized into the few
// helpers here, so that adding a third Encoding or Layout later fails to compile in exactly these
// spots instead of silently falling into whichever branch happened to be the `else`.
//
// Each chain therefore ends in a static_assert on dependent_false_v (core/types.hpp) rather than
// an unguarded `else`.

/**
 * @brief XOR constant that complements every 2-bit code in a word at once.
 *
 * Complementing swaps A <-> T and C <-> G. Which bit pattern that is depends on the encoding,
 * but it is a single XOR either way: under kACGT the codes are A=0,C=1,G=2,T=3, so the pairs are
 * 0<->3 and 1<->2, which is XOR with 0b11 (equivalent to complementing all bits); under kACTG they
 * are A=0,C=1,T=2,G=3, so the pairs are 0<->2 and 1<->3, which is XOR with 0b10.
 */
template <Encoding E>
inline constexpr std::uint64_t complement_xor_() noexcept
{
    if constexpr (E == Encoding::kACGT) {
        return ~std::uint64_t{0};
    } else if constexpr (E == Encoding::kACTG) {
        return 0xAAAAAAAAAAAAAAAAull;
    } else {
        static_assert(dependent_false_v<E>, "Unhandled Encoding in complement_xor_()");
        return 0;
    }
}

/**
 * @brief Right-shift that brings the 2-bit code of base `index` down into the lowest bits.
 */
template <Layout L>
inline constexpr unsigned base_shift_(std::size_t index, std::size_t width) noexcept
{
    if constexpr (L == Layout::kMSB) {
        return static_cast<unsigned>(2 * (width - 1 - index));
    } else if constexpr (L == Layout::kLSB) {
        return static_cast<unsigned>(2 * index);
    } else {
        static_assert(dependent_false_v<L>, "Unhandled Layout in base_shift_()");
        return 0;
    }
}

/**
 * @brief The nucleotide character that a 2-bit code stands for under a given encoding.
 */
template <Encoding E>
inline constexpr char code_to_char_(std::uint8_t code) noexcept
{
    if constexpr (E == Encoding::kACGT) {
        constexpr char lut[4] = {'A', 'C', 'G', 'T'};
        return lut[code];
    } else if constexpr (E == Encoding::kACTG) {
        constexpr char lut[4] = {'A', 'C', 'T', 'G'};
        return lut[code];
    } else {
        static_assert(dependent_false_v<E>, "Unhandled Encoding in code_to_char_()");
        return '\0';
    }
}

/**
 * @brief The 2-bit code that a nucleotide character stands for under a given encoding, or 4 for
 * anything that is not a nucleotide.
 */
template <Encoding E>
inline constexpr std::uint8_t char_to_code_(char c) noexcept
{
    if constexpr (E == Encoding::kACGT) {
        switch (c) {
            case 'A': case 'a': return 0;
            case 'C': case 'c': return 1;
            case 'G': case 'g': return 2;
            case 'T': case 't': return 3;
            default:            return 4;
        }
    } else if constexpr (E == Encoding::kACTG) {
        switch (c) {
            case 'A': case 'a': return 0;
            case 'C': case 'c': return 1;
            case 'T': case 't': return 2;
            case 'G': case 'g': return 3;
            default:            return 4;
        }
    } else {
        static_assert(dependent_false_v<E>, "Unhandled Encoding in char_to_code_()");
        return 4;
    }
}

/**
 * @brief Reverse the order of all 32 2-bit groups in a word, leaving each group's bits intact.
 *
 * Adapted from Kraken2's reverse-complement, which took it from the parallel bit reversal at
 * https://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel. The final three steps of
 * that sequence (swapping bytes, byte pairs, then word halves) are exactly a byte swap, so they
 * collapse into the single byte_swap_64() instruction instead of three mask-shift-or rounds.
 *
 * Note that this reverses the full 64-bit word, leaving the result in the high bits. Callers
 * shift it down by `64 - 2 * width` to recover the groups they care about.
 */
inline constexpr std::uint64_t reverse_bit_pairs_(std::uint64_t value) noexcept
{
    // Reverse bits (leaving bit pairs intact, as those represent nucleotides):
    // Swap consecutive pairs, then nibbles, then the whole byte order.
    value = ((value & 0xCCCCCCCCCCCCCCCCull) >> 2) | ((value & 0x3333333333333333ull) << 2);
    value = ((value & 0xF0F0F0F0F0F0F0F0ull) >> 4) | ((value & 0x0F0F0F0F0F0F0F0Full) << 4);
    return byte_swap_64(value);
}

// =================================================================================================
//     String Conversion
// =================================================================================================

/**
 * @brief Get the 2-bit code of the base at position `index`, counted from the start of the k-mer.
 *
 * Layout-aware, so that callers never have to work out the shift themselves.
 */
template <KmerType K>
[[nodiscard]] inline constexpr std::uint8_t base_at(
    K kmer, std::size_t index, std::size_t width
) noexcept {
    assert_width_(width);
    assert(index < width);
    return static_cast<std::uint8_t>((kmer.value >> base_shift_<K::layout>(index, width)) & 0x3u);
}

/**
 * @brief Get the string representation of a k-mer, as a sequence of upper case nucleotides.
 *
 * `width` is the number of bases to decode: k for a Kmer, the mask weight for a SpacedKmer.
 * The inverse of kmer_encode().
 */
template <KmerType K>
[[nodiscard]] inline std::string kmer_decode(K kmer, std::size_t width)
{
    // Not constexpr: some libc++ versions we still support (e.g. LLVM 14's bundled libc++ on macOS)
    // predate full support for constexpr std::string (P0980), which makes std::string not a literal
    // type there, and a constexpr function's return type must be a literal type. No caller needs
    // compile-time evaluation here anyway.
    assert_width_(width);

    std::string str;
    str.resize(width);
    for (std::size_t i = 0; i < width; ++i) {
        str[i] = code_to_char_<K::encoding>(base_at(kmer, i, width));
    }
    return str;
}

/**
 * @brief Build a k-mer from a nucleotide string, the inverse of kmer_decode().
 *
 * A convenience entry point for fixed, known sequences: writing out test cases, trying things out,
 * or turning a literal into a key. Not a bulk path; extracting the k-mers of a sequence is what
 * the for_each_kmer*() functions are for.
 *
 * The result type is named as the template argument and defaults to KmerAcgtMsb, as in
 * `kmer_encode(seq)` or `kmer_encode<KmerActgLsb>(seq)`. Spaced k-mers have a counterpart that
 * also takes the mask, see kmer_spaced/kmer_spaced.hpp.
 *
 * Unlike the operations around it this throws rather than asserting, on an empty sequence, one
 * longer than 32 bases, or any character that is not a nucleotide. Loud failure is the right
 * trade for a function used to construct known inputs, where a silently wrong k-mer would
 * undermine whatever is being checked with it. Upper and lower case are both accepted.
 */
template <ContiguousKmerType K = KmerAcgtMsb>
[[nodiscard]] inline constexpr K kmer_encode(std::string_view seq)
{
    if (seq.empty() || seq.size() > 32) {
        throw std::invalid_argument(
            "Invalid call to kmer_encode() with a sequence length not in [1, 32]"
        );
    }

    std::uint64_t value = 0;
    for (std::size_t i = 0; i < seq.size(); ++i) {
        std::uint8_t const code = char_to_code_<K::encoding>(seq[i]);
        if (code > 3) {
            throw std::invalid_argument(
                "Invalid call to kmer_encode() with a non-nucleotide character in the sequence"
            );
        }
        value |= std::uint64_t{code} << base_shift_<K::layout>(i, seq.size());
    }
    return K{value};
}

// =================================================================================================
//     Nucleotide Operations
// =================================================================================================

/**
 * @brief Complement each base of a k-mer in place, without reversing their order.
 *
 * Its own inverse: complementing twice returns the original k-mer.
 */
template <KmerType K>
[[nodiscard]] inline constexpr K complement(K kmer, std::size_t width) noexcept
{
    assert_width_(width);
    return K{(kmer.value ^ complement_xor_<K::encoding>()) & width_mask_(width)};
}

/**
 * @brief Reverse the order of the bases of a k-mer, without complementing them.
 *
 * Its own inverse. Worth noting for orientation: within a fixed layout, reversing the bases
 * produces the same bit pattern as leaving them alone and reinterpreting the word under the
 * opposite layout. The two mean entirely different things, which is precisely why both the
 * convention and the operation are spelled out in the type system rather than left to the reader
 * of a bare word.
 */
template <ContiguousKmerType K>
[[nodiscard]] inline constexpr K reverse(K kmer, std::size_t width) noexcept
{
    assert_width_(width);
    return K{reverse_bit_pairs_(kmer.value) >> (64 - 2 * width)};
}

/**
 * @brief Get the reverse complement of a k-mer: bases in reverse order, each complemented.
 *
 * Its own inverse. Fused rather than composed out of reverse() and complement(): the complement
 * XOR is applied to the full reversed word, and the shift that brings the k-mer's own bases down
 * then discards the complemented garbage above them, so no separate masking step is needed.
 */
template <ContiguousKmerType K>
[[nodiscard]] inline constexpr K reverse_complement(K kmer, std::size_t width) noexcept
{
    assert_width_(width);
    auto const reversed = reverse_bit_pairs_(kmer.value) ^ complement_xor_<K::encoding>();
    return K{reversed >> (64 - 2 * width)};
}

/**
 * @brief Get the canonical form of a k-mer: whichever of it and its reverse complement is smaller.
 *
 * Used to identify a k-mer with its reverse complement, so that both strands of a sequence map to
 * the same key. Idempotent, and agrees on a k-mer and its reverse complement, which is the
 * property that makes it usable as such a key. Note that "smaller" is by raw word, so which of the
 * two is picked depends on the conventions; it is consistent within one convention, which is all
 * that canonicalization needs.
 */
template <ContiguousKmerType K>
[[nodiscard]] inline constexpr K canonical(K kmer, std::size_t width) noexcept
{
    K const rev_comp = reverse_complement(kmer, width);
    return (kmer.value <= rev_comp.value) ? kmer : rev_comp;
}

// =================================================================================================
//     Convention Conversion
// =================================================================================================

/**
 * @brief Re-map the 2-bit codes of a word from one encoding to another.
 */
template <Encoding From, Encoding To>
inline constexpr std::uint64_t recode_(std::uint64_t value) noexcept
{
    if constexpr (From == To) {
        return value;
    } else if constexpr (
        (From == Encoding::kACGT && To == Encoding::kACTG) ||
        (From == Encoding::kACTG && To == Encoding::kACGT)
    ) {
        // The two encodings agree on A=0 and C=1 and differ only in exchanging codes 2 and 3,
        // that is, flipping the low bit of every group whose high bit is set. Groups above the
        // k-mer's width are zero and stay zero, so this needs no width and no masking. The
        // operation is its own inverse, hence one expression covering both directions.
        return value ^ ((value >> 1) & 0x5555555555555555ull);
    } else {
        static_assert(dependent_false_v<From>, "Unhandled Encoding pair in recode_()");
        return 0;
    }
}

/**
 * @brief Re-order the 2-bit codes of a word from one layout to another.
 */
template <Layout From, Layout To>
inline constexpr std::uint64_t relayout_(std::uint64_t value, std::size_t width) noexcept
{
    if constexpr (From == To) {
        return value;
    } else if constexpr (
        (From == Layout::kMSB && To == Layout::kLSB) ||
        (From == Layout::kLSB && To == Layout::kMSB)
    ) {
        // The two layouts differ in which end of the word the first base sits at, so converting
        // is simply reversing the order of the bases. Self-inverse, as for the encodings above.
        return reverse_bit_pairs_(value) >> (64 - 2 * width);
    } else {
        static_assert(dependent_false_v<From>, "Unhandled Layout pair in relayout_()");
        return 0;
    }
}

// Converting between conventions is spelled kmer_convert() in all three cases, distinguished only
// by which template arguments are supplied. Giving an Encoding selects the first overload, a
// Layout the second, and both the third: the other candidates drop out because a value of one enum
// cannot bind to a template parameter of the other, and because the unsupplied parameter of the
// two-axis form appears only in the return type and so cannot be deduced.
//
// Unlike kmer_cast(), these genuinely rewrite the bits, so the result really does spell the same
// nucleotides under the requested conventions.

/**
 * @brief Convert a k-mer to a different encoding, keeping its layout.
 *
 * Self-inverse, since the only re-coding between the two encodings is its own inverse.
 *
 * The unused, always-defaulted third parameter works around an MSVC linker bug (LNK1179
 * "duplicate COMDAT") where this overload's mangled name can collide with the Layout overload
 * below when the two enum values happen to share their underlying integer (e.g. both 0):
 * giving each overload a genuinely different type here keeps their mangled names apart.
 */
template <Encoding E2, KmerType K, typename = Encoding>
[[nodiscard]] inline constexpr typename K::template rebind<E2, K::layout> kmer_convert(
    K kmer, [[maybe_unused]] std::size_t width
) noexcept {
    assert_width_(width);
    return typename K::template rebind<E2, K::layout>{recode_<K::encoding, E2>(kmer.value)};
}

/**
 * @brief Convert a k-mer to a different layout, keeping its encoding.
 *
 * Self-inverse, since the only re-ordering between the two layouts is its own inverse.
 *
 * See the matching third-parameter comment on the Encoding overload above for why it's here.
 */
template <Layout L2, KmerType K, typename = Layout>
[[nodiscard]] inline constexpr typename K::template rebind<K::encoding, L2> kmer_convert(
    K kmer, std::size_t width
) noexcept {
    assert_width_(width);
    return typename K::template rebind<K::encoding, L2>{
        relayout_<K::layout, L2>(kmer.value, width)
    };
}

/**
 * @brief Convert a k-mer to a different encoding and layout at once.
 *
 * Exactly equivalent to composing the two single-axis conversions, and offered only so that a
 * caller changing both axes can say so in one call and name one result type. Re-coding is
 * per-group and re-ordering is positional, so the two commute and either order gives the same
 * result.
 */
template <Encoding E2, Layout L2, KmerType K>
[[nodiscard]] inline constexpr typename K::template rebind<E2, L2> kmer_convert(
    K kmer, std::size_t width
) noexcept {
    assert_width_(width);
    auto const recoded = recode_<K::encoding, E2>(kmer.value);
    return typename K::template rebind<E2, L2>{relayout_<K::layout, L2>(recoded, width)};
}

} // namespace fisk
