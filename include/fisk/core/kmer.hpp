#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/types.hpp"

namespace fisk {

// =================================================================================================
//     K-mer
// =================================================================================================

/**
 * @brief A single k-mer, densely encoded at 2 bits per base, tagged with its conventions.
 *
 * The 2-bit codes occupy the lowest `2 * width` bits of `value`; all higher bits are zero.
 * `E` states which base each code stands for (see Encoding), `L` in which order the bases are
 * laid out within the word (see Layout). Both are also exposed as the `encoding` and `layout`
 * static members, and as the `rebind` alias for expressing a change of convention.
 *
 * Deliberately an aggregate with a public member and no user-declared constructors: that keeps it
 * trivially copyable and standard layout, which is what makes the ABI pass it in a register
 * exactly like a bare `std::uint64_t`. The tags therefore cost nothing at runtime; they exist
 * purely so that the compiler can reject mixing k-mers of different conventions.
 *
 * `value` is public on purpose, as the sanctioned way out of the type for binary IO or for
 * interoperating with code that needs raw words; kmer_value() is the same thing under a name that
 * can be grepped for. To convert from raw words into typed k-mers, use kmer_cast().
 *
 * Note that the number of bases (`k`), here more generally called the `width`, is not part of the
 * type. It is a runtime parameter of the extraction loops, and templating on it would force every
 * caller that takes k at runtime to template its own code and dispatch 32 ways. It is therefore
 * passed alongside the k-mer to every operation that needs it.
 */
template <Encoding E, Layout L>
struct Kmer
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;

    /**
     * @brief The same k-mer type under different conventions.
     *
     * Used by kmer_convert() to name its result without hard-coding this template, so that other
     * k-mer-like types (see SpacedKmer) get the same conversions for free. Named as in the
     * standard library's allocator rebinding, which likewise re-instantiates a template with
     * different arguments rather than naming the type itself.
     */
    template <Encoding E2, Layout L2>
    using rebind = Kmer<E2, L2>;

    std::uint64_t value;

    /**
     * @brief Compare two k-mers of the same conventions, by their raw word.
     *
     * Defaulting the three-way comparison also implicitly defaults `operator==`, so equality and
     * all four relational operators come from this one declaration.
     *
     * The resulting order matches lexicographic order of the decoded string for Encoding::kACGT
     * combined with Layout::kMSB only. Under the other conventions, the order is still total and
     * deterministic, and so perfectly usable as a container key, just not alphabetical.
     */
    auto operator<=>(Kmer const&) const = default;
};

// -------------------------------------------------------------------------------------------------
//     Type Traits and Helpers
// -------------------------------------------------------------------------------------------------

/**
 * @brief A k-mer with A=0, C=1, G=2, T=3, earliest base in the most significant bits.
 *
 * The only combination whose integer order matches lexicographic order of the decoded string.
 */
using KmerAcgtMsb = Kmer<Encoding::kACGT, Layout::kMSB>;

/**
 * @brief A k-mer with A=0, C=1, G=2, T=3, earliest base in the least significant bits.
 *
 * Keeps the cheap complement of the ACGT encoding, and allows to use the more performant extraction
 * from (little-endian) bit packed sequences in PackedSequence.
 */
using KmerAcgtLsb = Kmer<Encoding::kACGT, Layout::kLSB>;

/**
 * @brief A k-mer with A=0, C=1, T=2, G=3, earliest base in the most significant bits.
 *
 * The ACTG encoding is cheaper to produce from ASCII, at the cost of no longer ordering
 * lexicographically.
 */
using KmerActgMsb = Kmer<Encoding::kACTG, Layout::kMSB>;

/**
 * @brief A k-mer with A=0, C=1, T=2, G=3, earliest base in the least significant bits.
 *
 * The cheapest combination to produce: both the encoding and the layout fall directly out of a
 * little-endian batch encode from ASCII with no further transformation.
 */
using KmerActgLsb = Kmer<Encoding::kACTG, Layout::kLSB>;

/**
 * @brief Anything shaped like a k-mer: convention tags plus a 2-bit-packed word.
 *
 * All operations below are written against this rather than against Kmer directly, so that
 * SpacedKmer (a spaced k-mer, where `width` is the mask weight rather than k) shares the entire
 * op set without any of it being duplicated.
 */
template <typename T>
concept KmerType = requires(T kmer) {
    { T::encoding } -> std::convertible_to<Encoding>;
    { T::layout }   -> std::convertible_to<Layout>;
    { kmer.value }  -> std::convertible_to<std::uint64_t>;
};

/**
 * @brief The k-mer type that a convention-tagged container yields, such as a PackedSequence.
 *
 * Anything exposing an `encoding` and a `layout` gets its matching k-mer type from this.
 */
template <typename Tagged>
using kmer_type_of = Kmer<Tagged::encoding, Tagged::layout>;

// =================================================================================================
//     Convention Dispatch
// =================================================================================================

// The operations further down branch on Encoding and Layout in several places. Rather than
// repeating an `if constexpr` chain in each of them, the branching is centralized into the few
// helpers here, so that adding a third Encoding or Layout later fails to compile in exactly these
// spots instead of silently falling into whichever branch happened to be the `else`.
//
// Each chain therefore ends in a static_assert rather than an unguarded `else`. It has to be made
// dependent on the template parameter: a plain `static_assert(false)` in a discarded if-constexpr
// branch is ill-formed in C++20 (only C++23 fixed that), and would fire even when never taken.

template <auto V>
inline constexpr bool dependent_false_v = false;

/**
 * @brief Mask of the lowest `2 * width` bits, i.e. the bits a k-mer of that width may occupy.
 */
inline constexpr std::uint64_t width_mask_(std::size_t width) noexcept
{
    return (width >= 32) ? ~std::uint64_t{0} : ((std::uint64_t{1} << (2 * width)) - 1u);
}

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
//     Construction and Access
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Raw Cast
// -------------------------------------------------------------------------------------------------

/**
 * @brief Adopt a raw 2-bit-packed word as a k-mer of a given convention.
 *
 * This is the one operation that can silently mislabel a k-mer, since nothing about a bare word
 * says which convention produced it. It is therefore a named function, so that every point where a
 * value enters the type system can be easily found. Its counterpart on the way out is kmer_value().
 *
 * The caller asserts that `raw` really is in convention `E`/`L` and holds `width` bases. Both are
 * checked only by assert(), so that release builds pay nothing; `width` is unused there.
 */
template <Encoding E, Layout L>
[[nodiscard]] inline constexpr Kmer<E, L> kmer_cast(
    std::uint64_t raw, [[maybe_unused]] std::size_t width
) noexcept {
    assert(width >= 1 && width <= 32);
    assert(width == 32 || (raw >> (2 * width)) == 0);
    return Kmer<E, L>{raw};
}

/**
 * @brief Adopt the first `count` lanes of a SIMD vector of raw words as k-mers, and return how
 * many were written.
 *
 * The vector-emitting extractors (kmer_extract/packed_simd.hpp) hand out several k-mers at once in
 * an ISA register, which no scalar wrapper can describe. This is the way back: it gives the same
 * conventions the producing PackedSequence carries, without the consumer re-deriving lane
 * extraction per ISA. `out` must have room for `count` k-mers.
 *
 * Deliberately ISA-agnostic, moving the register through memcpy rather than a per-ISA store
 * intrinsic. A vector type is an ordinary trivially copyable object, and memcpy imposes no
 * alignment requirement, so compilers lower this to the same unaligned store they would emit for
 * the intrinsic, and usually elide it altogether once inlined into the consumer.
 */
template <Encoding E, Layout L, typename Vec>
inline std::size_t kmer_cast(
    Vec vec, std::size_t count, std::size_t width, Kmer<E, L>* out
) noexcept {
    static_assert(
        sizeof(Vec) % sizeof(std::uint64_t) == 0,
        "kmer_cast() expects a vector of whole 64-bit lanes"
    );
    constexpr std::size_t lanes = sizeof(Vec) / sizeof(std::uint64_t);
    assert(count <= lanes);
    assert(out != nullptr);

    std::uint64_t buffer[lanes];
    std::memcpy(buffer, &vec, sizeof(Vec));
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = kmer_cast<E, L>(buffer[i], width);
    }
    return count;
}

/**
 * @brief Adopt every lane of a SIMD vector of raw words as k-mers.
 *
 * Convenience for the common full-vector case; see the overload above for the partial case.
 */
template <Encoding E, Layout L, typename Vec>
inline std::size_t kmer_cast(Vec vec, std::size_t width, Kmer<E, L>* out) noexcept
{
    return kmer_cast<E, L>(vec, sizeof(Vec) / sizeof(std::uint64_t), width, out);
}

/**
 * @brief Get the raw 2-bit-packed word of a k-mer.
 *
 * Identical to reading the public `value` member, and provided as the named counterpart to
 * kmer_cast(): every crossing of the type boundary, in either direction, is then one findable
 * function call.
 */
template <KmerType K>
[[nodiscard]] inline constexpr std::uint64_t kmer_value(K kmer) noexcept
{
    return kmer.value;
}

// -------------------------------------------------------------------------------------------------
//     String Conversion
// -------------------------------------------------------------------------------------------------

/**
 * @brief Get the 2-bit code of the base at position `index`, counted from the start of the k-mer.
 *
 * Layout-aware, so that callers never have to work out the shift themselves.
 */
template <KmerType K>
[[nodiscard]] inline constexpr std::uint8_t base_at(
    K kmer, std::size_t index, std::size_t width
) noexcept {
    assert(width >= 1 && width <= 32);
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
[[nodiscard]] inline constexpr std::string kmer_decode(K kmer, std::size_t width)
{
    assert(width >= 1 && width <= 32);

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
 * The conventions default to ACGT with an MSB layout, either axis can be named on its own, and
 * both axes can be given in either order: `kmer_encode(seq)`, `kmer_encode<Encoding::kACTG>(seq)`,
 * `kmer_encode<Layout::kLSB>(seq)`, `kmer_encode<Encoding::kACTG, Layout::kLSB>(seq)` and
 * `kmer_encode<Layout::kLSB, Encoding::kACTG>(seq)` all work.
 *
 * Unlike the operations around it this throws rather than asserting, on an empty sequence, one
 * longer than 32 bases, or any character that is not a nucleotide. Loud failure is the right
 * trade for a function used to construct known inputs, where a silently wrong k-mer would
 * undermine whatever is being checked with it. Upper and lower case are both accepted.
 */
template <Encoding E = Encoding::kACGT, Layout L = Layout::kMSB>
[[nodiscard]] inline constexpr Kmer<E, L> kmer_encode(std::string_view seq)
{
    if (seq.empty() || seq.size() > 32) {
        throw std::invalid_argument(
            "Invalid call to kmer_encode() with a sequence length not in [1, 32]"
        );
    }

    std::uint64_t value = 0;
    for (std::size_t i = 0; i < seq.size(); ++i) {
        std::uint8_t const code = char_to_code_<E>(seq[i]);
        if (code > 3) {
            throw std::invalid_argument(
                "Invalid call to kmer_encode() with a non-nucleotide character in the sequence"
            );
        }
        value |= std::uint64_t{code} << base_shift_<L>(i, seq.size());
    }
    return Kmer<E, L>{value};
}

/**
 * @brief Build a k-mer from a nucleotide string, with the layout named first.
 *
 * Exists so that both argument orders work, and so that the layout can be given on its own: a
 * Layout value passed as the *first* template argument would otherwise be read as an Encoding by
 * the overload above, and so needs a template parameter list that starts with Layout instead.
 * Since Encoding and Layout are unrelated types, a template argument list can only ever match the
 * overload whose parameters are declared in the order it supplies them, so this coexists with the
 * Encoding-first overload above without ambiguity. `E` is given a default here too, which is what
 * lets this single overload also cover the Layout-only call.
 */
template <Layout L, Encoding E = Encoding::kACGT>
[[nodiscard]] inline constexpr Kmer<E, L> kmer_encode(std::string_view seq)
{
    return kmer_encode<E, L>(seq);
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
    assert(width >= 1 && width <= 32);
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
template <KmerType K>
[[nodiscard]] inline constexpr K reverse(K kmer, std::size_t width) noexcept
{
    assert(width >= 1 && width <= 32);
    return K{reverse_bit_pairs_(kmer.value) >> (64 - 2 * width)};
}

/**
 * @brief Get the reverse complement of a k-mer: bases in reverse order, each complemented.
 *
 * Its own inverse. Fused rather than composed out of reverse() and complement(): the complement
 * XOR is applied to the full reversed word, and the shift that brings the k-mer's own bases down
 * then discards the complemented garbage above them, so no separate masking step is needed.
 */
template <KmerType K>
[[nodiscard]] inline constexpr K reverse_complement(K kmer, std::size_t width) noexcept
{
    assert(width >= 1 && width <= 32);
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
template <KmerType K>
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
 */
template <Encoding E2, KmerType K>
[[nodiscard]] inline constexpr typename K::template rebind<E2, K::layout> kmer_convert(
    K kmer, [[maybe_unused]] std::size_t width
) noexcept {
    assert(width >= 1 && width <= 32);
    return typename K::template rebind<E2, K::layout>{recode_<K::encoding, E2>(kmer.value)};
}

/**
 * @brief Convert a k-mer to a different layout, keeping its encoding.
 *
 * Self-inverse, since the only re-ordering between the two layouts is its own inverse.
 */
template <Layout L2, KmerType K>
[[nodiscard]] inline constexpr typename K::template rebind<K::encoding, L2> kmer_convert(
    K kmer, std::size_t width
) noexcept {
    assert(width >= 1 && width <= 32);
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
    assert(width >= 1 && width <= 32);
    auto const recoded = recode_<K::encoding, E2>(kmer.value);
    return typename K::template rebind<E2, L2>{relayout_<K::layout, L2>(recoded, width)};
}

// =================================================================================================
//     Hashing
// =================================================================================================

/**
 * @brief Hash a k-mer by its raw word, without mixing.
 *
 * Free, but note that k-mers adjacent in a sequence differ only by a shift, so their raw words
 * cluster heavily. Use this only when the k-mers are already well distributed, or when the
 * container does its own mixing. Otherwise prefer KmerHashMix, which is the default.
 */
struct KmerHashIdentity
{
    template <KmerType K>
    [[nodiscard]] std::size_t operator()(K kmer) const noexcept
    {
        return static_cast<std::size_t>(kmer.value);
    }
};

/**
 * @brief Hash a k-mer through the splitmix64 finalizer.
 *
 * Same mixing steps as Splitmix64::get_uint64() (see core/random.hpp), applied to the k-mer word
 * instead of to a counter. A handful of cycles, against the cache miss that the hash table lookup
 * around it costs anyway, in exchange for full avalanche.
 *
 * This is what std::hash<Kmer> uses. To opt out, name KmerHashIdentity as the container's hasher.
 */
struct KmerHashMix
{
    template <KmerType K>
    [[nodiscard]] std::size_t operator()(K kmer) const noexcept
    {
        std::uint64_t z = kmer.value;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return static_cast<std::size_t>(z ^ (z >> 31));
    }
};

} // namespace fisk

namespace std {

/**
 * @brief Make Kmer usable as a key in the standard unordered containers.
 *
 * Defaults to the mixing hash rather than the identity, so that the path a user gets without
 * thinking about it is the one that behaves. The cost is bounded and small; the cost of identity
 * hashing with clustered keys is neither.
 */
template <fisk::Encoding E, fisk::Layout L>
struct hash<fisk::Kmer<E, L>> : fisk::KmerHashMix
{};

} // namespace std
