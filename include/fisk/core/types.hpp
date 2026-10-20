#pragma once

#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "fisk/core/platform.hpp"
#include "fisk/core/simd.hpp"

namespace fisk {

// =================================================================================================
//     Conventions
// =================================================================================================

// Packed 2-bit data can differ along two independent axes: which base each 2-bit code stands for
// (Encoding), and in which order bases are laid out within a word (Layout). Both are
// tracked as template parameters on the types that carry such data, so that the compiler can
// enforce consistency between producer and consumer.

/**
 * @brief Encoding for which base each 2-bit code stands for.
 *
 * `kACGT`: A=0, C=1, G=2, T=3 (alphabetical). Complementary bases are bitwise complements of each
 * other, which makes reverse-complement handling cheap. Use this if you need to compute reverse
 * complements, or if you need lexicographic string order to match integer order.
 *
 * `kACTG`: A=0, C=1, T=2, G=3. Falls out directly of bits 1-2 of the ASCII code, making it cheaper
 * to encode, at the cost of the cheap-complement property. Use this if performance matters, but
 * lexicographically ordered encoding and reverse-complement handling are not required.
 */
enum class Encoding
{
    kACGT,
    kACTG
};

/**
 * @brief Layout in which order bases are stored within the bits of a packed word.
 *
 * `kMSB`: earlier bases of the input sequence occupy the most significant bits, matching the
 * left-rolling k-mer convention `kmer = (kmer << 2) | code`, and hence preserving lexicographic
 * string order as integer order.
 *
 * `kLSB`: earlier bases of the input sequence occupy the least significant bits, matching the
 * right-rolling k-mer convention `kmer = (kmer >> 2) | (code << (2 * (k - 1)))`. This is the order
 * that falls out directly of a little-endian bit extraction via batch encode of 8 bytes treated
 * as one 64 bit word with no extra transformation, and is therefore cheaper to produce with PEXT,
 * at the cost of not preserving lexicographic order.
 */
enum class Layout
{
    kMSB,
    kLSB
};

// Code that branches on these conventions does so with an `if constexpr` chain, and ends it in a
// static_assert rather than an unguarded `else`, so that adding a third Encoding or Layout later
// fails to compile at every site that has to be updated, instead of silently falling into whichever
// branch happened to be last. The condition has to be made dependent on the template parameter:
// a plain `static_assert(false)` in a discarded if-constexpr branch is ill-formed in C++20
// (only C++23 fixed that), and would fire even when never taken.
template <auto V>
inline constexpr bool dependent_false_v = false;

// =================================================================================================
//     Packed Sequence
// =================================================================================================

/**
 * @brief A nucleotide sequence, densely packed at 2 bits per base.
 *
 * `data` holds exactly `ceil(length / 4)` bytes of 4 bases each, with no trailing padding.
 * When `length` is not a multiple of 4, the unused bits of the last byte are zero.
 * Readers are responsible for bounds checking when reading from `data`.
 *
 * `E` states which base each 2-bit code stands for (see Encoding), and `L` how the 4 bases are
 * arranged within each byte (see Layout). Both are also exposed as the `encoding` and `layout`
 * static members.
 */
template <Encoding E, Layout L>
struct PackedSequence
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;

    std::vector<std::uint8_t> data;
    std::size_t length = 0;
};

/**
 * @brief A PackedSequence that also records which of its nucleotides are valid.
 *
 * Bit `j` of byte `b` of `valid` is set if nucleotide `8 * b + j` was one of `ACGTacgt` in the
 * input, for both layouts. The 2-bit code of an invalid nucleotide is unspecified.
 *
 * `valid` holds exactly `ceil(length / 8)` bytes, with no trailing padding, and its bits beyond
 * `length` are zero. Readers are responsible for bounds checking, as for `data`.
 *
 * Being a PackedSequence, it can be passed to everything that reads one. Packing into it through
 * a reference to the base however leaves `valid` as it was, see pack_sequence_with_validity().
 */
template <Encoding E, Layout L>
struct PackedSequenceWithValidity : PackedSequence<E, L>
{
    std::vector<std::uint8_t> valid;
};

/**
 * @brief Anything shaped like a PackedSequence: convention tags, packed codes and a length.
 *
 * The internal loops of the packed k-mer extraction are written against this, so that they take
 * both PackedSequence and PackedSequenceWithValidity.
 */
template <typename T>
concept PackedSequenceType = requires(T const& seq) {
    { T::encoding }       -> std::convertible_to<Encoding>;
    { T::layout }         -> std::convertible_to<Layout>;
    { seq.data.data() }   -> std::convertible_to<std::uint8_t const*>;
    { seq.length }        -> std::convertible_to<std::size_t>;
};

/**
 * @brief A packed sequence with validity bits, such as PackedSequenceWithValidity.
 *
 * The internal loops of the packed k-mer extraction check the validity of the nucleotides of
 * these, and extract all k-mers of the others.
 */
template <typename T>
concept PackedSequenceWithValidityType = PackedSequenceType<T> && requires(T const& seq) {
    { seq.valid.data() } -> std::convertible_to<std::uint8_t const*>;
};

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
 * The operations of core/kmer.hpp are written against this rather than against Kmer directly, so
 * that SpacedKmer (a spaced k-mer, where `width` is the mask weight rather than k) shares the
 * entire op set without any of it being duplicated. A vector of k-mers, see KmerVector, is not
 * one, also not with the single word of SimdScalar.
 */
template <typename T>
concept KmerType = requires(T kmer) {
    { T::encoding } -> std::convertible_to<Encoding>;
    { T::layout }   -> std::convertible_to<Layout>;
    { kmer.value }  -> std::convertible_to<std::uint64_t>;
} && !requires { T::lanes; };

/**
 * @brief A contiguous k-mer, i.e. a Kmer itself and not a look-alike such as SpacedKmer.
 *
 * Used to constrain the operations that depend on the bases being contiguous: reverse(), and
 * everything built on it, only mean something for a contiguous k-mer. The exact-type check is what
 * separates the two, as their shape is the same and so KmerType cannot.
 */
template <typename K>
concept ContiguousKmerType = KmerType<K> && std::same_as<K, Kmer<K::encoding, K::layout>>;

/**
 * @brief The k-mer type that a convention-tagged container yields, such as a PackedSequence.
 *
 * Anything exposing an `encoding` and a `layout` gets its matching k-mer type from this.
 */
template <typename Tagged>
using kmer_type_of = Kmer<Tagged::encoding, Tagged::layout>;

// =================================================================================================
//     Spaced K-mer
// =================================================================================================

/**
 * @brief A spaced k-mer: the bases kept as selected by a spaced seed mask, densely packed,
 * tagged with its Encoding and Layout conventions.
 *
 * Same representation and conventions as Kmer. Operations that depend only on the bases themselves
 * (complement, decoding, convention conversion, comparison, hashing) apply unchanged. The mask
 * weight, i.e. the number of kept bases, takes the place of Kmer's width. The operations that
 * would need the mask (reverse, reverse_complement, canonical) are not available for it, see
 * ContiguousKmerType.
 */
template <Encoding E, Layout L>
struct SpacedKmer
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;

    template <Encoding E2, Layout L2>
    using rebind = SpacedKmer<E2, L2>;

    std::uint64_t value;

    auto operator<=>(SpacedKmer const&) const = default;
};

// -------------------------------------------------------------------------------------------------
//     Type Traits and Helpers
// -------------------------------------------------------------------------------------------------

/**
 * @brief A spaced k-mer under the conventions of KmerAcgtMsb.
 */
using SpacedKmerAcgtMsb = SpacedKmer<Encoding::kACGT, Layout::kMSB>;

/**
 * @brief A spaced k-mer under the conventions of KmerAcgtLsb.
 */
using SpacedKmerAcgtLsb = SpacedKmer<Encoding::kACGT, Layout::kLSB>;

/**
 * @brief A spaced k-mer under the conventions of KmerActgMsb.
 */
using SpacedKmerActgMsb = SpacedKmer<Encoding::kACTG, Layout::kMSB>;

/**
 * @brief A spaced k-mer under the conventions of KmerActgLsb.
 */
using SpacedKmerActgLsb = SpacedKmer<Encoding::kACTG, Layout::kLSB>;

/**
 * @brief A spaced k-mer, i.e. a SpacedKmer itself and not a look-alike such as Kmer.
 *
 * The spaced counterpart of ContiguousKmerType, for the operations that only make sense for spaced
 * k-mers, such as encoding one from a window and its mask.
 */
template <typename K>
concept SpacedKmerType = KmerType<K> && std::same_as<K, SpacedKmer<K::encoding, K::layout>>;

// =================================================================================================
//     Wide K-mer
// =================================================================================================

/**
 * @brief A k-mer of up to `Capacity` bases, densely encoded at 2 bits per base over several words,
 * tagged with its conventions.
 *
 * The wide counterpart of Kmer, for k-mers longer than 32 bases, with the same Encoding and Layout
 * conventions and the same runtime width. The bases occupy the lowest `2 * width` bits of `words`,
 * with zero padding above them. Each word holds 32 two-bit slots: `words[0]` holds slots 0-31,
 * `words[1]` holds slots 32-63, etc. Under kLSB, base `i` occupies slot `i`; under kMSB it occupies
 * slot `width - 1 - i`.
 *
 * For width 40, from lowest to highest slot:
 *   kLSB: words[0] holds bases 0-31; words[1] holds bases 32-39, then zeros.
 *   kMSB: words[0] holds bases 39-8; words[1] holds bases 7-0, then zeros.
 *
 * `Capacity` is in bases and must be a multiple of 32, so that the type always fills its words. It
 * only bounds the width, which is passed to every operation at runtime, as for Kmer. Also as for
 * Kmer, this is a plain aggregate, and `words` is public as the sanctioned way out of the type.
 */
template <Encoding E, Layout L, std::size_t Capacity = 64>
struct WideKmer
{
    static_assert(
        Capacity > 0 && Capacity % 32 == 0, "WideKmer capacity must be a positive multiple of 32"
    );

    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t capacity = Capacity;
    static constexpr std::size_t word_count = Capacity / 32;

    /**
     * @brief The same wide k-mer type, with the same capacity, under different conventions.
     */
    template <Encoding E2, Layout L2>
    using rebind = WideKmer<E2, L2, Capacity>;

    std::array<std::uint64_t, word_count> words;

    [[nodiscard]] constexpr bool operator==(WideKmer const&) const noexcept = default;

    /**
     * @brief Compare as one unsigned integer, starting with the highest word.
     *
     * Not defaulted, as that would compare the array from `words[0]` up, i.e. lowest word first.
     * As for Kmer, the order matches lexicographic order of the decoded string for Encoding::kACGT
     * combined with Layout::kMSB only.
     */
    [[nodiscard]] constexpr std::strong_ordering operator<=>(WideKmer const& other) const noexcept
    {
        for (std::size_t i = word_count; i-- > 0;) {
            if (words[i] < other.words[i]) {
                return std::strong_ordering::less;
            }
            if (words[i] > other.words[i]) {
                return std::strong_ordering::greater;
            }
        }
        return std::strong_ordering::equal;
    }
};

// -------------------------------------------------------------------------------------------------
//     Type Traits and Helpers
// -------------------------------------------------------------------------------------------------

/**
 * @brief A wide k-mer of capacity 64 under the conventions of KmerAcgtMsb.
 */
using WideKmerAcgtMsb = WideKmer<Encoding::kACGT, Layout::kMSB>;

/**
 * @brief A wide k-mer of capacity 64 under the conventions of KmerAcgtLsb.
 */
using WideKmerAcgtLsb = WideKmer<Encoding::kACGT, Layout::kLSB>;

/**
 * @brief A wide k-mer of capacity 64 under the conventions of KmerActgMsb.
 */
using WideKmerActgMsb = WideKmer<Encoding::kACTG, Layout::kMSB>;

/**
 * @brief A wide k-mer of capacity 64 under the conventions of KmerActgLsb.
 */
using WideKmerActgLsb = WideKmer<Encoding::kACTG, Layout::kLSB>;

/**
 * @brief A wide k-mer, i.e. a WideKmer itself and not a look-alike with the same members.
 */
template <typename K>
concept WideKmerType = std::same_as<K, WideKmer<K::encoding, K::layout, K::capacity>>;

// =================================================================================================
//     Spaced K-mer Mask
// =================================================================================================

/**
 * @brief A spaced k-mer mask in the representation `M` of some bit extraction algorithm, tagged
 * with the Layout of the spaced k-mers that it extracts, and with its span.
 *
 * `M` is a bit extraction mask, table, or SIMD kernel, such as BitExtractMask,
 * BitExtractButterflyTable, or BitExtractKernelButterfly<SimdAVX2>. As this derives from `M`, it
 * can be passed to the generic bit extraction functions as is. The spaced k-mer extraction
 * functions on the other hand only accept masks tagged this way, and use the tag as the Layout of
 * the rolling window and of the emitted spaced k-mers. Hence, the bits of the mask have to be
 * arranged for that Layout.
 *
 * Create via make_spaced_kmer_mask() and the related functions, which take care of this.
 */
template <Layout L, typename M>
struct SpacedKmerMask : M
{
    static constexpr Layout layout = L;

    // Declared so that this is not an aggregate, as setting the members by hand is not safe.
    SpacedKmerMask() = default;

    // Number of positions that the mask spans, i.e., the k of the rolling window.
    std::size_t span_k = 0;
};

template <typename T>
struct is_spaced_kmer_mask_ : std::false_type {};

template <Layout L, typename M>
struct is_spaced_kmer_mask_<SpacedKmerMask<L, M>> : std::true_type {};

/**
 * @brief A spaced k-mer mask, i.e., a SpacedKmerMask, and not an untagged bit extraction mask.
 */
template <typename T>
concept SpacedKmerMaskType = is_spaced_kmer_mask_<T>::value;

// =================================================================================================
//     K-mer Vector
// =================================================================================================

/**
 * @brief A SIMD vector of k-mers, one per 64-bit lane, each encoded as in Kmer, tagged with their
 * conventions and the SIMD arch.
 *
 * The vector counterpart of Kmer, as handed out by the SIMD extraction functions, such as
 * for_each_kmer_packed_simd(). As for Kmer, this is an aggregate with the vector as its public
 * `value`, so that it is passed exactly like the bare vector. Use for_each_valid_lane() or
 * store_valid_lanes() to get the single k-mers, and `value` or kmer_value() for the vector, for
 * example for the operations of the arch.
 *
 * The template takes the arch rather than its vector type, as GCC warns about vector types that
 * are spelled out as template arguments.
 */
template <Encoding E, Layout L, SimdArch Simd>
struct KmerVector
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t lanes = Simd::lanes;
    using simd = Simd;

    typename Simd::simd_vector value;
};

/**
 * @brief A vector of k-mers, i.e. a KmerVector.
 */
template <typename T>
concept KmerVectorType = std::same_as<T, KmerVector<T::encoding, T::layout, typename T::simd>>;


/**
 * @brief The lanes of a SIMD vector of k-mers that hold valid k-mers, for the extraction functions
 * that check the validity of the nucleotides, such as for_each_kmer_packed_checked_simd().
 *
 * `value` is the lane mask in the form native to the arch, see SimdArch: a vector with all bits
 * set in the valid lanes and zero in the others, or the mask register of AVX512, which the masked
 * instructions of AVX512 take directly. bits() has bit `j` set for a valid lane `j`.
 */
template <SimdArch Simd>
struct KmerVectorValidLanes
{
    typename Simd::lane_mask value;

    std::uint64_t bits() const noexcept
    {
        return Simd::lane_mask_to_bits(value);
    }
};

} // namespace fisk
