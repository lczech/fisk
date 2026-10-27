#pragma once

#include <array>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/util/platform.hpp"
#include "fisk/simd/simd.hpp"

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
 * @brief A nucleotide sequence, densely packed at 2 bits per nucleotide.
 *
 * codes() holds exactly `ceil(length() / 4)` bytes of 4 codes each. When the length is not a
 * multiple of 4, the unused bits of the last byte are zero. For a length above 0, the codes are
 * followed by `codes_padding` zero bytes, which codes() does not include, so that loops can read
 * whole words up to the end of the codes without testing for it.
 *
 * `E` states which nucleotide each 2-bit code stands for (see Encoding), and `L` how the 4 codes
 * are arranged within each byte (see Layout). Both are also exposed as the `encoding` and `layout`
 * static members.
 *
 * The codes can be written through codes(), for example to pack a sequence by other means than
 * pack_sequence_unchecked(). Writers keep the unused bits of the last byte zero. The size of the
 * codes and the padding after them only change with resize(), which keeps both intact.
 */
template <Encoding E, Layout L>
class PackedSequence
{
public:

    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;

    /**
     * @brief The number of zero bytes after the codes, for a length above 0.
     */
    static constexpr std::size_t codes_padding = 16;

    /**
     * @brief An empty sequence, without storage.
     */
    PackedSequence() = default;

    /**
     * @brief A sequence of `length` nucleotides, all with code 0.
     */
    explicit PackedSequence(std::size_t length)
    {
        resize(length);
    }

    PackedSequence(PackedSequence const&) = default;
    PackedSequence& operator=(PackedSequence const&) = default;

    // A move takes the storage along, and leaves the moved-from sequence empty.
    PackedSequence(PackedSequence&& other) noexcept
        : storage_(std::move(other.storage_))
        , length_(std::exchange(other.length_, 0))
    {
        other.storage_.clear();
    }
    PackedSequence& operator=(PackedSequence&& other) noexcept
    {
        storage_ = std::move(other.storage_);
        length_ = std::exchange(other.length_, 0);
        other.storage_.clear();
        return *this;
    }

    ~PackedSequence() = default;

    /**
     * @brief The number of nucleotides.
     */
    std::size_t length() const noexcept
    {
        return length_;
    }

    /**
     * @brief The bytes of codes, `ceil(length() / 4)` of them, without the padding.
     */
    std::span<std::uint8_t> codes() noexcept
    {
        return {storage_.data(), code_bytes_(length_)};
    }

    /**
     * @brief The bytes of codes, `ceil(length() / 4)` of them, without the padding.
     */
    std::span<std::uint8_t const> codes() const noexcept
    {
        return {storage_.data(), code_bytes_(length_)};
    }

    /**
     * @brief Change the length to `length` nucleotides, reusing the storage where possible.
     *
     * As std::vector::resize(): the codes up to the new length keep their values, and new ones are
     * 0. The unused bits of the last byte and the padding after it are zero again afterwards, also
     * where they held codes before. A length of 0 keeps the storage, as it is never read.
     */
    void resize(std::size_t length)
    {
        length_ = length;
        if (length == 0) {
            return;
        }
        std::size_t const bytes = code_bytes_(length);
        storage_.resize(bytes + codes_padding);
        std::memset(storage_.data() + bytes, 0, codes_padding);

        // The codes beyond the length in the last byte, at its low bits for kMSB and at its high
        // bits for kLSB, see Layout.
        std::size_t const used = length % 4;
        if (used != 0) {
            if constexpr (L == Layout::kMSB) {
                storage_[bytes - 1] &= static_cast<std::uint8_t>(0xFF00u >> (2 * used));
            } else if constexpr (L == Layout::kLSB) {
                storage_[bytes - 1] &= static_cast<std::uint8_t>((1u << (2 * used)) - 1u);
            } else {
                static_assert(dependent_false_v<L>, "Unhandled Layout in PackedSequence::resize()");
            }
        }
    }

private:

    static constexpr std::size_t code_bytes_(std::size_t length) noexcept
    {
        return (length + 3) / 4;
    }

    // The codes and the padding after them.
    std::vector<std::uint8_t> storage_;
    std::size_t length_ = 0;
};

/**
 * @brief A PackedSequence that also records which of its nucleotides are valid.
 *
 * Bit `j` of byte `b` of validity() is set if nucleotide `8 * b + j` was one of `ACGTacgt` in the
 * input, for both layouts. The 2-bit code of an invalid nucleotide is unspecified. The codes follow
 * the Layout, so that a window of them is a k-mer as it is. The validity bits are only ever tested
 * against masks, such as the validity_mask of a SpacedKmerMask, and hence keep the order of the
 * sequence for both layouts.
 *
 * validity() holds exactly `ceil(length() / 8)` bytes, and its bits beyond the length are zero.
 * For a length above 0, they are followed by `validity_padding` zero bytes, as the codes are by
 * theirs, see PackedSequence. Writers of the validity bits keep the bits beyond the length zero.
 *
 * The codes and the length are those of sequence(), which can be passed on to everything that
 * reads a PackedSequence, and ignores the validity. The extraction functions that do not check
 * the validity take a PackedSequenceWithValidity as well, see PackedSequenceType.
 */
template <Encoding E, Layout L>
class PackedSequenceWithValidity
{
public:

    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;

    /**
     * @brief The number of zero bytes after the codes, for a length above 0.
     */
    static constexpr std::size_t codes_padding = PackedSequence<E, L>::codes_padding;

    /**
     * @brief The number of zero bytes after the validity bits, for a length above 0.
     */
    static constexpr std::size_t validity_padding = 8;

    /**
     * @brief An empty sequence, without storage.
     */
    PackedSequenceWithValidity() = default;

    /**
     * @brief A sequence of `length` nucleotides, all with code 0, and all invalid.
     */
    explicit PackedSequenceWithValidity(std::size_t length)
    {
        resize(length);
    }

    /**
     * @brief The number of nucleotides.
     */
    std::size_t length() const noexcept
    {
        return sequence_.length();
    }

    /**
     * @brief The bytes of codes, see PackedSequence::codes().
     */
    std::span<std::uint8_t> codes() noexcept
    {
        return sequence_.codes();
    }

    /**
     * @brief The bytes of codes, see PackedSequence::codes().
     */
    std::span<std::uint8_t const> codes() const noexcept
    {
        return sequence_.codes();
    }

    /**
     * @brief The bytes of validity bits, `ceil(length() / 8)` of them, without the padding.
     */
    std::span<std::uint8_t> validity() noexcept
    {
        return {validity_.data(), validity_bytes_(length())};
    }

    /**
     * @brief The bytes of validity bits, `ceil(length() / 8)` of them, without the padding.
     */
    std::span<std::uint8_t const> validity() const noexcept
    {
        return {validity_.data(), validity_bytes_(length())};
    }

    /**
     * @brief The sequence without its validity bits.
     */
    PackedSequence<E, L> const& sequence() const noexcept
    {
        return sequence_;
    }

    /**
     * @brief Change the length to `length` nucleotides, of the codes and of the validity bits.
     *
     * As PackedSequence::resize(), where new nucleotides are invalid, and the validity bits beyond
     * the length and the padding after them are zero again afterwards.
     */
    void resize(std::size_t length)
    {
        sequence_.resize(length);
        if (length == 0) {
            return;
        }
        std::size_t const bytes = validity_bytes_(length);
        validity_.resize(bytes + validity_padding);
        std::memset(validity_.data() + bytes, 0, validity_padding);
        if (length % 8 != 0) {
            validity_[bytes - 1] &= static_cast<std::uint8_t>((1u << (length % 8)) - 1u);
        }
    }

private:

    static constexpr std::size_t validity_bytes_(std::size_t length) noexcept
    {
        return (length + 7) / 8;
    }

    // The moves of the members leave a moved-from sequence empty, see PackedSequence.
    PackedSequence<E, L> sequence_;
    std::vector<std::uint8_t> validity_;
};

/**
 * @brief Anything shaped like a PackedSequence: convention tags, packed codes and a length.
 *
 * The packed k-mer extraction is written against this, so that it takes both PackedSequence and
 * PackedSequenceWithValidity.
 */
template <typename T>
concept PackedSequenceType = requires(T const& seq) {
    { T::encoding }    -> std::convertible_to<Encoding>;
    { T::layout }      -> std::convertible_to<Layout>;
    { seq.codes() }    -> std::convertible_to<std::span<std::uint8_t const>>;
    { seq.length() }   -> std::convertible_to<std::size_t>;
};

/**
 * @brief A packed sequence with validity bits, such as PackedSequenceWithValidity, and the
 * sequence without them.
 *
 * The internal loops of the packed k-mer extraction check the validity of the nucleotides of
 * these, and extract all k-mers of the others.
 */
template <typename T>
concept PackedSequenceWithValidityType = PackedSequenceType<T> && requires(T const& seq) {
    { seq.validity() } -> std::convertible_to<std::span<std::uint8_t const>>;
    requires PackedSequenceType<std::remove_cvref_t<decltype(seq.sequence())>>;
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
 * BitExtractButterflyMask, or BitExtractKernelButterfly<SimdAVX2>. As this derives from `M`, it
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

    // The representation of the bit extraction that this mask is, such as BitExtractMask.
    using bit_extract_type = M;

    // Declared so that this is not an aggregate, as setting the members by hand is not safe.
    SpacedKmerMask() = default;

    // Number of positions that the mask spans, i.e., the k of the rolling window.
    std::size_t span_k = 0;

    // The positions of the span that the mask keeps, in the order of the sequence for both
    // layouts: bit `i` is set if position `i` of the span is kept. This is the order of the
    // validity bits of a PackedSequenceWithValidity, so that a spaced k-mer at `pos` is valid if
    // the validity bits from `pos` on have all bits of this mask set.
    std::uint64_t validity_mask = 0;
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
//     K-mer Vectors
// =================================================================================================

/**
 * @brief A SIMD vector of k-mers, one per 64-bit lane, each encoded as in Kmer, tagged with their
 * conventions and the SIMD arch.
 *
 * The vector counterpart of Kmer, as handed out by the SIMD extraction functions, such as
 * for_each_kmer_packed_unchecked_bytewise_simd(). As for Kmer, this is an aggregate with the vector
 * as its public `value`, so that it is passed exactly like the bare vector. Use
 * for_each_valid_lane() or store_valid_lanes() to get the single k-mers, and `value` or
 * kmer_value() for the vector, for example for the operations of the arch.
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

    // The type of the single k-mers in the lanes.
    using kmer_type = Kmer<E, L>;

    typename Simd::simd_vector value;
};

/**
 * @brief A vector of contiguous k-mers, i.e. a KmerVector and not a SpacedKmerVector.
 */
template <typename T>
concept ContiguousKmerVectorType = std::same_as<
    T, KmerVector<T::encoding, T::layout, typename T::simd>
>;

/**
 * @brief A SIMD vector of spaced k-mers of one mask, one per 64-bit lane, each encoded as in
 * SpacedKmer, tagged with their conventions and the SIMD arch.
 *
 * The vector counterpart of SpacedKmer, as handed out by the SIMD spaced k-mer extraction
 * functions, such as for_each_spaced_kmer_packed_unchecked_bytewise_simd(), and otherwise the same
 * as KmerVector.
 */
template <Encoding E, Layout L, SimdArch Simd>
struct SpacedKmerVector
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t lanes = Simd::lanes;
    using simd = Simd;

    // The type of the single spaced k-mers in the lanes.
    using kmer_type = SpacedKmer<E, L>;

    typename Simd::simd_vector value;
};

/**
 * @brief A vector of spaced k-mers, i.e. a SpacedKmerVector.
 */
template <typename T>
concept SpacedKmerVectorType = std::same_as<
    T, SpacedKmerVector<T::encoding, T::layout, typename T::simd>
>;

/**
 * @brief A vector of k-mers of either kind, i.e. a KmerVector or a SpacedKmerVector, as KmerType
 * is for single k-mers.
 */
template <typename T>
concept KmerVectorType = ContiguousKmerVectorType<T> || SpacedKmerVectorType<T>;

/**
 * @brief The lanes of a SIMD vector of k-mers or spaced k-mers that hold valid ones, for the
 * extraction functions that check the validity of the nucleotides, such as
 * for_each_kmer_packed_checked_bytewise_simd().
 *
 * `bits` has bit `j` set for a valid lane `j`, and no bits above the lanes. lane_mask() gives the
 * lane mask in the form native to the arch, see SimdArch: a vector with all bits set in the valid
 * lanes and zero in the others, for blends and other vector operations, or the mask register of
 * AVX512, which holds the bits as they are.
 */
template <SimdArch Simd>
struct SimdVectorValidLanes
{
    std::uint64_t bits;

    typename Simd::lane_mask lane_mask() const noexcept
    {
        return Simd::lane_mask_from_bits(bits);
    }
};

} // namespace fisk
