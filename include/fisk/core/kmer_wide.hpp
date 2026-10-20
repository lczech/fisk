#pragma once

#include <array>
#include <cassert>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/types.hpp"

namespace fisk {

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

/**
 * @brief Debug check that a width is in [1, Capacity].
 */
template <std::size_t Capacity>
FISK_ALWAYS_INLINE inline constexpr void assert_wide_width_(
    [[maybe_unused]] std::size_t width
) noexcept {
    assert(width >= 1 && width <= Capacity);
}

/**
 * @brief Whether words hold no bits above `width` bases, i.e. they are a valid wide k-mer of that
 * width.
 */
template <std::size_t N>
inline constexpr bool wide_fits_width_(
    std::array<std::uint64_t, N> const& words, std::size_t width
) noexcept {
    for (std::size_t i = 0; i < N; ++i) {
        std::size_t const used = (width > 32 * i) ? width - 32 * i : 0;
        if ((words[i] & ~width_mask_(used)) != 0) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Debug check that words are a valid wide k-mer of `width` bases, see kmer_cast().
 */
template <std::size_t N>
FISK_ALWAYS_INLINE inline constexpr void assert_wide_fits_width_(
    [[maybe_unused]] std::array<std::uint64_t, N> const& words, [[maybe_unused]] std::size_t width
) noexcept {
    assert_wide_width_<32 * N>(width);
    assert(wide_fits_width_(words, width));
}

/**
 * @brief The two-bit slot that base `index` occupies under a layout convention.
 */
template <Layout L>
inline constexpr std::size_t wide_base_slot_(std::size_t index, std::size_t width) noexcept
{
    if constexpr (L == Layout::kMSB) {
        return width - 1 - index;
    } else if constexpr (L == Layout::kLSB) {
        return index;
    } else {
        static_assert(dependent_false_v<L>, "Unhandled Layout in wide_base_slot_()");
        return 0;
    }
}

/**
 * @brief Reverse the order of the used two-bit slots across all words, and XOR each with `Xor`.
 *
 * `Xor` is 0 for a plain reverse, or complement_xor_() for a reverse complement. Reverses the bit
 * pairs within each word and the order of the used words, then shifts the result down across word
 * boundaries by the padding above the width in the highest used word. As in reverse_complement()
 * for Kmer, the XOR is applied before that shift, which then discards the XOR-ed padding, so no
 * separate masking step is needed.
 */
template <std::uint64_t Xor, std::size_t N>
inline constexpr std::array<std::uint64_t, N> reverse_xor_wide_words_(
    std::array<std::uint64_t, N> const& words, std::size_t width
) noexcept {
    std::array<std::uint64_t, N> result{};
    std::size_t const used_words = (width - 1) / 32 + 1;
    unsigned const padding = static_cast<unsigned>(2 * (used_words * 32 - width));
    std::uint64_t carry = 0;
    for (std::size_t i = used_words; i-- > 0;) {
        std::uint64_t const reversed = reverse_bit_pairs_(words[used_words - 1 - i]) ^ Xor;
        if (padding == 0) {
            result[i] = reversed;
        } else {
            result[i] = (reversed >> padding) | carry;
            carry = reversed << (64 - padding);
        }
    }
    return result;
}

// =================================================================================================
//     Construction and Cast
// =================================================================================================

/**
 * @brief Get the raw words of a wide k-mer, the counterpart of kmer_cast().
 */
template <WideKmerType K>
[[nodiscard]] inline constexpr auto kmer_value(K const& kmer) noexcept
{
    return kmer.words;
}

/**
 * @brief Adopt raw words as a wide k-mer, without checking them against the width.
 *
 * For words that the caller already knows to be valid. Otherwise use kmer_cast(), which asserts the
 * width in debug builds.
 */
template <Encoding E, Layout L, std::size_t N>
[[nodiscard]] inline constexpr WideKmer<E, L, 32 * N> kmer_cast_unchecked(
    std::array<std::uint64_t, N> const& words
) noexcept {
    // A bare array of words, as Kmer is a bare word (see kmer_cast_unchecked() for Kmer).
    using K = WideKmer<E, L, 32 * N>;
    static_assert(sizeof(K) == N * sizeof(std::uint64_t));
    static_assert(alignof(K) == alignof(std::uint64_t));
    static_assert(std::is_trivially_copyable_v<K>);
    static_assert(std::is_standard_layout_v<K>);
    static_assert(std::is_aggregate_v<K>);
    return K{words};
}

/**
 * @brief Adopt raw words as a wide k-mer of a given convention.
 *
 * The wide counterpart of kmer_cast() for Kmer, with the same caveat that the caller asserts the
 * convention and the width, both checked by assert() only. The capacity is deduced from the number
 * of words, as `32 * N`.
 */
template <Encoding E, Layout L, std::size_t N>
[[nodiscard]] inline constexpr WideKmer<E, L, 32 * N> kmer_cast(
    std::array<std::uint64_t, N> const& words, [[maybe_unused]] std::size_t width
) noexcept {
    assert_wide_fits_width_(words, width);
    return kmer_cast_unchecked<E, L>(words);
}

// =================================================================================================
//     String Conversion
// =================================================================================================

/**
 * @brief Get the 2-bit code of the base at position `index`, counted from the start of the k-mer.
 */
template <WideKmerType K>
[[nodiscard]] inline constexpr std::uint8_t base_at(
    K const& kmer, std::size_t index, std::size_t width
) noexcept {
    assert_wide_width_<K::capacity>(width);
    assert(index < width);
    std::size_t const slot = wide_base_slot_<K::layout>(index, width);
    return static_cast<std::uint8_t>((kmer.words[slot / 32] >> (2 * (slot % 32))) & 0x3u);
}

/**
 * @brief Get the string representation of a wide k-mer, as a sequence of upper case nucleotides.
 *
 * The inverse of kmer_encode(). Not constexpr, for the same reason as kmer_decode() for Kmer.
 */
template <WideKmerType K>
[[nodiscard]] inline std::string kmer_decode(K const& kmer, std::size_t width)
{
    assert_wide_width_<K::capacity>(width);

    std::string str;
    str.resize(width);
    for (std::size_t i = 0; i < width; ++i) {
        str[i] = code_to_char_<K::encoding>(base_at(kmer, i, width));
    }
    return str;
}

/**
 * @brief Build a wide k-mer from a nucleotide string, the inverse of kmer_decode().
 *
 * The wide counterpart of kmer_encode() for Kmer, with the same purpose and the same loud failure,
 * here for sequences of 1 to `K::capacity` bases. The result type has no default and must be named,
 * as in `kmer_encode<WideKmerAcgtMsb>(seq)`.
 */
template <WideKmerType K>
[[nodiscard]] inline constexpr K kmer_encode(std::string_view seq)
{
    if (seq.empty() || seq.size() > K::capacity) {
        throw std::invalid_argument(
            "Invalid call to kmer_encode() with a sequence length not in [1, " +
            std::to_string(K::capacity) + "]"
        );
    }

    std::array<std::uint64_t, K::word_count> words{};
    for (std::size_t i = 0; i < seq.size(); ++i) {
        std::uint8_t const code = char_to_code_<K::encoding>(seq[i]);
        if (code > 3) {
            throw std::invalid_argument(
                "Invalid call to kmer_encode() with a non-nucleotide character in the sequence"
            );
        }
        std::size_t const slot = wide_base_slot_<K::layout>(i, seq.size());
        words[slot / 32] |= std::uint64_t{code} << (2 * (slot % 32));
    }
    return K{words};
}

// =================================================================================================
//     Nucleotide Operations
// =================================================================================================

/**
 * @brief Complement each base of a wide k-mer in place, without reversing their order.
 */
template <WideKmerType K>
[[nodiscard]] inline constexpr K complement(K const& kmer, std::size_t width) noexcept
{
    assert_wide_width_<K::capacity>(width);
    std::array<std::uint64_t, K::word_count> words{};
    std::size_t const used_words = (width - 1) / 32 + 1;
    for (std::size_t i = 0; i < used_words; ++i) {
        std::size_t const used = width - 32 * i;
        words[i] = (kmer.words[i] ^ complement_xor_<K::encoding>()) & width_mask_(used);
    }
    return K{words};
}

/**
 * @brief Reverse the order of the bases of a wide k-mer, without complementing them.
 */
template <WideKmerType K>
[[nodiscard]] inline constexpr K reverse(K const& kmer, std::size_t width) noexcept
{
    assert_wide_width_<K::capacity>(width);
    return K{reverse_xor_wide_words_<0>(kmer.words, width)};
}

/**
 * @brief Get the reverse complement of a wide k-mer: bases in reverse order, each complemented.
 *
 * Fused rather than composed out of reverse() and complement(), see reverse_xor_wide_words_().
 */
template <WideKmerType K>
[[nodiscard]] inline constexpr K reverse_complement(K const& kmer, std::size_t width) noexcept
{
    assert_wide_width_<K::capacity>(width);
    return K{reverse_xor_wide_words_<complement_xor_<K::encoding>()>(kmer.words, width)};
}

/**
 * @brief Get the canonical form of a wide k-mer: whichever of it and its reverse complement is
 * smaller, as for canonical() of a Kmer.
 */
template <WideKmerType K>
[[nodiscard]] inline constexpr K canonical(K const& kmer, std::size_t width) noexcept
{
    K const rev_comp = reverse_complement(kmer, width);
    return (kmer <= rev_comp) ? kmer : rev_comp;
}

// =================================================================================================
//     Convention Conversion
// =================================================================================================

/**
 * @brief Re-map the 2-bit codes of all words from one encoding to another, see recode_().
 */
template <Encoding From, Encoding To, std::size_t N>
inline constexpr std::array<std::uint64_t, N> recode_wide_words_(
    std::array<std::uint64_t, N> words
) noexcept {
    for (auto& word : words) {
        word = recode_<From, To>(word);
    }
    return words;
}

/**
 * @brief Re-order the 2-bit codes across all words from one layout to another, see relayout_().
 */
template <Layout From, Layout To, std::size_t N>
inline constexpr std::array<std::uint64_t, N> relayout_wide_words_(
    std::array<std::uint64_t, N> const& words, std::size_t width
) noexcept {
    if constexpr (From == To) {
        return words;
    } else if constexpr (
        (From == Layout::kMSB && To == Layout::kLSB) ||
        (From == Layout::kLSB && To == Layout::kMSB)
    ) {
        return reverse_xor_wide_words_<0>(words, width);
    } else {
        static_assert(dependent_false_v<From>, "Unhandled Layout pair in relayout_wide_words_()");
        return words;
    }
}

// As for Kmer, converting between conventions is spelled kmer_convert() in all three cases, and the
// result keeps the capacity.

/**
 * @brief Convert a wide k-mer to a different encoding, keeping its layout.
 *
 * The unused, always-defaulted third parameter is the MSVC workaround explained on kmer_convert()
 * for Kmer, in core/kmer.hpp.
 */
template <Encoding E2, WideKmerType K, typename = Encoding>
[[nodiscard]] inline constexpr typename K::template rebind<E2, K::layout> kmer_convert(
    K const& kmer, [[maybe_unused]] std::size_t width
) noexcept {
    assert_wide_width_<K::capacity>(width);
    return typename K::template rebind<E2, K::layout>{
        recode_wide_words_<K::encoding, E2>(kmer.words)
    };
}

/**
 * @brief Convert a wide k-mer to a different layout, keeping its encoding.
 *
 * See the matching third-parameter comment on the Encoding overload above for why it's here.
 */
template <Layout L2, WideKmerType K, typename = Layout>
[[nodiscard]] inline constexpr typename K::template rebind<K::encoding, L2> kmer_convert(
    K const& kmer, std::size_t width
) noexcept {
    assert_wide_width_<K::capacity>(width);
    return typename K::template rebind<K::encoding, L2>{
        relayout_wide_words_<K::layout, L2>(kmer.words, width)
    };
}

/**
 * @brief Convert a wide k-mer to a different encoding and layout at once.
 */
template <Encoding E2, Layout L2, WideKmerType K>
[[nodiscard]] inline constexpr typename K::template rebind<E2, L2> kmer_convert(
    K const& kmer, std::size_t width
) noexcept {
    assert_wide_width_<K::capacity>(width);
    auto const recoded = recode_wide_words_<K::encoding, E2>(kmer.words);
    return typename K::template rebind<E2, L2>{
        relayout_wide_words_<K::layout, L2>(recoded, width)
    };
}

} // namespace fisk
