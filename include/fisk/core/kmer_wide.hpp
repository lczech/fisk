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

#include "fisk/core/cast.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/types.hpp"

namespace fisk {

// =================================================================================================
//     Convention Dispatch
// =================================================================================================

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
