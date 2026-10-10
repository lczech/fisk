#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "fisk/core/kmer.hpp"

namespace fisk {

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
//     Construction and Cast
// =================================================================================================

/**
 * @brief Adopt a raw 2-bit-packed word of kept bases as a spaced k-mer of a given convention.
 *
 * The spaced counterpart of kmer_cast(): `weight` is the mask weight, i.e. the number of kept bases
 * in `raw`. Same caveat as kmer_cast(), that the caller asserts the convention and the weight.
 */
template <Encoding E, Layout L>
[[nodiscard]] inline constexpr SpacedKmer<E, L> spaced_kmer_cast(
    std::uint64_t raw, [[maybe_unused]] std::size_t weight
) noexcept {
    assert_fits_width_(raw, weight);
    return SpacedKmer<E, L>{raw};
}

/**
 * @brief Adopt a raw word of kept bases as a spaced k-mer, without checking it against the weight.
 *
 * For words that the caller already knows to be valid, such as the output of an extractor.
 * Otherwise use spaced_kmer_cast(), which asserts the weight in debug builds.
 */
template <Encoding E, Layout L>
[[nodiscard]] inline constexpr SpacedKmer<E, L> spaced_kmer_cast_unchecked(
    std::uint64_t raw
) noexcept
{
    return SpacedKmer<E, L>{raw};
}

/**
 * @brief Adopt the first `count` lanes of a SIMD vector of raw words as spaced k-mers, and return
 * how many were written.
 *
 * The spaced counterpart of the contiguous vector kmer_cast(), with the same memcpy-based,
 * ISA-agnostic lane handling. `weight` is the mask weight shared by all lanes.
 */
template <Encoding E, Layout L, typename Vec>
inline std::size_t spaced_kmer_cast(
    Vec vec, std::size_t count, std::size_t weight, SpacedKmer<E, L>* out
) noexcept {
    static_assert(
        sizeof(Vec) % sizeof(std::uint64_t) == 0,
        "spaced_kmer_cast() expects a vector of whole 64-bit lanes"
    );
    static_assert(
        std::is_trivially_copyable_v<Vec>,
        "spaced_kmer_cast() expects a trivially copyable vector"
    );
    constexpr std::size_t lanes = sizeof(Vec) / sizeof(std::uint64_t);
    assert(count <= lanes);
    assert(out != nullptr);

    std::uint64_t buffer[lanes];
    std::memcpy(buffer, &vec, sizeof(Vec));
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = spaced_kmer_cast<E, L>(buffer[i], weight);
    }
    return count;
}

/**
 * @brief Adopt every lane of a SIMD vector of raw words as spaced k-mers.
 */
template <Encoding E, Layout L, typename Vec>
inline std::size_t spaced_kmer_cast(Vec vec, std::size_t weight, SpacedKmer<E, L>* out) noexcept
{
    static_assert(
        sizeof(Vec) % sizeof(std::uint64_t) == 0,
        "spaced_kmer_cast() expects a vector of whole 64-bit lanes"
    );
    return spaced_kmer_cast<E, L>(vec, sizeof(Vec) / sizeof(std::uint64_t), weight, out);
}

/**
 * @brief Adopt the first `count` lanes of a SIMD vector of raw words as spaced k-mers, without
 * checking them against the weight. The unchecked counterpart of spaced_kmer_cast(vec, count,
 * weight, out).
 */
template <Encoding E, Layout L, typename Vec>
inline std::size_t spaced_kmer_cast_unchecked(
    Vec vec, std::size_t count, SpacedKmer<E, L>* out
) noexcept {
    static_assert(
        sizeof(Vec) % sizeof(std::uint64_t) == 0,
        "spaced_kmer_cast_unchecked() expects a vector of whole 64-bit lanes"
    );
    static_assert(
        std::is_trivially_copyable_v<Vec>,
        "spaced_kmer_cast_unchecked() expects a trivially copyable vector"
    );
    constexpr std::size_t lanes = sizeof(Vec) / sizeof(std::uint64_t);
    assert(count <= lanes);
    assert(out != nullptr);

    std::uint64_t buffer[lanes];
    std::memcpy(buffer, &vec, sizeof(Vec));
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = spaced_kmer_cast_unchecked<E, L>(buffer[i]);
    }
    return count;
}

/**
 * @brief Adopt every lane of a SIMD vector of raw words as spaced k-mers, without checking them
 * against the weight. The unchecked counterpart of spaced_kmer_cast(vec, weight, out).
 */
template <Encoding E, Layout L, typename Vec>
inline std::size_t spaced_kmer_cast_unchecked(Vec vec, SpacedKmer<E, L>* out) noexcept
{
    static_assert(
        sizeof(Vec) % sizeof(std::uint64_t) == 0,
        "spaced_kmer_cast_unchecked() expects a vector of whole 64-bit lanes"
    );
    return spaced_kmer_cast_unchecked<E, L>(vec, sizeof(Vec) / sizeof(std::uint64_t), out);
}

// =================================================================================================
//     Spaced K-mer Mask
// =================================================================================================

/**
 * @brief A spaced k-mer mask in the representation `M` of some bit extraction algorithm, tagged
 * with the Layout of the spaced k-mers that it extracts, and with its span.
 *
 * `M` is a bit extraction mask, table, or SIMD kernel, such as BitExtractMask,
 * BitExtractButterflyTable, or BitExtractKernelButterflyAVX2. As this derives from `M`, it can be
 * passed to the generic bit extraction functions as is. The spaced k-mer extraction functions on
 * the other hand only accept masks tagged this way, and use the tag as the Layout of the rolling
 * window and of the emitted spaced k-mers. Hence, the bits of the mask have to be arranged for that
 * Layout.
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

} // namespace fisk
