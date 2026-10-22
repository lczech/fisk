#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/types.hpp"

namespace fisk {

// =================================================================================================
//     Width Checks
// =================================================================================================

/**
 * @brief Mask of the lowest `2 * width` bits, i.e. the bits a k-mer of that width may occupy.
 */
inline constexpr std::uint64_t width_mask_(std::size_t width) noexcept
{
    return (width >= 32) ? ~std::uint64_t{0} : ((std::uint64_t{1} << (2 * width)) - 1u);
}

/**
 * @brief Debug check that a width (k-mer length, or mask weight) is in [1, 32].
 *
 * Compiled out in release builds, as are all debug checks in this file.
 */
FISK_ALWAYS_INLINE inline constexpr void assert_width_(
    [[maybe_unused]] std::size_t width
) noexcept {
    assert(width >= 1 && width <= 32);
}

/**
 * @brief Whether a word holds no bits above `width` bases, i.e. it is a valid k-mer or spaced k-mer
 * of that width.
 */
inline constexpr bool fits_width_(std::uint64_t word, std::size_t width) noexcept
{
    return (word & ~width_mask_(width)) == 0;
}

/**
 * @brief Debug check that a word is a valid k-mer or spaced k-mer of `width` bases.
 *
 * Producers of k-mer words call this with the word they emit.
 */
FISK_ALWAYS_INLINE inline constexpr void assert_fits_width_(
    [[maybe_unused]] std::uint64_t word, [[maybe_unused]] std::size_t width
) noexcept {
    assert_width_(width);
    assert(fits_width_(word, width));
}

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
 * @brief Debug check that words are a valid wide k-mer of `width` bases, see wide_kmer_cast().
 */
template <std::size_t N>
FISK_ALWAYS_INLINE inline constexpr void assert_wide_fits_width_(
    [[maybe_unused]] std::array<std::uint64_t, N> const& words, [[maybe_unused]] std::size_t width
) noexcept {
    assert_wide_width_<32 * N>(width);
    assert(wide_fits_width_(words, width));
}

// =================================================================================================
//     K-mer Cast
// =================================================================================================

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

/**
 * @brief Adopt a raw word as a k-mer of a given convention, without checking it against the width.
 *
 * For words that the caller already knows to be valid, such as the output of an extractor.
 * Otherwise use kmer_cast(), which asserts the width in debug builds.
 */
template <Encoding E, Layout L>
[[nodiscard]] inline constexpr Kmer<E, L> kmer_cast_unchecked(std::uint64_t raw) noexcept
{
    // The type must stay a bare word as far as the ABI is concerned: an aggregate with no
    // user-declared constructors, so that it is passed in a register exactly like the uint64_t it
    // wraps. Checked here, as every cast passes through, and the type is incomplete in its body.
    using K = Kmer<E, L>;
    static_assert(sizeof(K) == sizeof(std::uint64_t));
    static_assert(alignof(K) == alignof(std::uint64_t));
    static_assert(std::is_trivially_copyable_v<K>);
    static_assert(std::is_standard_layout_v<K>);
    static_assert(std::is_aggregate_v<K>);
    return K{raw};
}

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
    assert_fits_width_(raw, width);
    return kmer_cast_unchecked<E, L>(raw);
}

// =================================================================================================
//     K-mer Vector Cast
// =================================================================================================

// Adopt a SIMD vector of raw words as a KmerVector or SpacedKmerVector `V`, without checking them.
template <KmerVectorType V>
[[nodiscard]] inline V kmer_vector_cast_unchecked_(typename V::simd::simd_vector vec) noexcept
{
    // A bare vector as far as the ABI is concerned, as Kmer is a bare word (see
    // kmer_cast_unchecked()), so that it is passed in a vector register.
    static_assert(sizeof(V) == sizeof(vec));
    static_assert(alignof(V) == alignof(typename V::simd::simd_vector));
    static_assert(std::is_trivially_copyable_v<V>);
    static_assert(std::is_standard_layout_v<V>);
    static_assert(std::is_aggregate_v<V>);
    return V{vec};
}

// Adopt a SIMD vector of raw words as a KmerVector or SpacedKmerVector `V`, asserting that every
// lane fits `width` nucleotides.
template <KmerVectorType V>
[[nodiscard]] inline V kmer_vector_cast_(
    typename V::simd::simd_vector vec, [[maybe_unused]] std::size_t width
) noexcept {
    static_assert(sizeof(vec) == V::lanes * sizeof(std::uint64_t));
    std::uint64_t words[V::lanes];
    std::memcpy(words, &vec, sizeof(vec));
    for (std::uint64_t const word : words) {
        assert_fits_width_(word, width);
    }
    return kmer_vector_cast_unchecked_<V>(vec);
}

/**
 * @brief Adopt a SIMD vector of raw words as a vector of k-mers of a given convention, without
 * checking them against the width.
 *
 * The vector counterpart of kmer_cast_unchecked(), for vectors that the caller already knows to
 * hold valid k-mers. Otherwise use kmer_vector_cast(), which asserts the width in debug builds.
 */
template <Encoding E, Layout L, SimdArch Simd>
[[nodiscard]] inline KmerVector<E, L, Simd> kmer_vector_cast_unchecked(
    typename Simd::simd_vector vec
) noexcept {
    return kmer_vector_cast_unchecked_<KmerVector<E, L, Simd>>(vec);
}

/**
 * @brief Adopt a SIMD vector of raw words as a vector of k-mers of a given convention.
 *
 * The vector counterpart of kmer_cast(), with the same caveat that the caller asserts the
 * convention, and that every lane holds a k-mer of `width` bases, both checked by assert() only.
 */
template <Encoding E, Layout L, SimdArch Simd>
[[nodiscard]] inline KmerVector<E, L, Simd> kmer_vector_cast(
    typename Simd::simd_vector vec, std::size_t width
) noexcept {
    return kmer_vector_cast_<KmerVector<E, L, Simd>>(vec, width);
}

/**
 * @brief Get the SIMD vector of a vector of k-mers or spaced k-mers, the counterpart of
 * kmer_vector_cast() and spaced_kmer_vector_cast().
 */
template <KmerVectorType V>
[[nodiscard]] inline typename V::simd::simd_vector kmer_value(V kmers) noexcept
{
    return kmers.value;
}

// =================================================================================================
//     Spaced K-mer Cast
// =================================================================================================

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
    // A bare word as far as the ABI is concerned, as for Kmer (see kmer_cast_unchecked()).
    using K = SpacedKmer<E, L>;
    static_assert(sizeof(K) == sizeof(std::uint64_t));
    static_assert(alignof(K) == alignof(std::uint64_t));
    static_assert(std::is_trivially_copyable_v<K>);
    static_assert(std::is_standard_layout_v<K>);
    static_assert(std::is_aggregate_v<K>);
    return K{raw};
}

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
    return spaced_kmer_cast_unchecked<E, L>(raw);
}

/**
 * @brief Adopt a SIMD vector of raw words as a vector of spaced k-mers of a given convention,
 * without checking them against the weight.
 *
 * The vector counterpart of spaced_kmer_cast_unchecked(), for vectors that the caller already
 * knows to hold valid spaced k-mers. Otherwise use spaced_kmer_vector_cast(), which asserts the
 * weight in debug builds.
 */
template <Encoding E, Layout L, SimdArch Simd>
[[nodiscard]] inline SpacedKmerVector<E, L, Simd> spaced_kmer_vector_cast_unchecked(
    typename Simd::simd_vector vec
) noexcept {
    return kmer_vector_cast_unchecked_<SpacedKmerVector<E, L, Simd>>(vec);
}

/**
 * @brief Adopt a SIMD vector of raw words as a vector of spaced k-mers of a given convention.
 *
 * The vector counterpart of spaced_kmer_cast(), with the same caveat that the caller asserts the
 * convention, and that every lane holds a spaced k-mer of `weight` kept nucleotides, both checked
 * by assert() only.
 */
template <Encoding E, Layout L, SimdArch Simd>
[[nodiscard]] inline SpacedKmerVector<E, L, Simd> spaced_kmer_vector_cast(
    typename Simd::simd_vector vec, std::size_t weight
) noexcept {
    return kmer_vector_cast_<SpacedKmerVector<E, L, Simd>>(vec, weight);
}

// =================================================================================================
//     Wide K-mer Cast
// =================================================================================================

/**
 * @brief Get the raw words of a wide k-mer, the counterpart of wide_kmer_cast().
 */
template <WideKmerType K>
[[nodiscard]] inline constexpr auto kmer_value(K const& kmer) noexcept
{
    return kmer.words;
}

/**
 * @brief Adopt raw words as a wide k-mer, without checking them against the width.
 *
 * For words that the caller already knows to be valid. Otherwise use wide_kmer_cast(), which
 * asserts the width in debug builds.
 */
template <Encoding E, Layout L, std::size_t N>
[[nodiscard]] inline constexpr WideKmer<E, L, 32 * N> wide_kmer_cast_unchecked(
    std::array<std::uint64_t, N> const& words
) noexcept {
    // A bare array of words, as Kmer is a bare word (see kmer_cast_unchecked()).
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
[[nodiscard]] inline constexpr WideKmer<E, L, 32 * N> wide_kmer_cast(
    std::array<std::uint64_t, N> const& words, [[maybe_unused]] std::size_t width
) noexcept {
    assert_wide_fits_width_(words, width);
    return wide_kmer_cast_unchecked<E, L>(words);
}

} // namespace fisk
