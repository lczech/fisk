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
    static_assert(
        std::is_trivially_copyable_v<Vec>,
        "kmer_cast() expects a trivially copyable vector"
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
    static_assert(
        sizeof(Vec) % sizeof(std::uint64_t) == 0,
        "kmer_cast() expects a vector of whole 64-bit lanes"
    );
    return kmer_cast<E, L>(vec, sizeof(Vec) / sizeof(std::uint64_t), width, out);
}

/**
 * @brief Adopt the first `count` lanes of a SIMD vector of raw words as k-mers, without checking
 * them against the width. The unchecked counterpart of kmer_cast(vec, count, width, out).
 */
template <Encoding E, Layout L, typename Vec>
inline std::size_t kmer_cast_unchecked(Vec vec, std::size_t count, Kmer<E, L>* out) noexcept
{
    static_assert(
        sizeof(Vec) % sizeof(std::uint64_t) == 0,
        "kmer_cast_unchecked() expects a vector of whole 64-bit lanes"
    );
    static_assert(
        std::is_trivially_copyable_v<Vec>,
        "kmer_cast_unchecked() expects a trivially copyable vector"
    );
    constexpr std::size_t lanes = sizeof(Vec) / sizeof(std::uint64_t);
    assert(count <= lanes);
    assert(out != nullptr);

    std::uint64_t buffer[lanes];
    std::memcpy(buffer, &vec, sizeof(Vec));
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = kmer_cast_unchecked<E, L>(buffer[i]);
    }
    return count;
}

/**
 * @brief Adopt every lane of a SIMD vector of raw words as k-mers, without checking them against
 * the width. The unchecked counterpart of kmer_cast(vec, width, out).
 */
template <Encoding E, Layout L, typename Vec>
inline std::size_t kmer_cast_unchecked(Vec vec, Kmer<E, L>* out) noexcept
{
    static_assert(
        sizeof(Vec) % sizeof(std::uint64_t) == 0,
        "kmer_cast_unchecked() expects a vector of whole 64-bit lanes"
    );
    return kmer_cast_unchecked<E, L>(vec, sizeof(Vec) / sizeof(std::uint64_t), out);
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
