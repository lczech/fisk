#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#include "fisk/core/cast.hpp"
#include "fisk/util/intrinsics.hpp"
#include "fisk/simd/simd.hpp"
#include "fisk/core/types.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction Callback Concepts
// =================================================================================================

// The forms of callbacks that the k-mer extraction functions accept, as dispatched by the
// invoke_*() functions below. The vector forms take a KmerVector or a SpacedKmerVector, and always
// take the position: the forms of single k-mers and of vectors are then told apart by their number
// of arguments alone, also for generic lambdas.

// -------------------------------------------------------------------------------------------------
//     Parameter Widths
// -------------------------------------------------------------------------------------------------

// Positions, mask indices, and counts are std::size_t, and a callback that takes one of them as a
// narrower integer, such as `int` or `std::uint32_t`, would silently truncate it, for example the
// positions beyond 2^32 of a long sequence. check_callback_parameters_() rejects such callbacks at
// compile time, where their parameter types can be read from their type: functions, function
// pointers, and classes with one `operator()` that is not a template, such as lambdas without
// `auto` parameters and std::function. The parts of a callback_overload are checked each. Not
// checked are lambdas with an `auto` parameter, even where another one is a narrow integer, and
// classes with several `operator()`: their `operator()` is a template or an overload set, whose
// parameters cannot be read from the type, and trying them with other argument types would
// compile the bodies of generic lambdas with those types. An `auto` position takes the
// std::size_t as it is passed.

// Whether any parameter of a function type is an integer narrower than 64 bits.
template <typename F>
struct callback_narrow_parameters_ : std::false_type {};

template <typename R, typename... Args>
struct callback_narrow_parameters_<R(Args...)>
    : std::bool_constant<(
        (std::integral<std::remove_cvref_t<Args>> && sizeof(std::remove_cvref_t<Args>) < 8) || ...
    )>
{};

template <typename R, typename... Args>
struct callback_narrow_parameters_<R(Args...) noexcept>
    : callback_narrow_parameters_<R(Args...)>
{};

template <typename R, typename C, typename... Args>
struct callback_narrow_parameters_<R (C::*)(Args...)> : callback_narrow_parameters_<R(Args...)> {};

template <typename R, typename C, typename... Args>
struct callback_narrow_parameters_<R (C::*)(Args...) const>
    : callback_narrow_parameters_<R(Args...)>
{};

template <typename R, typename C, typename... Args>
struct callback_narrow_parameters_<R (C::*)(Args...) noexcept>
    : callback_narrow_parameters_<R(Args...)>
{};

template <typename R, typename C, typename... Args>
struct callback_narrow_parameters_<R (C::*)(Args...) const noexcept>
    : callback_narrow_parameters_<R(Args...)>
{};

// Whether `F` has one `operator()` that is not a template. A trait instead of a requires-expression:
// Clang 13 reports the address of a template or overloaded `operator()` as a hard error there.
template <typename F, typename = void>
struct callback_has_single_call_operator_ : std::false_type {};

template <typename F>
struct callback_has_single_call_operator_<F, std::void_t<decltype(&F::operator())>>
    : std::true_type
{};

// Whether `Callback` takes an integer narrower than 64 bits, as far as its type tells, see above.
template <typename Callback>
consteval bool callback_has_narrow_parameters_()
{
    using F = std::remove_pointer_t<std::remove_cvref_t<Callback>>;
    if constexpr (std::is_function_v<F>) {
        return callback_narrow_parameters_<F>::value;
    } else if constexpr (callback_has_single_call_operator_<F>::value) {
        return callback_narrow_parameters_<decltype(&F::operator())>::value;
    } else {
        return false;
    }
}

// Reject a callback that takes an integer narrower than 64 bits, see above.
template <typename Callback>
FISK_ALWAYS_INLINE
constexpr void check_callback_parameters_() noexcept
{
    static_assert(
        !callback_has_narrow_parameters_<Callback>(),
        "Callback parameters for positions, mask indices, and counts must be 64-bit integers, "
        "such as std::size_t: a narrower type would truncate them."
    );
}

// -------------------------------------------------------------------------------------------------
//     Contiguous K-mers
// -------------------------------------------------------------------------------------------------

/**
 * @brief A callback for single k-mers, as `callback(pos, kmer)` or `callback(kmer)`.
 */
template <typename Callback, Encoding E, Layout L>
concept KmerCallback =
    std::invocable<Callback, std::size_t, Kmer<E, L> const&> ||
    std::invocable<Callback, Kmer<E, L> const&>;

/**
 * @brief A callback for vectors of k-mers, as `callback(pos, kmers, valid_count)`, see KmerVector.
 */
template <typename Callback, Encoding E, Layout L, typename Simd>
concept KmerVectorCallback =
    std::invocable<Callback, std::size_t, KmerVector<E, L, Simd>, std::size_t>;

/**
 * @brief A callback for vectors of k-mers with their valid lanes, as
 * `callback(pos, kmers, valid_lanes)`, see KmerVector and SimdVectorValidLanes.
 */
template <typename Callback, Encoding E, Layout L, typename Simd>
concept KmerVectorValidLanesCallback =
    std::invocable<Callback, std::size_t, KmerVector<E, L, Simd>, SimdVectorValidLanes<Simd>>;

// -------------------------------------------------------------------------------------------------
//     Spaced K-mers
// -------------------------------------------------------------------------------------------------

/**
 * @brief A callback for single spaced k-mers, as `callback(pos, mask_idx, spaced_kmer)`, or, when
 * `SingleMask` is true, also as `callback(pos, spaced_kmer)` or `callback(spaced_kmer)`.
 */
template <typename Callback, Encoding E, Layout L, bool SingleMask>
concept SpacedKmerCallback =
    std::invocable<Callback, std::size_t, std::size_t, SpacedKmer<E, L> const&> ||
    (SingleMask && std::invocable<Callback, std::size_t, SpacedKmer<E, L> const&>) ||
    (SingleMask && std::invocable<Callback, SpacedKmer<E, L> const&>);

/**
 * @brief A callback for vectors of spaced k-mers, as `callback(pos, mask_idx, vec, valid_count)`,
 * or, when `SingleMask` is true, also as `callback(pos, vec, valid_count)`, see SpacedKmerVector.
 */
template <typename Callback, Encoding E, Layout L, typename Simd, bool SingleMask>
concept SpacedKmerVectorCallback =
    std::invocable<Callback, std::size_t, std::size_t, SpacedKmerVector<E, L, Simd>, std::size_t> ||
    (
        SingleMask &&
        std::invocable<Callback, std::size_t, SpacedKmerVector<E, L, Simd>, std::size_t>
    );

/**
 * @brief A callback for vectors of spaced k-mers with their valid lanes, as
 * `callback(pos, mask_idx, vec, valid_lanes)`, or, when `SingleMask` is true, also as
 * `callback(pos, vec, valid_lanes)`, see SpacedKmerVector and SimdVectorValidLanes.
 */
template <typename Callback, Encoding E, Layout L, typename Simd, bool SingleMask>
concept SpacedKmerVectorValidLanesCallback =
    std::invocable<
        Callback, std::size_t, std::size_t, SpacedKmerVector<E, L, Simd>, SimdVectorValidLanes<Simd>
    > ||
    (
        SingleMask && std::invocable<
            Callback, std::size_t, SpacedKmerVector<E, L, Simd>, SimdVectorValidLanes<Simd>
        >
    );

/**
 * @brief A callback for the single k-mers or spaced k-mers in the lanes of a vector `V`, as
 * `func(pos, kmer)` or `func(kmer)`, see for_each_valid_lane().
 */
template <typename Func, typename V>
concept KmerVectorLaneCallback =
    std::invocable<Func, std::size_t, typename V::kmer_type const&> ||
    std::invocable<Func, typename V::kmer_type const&>;

// =================================================================================================
//     K-mer Extraction Callback Dispatch
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Contiguous K-mers
// -------------------------------------------------------------------------------------------------

/**
 * @brief Invoke a k-mer extraction callback with the k-mer of a raw word, with or without a leading
 * position argument.
 *
 * Accepts either `callback(pos, kmer)` or `callback(kmer)`, see KmerCallback. This is resolved at
 * compile time, so a callback without `pos` does not pay for it. The word must already be a valid
 * k-mer, see kmer_cast_unchecked().
 */
template <Encoding E, Layout L, typename Callback>
FISK_ALWAYS_INLINE
inline void invoke_kmer_callback(Callback&& callback, std::size_t pos, std::uint64_t raw)
{
    check_callback_parameters_<Callback>();
    using Value = Kmer<E, L>;
    auto const value = kmer_cast_unchecked<E, L>(raw);
    if constexpr (std::invocable<Callback, std::size_t, Value const&>) {
        callback(pos, value);
    } else if constexpr (std::invocable<Callback, Value const&>) {
        callback(value);
    } else {
        static_assert(
            KmerCallback<Callback, E, L>,
            "Callback must be callable as callback(pos, kmer) or callback(kmer)."
        );
    }
}

/**
 * @brief Invoke a k-mer extraction callback with a vector of consecutive k-mers.
 *
 * Accepts `callback(pos, kmers, valid_count)`, see KmerVectorCallback. `pos` is the start position
 * of the k-mer in lane 0; lane `j` holds the k-mer starting at `pos + j`, for `j < valid_count`.
 * As for invoke_kmer_callback(), a callback that ignores the position costs nothing.
 */
template <Encoding E, Layout L, SimdArch Simd, typename Callback>
FISK_ALWAYS_INLINE
inline void invoke_kmer_vector_callback(
    Callback&& callback, std::size_t pos, KmerVector<E, L, Simd> kmers, std::size_t valid_count
) {
    check_callback_parameters_<Callback>();
    if constexpr (KmerVectorCallback<Callback, E, L, Simd>) {
        callback(pos, kmers, valid_count);
    } else {
        static_assert(
            KmerVectorCallback<Callback, E, L, Simd>,
            "Callback must be callable as callback(pos, kmers, valid_count)."
        );
    }
}

/**
 * @brief Invoke a k-mer extraction callback with a vector of consecutive k-mers and its valid
 * lanes.
 *
 * Accepts `callback(pos, kmers, valid_lanes)`, see KmerVectorValidLanesCallback, where lane `j` of
 * `kmers` holds the k-mer starting at `pos + j`, valid if bit `j` of `valid_lanes.bits` is set.
 */
template <Encoding E, Layout L, SimdArch Simd, typename Callback>
FISK_ALWAYS_INLINE
inline void invoke_kmer_vector_valid_lanes_callback(
    Callback&& callback, std::size_t pos, KmerVector<E, L, Simd> kmers,
    SimdVectorValidLanes<Simd> valid_lanes
) {
    check_callback_parameters_<Callback>();
    if constexpr (KmerVectorValidLanesCallback<Callback, E, L, Simd>) {
        callback(pos, kmers, valid_lanes);
    } else {
        static_assert(
            KmerVectorValidLanesCallback<Callback, E, L, Simd>,
            "Callback must be callable as callback(pos, kmers, valid_lanes)."
        );
    }
}

// -------------------------------------------------------------------------------------------------
//     Spaced K-mers
// -------------------------------------------------------------------------------------------------

/**
 * @brief Invoke a spaced k-mer extraction callback with the spaced k-mer of a raw word, with
 * optional shorthands for a single mask.
 *
 * Accepts `callback(pos, mask_idx, spaced_kmer)`. When `SingleMask` is true (exactly one mask,
 * known at compile time), the shorthands `callback(pos, spaced_kmer)` and `callback(spaced_kmer)`
 * are also accepted, as `mask_idx` is always 0 then, see SpacedKmerCallback. As for
 * invoke_kmer_callback(), a callback without `pos` does not pay for it. The word must already be a
 * valid spaced k-mer, see spaced_kmer_cast_unchecked().
 *
 * @tparam SingleMask Whether the caller is known at compile time to iterate exactly one mask.
 */
template <Encoding E, Layout L, bool SingleMask, typename Callback>
FISK_ALWAYS_INLINE
inline void invoke_spaced_kmer_callback(
    Callback&& callback, std::size_t mask_idx, std::size_t pos, std::uint64_t raw
) {
    check_callback_parameters_<Callback>();
    using Value = SpacedKmer<E, L>;
    auto const value = spaced_kmer_cast_unchecked<E, L>(raw);

    // There is no `callback(mask_idx, spaced_kmer)` shorthand for multiple masks: it would have the
    // same shape as the single-mask shorthand, so a `(std::size_t, Value)` callback would receive
    // either pos or mask_idx, depending on how many masks are used at the call site. And
    // `callback(spaced_kmer)` is only accepted for a single mask, as with multiple masks it would
    // lose which mask a spaced k-mer belongs to.
    if constexpr (std::invocable<Callback, std::size_t, std::size_t, Value const&>) {
        callback(pos, mask_idx, value);
    } else if constexpr (SingleMask && std::invocable<Callback, std::size_t, Value const&>) {
        callback(pos, value);
    } else if constexpr (SingleMask && std::invocable<Callback, Value const&>) {
        callback(value);
    } else if constexpr (SingleMask) {
        static_assert(
            SpacedKmerCallback<Callback, E, L, SingleMask>,
            "Callback must be callable as callback(pos, mask_idx, value), callback(pos, value), "
            "or callback(value)."
        );
    } else {
        static_assert(
            SpacedKmerCallback<Callback, E, L, SingleMask>,
            "Callback must be callable as callback(pos, mask_idx, value). The shorthands "
            "callback(pos, value) and callback(value) are only available for a single mask (known "
            "at compile time), since with multiple masks the mask index must be explicit."
        );
    }
}

/**
 * @brief Invoke a spaced k-mer extraction callback with a vector of the spaced k-mers of one mask
 * at consecutive positions.
 *
 * Accepts `callback(pos, mask_idx, spaced_kmers, valid_count)`, and, when `SingleMask` is true,
 * also the shorthand `callback(pos, spaced_kmers, valid_count)`, see SpacedKmerVectorCallback.
 * Lane `j` holds the spaced k-mer of mask `mask_idx` starting at `pos + j`, for `j < valid_count`;
 * a vector never mixes masks.
 */
template <Encoding E, Layout L, SimdArch Simd, bool SingleMask, typename Callback>
FISK_ALWAYS_INLINE
inline void invoke_spaced_kmer_vector_callback(
    Callback&& callback, std::size_t mask_idx, std::size_t pos,
    SpacedKmerVector<E, L, Simd> spaced_kmers, std::size_t valid_count
) {
    check_callback_parameters_<Callback>();
    using V = SpacedKmerVector<E, L, Simd>;
    if constexpr (std::invocable<Callback, std::size_t, std::size_t, V, std::size_t>) {
        callback(pos, mask_idx, spaced_kmers, valid_count);
    } else if constexpr (SingleMask && std::invocable<Callback, std::size_t, V, std::size_t>) {
        callback(pos, spaced_kmers, valid_count);
    } else {
        static_assert(
            SpacedKmerVectorCallback<Callback, E, L, Simd, SingleMask>,
            "Callback must be callable as callback(pos, mask_idx, spaced_kmers, valid_count), or, "
            "for a single mask, as callback(pos, spaced_kmers, valid_count)."
        );
    }
}

/**
 * @brief Invoke a spaced k-mer extraction callback with a vector of the spaced k-mers of one mask
 * at consecutive positions and its valid lanes.
 *
 * Accepts `callback(pos, mask_idx, spaced_kmers, valid_lanes)`, and, when `SingleMask` is true,
 * also the shorthand `callback(pos, spaced_kmers, valid_lanes)`, see
 * SpacedKmerVectorValidLanesCallback. Lane `j` holds the spaced k-mer of mask `mask_idx` starting
 * at `pos + j`, valid if bit `j` of `valid_lanes.bits` is set.
 */
template <Encoding E, Layout L, SimdArch Simd, bool SingleMask, typename Callback>
FISK_ALWAYS_INLINE
inline void invoke_spaced_kmer_vector_valid_lanes_callback(
    Callback&& callback, std::size_t mask_idx, std::size_t pos,
    SpacedKmerVector<E, L, Simd> spaced_kmers, SimdVectorValidLanes<Simd> valid_lanes
) {
    check_callback_parameters_<Callback>();
    using V = SpacedKmerVector<E, L, Simd>;
    using Lanes = SimdVectorValidLanes<Simd>;
    if constexpr (std::invocable<Callback, std::size_t, std::size_t, V, Lanes>) {
        callback(pos, mask_idx, spaced_kmers, valid_lanes);
    } else if constexpr (SingleMask && std::invocable<Callback, std::size_t, V, Lanes>) {
        callback(pos, spaced_kmers, valid_lanes);
    } else {
        static_assert(
            SpacedKmerVectorValidLanesCallback<Callback, E, L, Simd, SingleMask>,
            "Callback must be callable as callback(pos, mask_idx, spaced_kmers, valid_lanes), or, "
            "for a single mask, as callback(pos, spaced_kmers, valid_lanes)."
        );
    }
}

// =================================================================================================
//     Valid Lanes of K-mer Vectors
// =================================================================================================

// Call `func` with the single k-mer or spaced k-mer `raw` of a vector `V`, with or without `pos`.
template <KmerVectorType V, typename Func>
FISK_ALWAYS_INLINE
inline void invoke_kmer_vector_lane_callback_(Func& func, std::size_t pos, std::uint64_t raw)
{
    check_callback_parameters_<Func>();
    using K = typename V::kmer_type;
    K const kmer{raw};
    if constexpr (std::invocable<Func&, std::size_t, K const&>) {
        func(pos, kmer);
    } else {
        func(kmer);
    }
}

// Copy the lanes of a vector of k-mers or spaced k-mers into `words`. Clang 16 and older (but
// not Apple Clang) crash in their SLP vectorizer on some loops that sum the lanes of these
// vectors, for several vectors per iteration; there, an empty asm keeps the lanes in memory,
// which avoids that.
template <KmerVectorType V>
FISK_ALWAYS_INLINE
inline void copy_kmer_vector_lanes_(V const& kmers, std::uint64_t (&words)[V::lanes])
{
    std::memcpy(words, &kmers.value, sizeof(words));
    #if defined(__clang__) && !defined(__apple_build_version__) && __clang_major__ <= 16
    __asm__ volatile("" : "+m"(words));
    #endif
}

/**
 * @brief Call `func` with each of the first `valid_count` k-mers of a vector of k-mers or spaced
 * k-mers, in lane order, as handed out by for_each_kmer_packed_unchecked_bytewise_simd() or
 * for_each_spaced_kmer_packed_unchecked_bytewise_simd().
 *
 * `func` takes single k-mers or spaced k-mers, as `func(pos, kmer)` or `func(kmer)`, see
 * KmerVectorLaneCallback, where the k-mer in lane `j` starts at `pos + j`.
 */
template <KmerVectorType V, typename Func>
    requires KmerVectorLaneCallback<Func, V>
FISK_ALWAYS_INLINE
inline void for_each_valid_lane(std::size_t pos, V kmers, std::size_t valid_count, Func&& func)
{
    assert(valid_count <= V::lanes);
    std::uint64_t words[V::lanes];
    copy_kmer_vector_lanes_(kmers, words);
    for (std::size_t j = 0; j < valid_count; ++j) {
        invoke_kmer_vector_lane_callback_<V>(func, pos + j, words[j]);
    }
}

/**
 * @brief Call `func` with each valid k-mer or spaced k-mer of a vector, in lane order, as handed
 * out by for_each_kmer_packed_checked_bytewise_simd() or
 * for_each_spaced_kmer_ascii_checked_rolling_simd().
 *
 * As the overload for a count of valid lanes, for the k-mers in the lanes whose bits are set in
 * `valid_lanes.bits`.
 */
template <KmerVectorType V, typename Func>
    requires KmerVectorLaneCallback<Func, V>
FISK_ALWAYS_INLINE
inline void for_each_valid_lane(
    std::size_t pos, V kmers, SimdVectorValidLanes<typename V::simd> valid_lanes, Func&& func
) {
    std::uint64_t words[V::lanes];
    copy_kmer_vector_lanes_(kmers, words);
    std::uint64_t const bits = valid_lanes.bits;

    // A loop with a fixed count over all lanes, which tests the bit of each, rather than a loop
    // over the set bits, whose number of iterations depends on the valid lanes.
    for (std::size_t j = 0; j < V::lanes; ++j) {
        if ((bits >> j) & 1u) {
            invoke_kmer_vector_lane_callback_<V>(func, pos + j, words[j]);
        }
    }
}

/**
 * @brief Store the valid k-mers or spaced k-mers of a vector at `out`, in lane order, and return
 * how many were stored.
 *
 * `valid` is the count of valid lanes or the SimdVectorValidLanes that the vector was handed out
 * with, see for_each_valid_lane(). `out` must have room for `V::lanes` k-mers. The positions are
 * not stored: with a count, the stored k-mer `i` starts at `pos + i`, but valid lanes can leave
 * gaps, so for their positions, use for_each_valid_lane() with `func(pos, kmer)` instead.
 */
template <KmerVectorType V, typename Valid>
    requires std::same_as<Valid, std::size_t> ||
             std::same_as<Valid, SimdVectorValidLanes<typename V::simd>>
inline std::size_t store_valid_lanes(
    V kmers, Valid valid, typename V::kmer_type* out
) noexcept {
    using K = typename V::kmer_type;
    std::size_t count = 0;
    for_each_valid_lane(0, kmers, valid, [&](K const& kmer) FISK_ALWAYS_INLINE_LAMBDA {
        out[count++] = kmer;
    });
    return count;
}

// =================================================================================================
//     Callback Overload Sets
// =================================================================================================

/**
 * @brief A callback made of several callables, such as lambdas, with the `operator()` of each.
 *
 * For instance, a callback for single k-mers and one for vectors of k-mers, for the extraction
 * functions that hand vectors to callbacks that take them, see for_each_kmer():
 *
 *     auto callback = callback_overload{
 *         [&](std::size_t pos, KmerAcgtMsb kmer) { ... },
 *         [&](std::size_t pos, auto kmers, auto valid) { ... }
 *     };
 */
template <typename... Fs>
struct callback_overload : Fs...
{
    // Each part on its own, see check_callback_parameters_(): the overload set as a whole has no
    // single `operator()`.
    static_assert(
        (!callback_has_narrow_parameters_<Fs>() && ...),
        "Callback parameters for positions, mask indices, and counts must be 64-bit integers, "
        "such as std::size_t: a narrower type would truncate them."
    );

    using Fs::operator()...;
};

// Deduction guide, for compilers without the deduction for aggregates of C++20, such as Clang
// before 17.
template <typename... Fs>
callback_overload(Fs...) -> callback_overload<Fs...>;

} // namespace fisk
