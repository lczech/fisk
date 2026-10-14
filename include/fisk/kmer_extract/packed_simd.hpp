#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <utility>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/simd.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "fisk/kmer_extract/packed.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from a PackedSequence, SIMD
// =================================================================================================

// Extractors for PackedSequence values, generic over the SIMD arch, see core/simd.hpp.
// for_each_kmer_packed_simd() accepts k in [1, 32] and calls `func(pos, vec, valid_count)` or
// `func(vec, valid_count)` with consecutive k-mers in the vector of the arch: `pos` is the start
// position of the k-mer in lane 0, and lane `j` holds the one starting at `pos + j`. Full vectors
// have `valid_count == Simd::lanes`; the final vector may be zero-padded in high lanes.
//
// The vectors hold raw k-mer words rather than Kmer values (core/kmer.hpp), with the conventions
// of the PackedSequence. Use the vector overloads of kmer_cast() to get Kmer values from them.

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// Four k-mers start in each byte of a packed sequence, at the positions 0 to 3 of that starting
// byte, and are extracted from the word of 8 bytes loaded there. Archs with shifts per lane, see
// SimdArchWithLaneShifts, hold consecutive k-mers in their lanes: the word of a starting byte is
// in four lanes, each shifted by the position of its k-mer. SSE2 can only shift all lanes by the
// same count. Its lanes hold the words of two consecutive starting bytes instead, which are
// shifted once per position, and then interleaved into the order of the k-mers.

// Number of lanes that hold k-mers of the same starting byte, as far as the arch has that many.
template <SimdArch Simd>
inline constexpr std::size_t packed_simd_lanes_per_byte_v = SimdArchWithLaneShifts<Simd> ? 4 : 1;

// Number of starting bytes that an iteration of the extraction loops consumes, and number of
// vectors that it emits.
template <SimdArch Simd>
inline constexpr std::size_t packed_simd_step_bytes_v
    = (Simd::lanes + packed_simd_lanes_per_byte_v<Simd> - 1) / packed_simd_lanes_per_byte_v<Simd>;
template <SimdArch Simd>
inline constexpr std::size_t packed_simd_step_vectors_v
    = 4 * packed_simd_step_bytes_v<Simd> / Simd::lanes;

// The 8 bytes at `p` as a word with the bases in order: the first base is in the lowest bits for
// kLSB, and in the highest bits for kMSB.
template <Layout L>
FISK_ALWAYS_INLINE
inline std::uint64_t load_packed_word_(std::uint8_t const* p) noexcept
{
    std::uint64_t word;
    std::memcpy(&word, p, 8);
    if constexpr (L == Layout::kMSB) {
        return byte_swap_64(word);
    } else if constexpr (L == Layout::kLSB) {
        return word;
    } else {
        static_assert(dependent_false_v<L>, "Unhandled Layout in load_packed_word_()");
    }
}

// The words of the starting bytes of an iteration, see load_packed_word_(): lane `j` has the word
// that starts `j / packed_simd_lanes_per_byte_v` bytes after `p`.
template <SimdArch Simd, Layout L>
FISK_ALWAYS_INLINE
inline typename Simd::simd_vector load_packed_words_(std::uint8_t const* p) noexcept
{
    constexpr std::size_t lanes_per_byte = packed_simd_lanes_per_byte_v<Simd>;

    // With a byte shuffle, swapping the bytes for kMSB is a single instruction on the vector, and
    // leaves the words to be loaded from memory as they are. Otherwise, swap each word on its own.
    constexpr Layout load_layout = SimdArchWithByteShuffle<Simd> ? Layout::kLSB : L;

    typename Simd::simd_vector words;
    if constexpr (Simd::lanes <= lanes_per_byte) {
        words = Simd::broadcast(load_packed_word_<load_layout>(p));
    } else {
        std::array<std::uint64_t, Simd::lanes> lane_words;
        for (std::size_t j = 0; j < Simd::lanes; ++j) {
            lane_words[j] = load_packed_word_<load_layout>(p + j / lanes_per_byte);
        }
        words = Simd::from_values(lane_words);
    }
    if constexpr (load_layout != L) {
        words = Simd::byte_swap_lanes(words);
    }
    return words;
}

// Shift counts for the vectors of an iteration, from the count for each position `local` in
// [0, 4) of a starting byte. With shifts per lane, `at[v]` has the counts for the lanes of vector
// `v` of the iteration. Otherwise, `at[local]` is the count for all lanes at that position.
template <SimdArch Simd, typename CountOf>
FISK_ALWAYS_INLINE
inline auto make_packed_simd_counts_(CountOf&& count_of)
{
    constexpr std::size_t vectors = packed_simd_step_vectors_v<Simd>;
    if constexpr (SimdArchWithLaneShifts<Simd>) {
        // Not a std::array, as GCC warns about vector types as template arguments.
        struct Counts { typename Simd::simd_vector at[vectors]; } counts;
        for (std::size_t v = 0; v < vectors; ++v) {
            std::array<std::uint64_t, Simd::lanes> lane_counts;
            for (std::size_t j = 0; j < Simd::lanes; ++j) {
                lane_counts[j] = count_of((v * Simd::lanes + j) % 4);
            }
            counts.at[v] = Simd::from_values(lane_counts);
        }
        return counts;
    } else {
        static_assert(vectors == 4, "Without shifts per lane, each vector is one position.");
        struct Counts { typename Simd::shift_count at[vectors]; } counts;
        for (std::size_t local = 0; local < vectors; ++local) {
            counts.at[local] = Simd::make_shift_count(count_of(local));
        }
        return counts;
    }
}

// Shift by a count of make_packed_simd_counts_().
template <SimdArch Simd, typename Count>
FISK_ALWAYS_INLINE
inline typename Simd::simd_vector packed_simd_shift_right_(
    typename Simd::simd_vector v, Count const& count
) noexcept {
    if constexpr (SimdArchWithLaneShifts<Simd>) {
        return Simd::shift_right_lanes(v, count);
    } else {
        return Simd::shift_right(v, count);
    }
}

template <SimdArch Simd, typename Count>
FISK_ALWAYS_INLINE
inline typename Simd::simd_vector packed_simd_shift_left_(
    typename Simd::simd_vector v, Count const& count
) noexcept {
    if constexpr (SimdArchWithLaneShifts<Simd>) {
        return Simd::shift_left_lanes(v, count);
    } else {
        return Simd::shift_left(v, count);
    }
}

// Emit the vectors of the iteration that starts at position `pos`, where `kmers(i)` computes them
// from the shift counts at index `i`, see make_packed_simd_counts_(). With shifts per lane, that
// is vector `i` of the iteration. Otherwise, it has position `i` of two starting bytes, and is
// interleaved with the next position to get consecutive k-mers.
template <SimdArch Simd, typename Kmers, typename Func>
FISK_ALWAYS_INLINE
inline void emit_packed_simd_step_(std::size_t pos, Kmers&& kmers, Func&& func)
{
    constexpr std::size_t lanes = Simd::lanes;
    if constexpr (SimdArchWithLaneShifts<Simd>) {
        for (std::size_t v = 0; v < packed_simd_step_vectors_v<Simd>; ++v) {
            invoke_kmer_vector_callback(func, pos + v * lanes, kmers(v), lanes);
        }
    } else {
        static_assert(lanes == 2, "Interleaving the positions needs an arch with two lanes.");
        auto const k0 = kmers(0);
        auto const k1 = kmers(1);
        auto const k2 = kmers(2);
        auto const k3 = kmers(3);
        invoke_kmer_vector_callback(func, pos + 0, Simd::interleave_low(k0, k1), lanes);
        invoke_kmer_vector_callback(func, pos + 2, Simd::interleave_low(k2, k3), lanes);
        invoke_kmer_vector_callback(func, pos + 4, Simd::interleave_high(k0, k1), lanes);
        invoke_kmer_vector_callback(func, pos + 6, Simd::interleave_high(k2, k3), lanes);
    }
}

// Emit the k-mers at the end of the sequence, which the extraction loops leave out, zero-padding
// the last vector. `vals[0]` is the k-mer starting at sequence position `base`, and the following
// ones are consecutive.
template <SimdArch Simd, typename Func>
inline void emit_tail_packed_(
    std::uint64_t const* vals, std::size_t n, std::size_t base, Func&& func
) {
    constexpr std::size_t lanes = Simd::lanes;
    std::size_t i = 0;
    while (i < n) {
        std::size_t const chunk = std::min(lanes, n - i);
        std::array<std::uint64_t, lanes> buf{};
        for (std::size_t j = 0; j < chunk; ++j) {
            buf[j] = vals[i + j];
        }
        invoke_kmer_vector_callback(func, base + i, Simd::from_values(buf), chunk);
        i += chunk;
    }
}

// -------------------------------------------------------------------------------------------------
//     Narrow Width
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extracts k in [1, 29] as vectors of k-mers.
 *
 * Any such k-mer is within the word of its starting byte, which needs one shift and a mask. With
 * `Masked` false, the mask is left out, so that the bits above each k-mer hold the bases next to
 * it, for callers that select bits with a mask of their own, such as spaced k-mer extraction.
 */
template <SimdArch Simd, bool Masked = true, Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_narrow_width_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 29) {
        throw_invalid_kmer_k_(29);
    }
    if (seq.length < k) {
        return;
    }

    using simd_vector = typename Simd::simd_vector;
    constexpr std::size_t step_bytes = packed_simd_step_bytes_v<Simd>;

    std::uint64_t const mask = (std::uint64_t{1} << (2 * k)) - 1u;
    unsigned const k32 = static_cast<unsigned>(k);

    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    // Every starting byte of the loop has a safe 8-byte read. Starting bytes that do not fill a
    // whole iteration are left to the tail.
    std::size_t fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );
    fast_bytes -= fast_bytes % step_bytes;

    // Right shift that moves the k-mer at position `local` of a starting byte to the low bits.
    simd_vector const mask_v = Simd::broadcast(mask);
    auto const counts = make_packed_simd_counts_<Simd>([&](std::size_t local) -> std::uint64_t {
        if constexpr (L == Layout::kMSB) {
            return (64u - 2u * k32) - 2u * local;
        } else if constexpr (L == Layout::kLSB) {
            return 2u * local;
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_kmer_packed_simd_narrow_width_()"
            );
        }
    });

    for (std::size_t b = 0; b < fast_bytes; b += step_bytes) {
        simd_vector const words = load_packed_words_<Simd, L>(&seq.data[b]);
        emit_packed_simd_step_<Simd>(4 * b, [&](std::size_t i) {
            simd_vector const kmers = packed_simd_shift_right_<Simd>(words, counts.at[i]);
            return Masked ? Simd::bitwise_and(kmers, mask_v) : kmers;
        }, func);
    }

    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = collect_kmers_packed_tail_<E, L>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    emit_tail_packed_<Simd>(tail_vals.data(), tail_n, 4 * fast_bytes, func);
}

// -------------------------------------------------------------------------------------------------
//     Full Width
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extracts k in [1, 32] as vectors of k-mers.
 *
 * A k-mer of more than 29 bases can reach into a ninth byte, so that this combines two words.
 * `Masked` is as for for_each_kmer_packed_simd_narrow_width_(). It only affects kLSB, as the
 * shift down of kMSB leaves nothing above the k-mer.
 */
template <SimdArch Simd, bool Masked = true, Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_full_width_(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }

    using simd_vector = typename Simd::simd_vector;
    constexpr std::size_t step_bytes = packed_simd_step_bytes_v<Simd>;

    std::uint64_t const mask = (k == 32) ? ~std::uint64_t{0} : ((std::uint64_t{1} << (2 * k)) - 1u);
    unsigned const k32 = static_cast<unsigned>(k);

    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    // An iteration also loads the words of the next iteration, see below. The bound keeps these
    // 8-byte reads within the packed input.
    std::size_t fast_bytes = (num_bytes >= 7 + step_bytes)
        ? std::min(full_bytes, num_bytes - 7 - step_bytes)
        : std::size_t{0}
    ;
    fast_bytes -= fast_bytes % step_bytes;

    // The k-mer at position `local` of a starting byte spans up to nine bytes: the word at the
    // starting byte, and a ninth byte, for which we take the word `step_bytes` after it. That is
    // the word of the same lane in the next iteration, so that each iteration only loads one
    // vector of words. The word is shifted by `2 * local`, so that the k-mer starts at the first
    // base, and the next word by the remaining `8 * step_bytes - 2 * local` in the other
    // direction, which fills in the bases that the first shift made room for. Where the two
    // overlap, they hold the same bases.
    //
    // kLSB has the first base in the lowest bits, and masks what exceeds the k-mer. kMSB has it
    // in the highest bits, and moves the k-mer down by `64 - 2k`, which leaves nothing above it.
    simd_vector const mask_v = Simd::broadcast(mask);
    auto const word_counts = make_packed_simd_counts_<Simd>(
        [](std::size_t local) -> std::uint64_t { return 2 * local; }
    );
    auto const next_counts = make_packed_simd_counts_<Simd>(
        [](std::size_t local) -> std::uint64_t { return 8 * step_bytes - 2 * local; }
    );
    auto const down_counts = make_packed_simd_counts_<Simd>(
        [&](std::size_t) -> std::uint64_t { return 64u - 2u * k32; }
    );

    if (fast_bytes > 0) {
        simd_vector next = load_packed_words_<Simd, L>(&seq.data[0]);
        for (std::size_t b = 0; b < fast_bytes; b += step_bytes) {
            simd_vector const words = next;
            next = load_packed_words_<Simd, L>(&seq.data[b + step_bytes]);
            emit_packed_simd_step_<Simd>(4 * b, [&](std::size_t i) {
                // Without shifts per lane, the first vector has position 0 of the starting bytes
                // in all lanes, where the words hold the whole k-mer already.
                bool const whole = !SimdArchWithLaneShifts<Simd> && i == 0;
                if constexpr (L == Layout::kMSB) {
                    simd_vector const kmers = whole ? words : Simd::bitwise_or(
                        packed_simd_shift_left_<Simd>(words, word_counts.at[i]),
                        packed_simd_shift_right_<Simd>(next, next_counts.at[i])
                    );
                    return packed_simd_shift_right_<Simd>(kmers, down_counts.at[i]);
                } else if constexpr (L == Layout::kLSB) {
                    simd_vector const kmers = whole ? words : Simd::bitwise_or(
                        packed_simd_shift_right_<Simd>(words, word_counts.at[i]),
                        packed_simd_shift_left_<Simd>(next, next_counts.at[i])
                    );
                    return Masked ? Simd::bitwise_and(kmers, mask_v) : kmers;
                } else {
                    static_assert(
                        dependent_false_v<L>,
                        "Unhandled Layout in for_each_kmer_packed_simd_full_width_()"
                    );
                }
            }, func);
        }
    }

    std::array<std::uint64_t, 64> tail_vals;
    std::size_t const tail_n = collect_kmers_packed_tail_<E, L>(
        seq, 4 * fast_bytes, p_max, k32, tail_vals
    );
    emit_tail_packed_<Simd>(tail_vals.data(), tail_n, 4 * fast_bytes, func);
}

// -------------------------------------------------------------------------------------------------
//     Dispatch
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extracts all k-mers for k in [1, 32] from a PackedSequence with the given SIMD arch,
 * yielding consecutive k-mers in the vectors of that arch.
 *
 * Func may be called either as `func(pos, vec, valid_count)` or as `func(vec, valid_count)`, see
 * invoke_kmer_vector_callback().
 *
 * @tparam Simd  SIMD arch, such as SimdAVX2, to be given explicitly.
 */
template <SimdArch Simd, Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd(
    PackedSequence<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (k <= 29) {
        for_each_kmer_packed_simd_narrow_width_<Simd>(seq, k, std::forward<Func>(func));
    } else {
        for_each_kmer_packed_simd_full_width_<Simd>(seq, k, std::forward<Func>(func));
    }
}

} // namespace fisk
