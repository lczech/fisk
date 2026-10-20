#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <type_traits>
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
// for_each_kmer_packed_simd() accepts k in [1, 32] and calls `func(pos, vec, valid_count)` with
// consecutive k-mers in the vector of the arch: `pos` is the start position of the k-mer in lane 0,
// and lane `j` holds the one starting at `pos + j`. Full vectors have `valid_count == Simd::lanes`;
// the final vector may be zero-padded in high lanes. for_each_kmer_packed_checked_simd() extracts
// from a PackedSequenceWithValidity, and calls `func(pos, vec, valid_lanes)` instead.
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

// The words of the starting bytes of an iteration, see load_packed_word_() (packed.hpp): lane `j`
// has the word that starts `j / packed_simd_lanes_per_byte_v` bytes after `p`.
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

// Emit the vectors of the iteration that starts at position `pos` as `emit(pos, vec)`, where
// `kmers(i)` computes them from the shift counts at index `i`, see make_packed_simd_counts_(). With
// shifts per lane, that is vector `i` of the iteration. Otherwise, it has position `i` of two
// starting bytes, and is interleaved with the next position to get consecutive k-mers.
template <SimdArch Simd, typename Kmers, typename Emit>
FISK_ALWAYS_INLINE
inline void emit_packed_simd_step_(std::size_t pos, Kmers&& kmers, Emit&& emit)
{
    constexpr std::size_t lanes = Simd::lanes;
    if constexpr (SimdArchWithLaneShifts<Simd>) {
        for (std::size_t v = 0; v < packed_simd_step_vectors_v<Simd>; ++v) {
            emit(pos + v * lanes, kmers(v));
        }
    } else {
        static_assert(lanes == 2, "Interleaving the positions needs an arch with two lanes.");
        auto const k0 = kmers(0);
        auto const k1 = kmers(1);
        auto const k2 = kmers(2);
        auto const k3 = kmers(3);
        emit(pos + 0, Simd::interleave_low(k0, k1));
        emit(pos + 2, Simd::interleave_low(k2, k3));
        emit(pos + 4, Simd::interleave_high(k0, k1));
        emit(pos + 6, Simd::interleave_high(k2, k3));
    }
}

// The valid lanes of a vector of consecutive k-mers, from the validity window `w` at the k-mer in
// lane 0, see packed.hpp: lane `j` is valid if the bits of `test` are all set in `w >> j`. `test`
// has a bit for each nucleotide of a k-mer that has to be valid, the low k bits for k-mers, or
// just bit 0 for bits that are set for each valid k-mer.
template <SimdArch Simd>
FISK_ALWAYS_INLINE
inline SimdValidLanes<Simd> valid_lanes_(std::uint64_t w, std::uint64_t test) noexcept
{
    typename Simd::simd_vector windows;
    if constexpr (SimdArchWithLaneShifts<Simd>) {
        std::array<std::uint64_t, Simd::lanes> lane_index;
        for (std::size_t j = 0; j < Simd::lanes; ++j) {
            lane_index[j] = j;
        }
        windows = Simd::shift_right_lanes(Simd::broadcast(w), Simd::from_values(lane_index));
    } else {
        std::array<std::uint64_t, Simd::lanes> lane_windows;
        for (std::size_t j = 0; j < Simd::lanes; ++j) {
            lane_windows[j] = w >> j;
        }
        windows = Simd::from_values(lane_windows);
    }
    auto const test_v = Simd::broadcast(test);
    return {Simd::compare_eq_lanes(Simd::bitwise_and(windows, test_v), test_v)};
}

// All lanes, for the k-mers of a block of valid nucleotides, see run_packed_blocks_() (packed.hpp).
template <SimdArch Simd>
FISK_ALWAYS_INLINE
inline SimdValidLanes<Simd> valid_lanes_(AllValid, std::uint64_t) noexcept
{
    return {Simd::lane_mask_all()};
}

// Store the k-mers from `first` on into `kmers`, see emit_packed_simd_tail_(), with the rolling
// loop of packed.hpp, and return the bits of the valid ones: bit `i` for the k-mer at `first + i`.
//
// There are at most 56 k-mers after the fast region of the loops below, for any length of the
// sequence: the region ends at most 40 positions before the end, as its 8-byte loads have to be
// within the codes, and at most 56 for the checked loops, see checked_fast_bytes_() (packed.hpp).
// With the zero padding of the last vector, they fit into 64 entries.
template <bool FullWidth, PackedSequenceType Seq>
FISK_NOINLINE
std::uint64_t collect_kmers_packed_simd_tail_(
    Seq const& seq, std::size_t k, std::size_t first, std::array<std::uint64_t, 64>& kmers
) {
    // The emit of the rolling loop: store the k-mer at its index from `first`, and set its bit.
    std::uint64_t bits = 0;
    auto const store_ = [&](std::size_t pos, std::uint64_t kmer) FISK_ALWAYS_INLINE_LAMBDA {
        kmers[pos - first] = kmer;
        bits |= std::uint64_t{1} << (pos - first);
    };
    if constexpr (FullWidth) {
        for_each_kmer_packed_rolling_full_width_(seq, k, first, store_);
    } else {
        for_each_kmer_packed_rolling_narrow_width_(seq, k, first, store_);
    }
    return bits;
}

// Emit the k-mers from `first` to `p_max`, which the extraction loops leave out, as vectors as in
// the loops below, zero-padding the last one. The k-mers are stored first, out of line, and
// emitted from a single loop, so that the code of the extraction loops does not grow by the tail.
template <SimdArch Simd, bool FullWidth, PackedSequenceType Seq, typename Emit>
inline void emit_packed_simd_tail_(
    Seq const& seq, std::size_t k, std::size_t first, std::size_t p_max, Emit&& emit
) {
    constexpr std::size_t lanes = Simd::lanes;
    if (first > p_max) {
        return;
    }

    // The `n` k-mers from `first` on, zero where the rolling loop emits none: for invalid k-mers,
    // and after `p_max`, where the last vector can reach.
    std::array<std::uint64_t, 64> kmers{};
    std::uint64_t const bits = collect_kmers_packed_simd_tail_<FullWidth>(seq, k, first, kmers);
    std::size_t const n = p_max + 1 - first;
    assert(n + lanes - 1 <= kmers.size());

    // One vector of `lanes` consecutive k-mers at a time, of which `count` are in the sequence.
    // With validity bits, its valid lanes are those whose bit is set in `bits`.
    for (std::size_t i = 0; i < n; i += lanes) {
        std::array<std::uint64_t, lanes> vec;
        std::copy_n(kmers.begin() + static_cast<std::ptrdiff_t>(i), lanes, vec.begin());
        if constexpr (PackedSequenceWithValidityType<Seq>) {
            emit(first + i, Simd::from_values(vec), valid_lanes_<Simd>(bits >> i, 1));
        } else {
            emit(first + i, Simd::from_values(vec), std::min(lanes, n - i));
        }
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
 *
 * The vectors are emitted as `emit(pos, vec, valid_count)`, as for for_each_kmer_packed_simd().
 * For a sequence with validity bits, see PackedSequenceWithValidityType (core/types.hpp), they are
 * emitted as `emit(pos, vec, valid_lanes)` instead, as for for_each_kmer_packed_checked_simd().
 */
template <SimdArch Simd, bool Masked = true, PackedSequenceType Seq, typename Emit>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_narrow_width_(Seq const& seq, std::size_t k, Emit&& emit)
{
    if (k == 0 || k > 29) {
        throw_invalid_kmer_k_(29);
    }
    if (seq.length < k) {
        return;
    }

    // A step covers the k-mers that start in `step_bytes` bytes, see packed_simd_step_bytes_v.
    using simd_vector = typename Simd::simd_vector;
    constexpr std::size_t step_bytes = packed_simd_step_bytes_v<Simd>;

    // The 2k bits of a k-mer in the codes, and its k bits in the validity bits.
    constexpr Layout L = Seq::layout;
    std::uint64_t const mask = (std::uint64_t{1} << (2 * k)) - 1u;
    std::uint64_t const kbits = (std::uint64_t{1} << k) - 1u;
    unsigned const k32 = static_cast<unsigned>(k);

    // As in the aligned loops of packed.hpp: the last k-mer starts at `p_max`, so that the 4 k-mers
    // of each of the first `full_bytes` bytes are within the sequence.
    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    // The fast region: every starting byte of the loop has a safe 8-byte read. Starting bytes that
    // do not fill a whole step are left to the tail.
    std::size_t fast_bytes = std::min(
        full_bytes, num_bytes >= 8 ? num_bytes - 8 : std::size_t{0}
    );
    if constexpr (PackedSequenceWithValidityType<Seq>) {
        fast_bytes = checked_fast_bytes_(fast_bytes, (seq.length + 7) / 8);
    }
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

    // One step: the k-mers that start in the `step_bytes` bytes from `b`, as vectors of consecutive
    // k-mers, each with the validity of its k-mer in lane 0, see run_packed_blocks_() (packed.hpp).
    auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
        simd_vector const words = load_packed_words_<Simd, L>(&seq.data[b]);
        emit_packed_simd_step_<Simd>(4 * b, [&](std::size_t i) {
            simd_vector const kmers = packed_simd_shift_right_<Simd>(words, counts.at[i]);
            return Masked ? Simd::bitwise_and(kmers, mask_v) : kmers;
        }, [&](std::size_t pos, simd_vector vec) FISK_ALWAYS_INLINE_LAMBDA {
            if constexpr (PackedSequenceWithValidityType<Seq>) {
                emit(pos, vec, valid_lanes_<Simd>(validity_at(pos), kbits));
            } else {
                emit(pos, vec, Simd::lanes);
            }
        });
    };

    // The steps over the fast region, with one validity test per block of 32 positions. Then the
    // k-mers after it, from the rolling loop, as vectors, see emit_packed_simd_tail_().
    run_packed_blocks_<step_bytes>(seq, fast_bytes, (std::uint64_t{1} << (31 + k)) - 1u, step_);
    emit_packed_simd_tail_<Simd, false>(seq, k, 4 * fast_bytes, p_max, emit);
}

// -------------------------------------------------------------------------------------------------
//     Full Width
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extracts k in [1, 32] as vectors of k-mers.
 *
 * A k-mer of more than 29 bases can reach into a ninth byte, so that this combines two words.
 * `Masked` and `emit` are as for for_each_kmer_packed_simd_narrow_width_().
 * `Masked` only affects kLSB, as the shift down of kMSB leaves nothing above the k-mer.
 */
template <SimdArch Simd, bool Masked = true, PackedSequenceType Seq, typename Emit>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_simd_full_width_(Seq const& seq, std::size_t k, Emit&& emit)
{
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length < k) {
        return;
    }

    // As in the narrow-width loop.
    using simd_vector = typename Simd::simd_vector;
    constexpr std::size_t step_bytes = packed_simd_step_bytes_v<Simd>;

    constexpr Layout L = Seq::layout;
    std::uint64_t const mask = (k == 32) ? ~std::uint64_t{0} : ((std::uint64_t{1} << (2 * k)) - 1u);
    std::uint64_t const kbits = (std::uint64_t{1} << k) - 1u;
    unsigned const k32 = static_cast<unsigned>(k);

    std::size_t const p_max = seq.length - k;
    std::size_t const full_bytes = (p_max + 1) / 4;
    std::size_t const num_bytes  = (seq.length + 3) / 4;
    // The fast region: a step also loads the words of the next step, see below. The bound keeps
    // these 8-byte reads within the packed input.
    std::size_t fast_bytes = (num_bytes >= 7 + step_bytes)
        ? std::min(full_bytes, num_bytes - 7 - step_bytes)
        : std::size_t{0}
    ;
    if constexpr (PackedSequenceWithValidityType<Seq>) {
        fast_bytes = checked_fast_bytes_(fast_bytes, (seq.length + 7) / 8);
    }
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
        // One step, as in the narrow-width loop. `next` holds the words of the next step, which
        // the step loads for its ninth bytes, and which the next step then starts with.
        simd_vector next = load_packed_words_<Simd, L>(&seq.data[0]);
        auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
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
            }, [&](std::size_t pos, simd_vector vec) FISK_ALWAYS_INLINE_LAMBDA {
                if constexpr (PackedSequenceWithValidityType<Seq>) {
                    emit(pos, vec, valid_lanes_<Simd>(validity_at(pos), kbits));
                } else {
                    emit(pos, vec, Simd::lanes);
                }
            });
        };
        // The steps over the fast region, as in the narrow-width loop.
        run_packed_blocks_<step_bytes>(
            seq, fast_bytes, (std::uint64_t{1} << (31 + k)) - 1u, step_
        );
    }

    // The k-mers after the fast region, from the rolling loop, as vectors, see
    // emit_packed_simd_tail_().
    emit_packed_simd_tail_<Simd, true>(seq, k, 4 * fast_bytes, p_max, emit);
}

// -------------------------------------------------------------------------------------------------
//     Dispatch
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extracts all k-mers for k in [1, 32] from a PackedSequence with the given SIMD arch,
 * yielding consecutive k-mers in the vectors of that arch.
 *
 * Func is called as `func(pos, vec, valid_count)`, see invoke_kmer_vector_callback().
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
    // Unchecked, all k-mers are valid, and the vectors go to `func` with their count of k-mers.
    auto const emit = [&](
        std::size_t pos, typename Simd::simd_vector vec, std::size_t valid_count
    ) FISK_ALWAYS_INLINE_LAMBDA {
        invoke_kmer_vector_callback(func, pos, vec, valid_count);
    };
    if (k <= 29) {
        for_each_kmer_packed_simd_narrow_width_<Simd>(seq, k, emit);
    } else {
        for_each_kmer_packed_simd_full_width_<Simd>(seq, k, emit);
    }
}

/**
 * @brief Extracts the valid k-mers for k in [1, 32] from a PackedSequenceWithValidity with the
 * given SIMD arch, yielding consecutive k-mers in the vectors of that arch, with their valid lanes.
 *
 * As for_each_kmer_packed_simd(), checking the validity of the nucleotides by the validity bits of
 * `seq`, see pack_sequence_with_validity(). Func is called as `func(pos, vec, valid_lanes)`: lane
 * `j` of `vec` holds the k-mer starting at `pos + j`, and is valid if it is set in `valid_lanes`,
 * see SimdValidLanes. That covers the end of the sequence as well, where the last vector can have
 * lanes beyond it. For example, with AVX512, the valid k-mers can be stored contiguously with
 *
 *     _mm512_mask_compressstoreu_epi64(out, valid_lanes.value, vec);
 *
 * For single k-mers, use for_each_kmer_packed_checked_aligned(), which extracts them directly,
 * instead of taking them out of the vectors one at a time.
 *
 * The loop tests a block of 32 positions at once, and the blocks whose nucleotides are all valid
 * get a constant `valid_lanes`, as do most blocks of sequences whose invalid nucleotides come in
 * runs, such as assemblies.
 *
 * @tparam Simd  SIMD arch, such as SimdAVX2, to be given explicitly.
 * @throws std::invalid_argument if `seq.valid` does not have `(seq.length + 7) / 8` bytes.
 */
template <SimdArch Simd, Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_checked_simd(
    PackedSequenceWithValidity<E, L> const& seq, std::size_t k, Func&& func
) {
    check_validity_size_(seq);
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    auto const emit = [&](
        std::size_t pos, typename Simd::simd_vector vec, SimdValidLanes<Simd> valid_lanes
    ) FISK_ALWAYS_INLINE_LAMBDA {
        invoke_kmer_valid_lanes_callback(func, pos, vec, valid_lanes);
    };
    if (k <= 29) {
        for_each_kmer_packed_simd_narrow_width_<Simd>(seq, k, emit);
    } else {
        for_each_kmer_packed_simd_full_width_<Simd>(seq, k, emit);
    }
}

} // namespace fisk
