#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fisk/core/cast.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/simd.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from a PackedSequence, Bytewise, SIMD
// =================================================================================================

// Extractors for PackedSequence values, generic over the SIMD arch, see core/simd.hpp, and
// packed_bytewise_scalar.hpp for an overview of the extraction from a PackedSequence.
// for_each_kmer_packed_unchecked_bytewise_simd() accepts k in [1, 32] and calls `func(pos, kmers,
// valid_count)` with consecutive k-mers in a KmerVector of the arch: `pos` is the start position of
// the k-mer in lane 0, and lane `j` holds the one starting at `pos + j`, for `j < valid_count`.
// Only the last vector of a sequence can have a `valid_count` below `Simd::lanes`, and its other
// lanes are unspecified. for_each_kmer_packed_checked_bytewise_simd() extracts from a
// PackedSequenceWithValidity, and calls `func(pos, kmers, valid_lanes)` instead.
// for_each_valid_lane() and store_valid_lanes() take the single k-mers out of the vectors.
//
// The loops below work on the raw vectors of the arch, which the dispatchers at the end hand on as
// a KmerVector with the conventions of the PackedSequence.

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
inline constexpr std::size_t packed_bytewise_simd_lanes_per_byte_v
    = SimdArchWithLaneShifts<Simd> ? 4 : 1;

// Number of starting bytes that an iteration of the extraction loops consumes, and number of
// vectors that it emits.
template <SimdArch Simd>
inline constexpr std::size_t packed_bytewise_simd_step_bytes_v
    = (Simd::lanes + packed_bytewise_simd_lanes_per_byte_v<Simd> - 1)
    / packed_bytewise_simd_lanes_per_byte_v<Simd>;
template <SimdArch Simd>
inline constexpr std::size_t packed_bytewise_simd_step_vectors_v
    = 4 * packed_bytewise_simd_step_bytes_v<Simd> / Simd::lanes;

// The words of the starting bytes of an iteration, see load_packed_word_() (internal.hpp): lane `j`
// has the word that starts `j / packed_bytewise_simd_lanes_per_byte_v` bytes after `p`.
template <SimdArch Simd, Layout L>
FISK_ALWAYS_INLINE
inline typename Simd::simd_vector load_packed_words_(std::uint8_t const* p) noexcept
{
    constexpr std::size_t lanes_per_byte = packed_bytewise_simd_lanes_per_byte_v<Simd>;

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
inline auto make_packed_bytewise_simd_counts_(CountOf&& count_of)
{
    constexpr std::size_t vectors = packed_bytewise_simd_step_vectors_v<Simd>;
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

// Shift by a count of make_packed_bytewise_simd_counts_().
template <SimdArch Simd, typename Count>
FISK_ALWAYS_INLINE
inline typename Simd::simd_vector packed_bytewise_simd_shift_right_(
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
inline typename Simd::simd_vector packed_bytewise_simd_shift_left_(
    typename Simd::simd_vector v, Count const& count
) noexcept {
    if constexpr (SimdArchWithLaneShifts<Simd>) {
        return Simd::shift_left_lanes(v, count);
    } else {
        return Simd::shift_left(v, count);
    }
}

// Emit the vectors of the iteration that starts at position `pos` as `emit(pos, vec, validity)`,
// where `kmers(i)` computes them from the shift counts at index `i`, see
// make_packed_bytewise_simd_counts_(), and `validity_at(pos)` gives the validity of the k-mer in
// lane 0, see run_packed_bytewise_blocks_() (internal.hpp). With shifts per lane, `kmers(i)` is
// vector `i` of the iteration. Otherwise, it has position `i` of two starting bytes, and is
// interleaved with the next position to get consecutive k-mers.
template <SimdArch Simd, typename Kmers, typename ValidityAt, typename Emit>
FISK_ALWAYS_INLINE
inline void emit_packed_bytewise_simd_step_(
    std::size_t pos, Kmers&& kmers, ValidityAt const& validity_at, Emit&& emit
) {
    constexpr std::size_t lanes = Simd::lanes;
    auto const emit_at_ = [&](
        std::size_t at, typename Simd::simd_vector vec
    ) FISK_ALWAYS_INLINE_LAMBDA {
        emit(at, vec, validity_at(at));
    };
    if constexpr (SimdArchWithLaneShifts<Simd>) {
        for (std::size_t v = 0; v < packed_bytewise_simd_step_vectors_v<Simd>; ++v) {
            emit_at_(pos + v * lanes, kmers(v));
        }
    } else {
        static_assert(lanes == 2, "Interleaving the positions needs an arch with two lanes.");
        auto const k0 = kmers(0);
        auto const k1 = kmers(1);
        auto const k2 = kmers(2);
        auto const k3 = kmers(3);
        emit_at_(pos + 0, Simd::interleave_low(k0, k1));
        emit_at_(pos + 2, Simd::interleave_low(k2, k3));
        emit_at_(pos + 4, Simd::interleave_high(k0, k1));
        emit_at_(pos + 6, Simd::interleave_high(k2, k3));
    }
}

// The emit of the steps of the loops below, see emit_packed_bytewise_simd_step_(), which passes
// each vector of the k-mers from `pos` on to `emit` with the validity of the k-mers from `pos` on.
// With validity bits, the vector goes on with its valid lanes: all of them in a block of valid
// nucleotides, and otherwise the low bits of the valid starts, see run_packed_bytewise_blocks_()
// (internal.hpp). With PackedValidity::kNucleotides, the vector goes on with the validity as it is
// instead, AllValid or the validity window from `pos` on, for the caller to test its lanes against.
// Without validity bits, it goes on with its count of k-mers, which is below all lanes only in the
// step after the k-mers within the sequence. There, the count comes from the length, and the
// vectors after the last k-mer are left out.
template <
    SimdArch Simd, PackedValidity Form = PackedValidity::kStarts,
    PackedSequenceType Seq, typename Emit
>
FISK_ALWAYS_INLINE
inline auto packed_bytewise_simd_emit_(Seq const& seq, std::size_t k, Emit& emit)
{
    std::size_t const p_max = seq.length() - k;
    return [&emit, p_max](
        std::size_t pos, typename Simd::simd_vector vec, auto validity
    ) FISK_ALWAYS_INLINE_LAMBDA {
        (void) p_max;
        if constexpr (PackedSequenceWithValidityType<Seq> && Form != PackedValidity::kStarts) {
            // The validity window as it is, for the spaced k-mer extraction, which tests the lanes
            // of each mask against it.
            emit(pos, vec, validity);
        } else if constexpr (PackedSequenceWithValidityType<Seq>) {
            std::uint64_t const all_lanes = (std::uint64_t{1} << Simd::lanes) - 1u;
            if constexpr (std::is_same_v<decltype(validity), AllValid>) {
                emit(pos, vec, SimdVectorValidLanes<Simd>{all_lanes});
            } else {
                emit(pos, vec, SimdVectorValidLanes<Simd>{validity & all_lanes});
            }
        } else if constexpr (std::is_same_v<decltype(validity), AllValid>) {
            emit(pos, vec, Simd::lanes);
        } else if (pos <= p_max) {
            std::size_t const count = std::min(Simd::lanes, p_max + 1 - pos);

            // The benchmarks sum whole vectors, also this one, and need its lanes past the count to
            // be zero, so that the sum matches that of the other extraction functions. Only built
            // for them, as the do_not_optimize barriers, see internal.hpp.
            #if defined(FISK_BENCHMARK_SINK_MODE)
            alignas(64) std::uint64_t values[Simd::lanes];
            Simd::store_aligned(vec, values);
            for (std::size_t j = count; j < Simd::lanes; ++j) {
                values[j] = 0;
            }
            vec = Simd::load_aligned(values);
            #endif

            emit(pos, vec, count);
        }
    };
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
 * The vectors are emitted as `emit(pos, vec, valid_count)`, as for
 * for_each_kmer_packed_unchecked_bytewise_simd(). For a sequence with validity bits, see
 * PackedSequenceWithValidityType (core/types.hpp), they are emitted as `emit(pos, vec,
 * valid_lanes)` instead, as for for_each_kmer_packed_checked_bytewise_simd(), or, with `Form`
 * PackedValidity::kNucleotides, as `emit(pos, vec, validity)`, see packed_bytewise_simd_emit_().
 */
template <
    SimdArch Simd, bool Masked = true, PackedValidity Form = PackedValidity::kStarts,
    PackedSequenceType Seq, typename Emit
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_bytewise_simd_narrow_width_(
    Seq const& seq, std::size_t k, Emit&& emit
) {
    if (k == 0 || k > 29) {
        throw_invalid_kmer_k_(29);
    }
    if (seq.length() < k) {
        return;
    }

    // A step covers the k-mers that start in `step_bytes` bytes, see
    // packed_bytewise_simd_step_bytes_v.
    using simd_vector = typename Simd::simd_vector;
    constexpr std::size_t step_bytes = packed_bytewise_simd_step_bytes_v<Simd>;

    // The 2k bits of a k-mer in the codes. The codes are read up into their padding, see
    // run_packed_bytewise_blocks_() (internal.hpp).
    constexpr Layout L = Seq::layout;
    std::uint64_t const mask = width_mask_(k);
    unsigned const k32 = static_cast<unsigned>(k);
    std::uint8_t const* const codes = seq.codes().data();
    auto const emit_ = packed_bytewise_simd_emit_<Simd, Form>(seq, k, emit);

    // Right shift that moves the k-mer at position `local` of a starting byte to the low bits.
    simd_vector const mask_v = Simd::broadcast(mask);
    auto const counts = make_packed_bytewise_simd_counts_<Simd>(
        [&](std::size_t local) -> std::uint64_t {
            if constexpr (L == Layout::kMSB) {
                return (64u - 2u * k32) - 2u * local;
            } else if constexpr (L == Layout::kLSB) {
                return 2u * local;
            } else {
                static_assert(
                    dependent_false_v<L>,
                    "Unhandled Layout in for_each_kmer_packed_bytewise_simd_narrow_width_()"
                );
            }
        }
    );

    // One step: the k-mers that start in the `step_bytes` bytes from `b`, as vectors of consecutive
    // k-mers, each with the validity of its k-mer in lane 0, see packed_bytewise_simd_emit_().
    auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
        simd_vector const words = load_packed_words_<Simd, L>(codes + b);
        emit_packed_bytewise_simd_step_<Simd>(4 * b, [&](std::size_t i) {
            simd_vector const kmers = packed_bytewise_simd_shift_right_<Simd>(words, counts.at[i]);
            return Masked ? Simd::bitwise_and(kmers, mask_v) : kmers;
        }, validity_at, emit_);
    };

    // The steps over all bytes in which a k-mer starts, see run_packed_bytewise_blocks_()
    // (internal.hpp).
    run_packed_bytewise_blocks_<step_bytes, Form>(seq, k, step_);
}

// -------------------------------------------------------------------------------------------------
//     Full Width
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extracts k in [1, 32] as vectors of k-mers.
 *
 * A k-mer of more than 29 bases can reach into a ninth byte, so that this combines two words.
 * `Masked`, `Form`, and `emit` are as for for_each_kmer_packed_bytewise_simd_narrow_width_().
 * `Masked` only affects kLSB, as the shift down of kMSB leaves nothing above the k-mer.
 */
template <
    SimdArch Simd, bool Masked = true, PackedValidity Form = PackedValidity::kStarts,
    PackedSequenceType Seq, typename Emit
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_bytewise_simd_full_width_(
    Seq const& seq, std::size_t k, Emit&& emit
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.length() < k) {
        return;
    }

    // As in the narrow-width loop.
    using simd_vector = typename Simd::simd_vector;
    constexpr std::size_t step_bytes = packed_bytewise_simd_step_bytes_v<Simd>;

    constexpr Layout L = Seq::layout;
    std::uint64_t const mask = width_mask_(k);
    unsigned const k32 = static_cast<unsigned>(k);
    std::uint8_t const* const codes = seq.codes().data();
    auto const emit_ = packed_bytewise_simd_emit_<Simd, Form>(seq, k, emit);

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
    auto const word_counts = make_packed_bytewise_simd_counts_<Simd>(
        [](std::size_t local) -> std::uint64_t { return 2 * local; }
    );
    auto const next_counts = make_packed_bytewise_simd_counts_<Simd>(
        [](std::size_t local) -> std::uint64_t { return 8 * step_bytes - 2 * local; }
    );
    auto const down_counts = make_packed_bytewise_simd_counts_<Simd>(
        [&](std::size_t) -> std::uint64_t { return 64u - 2u * k32; }
    );

    // One step, as in the narrow-width loop. `next` holds the words of the next step, which
    // the step loads for its ninth bytes, and which the next step then starts with.
    simd_vector next = load_packed_words_<Simd, L>(codes);
    auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
        simd_vector const words = next;
        next = load_packed_words_<Simd, L>(codes + b + step_bytes);
        emit_packed_bytewise_simd_step_<Simd>(4 * b, [&](std::size_t i) {
            // Without shifts per lane, the first vector has position 0 of the starting bytes
            // in all lanes, where the words hold the whole k-mer already.
            bool const whole = !SimdArchWithLaneShifts<Simd> && i == 0;
            if constexpr (L == Layout::kMSB) {
                simd_vector const kmers = whole ? words : Simd::bitwise_or(
                    packed_bytewise_simd_shift_left_<Simd>(words, word_counts.at[i]),
                    packed_bytewise_simd_shift_right_<Simd>(next, next_counts.at[i])
                );
                return packed_bytewise_simd_shift_right_<Simd>(kmers, down_counts.at[i]);
            } else if constexpr (L == Layout::kLSB) {
                simd_vector const kmers = whole ? words : Simd::bitwise_or(
                    packed_bytewise_simd_shift_right_<Simd>(words, word_counts.at[i]),
                    packed_bytewise_simd_shift_left_<Simd>(next, next_counts.at[i])
                );
                return Masked ? Simd::bitwise_and(kmers, mask_v) : kmers;
            } else {
                static_assert(
                    dependent_false_v<L>,
                    "Unhandled Layout in for_each_kmer_packed_bytewise_simd_full_width_()"
                );
            }
        }, validity_at, emit_);
    };

    // As in the narrow-width loop.
    run_packed_bytewise_blocks_<step_bytes, Form>(seq, k, step_);
}

// -------------------------------------------------------------------------------------------------
//     Dispatch
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extracts all k-mers for k in [1, 32] from a PackedSequence with the given SIMD arch,
 * yielding consecutive k-mers in the vectors of that arch.
 *
 * Func is called as `func(pos, kmers, valid_count)`, with a KmerVector<E, L, Simd>: lane `j` of
 * `kmers` holds the k-mer starting at `pos + j`, for `j < valid_count`. The single k-mers can be
 * taken out with for_each_valid_lane() or store_valid_lanes().
 *
 * A PackedSequenceWithValidity is taken as well, ignoring its validity bits, see
 * for_each_kmer_packed_checked_bytewise_simd() for its valid k-mers only.
 *
 * @tparam Simd  SIMD arch, such as SimdAVX2, to be given explicitly.
 */
template <SimdArch Simd, PackedSequenceType Seq, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_unchecked_bytewise_simd(Seq const& seq, std::size_t k, Func&& func)
{
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Seq::layout;
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    // Unchecked, all k-mers are valid, and the vectors go to `func` with their count of k-mers.
    auto const emit = [&](
        std::size_t pos, typename Simd::simd_vector vec, std::size_t valid_count
    ) FISK_ALWAYS_INLINE_LAMBDA {
        invoke_kmer_vector_callback(
            func, pos, kmer_vector_cast_unchecked<E, L, Simd>(vec), valid_count
        );
    };
    if (k <= 29) {
        for_each_kmer_packed_bytewise_simd_narrow_width_<Simd>(without_validity_(seq), k, emit);
    } else {
        for_each_kmer_packed_bytewise_simd_full_width_<Simd>(without_validity_(seq), k, emit);
    }
}

/**
 * @brief Extracts the valid k-mers for k in [1, 32] from a PackedSequenceWithValidity with the
 * given SIMD arch, yielding consecutive k-mers in the vectors of that arch, with their valid lanes.
 *
 * As for_each_kmer_packed_unchecked_bytewise_simd(), checking the validity of the nucleotides by
 * the validity bits of `seq`, see pack_sequence_checked(). Func is called as `func(pos, kmers,
 * valid_lanes)`, with a KmerVector<E, L, Simd>: lane `j` of `kmers` holds the k-mer starting at
 * `pos + j`, and is valid if it is set in `valid_lanes`, see SimdVectorValidLanes. That covers the
 * end of the sequence as well, where the last vector can have lanes beyond it. The valid k-mers can
 * be taken out with for_each_valid_lane() or store_valid_lanes(), or, for example with AVX512,
 * stored contiguously with
 *
 *     _mm512_mask_compressstoreu_epi64(out, valid_lanes.lane_mask(), kmers.value);
 *
 * For single k-mers, use for_each_kmer_packed_checked_bytewise_scalar(), which extracts them
 * directly, instead of taking them out of the vectors one at a time.
 *
 * The loop tests a block of 32 positions at once, and the blocks whose nucleotides are all valid
 * get a constant `valid_lanes`, as do most blocks of sequences whose invalid nucleotides come in
 * runs, such as assemblies.
 *
 * @tparam Simd  SIMD arch, such as SimdAVX2, to be given explicitly.
 */
template <SimdArch Simd, Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_packed_checked_bytewise_simd(
    PackedSequenceWithValidity<E, L> const& seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    auto const emit = [&](
        std::size_t pos, typename Simd::simd_vector vec, SimdVectorValidLanes<Simd> valid_lanes
    ) FISK_ALWAYS_INLINE_LAMBDA {
        invoke_kmer_vector_valid_lanes_callback(
            func, pos, kmer_vector_cast_unchecked<E, L, Simd>(vec), valid_lanes
        );
    };
    if (k <= 29) {
        for_each_kmer_packed_bytewise_simd_narrow_width_<Simd>(seq, k, emit);
    } else {
        for_each_kmer_packed_bytewise_simd_full_width_<Simd>(seq, k, emit);
    }
}

} // namespace fisk
