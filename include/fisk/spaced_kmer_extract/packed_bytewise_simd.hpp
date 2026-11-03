#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/internal.hpp"
#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "fisk/spaced_kmer_extract/ascii_rolling_simd.hpp"
#include "fisk/spaced_kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Extraction from a PackedSequence, Bytewise, SIMD
// =================================================================================================

// The SIMD counterpart of for_each_spaced_kmer_packed_unchecked_bytewise_scalar()
// (packed_bytewise_scalar.hpp), for the bit extraction kernels of a SIMD arch
// (bit_extract/simd.hpp), such as BitExtractKernelButterfly<SimdAVX2>, created via
// make_spaced_kmer_kernel(). The loop below holds the windows of consecutive spans in the lanes of
// its vectors, as the k-mer extraction from packed sequences holds its k-mers, see
// kmer_extract/packed_bytewise_simd.hpp, and each kernel extracts its spaced k-mers from all of
// them at once. As in
// for_each_spaced_kmer_packed_unchecked_bytewise_scalar(), the windows still hold the neighboring
// bases above the span, which the bit extraction does not take.
//
// The callback is called with vectors, as `callback(pos, mask_idx, spaced_kmers, valid_count)`,
// or, for a single kernel, also as `callback(pos, spaced_kmers, valid_count)`: lane `j` of the
// SpacedKmerVector `spaced_kmers` holds the spaced k-mer of mask `mask_idx` at position `pos + j`,
// for the first `valid_count` lanes, and the remaining lanes are unspecified. Per vector of
// positions, the masks are emitted in their order. The spaced k-mers have the Encoding and Layout
// of the sequence. For single spaced k-mers, use for_each_valid_lane() in the callback.
//
// The checked counterpart, for_each_spaced_kmer_packed_checked_bytewise_simd(), takes a
// PackedSequenceWithValidity, and calls back with the valid lanes of each vector instead of a
// count, as `callback(pos, mask_idx, spaced_kmers, valid_lanes)`, or, for a single kernel, also as
// `callback(pos, spaced_kmers, valid_lanes)`, as for_each_spaced_kmer_ascii_checked_rolling_simd()
// (ascii_rolling_simd.hpp): lane `j` is valid if all positions that mask `mask_idx` keeps are valid
// nucleotides in the span at `pos + j`. The vectors cover all positions in their order, and those
// after the last spaced k-mer, up to the end of the last step of the loop, have no valid lanes.

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// Extract and emit the spaced k-mers of all kernels from the `windows` of the spans at `pos` and
// after, one kernel after the other: a loop with the compile-time trip count N, which the compiler
// unrolls where the kernels are cheap, as for_each_mask_() (internal.hpp). `Vector` is deduced
// instead of taken from the kernel, as GCC warns about vector types that are spelled out as
// template arguments.
template <
    Encoding E, Layout L, typename Kernel, std::size_t N, typename Callback, typename Vector
>
FISK_ALWAYS_INLINE
inline void emit_spaced_kmers_packed_unchecked_bytewise_simd_(
    std::span<Kernel, N> kernels, std::size_t pos, Vector windows,
    std::size_t valid_count, Callback& callback
) {
    using Simd = typename Kernel::bit_extract_type::simd_arch;
    static_assert(
        SpacedKmerVectorCallback<Callback&, E, L, Simd, N == 1>,
        "Callback must be callable as callback(pos, mask_idx, spaced_kmers, valid_count), or, "
        "for a single kernel, as callback(pos, spaced_kmers, valid_count)."
    );
    for (std::size_t m = 0; m < N; ++m) {
        invoke_spaced_kmer_vector_callback<E, L, Simd, N == 1>(
            callback, m, pos,
            spaced_kmer_vector_cast_unchecked<E, L, Simd>(
                kernels[m].mask_pieces[0].bit_extract(windows)
            ),
            valid_count
        );
    }
}

// As emit_spaced_kmers_packed_unchecked_bytewise_simd_(), with the valid lanes of each kernel, from
// the `validity` of the windows, see run_packed_bytewise_blocks_() (kmer_extract/internal.hpp):
// AllValid, or the validity window at `pos`, as PackedValidity::kNucleotides, against which each
// lane tests the positions that the mask keeps, its validity_mask, see SpacedKmerMask.
template <
    Encoding E, Layout L, typename Kernel, std::size_t N, typename Callback, typename Vector,
    typename Validity
>
FISK_ALWAYS_INLINE
inline void emit_spaced_kmers_packed_checked_bytewise_simd_(
    std::span<Kernel, N> kernels, std::size_t pos, Vector windows,
    Validity validity, Callback& callback
) {
    using Simd = typename Kernel::bit_extract_type::simd_arch;
    constexpr std::size_t lanes = Simd::lanes;
    static_assert(
        SpacedKmerVectorValidLanesCallback<Callback&, E, L, Simd, N == 1>,
        "Callback must be callable as callback(pos, mask_idx, spaced_kmers, valid_lanes), or, "
        "for a single kernel, as callback(pos, spaced_kmers, valid_lanes)."
    );
    auto const lanes_of = [&](std::size_t m) FISK_ALWAYS_INLINE_LAMBDA -> std::uint64_t {
        if constexpr (std::is_same_v<Validity, AllValid>) {
            (void) m;
            return (std::uint64_t{1} << lanes) - 1u;
        } else {
            std::uint64_t const vm = kernels[m].validity_pieces[0];
            std::uint64_t bits = 0;
            for (std::size_t j = 0; j < lanes; ++j) {
                bits |= static_cast<std::uint64_t>(((validity >> j) & vm) == vm) << j;
            }
            return bits;
        }
    };
    for (std::size_t m = 0; m < N; ++m) {
        invoke_spaced_kmer_vector_valid_lanes_callback<E, L, Simd, N == 1>(
            callback, m, pos,
            spaced_kmer_vector_cast_unchecked<E, L, Simd>(
                kernels[m].mask_pieces[0].bit_extract(windows)
            ),
            SimdVectorValidLanes<Simd>{lanes_of(m)}
        );
    }
}

// -------------------------------------------------------------------------------------------------
//     Loop
// -------------------------------------------------------------------------------------------------

// The loop over the windows of the spans, in vectors of consecutive spans, see the overview of
// kmer_extract/packed_bytewise_simd.hpp: a span of up to 29 is within the word of its starting
// byte, which a shift per lane moves to the lowest bits. With `Full`, a span of 30 to 32 can reach
// into a ninth byte, whose bases are shifted in from the words of the next step, which that step
// starts with, as in for_each_kmer_packed_bytewise_simd_full_width_(). The vectors are emitted as
// `emit(pos, windows, validity)`, with the validity of `Form`, see packed_bytewise_simd_emit_().
template <
    bool Full, typename Simd, PackedValidity Form, PackedSequenceType Seq, typename Emit
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_bytewise_simd_loop_(
    Seq const& seq, std::size_t span_k, Emit& emit
) {
    using simd_vector = typename Simd::simd_vector;
    constexpr std::size_t step_bytes = packed_bytewise_simd_step_bytes_v<Simd>;
    constexpr Layout L = Seq::layout;
    unsigned const span32 = static_cast<unsigned>(span_k);
    std::uint8_t const* const codes = seq.codes().data();
    auto const emit_ = packed_bytewise_simd_emit_<Simd, Form>(seq, span_k, emit);

    if constexpr (!Full) {
        // Right shift that moves the span at position `local` of a starting byte to the low bits.
        auto const counts = make_packed_bytewise_simd_counts_<Simd>(
            [&](std::size_t local) -> std::uint64_t {
                return L == Layout::kMSB ? (64u - 2u * span32) - 2u * local : 2u * local;
            }
        );
        auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
            simd_vector const words = load_packed_words_<Simd, L>(codes + b);
            emit_packed_bytewise_simd_step_<Simd>(4 * b, [&](std::size_t i) {
                return packed_bytewise_simd_shift_right_<Simd>(words, counts.at[i]);
            }, validity_at, emit_);
        };
        run_packed_bytewise_blocks_<step_bytes, Form>(seq, span_k, step_);
    } else {
        // The word of the starting byte, shifted by `2 * local` so that the span starts at its
        // first base, and the word `step_bytes` after it, shifted the other way, as in
        // for_each_kmer_packed_bytewise_simd_full_width_(). kMSB then moves the span down by
        // `64 - 2 * span_k`.
        auto const word_counts = make_packed_bytewise_simd_counts_<Simd>(
            [](std::size_t local) -> std::uint64_t { return 2 * local; }
        );
        auto const next_counts = make_packed_bytewise_simd_counts_<Simd>(
            [](std::size_t local) -> std::uint64_t { return 8 * step_bytes - 2 * local; }
        );
        auto const down_counts = make_packed_bytewise_simd_counts_<Simd>(
            [&](std::size_t) -> std::uint64_t { return 64u - 2u * span32; }
        );
        simd_vector next = load_packed_words_<Simd, L>(codes);
        auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
            simd_vector const words = next;
            next = load_packed_words_<Simd, L>(codes + b + step_bytes);
            emit_packed_bytewise_simd_step_<Simd>(4 * b, [&](std::size_t i) {
                // Without shifts per lane, the first vector has position 0 of the starting bytes
                // in all lanes, where the words hold the whole span already.
                bool const whole = !SimdArchWithLaneShifts<Simd> && i == 0;
                if constexpr (L == Layout::kMSB) {
                    simd_vector const windows = whole ? words : Simd::bitwise_or(
                        packed_bytewise_simd_shift_left_<Simd>(words, word_counts.at[i]),
                        packed_bytewise_simd_shift_right_<Simd>(next, next_counts.at[i])
                    );
                    return packed_bytewise_simd_shift_right_<Simd>(windows, down_counts.at[i]);
                } else {
                    return whole ? words : Simd::bitwise_or(
                        packed_bytewise_simd_shift_right_<Simd>(words, word_counts.at[i]),
                        packed_bytewise_simd_shift_left_<Simd>(next, next_counts.at[i])
                    );
                }
            }, validity_at, emit_);
        };
        run_packed_bytewise_blocks_<step_bytes, Form>(seq, span_k, step_);
    }
}

// The loop for the spans of the kernels, see above.
template <typename Simd, PackedValidity Form, PackedSequenceType Seq, typename Emit>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_bytewise_simd_(
    Seq const& seq, std::size_t span_k, Emit& emit
) {
    if (span_k <= 29) {
        for_each_spaced_kmer_packed_bytewise_simd_loop_<false, Simd, Form>(seq, span_k, emit);
    } else {
        for_each_spaced_kmer_packed_bytewise_simd_loop_<true, Simd, Form>(seq, span_k, emit);
    }
}

// -------------------------------------------------------------------------------------------------
//     Extraction
// -------------------------------------------------------------------------------------------------

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it with SIMD bit extraction, and
 * call a callback on them, for a set of kernels.
 *
 * As the sequence holds only valid bases, a spaced k-mer is emitted at every position, for every
 * kernel. It has the Encoding and Layout of the sequence. See above for the form of the callback.
 * A PackedSequenceWithValidity is taken as well, ignoring its validity bits.
 *
 * @tparam Kernel   Bit extraction kernel of a SIMD arch, see BitExtractKernel, tagged as spaced
 *                  k-mer mask with the Layout of the sequence, see make_spaced_kmer_kernel(), and
 *                  possibly const. The spans of all kernels must be the same, and in [1, 32].
 * @tparam N        Number of kernels, fixed at compile time; see dispatch_static_span() for a
 *                  std::vector of kernels.
 * @tparam Callback Callback function, called with vectors of spaced k-mers, see above.
 */
template <PackedSequenceType Seq, typename Kernel, std::size_t N, typename Callback>
    requires SpacedKmerMaskType<std::remove_const_t<Kernel>>
        && BitExtractKernel<typename Kernel::bit_extract_type>
        && (Kernel::layout == Seq::layout) && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_unchecked_bytewise_simd(
    Seq const& seq,
    std::span<Kernel, N> kernels,
    Callback&& callback
) {
    static_assert(N > 0, "Need at least one kernel.");
    using Simd = typename Kernel::bit_extract_type::simd_arch;
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Kernel::layout;

    std::size_t const span_k = common_spaced_kmer_span_(kernels);
    for ([[maybe_unused]] auto const& kernel : kernels) {
        assert(is_valid_spaced_kmer_mask(kernel));
    }
    if (seq.length() < span_k) {
        return;
    }

    auto const emit = [&](
        std::size_t pos, auto windows, std::size_t valid_count
    ) FISK_ALWAYS_INLINE_LAMBDA {
        emit_spaced_kmers_packed_unchecked_bytewise_simd_<E, L>(
            kernels, pos, windows, valid_count, callback
        );
    };
    for_each_spaced_kmer_packed_bytewise_simd_<Simd, PackedValidity::kStarts>(
        without_validity_(seq), span_k, emit
    );
}

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it with SIMD bit extraction, and
 * call a callback on them, for a single kernel.
 *
 * Same as the overload above, for a single kernel, with `mask_idx` always 0. The callback can also
 * be called without it, as callback(pos, spaced_kmers, valid_count).
 */
template <PackedSequenceType Seq, Layout L, typename Kernel, typename Callback>
    requires BitExtractKernel<Kernel> && (L == Seq::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_unchecked_bytewise_simd(
    Seq const& seq,
    SpacedKmerMask<L, Kernel> const& kernel,
    Callback&& callback
) {
    // A copy of the kernel of the caller, so that the loop can keep its values in registers: with
    // a reference, it would have to reload them after each store, which might change them.
    std::array<SpacedKmerMask<L, Kernel>, 1> const kernels{{kernel}};
    for_each_spaced_kmer_packed_unchecked_bytewise_simd(
        seq, std::span{kernels}, std::forward<Callback>(callback)
    );
}

/**
 * @brief Iterate a PackedSequenceWithValidity, extract its spaced k-mers with SIMD bit extraction,
 * and call a callback on them with their valid lanes, for a set of kernels.
 *
 * As for_each_spaced_kmer_packed_unchecked_bytewise_simd(), with the valid lanes of each vector
 * instead of its count, see above: a spaced k-mer is valid if the positions that its mask keeps are
 * valid by the validity bits of `seq`, see pack_sequence_checked(). As in
 * for_each_spaced_kmer_packed_checked_bytewise_scalar(), the loop tests a block of 32 positions at
 * once, and the blocks whose nucleotides are all valid run without a test per lane.
 *
 * @tparam Kernel   Bit extraction kernel of a SIMD arch, see BitExtractKernel, tagged as spaced
 *                  k-mer mask with the Layout of the sequence, see make_spaced_kmer_kernel(), and
 *                  possibly const. The spans of all kernels must be the same, and in [1, 32].
 * @tparam N        Number of kernels, fixed at compile time; see dispatch_static_span() for a
 *                  std::vector of kernels.
 * @tparam Callback Callback function, called with vectors of spaced k-mers, see above.
 */
template <Encoding E, Layout L, typename Kernel, std::size_t N, typename Callback>
    requires SpacedKmerMaskType<std::remove_const_t<Kernel>>
        && BitExtractKernel<typename Kernel::bit_extract_type>
        && (Kernel::layout == L) && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_checked_bytewise_simd(
    PackedSequenceWithValidity<E, L> const& seq,
    std::span<Kernel, N> kernels,
    Callback&& callback
) {
    static_assert(N > 0, "Need at least one kernel.");
    using Simd = typename Kernel::bit_extract_type::simd_arch;

    std::size_t const span_k = common_spaced_kmer_span_(kernels);
    for ([[maybe_unused]] auto const& kernel : kernels) {
        assert(is_valid_spaced_kmer_mask(kernel));
    }
    if (seq.length() < span_k) {
        return;
    }

    auto const emit = [&](
        std::size_t pos, auto windows, auto validity
    ) FISK_ALWAYS_INLINE_LAMBDA {
        emit_spaced_kmers_packed_checked_bytewise_simd_<E, L>(
            kernels, pos, windows, validity, callback
        );
    };
    for_each_spaced_kmer_packed_bytewise_simd_<Simd, PackedValidity::kNucleotides>(
        seq, span_k, emit
    );
}

/**
 * @brief Iterate a PackedSequenceWithValidity, extract its spaced k-mers with SIMD bit extraction,
 * and call a callback on them with their valid lanes, for a single kernel.
 *
 * Same as the overload above, for a single kernel, with `mask_idx` always 0. The callback can also
 * be called without it, as callback(pos, spaced_kmers, valid_lanes).
 */
template <Encoding E, Layout L, typename Kernel, typename Callback>
    requires BitExtractKernel<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_checked_bytewise_simd(
    PackedSequenceWithValidity<E, L> const& seq,
    SpacedKmerMask<L, Kernel> const& kernel,
    Callback&& callback
) {
    // A copy of the kernel of the caller, so that the loop can keep its values in registers: with
    // a reference, it would have to reload them after each store, which might change them.
    std::array<SpacedKmerMask<L, Kernel>, 1> const kernels{{kernel}};
    for_each_spaced_kmer_packed_checked_bytewise_simd(
        seq, std::span{kernels}, std::forward<Callback>(callback)
    );
}

} // namespace fisk
