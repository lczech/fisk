#pragma once

#include <algorithm>
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
// them at once. As in the scalar extraction, a span is taken in the pieces of its mask, see
// SpacedKmerMask (core/types.hpp): piece `j` of the spans in a vector is taken from the words
// loaded `7 * j` bytes after those of the first piece, and the spaced k-mer is the OR of the bits
// extracted from each piece, shifted to its place by the mask shift of the piece. The windows still
// hold the neighboring bases above each piece, which the bit extraction does not take.
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
//
// As in the scalar extraction, the loop has a copy of its own for each number of pieces up to
// `MaxCompiledPieces`, see kDefaultMaxCompiledPieces (internal.hpp), and masks with more pieces
// run in one loop over the pieces at runtime.

// -------------------------------------------------------------------------------------------------
//     Loop
// -------------------------------------------------------------------------------------------------

// The loop over the spans of `Count` pieces, or of the piece count of the kernels at runtime if
// `Count` is 0, see above. The windows of a vector of positions are a SimdVectorSet, one vector per
// piece: the leading pieces of 28 bases are within the words loaded at their byte, which a shift
// per lane moves to the lowest bits, and so is a tail of up to 29. With `Full`, a single piece of
// 30 to 32 can reach into a ninth byte, whose bases are shifted in from the words of the next step,
// which that step starts with, as in for_each_kmer_packed_bytewise_simd_full_width_(). Checked, a
// lane is valid if each piece is valid by the validity window at its own start.
template <
    std::size_t Count, bool Full, PackedSequenceType Seq, typename Kernel, std::size_t N,
    typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_bytewise_simd_loop_(
    Seq const& seq, std::size_t span_k, std::span<Kernel, N> kernels, Callback& callback
) {
    using Simd = typename Kernel::bit_extract_type::simd_arch;
    using Mask = std::remove_const_t<Kernel>;
    using simd_vector = typename Simd::simd_vector;
    static_assert(!Full || Count == 1);
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Seq::layout;
    static_assert(
        L == Layout::kMSB || L == Layout::kLSB,
        "Unhandled Layout in for_each_spaced_kmer_packed_bytewise_simd_loop_()"
    );
    constexpr bool Checked = PackedSequenceWithValidityType<Seq>;
    constexpr std::size_t lanes = Simd::lanes;
    constexpr std::size_t step_bytes = packed_bytewise_simd_step_bytes_v<Simd>;
    constexpr std::size_t Pieces = Count != 0 ? Count : Mask::max_pieces;
    using Windows = SimdVectorSet<Simd, Pieces>;
    std::size_t const count = Count != 0 ? Count : spaced_kmer_mask_piece_count(span_k);
    std::size_t const lead = count - 1;
    assert(count == spaced_kmer_mask_piece_count(span_k) && count <= Pieces);
    assert(!Full || (span_k >= 30 && span_k <= 32));
    unsigned const tail = static_cast<unsigned>(spaced_kmer_mask_tail_span(span_k));
    std::uint8_t const* const codes = seq.codes().data();

    // The mask shift of each piece of each kernel, as a count for all lanes.
    auto const uniform_count_ = [](std::uint64_t value) {
        if constexpr (SimdArchWithLaneShifts<Simd>) {
            return Simd::broadcast(value);
        } else {
            return Simd::make_shift_count(value);
        }
    };
    struct Shifts { decltype(uniform_count_(0)) at[N][Pieces]; } shifts{};
    if constexpr (Count != 1) {
        for (std::size_t m = 0; m < N; ++m) {
            for (std::size_t j = 0; j < Pieces; ++j) {
                shifts.at[m][j] = uniform_count_(kernels[m].mask_shifts[j]);
            }
        }
    }

    // The spaced k-mers of kernel `m` from the `windows`: the bits of its single piece, or the OR
    // of those of each piece, shifted to their place. The loops over the pieces here and below run
    // to `Pieces`, known at compile time, and skip those past the count, so that the windows are
    // indexed by constants and stay in registers.
    auto const extract_ = [&](std::size_t m, Windows const& windows) FISK_ALWAYS_INLINE_LAMBDA {
        auto const& kernel = kernels[m];
        if constexpr (Count == 1) {
            return spaced_kmer_vector_cast_unchecked<E, L, Simd>(
                kernel.mask_pieces[0].bit_extract(windows.at[0])
            );
        } else {
            simd_vector out = Simd::zero();
            for (std::size_t j = 0; j < Pieces; ++j) {
                if (j < count) {
                    out = Simd::bitwise_or(out, packed_bytewise_simd_shift_left_<Simd>(
                        kernel.mask_pieces[j].bit_extract(windows.at[j]), shifts.at[m][j]
                    ));
                }
            }
            return spaced_kmer_vector_cast_unchecked<E, L, Simd>(out);
        }
    };

    // The valid lanes of kernel `m` from the validity windows at the start of each piece, see
    // `piece_validity_` below: AllValid, or a window per piece, against which each lane tests the
    // positions of the piece that the mask keeps.
    auto const valid_lanes_ = [&](std::size_t m, auto const& validity) FISK_ALWAYS_INLINE_LAMBDA {
        std::uint64_t bits = (std::uint64_t{1} << lanes) - 1u;
        if constexpr (!std::is_same_v<std::remove_cvref_t<decltype(validity)>, AllValid>) {
            for (std::size_t j = 0; j < Pieces; ++j) {
                if (j < count) {
                    std::uint64_t const vm = kernels[m].validity_pieces[j];
                    std::uint64_t piece_bits = 0;
                    for (std::size_t l = 0; l < lanes; ++l) {
                        piece_bits |= static_cast<std::uint64_t>(
                            ((validity[j] >> l) & vm) == vm
                        ) << l;
                    }
                    bits &= piece_bits;
                }
            }
        } else {
            (void) m;
        }
        return SimdVectorValidLanes<Simd>{bits};
    };

    // Extract and emit the spaced k-mers of all kernels from the `windows` of the spans at `pos`
    // and after, with their count or validity, see packed_bytewise_simd_emit_(), one kernel after
    // the other: a loop with the compile-time trip count N, which the compiler unrolls where the
    // kernels are cheap, as for_each_mask_() (internal.hpp).
    auto const emit = [&](
        std::size_t pos, Windows const& windows, auto validity
    ) FISK_ALWAYS_INLINE_LAMBDA {
        for (std::size_t m = 0; m < N; ++m) {
            if constexpr (Checked) {
                invoke_spaced_kmer_vector_valid_lanes_callback<E, L, Simd, N == 1>(
                    callback, m, pos, extract_(m, windows), valid_lanes_(m, validity)
                );
            } else {
                invoke_spaced_kmer_vector_callback<E, L, Simd, N == 1>(
                    callback, m, pos, extract_(m, windows), validity
                );
            }
        }
    };
    auto const emit_ = packed_bytewise_simd_emit_<Simd, PackedValidity::kNucleotides>(
        seq, span_k, emit
    );

    // Checked, the validity windows at the start of each piece of the spans from `pos` on, from
    // `validity_at`, see run_packed_bytewise_blocks_() (kmer_extract/internal.hpp), or AllValid.
    auto const piece_validity_ = [&](auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
        return [&](std::size_t pos) FISK_ALWAYS_INLINE_LAMBDA {
            using Validity = decltype(validity_at(pos));
            if constexpr (!Checked || std::is_same_v<Validity, AllValid>) {
                return validity_at(pos);
            } else {
                std::array<std::uint64_t, Pieces> windows{};
                for (std::size_t j = 0; j < Pieces; ++j) {
                    if (j < count) {
                        windows[j] = validity_at(pos + 28 * j);
                    }
                }
                return windows;
            }
        };
    };

    if constexpr (!Full) {
        // Right shifts that move the piece at position `local` of a starting byte to the low bits:
        // for the tail, as for a span of its length, and for the leading pieces of 28, the same
        // for all spans, from bit `8 - 2 * local` for kMSB and from bit `2 * local` for kLSB.
        auto const tail_counts = make_packed_bytewise_simd_counts_<Simd>(
            [&](std::size_t local) -> std::uint64_t {
                return L == Layout::kMSB ? (64u - 2u * tail) - 2u * local : 2u * local;
            }
        );
        auto const lead_counts = make_packed_bytewise_simd_counts_<Simd>(
            [](std::size_t local) -> std::uint64_t {
                return L == Layout::kMSB ? 8u - 2u * local : 2u * local;
            }
        );
        auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
            emit_packed_bytewise_simd_step_<Simd>(4 * b, [&](std::size_t i) {
                Windows windows{};
                for (std::size_t j = 0; j < Pieces; ++j) {
                    if (j < count) {
                        windows.at[j] = packed_bytewise_simd_shift_right_<Simd>(
                            load_packed_words_<Simd, L>(codes + b + 7 * j),
                            j == lead ? tail_counts.at[i] : lead_counts.at[i]
                        );
                    }
                }
                return windows;
            }, piece_validity_(validity_at), emit_);
        };

        // For more than one piece, the block test covers the longest span of `Count` pieces in a
        // number of validity words that is known at compile time, see
        // for_each_spaced_kmer_packed_bytewise_scalar_loop_().
        constexpr std::size_t BlockWords = Count <= 1 ? 0 : (31 + 28 * Count + 1 + 63) / 64;
        run_packed_bytewise_blocks_<
            step_bytes, PackedValidity::kNucleotides, Count != 1, BlockWords
        >(seq, span_k, step_);
    } else {
        // The word of the starting byte, shifted by `2 * local` so that the span starts at its
        // first base, and the word `step_bytes` after it, shifted the other way, as in
        // for_each_kmer_packed_bytewise_simd_full_width_(). kMSB then moves the span down by
        // `64 - 2 * span_k`.
        unsigned const span32 = static_cast<unsigned>(span_k);
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
                Windows windows;
                if constexpr (L == Layout::kMSB) {
                    simd_vector const span_words = whole ? words : Simd::bitwise_or(
                        packed_bytewise_simd_shift_left_<Simd>(words, word_counts.at[i]),
                        packed_bytewise_simd_shift_right_<Simd>(next, next_counts.at[i])
                    );
                    windows.at[0] = packed_bytewise_simd_shift_right_<Simd>(
                        span_words, down_counts.at[i]
                    );
                } else {
                    windows.at[0] = whole ? words : Simd::bitwise_or(
                        packed_bytewise_simd_shift_right_<Simd>(words, word_counts.at[i]),
                        packed_bytewise_simd_shift_left_<Simd>(next, next_counts.at[i])
                    );
                }
                return windows;
            }, piece_validity_(validity_at), emit_);
        };
        run_packed_bytewise_blocks_<step_bytes, PackedValidity::kNucleotides>(seq, span_k, step_);
    }
}

// -------------------------------------------------------------------------------------------------
//     Dispatch
// -------------------------------------------------------------------------------------------------

// The loop for the piece count of the kernels, whose common span is `span_k`, see above: a copy
// for each piece count up to `MaxCompiledPieces`, and one for spans of 30 to 32, which can reach
// past the word of their starting byte, and the loop over the pieces at runtime above.
template <
    std::size_t MaxCompiledPieces, PackedSequenceType Seq, typename Kernel, std::size_t N,
    typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_bytewise_simd_(
    Seq const& seq, std::size_t span_k, std::span<Kernel, N> kernels, Callback& callback
) {
    using Mask = std::remove_const_t<Kernel>;
    static_assert(MaxCompiledPieces >= 1, "MaxCompiledPieces must be at least 1");
    for ([[maybe_unused]] auto const& kernel : kernels) {
        assert(is_valid_spaced_kmer_mask(kernel));
    }
    if (seq.length() < span_k) {
        return;
    }

    std::size_t const count = spaced_kmer_mask_piece_count(span_k);
    if (count == 1) {
        if (span_k <= 29) {
            for_each_spaced_kmer_packed_bytewise_simd_loop_<1, false>(
                seq, span_k, kernels, callback
            );
        } else {
            for_each_spaced_kmer_packed_bytewise_simd_loop_<1, true>(
                seq, span_k, kernels, callback
            );
        }
        return;
    }
    constexpr std::size_t Compiled = std::min(MaxCompiledPieces, Mask::max_pieces);
    bool const done = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return ((
            count == Is + 2 && (
                for_each_spaced_kmer_packed_bytewise_simd_loop_<Is + 2, false>(
                    seq, span_k, kernels, callback
                ), true
            )
        ) || ...);
    }(std::make_index_sequence<Compiled - 1>{});
    if constexpr (Compiled < Mask::max_pieces) {
        if (!done) {
            for_each_spaced_kmer_packed_bytewise_simd_loop_<0, false>(
                seq, span_k, kernels, callback
            );
        }
    } else {
        (void) done;
        assert(done);
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
 * @tparam MaxCompiledPieces Largest number of pieces of the kernels, see SpacedKmerMask, for which
 *                  the loop has a copy of its own, see kDefaultMaxCompiledPieces.
 * @tparam Kernel   Bit extraction kernel of a SIMD arch, see BitExtractKernel, tagged as spaced
 *                  k-mer mask with the Layout of the sequence, see make_spaced_kmer_kernel(), and
 *                  possibly const. The spans of all kernels must be the same, and in
 *                  [1, Kernel::max_span].
 * @tparam N        Number of kernels, fixed at compile time; see dispatch_static_span() for a
 *                  std::vector of kernels.
 * @tparam Callback Callback function, called with vectors of spaced k-mers, see above.
 */
template <
    std::size_t MaxCompiledPieces = kDefaultMaxCompiledPieces,
    PackedSequenceType Seq, typename Kernel, std::size_t N, typename Callback
>
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
    static_assert(
        SpacedKmerVectorCallback<Callback&, Seq::encoding, Seq::layout, Simd, N == 1>,
        "Callback must be callable as callback(pos, mask_idx, spaced_kmers, valid_count), or, "
        "for a single kernel, as callback(pos, spaced_kmers, valid_count)."
    );
    for_each_spaced_kmer_packed_bytewise_simd_<MaxCompiledPieces>(
        without_validity_(seq),
        common_spaced_kmer_span_<std::remove_const_t<Kernel>::max_span>(kernels), kernels,
        callback
    );
}

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it with SIMD bit extraction, and
 * call a callback on them, for a single kernel.
 *
 * Same as the overload above, for a single kernel, with `mask_idx` always 0. The callback can also
 * be called without it, as callback(pos, spaced_kmers, valid_count).
 */
template <
    std::size_t MaxCompiledPieces = kDefaultMaxCompiledPieces,
    PackedSequenceType Seq, Layout L, typename Kernel, std::size_t MaxSpan, typename Callback
>
    requires BitExtractKernel<Kernel> && (L == Seq::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_unchecked_bytewise_simd(
    Seq const& seq,
    SpacedKmerMask<L, Kernel, MaxSpan> const& kernel,
    Callback&& callback
) {
    // A copy of the kernel of the caller, so that the loop can keep its values in registers: with
    // a reference, it would have to reload them after each store, which might change them.
    std::array<SpacedKmerMask<L, Kernel, MaxSpan>, 1> const kernels{{kernel}};
    for_each_spaced_kmer_packed_unchecked_bytewise_simd<MaxCompiledPieces>(
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
 * @tparam MaxCompiledPieces Largest number of pieces of the kernels, see SpacedKmerMask, for which
 *                  the loop has a copy of its own, see kDefaultMaxCompiledPieces.
 * @tparam Kernel   Bit extraction kernel of a SIMD arch, see BitExtractKernel, tagged as spaced
 *                  k-mer mask with the Layout of the sequence, see make_spaced_kmer_kernel(), and
 *                  possibly const. The spans of all kernels must be the same, and in
 *                  [1, Kernel::max_span].
 * @tparam N        Number of kernels, fixed at compile time; see dispatch_static_span() for a
 *                  std::vector of kernels.
 * @tparam Callback Callback function, called with vectors of spaced k-mers, see above.
 */
template <
    std::size_t MaxCompiledPieces = kDefaultMaxCompiledPieces,
    Encoding E, Layout L, typename Kernel, std::size_t N, typename Callback
>
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
    static_assert(
        SpacedKmerVectorValidLanesCallback<Callback&, E, L, Simd, N == 1>,
        "Callback must be callable as callback(pos, mask_idx, spaced_kmers, valid_lanes), or, "
        "for a single kernel, as callback(pos, spaced_kmers, valid_lanes)."
    );
    for_each_spaced_kmer_packed_bytewise_simd_<MaxCompiledPieces>(
        seq, common_spaced_kmer_span_<std::remove_const_t<Kernel>::max_span>(kernels), kernels,
        callback
    );
}

/**
 * @brief Iterate a PackedSequenceWithValidity, extract its spaced k-mers with SIMD bit extraction,
 * and call a callback on them with their valid lanes, for a single kernel.
 *
 * Same as the overload above, for a single kernel, with `mask_idx` always 0. The callback can also
 * be called without it, as callback(pos, spaced_kmers, valid_lanes).
 */
template <
    std::size_t MaxCompiledPieces = kDefaultMaxCompiledPieces,
    Encoding E, Layout L, typename Kernel, std::size_t MaxSpan, typename Callback
>
    requires BitExtractKernel<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_checked_bytewise_simd(
    PackedSequenceWithValidity<E, L> const& seq,
    SpacedKmerMask<L, Kernel, MaxSpan> const& kernel,
    Callback&& callback
) {
    // A copy of the kernel of the caller, so that the loop can keep its values in registers: with
    // a reference, it would have to reload them after each store, which might change them.
    std::array<SpacedKmerMask<L, Kernel, MaxSpan>, 1> const kernels{{kernel}};
    for_each_spaced_kmer_packed_checked_bytewise_simd<MaxCompiledPieces>(
        seq, std::span{kernels}, std::forward<Callback>(callback)
    );
}

} // namespace fisk
