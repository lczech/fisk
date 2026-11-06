#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/internal.hpp"
#include "fisk/spaced_kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Extraction from a PackedSequence, Bytewise, Scalar
// =================================================================================================

// Unlike for_each_spaced_kmer_ascii_checked_rolling_scalar() (ascii_rolling_scalar.hpp), which
// takes ASCII input and checks character validity, the functions here read directly from a
// PackedSequence (sequence_pack/pack_sequence.hpp), and assume that it holds only valid bases.
// Their checked counterparts take a PackedSequenceWithValidity, and emit only the spaced k-mers
// whose kept positions are valid by its validity bits, as for ASCII input: an invalid nucleotide at
// a position that the mask skips does not make the spaced k-mer invalid. The masks need to have the
// Layout of the sequence, and the spaced k-mers are emitted with its Encoding and Layout.
//
// Each byte of the sequence holds four bases, and a span starts at each of them. A span is taken in
// the pieces of its mask, see SpacedKmerMask (core/types.hpp): piece `j` of the span that starts at
// base `local` of byte `b` starts at base `local` of byte `b + 7 * j`, so that each piece is taken
// from the word of the 8 bytes loaded there, as a window whose lowest bits hold the piece, as for
// the rolling window of for_each_spaced_kmer_ascii_checked_rolling_scalar(). The bit extraction
// masks of the pieces are hence used as they are. The leading pieces of 28 bases are within their
// word for all four bases of the byte, and so is a tail of up to 29. A span of 30 to 32 is one
// piece, which can reach past its word, and takes the bases after it from the next bytes. Above
// each piece, the window still holds the neighboring bases. They need no clearing, as the bit
// extraction only takes the bits of the bit extraction mask of the piece, see `bit_ext` below. The
// spaced k-mer is the OR of the bits extracted from each piece, each shifted to its place by the
// mask shift of the piece.
//
// For kMSB, each load needs a byte swap, see load_packed_word_() (kmer_extract/internal.hpp),
// which on Intel competes with PEXT for the same port. A step hence covers 8 bytes with one load
// per piece: the word of each following byte is the one of the byte before it, shifted by a byte,
// with the byte after its end shifted in, see spaced_kmer_packed_bytewise_step_bytes_v. Each step
// starts with a load of its own. For kLSB, and for kMSB in the loop over the pieces at runtime,
// whose words would be kept in memory, each piece is loaded where its bits are extracted.
//
// The loop has a copy of its own for each number of pieces up to `MaxCompiledPieces`, with the loop
// over the pieces unrolled, see kDefaultMaxCompiledPieces (internal.hpp). Masks with more pieces
// run in one loop over the pieces at runtime.

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// The number of bytes of a step, see above. Other layouts than kMSB and kLSB fail the static_assert
// of the loop below.
template <Layout L>
inline constexpr std::size_t spaced_kmer_packed_bytewise_step_bytes_v = L == Layout::kMSB ? 8 : 1;

// Emit the spaced k-mers at `pos` of the masks that are valid by `valid(mask)`, with the spaced
// k-mer of each from `extract(mask, mask_idx)`.
template <
    bool SingleMask, Encoding E, Layout L,
    typename MaskOrMasks, typename Extract, typename Valid, typename Callback
>
FISK_ALWAYS_INLINE
inline void emit_spaced_kmers_packed_bytewise_scalar_(
    MaskOrMasks const& masks, std::size_t pos, Extract const& extract, Valid const& valid,
    Callback& callback
) {
    std::size_t mask_idx = 0;
    for_each_mask_(masks, [&](auto const& mask) FISK_ALWAYS_INLINE_LAMBDA {
        if (valid(mask)) {
            auto const word = extract(mask, mask_idx);
            assert_fits_width_(word, spaced_kmer_weight(mask));
            invoke_spaced_kmer_callback<E, L, SingleMask>(callback, mask_idx, pos, word);
        }
        ++mask_idx;
    });
}

// The window of a span of 30 to 32 starting at base `Local` (0 to 3) of its byte, from the
// `word` loaded at that byte, and the bases after it in `next`: for kMSB the ninth byte, and for
// kLSB the word loaded at the byte after `word`. Where the span reaches past `word`, the bases from
// `next` are shifted in, such that the two shifts do not add up to 64, which compilers would turn
// into a double shift (shld, shrd), which on Intel competes with PEXT for the same port. For kLSB,
// the bases of `next` that overlap with `word` are the same, so that the OR keeps them.
template <Layout L, unsigned K, unsigned Local>
FISK_ALWAYS_INLINE
inline std::uint64_t spaced_kmer_packed_bytewise_full_width_window_(
    std::uint64_t word, std::uint64_t next
) {
    constexpr bool in_word = 2 * K + 2 * Local <= 64;
    if constexpr (L == Layout::kMSB) {
        if constexpr (in_word) {
            return word >> (64 - 2 * K - 2 * Local);
        } else {
            constexpr unsigned shift = 2 * K + 2 * Local - 64;
            return (word << shift) | (next >> (8 - shift));
        }
    } else if constexpr (L == Layout::kLSB) {
        if constexpr (in_word) {
            return word >> (2 * Local);
        } else {
            return (word >> (2 * Local)) | (next << (8 - 2 * Local));
        }
    } else {
        static_assert(
            dependent_false_v<L>,
            "Unhandled Layout in spaced_kmer_packed_bytewise_full_width_window_()"
        );
    }
}

// -------------------------------------------------------------------------------------------------
//     Loop
// -------------------------------------------------------------------------------------------------

// The loop over the spans of `Count` pieces, or of the piece count of the masks at runtime if
// `Count` is 0, see above. `Tail` is the span of a single piece of 30 to 32, known at compile time,
// so that it is known which positions need the bases after their word; otherwise it is 0, and the
// tail, of up to 29 bases, has its span at runtime, as the shifts per position do not depend on
// it. With validity bits, see PackedSequenceWithValidityType, only the valid spaced k-mers are
// emitted.
template <
    std::size_t Count, unsigned Tail, bool SingleMask, PackedSequenceType Seq,
    typename MaskOrMasks, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_bytewise_scalar_loop_(
    Seq const& seq, std::size_t span_k,
    MaskOrMasks const& masks, BitExtract& bit_ext, Callback& callback
) {
    using Mask = spaced_kmer_mask_of_t_<MaskOrMasks>;
    static_assert(Tail == 0 || (Count == 1 && Tail >= 30 && Tail <= 32));
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Seq::layout;
    static_assert(
        L == Layout::kMSB || L == Layout::kLSB,
        "Unhandled Layout in for_each_spaced_kmer_packed_bytewise_scalar_loop_()"
    );
    constexpr bool Checked = PackedSequenceWithValidityType<Seq>;
    constexpr std::size_t StepBytes = spaced_kmer_packed_bytewise_step_bytes_v<L>;
    constexpr std::size_t Pieces = Count != 0 ? Count : Mask::max_pieces;
    constexpr std::size_t AccumulatedMasks
        = Count != 1 ? spaced_kmer_mask_of_<MaskOrMasks>::count : 1;
    std::size_t const count = Count != 0 ? Count : spaced_kmer_mask_piece_count(span_k);
    std::size_t const lead = count - 1;
    assert(count == spaced_kmer_mask_piece_count(span_k) && count <= Pieces);
    assert(Tail == 0 ? spaced_kmer_mask_tail_span(span_k) <= 29 : span_k == Tail);
    std::uint8_t const* const codes = seq.codes().data();

    // For kMSB, moves a tail of up to 29 at position 3 to the lowest bits, so that the shifts per
    // position are constants, as in for_each_kmer_packed_bytewise_scalar_narrow_width_().
    [[maybe_unused]] unsigned const tail_align = static_cast<unsigned>(
        58 - 2 * spaced_kmer_mask_tail_span(span_k)
    );

    // One step of run_packed_bytewise_blocks_() (kmer_extract/internal.hpp): the spans that start
    // in the `StepBytes` bytes from `b`, of which those that are valid by `validity_at` are
    // emitted. Unchecked, that is a test of the whole span, for all masks at once, and checked, a
    // test per mask, of each of its pieces.
    auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
        // For kMSB with a compiled piece count, the words roll over the bytes of the step.
        constexpr bool Rolling = L == Layout::kMSB && Count != 0;
        [[maybe_unused]] std::array<std::uint64_t, Pieces> words;
        if constexpr (Rolling) {
            for (std::size_t j = 0; j < count; ++j) {
                words[j] = load_packed_word_<L>(codes + b + 7 * j);
            }
        }
        for (std::size_t i = 0; i < StepBytes; ++i) {
            std::size_t const at = b + i;
            if constexpr (Rolling) {
                if (i > 0) {
                    for (std::size_t j = 0; j < count; ++j) {
                        words[j] = (words[j] << 8) | codes[at + 7 * j + 7];
                    }
                }
            }

            // The word of piece `j`: otherwise loaded where it is used.
            auto const word_ = [&](std::size_t j) FISK_ALWAYS_INLINE_LAMBDA {
                if constexpr (Rolling) {
                    return words[j];
                } else {
                    return load_packed_word_<L>(codes + at + 7 * j);
                }
            };

            // The window of the tail at base `Local` of the byte, computed where it is used, so
            // that the bit extraction has the registers for its tables. The base is passed as a
            // std::integral_constant, not as a template parameter, as Clang 13 fails on lambdas
            // with template parameters here.
            std::uint64_t const tail_word = word_(lead);
            [[maybe_unused]] std::uint64_t next = 0;
            if constexpr (Tail != 0) {
                next = L == Layout::kMSB
                    ? std::uint64_t{codes[at + 8]} : load_packed_word_<L>(codes + at + 1);
            }
            auto const tail_window_ = [&](auto local) FISK_ALWAYS_INLINE_LAMBDA {
                constexpr unsigned Local = decltype(local)::value;
                if constexpr (Tail != 0) {
                    return spaced_kmer_packed_bytewise_full_width_window_<L, Tail, Local>(
                        tail_word, next
                    );
                } else if constexpr (L == Layout::kMSB) {
                    return (tail_word >> tail_align) >> (6 - 2 * Local);
                } else if constexpr (L == Layout::kLSB) {
                    return tail_word >> (2 * Local);
                } else {
                    static_assert(
                        dependent_false_v<L>,
                        "Unhandled Layout in for_each_spaced_kmer_packed_bytewise_scalar_loop_()"
                    );
                }
            };

            // For more than one piece, the spaced k-mers of all masks at the four bases of the
            // byte, computed here ahead of emit_() below, from the tail and the leading pieces of
            // 28, one piece after the other, so that the word and the bit extraction mask of each
            // piece serve all four bases at once. The windows of the leading pieces end at bit
            // `8 - 2 * local` of their word for kMSB, and start at bit `2 * local` for kLSB.
            std::array<std::array<std::uint64_t, 4>, AccumulatedMasks> spaced_kmers;
            if constexpr (Count != 1) {
                std::size_t m = 0;
                for_each_mask_(masks, [&](auto const& mask) FISK_ALWAYS_INLINE_LAMBDA {
                    for (unsigned local = 0; local < 4; ++local) {
                        std::uint64_t const window = L == Layout::kMSB
                            ? (tail_word >> tail_align) >> (6 - 2 * local)
                            : tail_word >> (2 * local);
                        spaced_kmers[m][local] = bit_ext(
                            window, mask.mask_pieces[lead]
                        ) << mask.mask_shifts[lead];
                    }
                    for (std::size_t j = 0; j < lead; ++j) {
                        std::uint64_t const word = word_(j);
                        for (unsigned local = 0; local < 4; ++local) {
                            unsigned const shift = L == Layout::kMSB ? 8 - 2 * local : 2 * local;
                            spaced_kmers[m][local] |= bit_ext(
                                word >> shift, mask.mask_pieces[j]
                            ) << mask.mask_shifts[j];
                        }
                    }
                    ++m;
                });
            }

            // The spaced k-mers at base `local` of the byte, from the `window` of a single piece.
            // Not a template over `local`, so that the four calls share one instantiation; each
            // call is inlined, with the extraction and the callbacks.
            auto const emit_ = [&](
                unsigned local, [[maybe_unused]] std::uint64_t window
            ) FISK_ALWAYS_INLINE_LAMBDA {
                std::size_t const pos = 4 * at + local;
                if constexpr (!Checked) {
                    if (!packed_kmer_is_valid_(validity_at(pos))) {
                        return;
                    }
                }

                // The spaced k-mer of a mask: for one piece, the bits extracted here from the
                // window, and for more, the spaced k-mer computed ahead, see `spaced_kmers` above.
                auto const extract = [&](
                    auto const& mask, std::size_t mask_idx
                ) FISK_ALWAYS_INLINE_LAMBDA {
                    if constexpr (Count == 1) {
                        (void) mask_idx;
                        return bit_ext(window, mask.mask_pieces[0]);
                    } else {
                        (void) mask;
                        return spaced_kmers[mask_idx][local];
                    }
                };

                // Whether the spaced k-mer of a mask is valid: checked, and outside of the blocks
                // that are all valid, by its validity window at the start of each piece.
                auto const valid = [&](auto const& mask) FISK_ALWAYS_INLINE_LAMBDA {
                    using Validity = decltype(validity_at(pos));
                    if constexpr (!Checked || std::is_same_v<Validity, AllValid>) {
                        (void) mask;
                        return true;
                    } else if constexpr (Count == 1) {
                        auto const v = mask.validity_pieces[0];
                        return (validity_at(pos) & v) == v;
                    } else {
                        bool all = true;
                        for (std::size_t j = 0; j < count; ++j) {
                            auto const v = mask.validity_pieces[j];
                            all &= (validity_at(pos + 28 * j) & v) == v;
                        }
                        return all;
                    }
                };
                emit_spaced_kmers_packed_bytewise_scalar_<SingleMask, E, L>(
                    masks, pos, extract, valid, callback
                );
            };
            [&]<unsigned... Locals>(
                std::integer_sequence<unsigned, Locals...>
            ) FISK_ALWAYS_INLINE_LAMBDA {
                (emit_(Locals, Count == 1 ? tail_window_(std::integral_constant<unsigned, Locals>{}) : 0), ...);
            }(std::make_integer_sequence<unsigned, 4>{});
        }
    };

    // For more than one piece, the block test covers the longest span of `Count` pieces in a
    // number of validity words that is known at compile time.
    constexpr std::size_t BlockWords = Count <= 1 ? 0 : (31 + 28 * Count + 1 + 63) / 64;
    run_packed_bytewise_blocks_<StepBytes, PackedValidity::kNucleotides, Count != 1, BlockWords>(
        seq, span_k, step_
    );
}

// -------------------------------------------------------------------------------------------------
//     Dispatch
// -------------------------------------------------------------------------------------------------

/**
 * @brief Shared implementation behind the for_each_spaced_kmer_packed_unchecked_bytewise_scalar()
 * and for_each_spaced_kmer_packed_checked_bytewise_scalar() overloads below.
 *
 * `SingleMask` is set by each overload, as for
 * for_each_spaced_kmer_ascii_checked_rolling_scalar_(), and `span_k` is the common span of the
 * masks, as checked by the overloads. Each piece count up to `MaxCompiledPieces` has a copy of the
 * loop of its own, and so does each span of 30 to 32. With validity bits, see
 * PackedSequenceWithValidityType, only the valid spaced k-mers are emitted.
 */
template <
    std::size_t MaxCompiledPieces, bool SingleMask, PackedSequenceType Seq,
    typename MaskOrMasks, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_bytewise_scalar_(
    Seq const& seq, std::size_t span_k,
    MaskOrMasks const& masks, BitExtract& bit_ext, Callback& callback
) {
    using Mask = spaced_kmer_mask_of_t_<MaskOrMasks>;
    static_assert(MaxCompiledPieces >= 1, "MaxCompiledPieces must be at least 1");
    for_each_mask_(masks, [&]([[maybe_unused]] auto const& mask) {
        assert(is_valid_spaced_kmer_mask(mask));
        assert(
            bit_ext(~std::uint64_t{0}, mask.mask_pieces[0])
            == bit_ext(spaced_kmer_mask_bits_(mask), mask.mask_pieces[0])
        );
    });
    if (seq.length() < span_k) {
        return;
    }

    std::size_t const count = spaced_kmer_mask_piece_count(span_k);
    if (count == 1) {
        switch (span_k) {
            case 30: {
                for_each_spaced_kmer_packed_bytewise_scalar_loop_<1, 30, SingleMask>(
                    seq, span_k, masks, bit_ext, callback
                );
                break;
            }
            case 31: {
                for_each_spaced_kmer_packed_bytewise_scalar_loop_<1, 31, SingleMask>(
                    seq, span_k, masks, bit_ext, callback
                );
                break;
            }
            case 32: {
                for_each_spaced_kmer_packed_bytewise_scalar_loop_<1, 32, SingleMask>(
                    seq, span_k, masks, bit_ext, callback
                );
                break;
            }
            default: {
                for_each_spaced_kmer_packed_bytewise_scalar_loop_<1, 0, SingleMask>(
                    seq, span_k, masks, bit_ext, callback
                );
                break;
            }
        }
        return;
    }

    // The copies for 2 to MaxCompiledPieces pieces, as far as the masks can have that many, and the
    // loop over the pieces at runtime for masks that can have more.
    constexpr std::size_t Compiled = std::min(MaxCompiledPieces, Mask::max_pieces);
    bool const done = [&]<std::size_t... Is>(std::index_sequence<Is...>) FISK_ALWAYS_INLINE_LAMBDA {
        return ((
            count == Is + 1 && (
                for_each_spaced_kmer_packed_bytewise_scalar_loop_<Is + 1, 0, SingleMask>(
                    seq, span_k, masks, bit_ext, callback
                ), true
            )
        ) || ...);
    }(std::make_index_sequence<Compiled>{});
    if constexpr (Compiled < Mask::max_pieces) {
        if (!done) {
            for_each_spaced_kmer_packed_bytewise_scalar_loop_<0, 0, SingleMask>(
                seq, span_k, masks, bit_ext, callback
            );
        }
    } else {
        (void) done;
        assert(done);
    }
}

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it, and call a callback on each
 * spaced k-mer, for a single mask.
 *
 * The packed counterpart of for_each_spaced_kmer_ascii_checked_rolling_scalar() for ASCII input. As
 * the sequence holds only valid bases, a spaced k-mer is emitted at every position. It has the
 * Encoding and Layout of the sequence. A PackedSequenceWithValidity is taken as well, ignoring its
 * validity bits, see for_each_spaced_kmer_packed_checked_bytewise_scalar() for its valid spaced
 * k-mers only.
 *
 * @tparam MaxCompiledPieces Largest number of pieces of the mask, see SpacedKmerMask, for which
 *                  the loop has a copy of its own, see kDefaultMaxCompiledPieces.
 * @tparam Mask     Spaced k-mer mask, see SpacedKmerMask and make_spaced_kmer_mask(), whose Layout
 *                  has to be the one of the sequence.
 * @tparam Callback Callback function called with each extracted spaced k-mer.
 *
 * @param seq       Input sequence.
 * @param mask      Spaced k-mer mask, in the form that `bit_ext` expects, such as
 *                  BitExtractMask, BitExtractBlockShiftMask, or BitExtractButterflyMask.
 *                  Its span must be in [1, Mask::max_span].
 * @param bit_ext   Bit extraction functor, called as bit_ext(word, piece) with the bit extraction
 *                  mask of each piece of `mask`. It must ignore all bits of `word` outside of that
 *                  bit extraction mask, as PEXT and all fisk bit extraction functions do.
 * @param callback  Callback functor, called as callback(pos, mask_idx, spaced_kmer), or as
 *                  callback(pos, spaced_kmer) or callback(spaced_kmer). `pos` is the start
 *                  position of the spaced k-mer's span in `seq`. See invoke_spaced_kmer_callback().
 */
template <
    std::size_t MaxCompiledPieces = kDefaultMaxCompiledPieces,
    PackedSequenceType Seq, typename Mask, typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<Mask> && (Mask::layout == Seq::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
    Seq const& seq,
    Mask const& mask,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    for_each_spaced_kmer_packed_bytewise_scalar_<MaxCompiledPieces, true>(
        without_validity_(seq), check_spaced_kmer_span_<Mask::max_span>(mask.span_k), mask,
        bit_ext, callback
    );
}

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it, and call a callback on each
 * spaced k-mer, across a set of masks.
 *
 * Same as the single-mask overload above, but for a set of masks, which all need to have the same
 * span, as a std::span of compile-time size, possibly of const masks; see dispatch_static_span()
 * for a std::vector of masks. At each position, the spaced k-mers of all
 * masks are emitted, in the order of the masks. The callback must be callable as callback(pos,
 * mask_idx, spaced_kmer); the shorthands callback(pos, spaced_kmer) and callback(spaced_kmer) are
 * only available for a single mask.
 */
template <
    std::size_t MaxCompiledPieces = kDefaultMaxCompiledPieces,
    PackedSequenceType Seq, typename Mask, std::size_t N, typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<std::remove_const_t<Mask>> && (Mask::layout == Seq::layout)
        && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
    Seq const& seq,
    std::span<Mask, N> masks,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    static_assert(N > 0, "Need at least one mask.");
    for_each_spaced_kmer_packed_bytewise_scalar_<MaxCompiledPieces, false>(
        without_validity_(seq), common_spaced_kmer_span_<Mask::max_span>(masks), masks, bit_ext,
        callback
    );
}

/**
 * @brief Iterate a PackedSequenceWithValidity, extract its valid spaced k-mers, and call a callback
 * on each, for a single mask.
 *
 * As for_each_spaced_kmer_packed_unchecked_bytewise_scalar(), emitting only the spaced k-mers whose
 * kept positions are valid by the validity bits of `seq`, see pack_sequence_checked(). Invalid
 * nucleotides at the positions that the mask skips are allowed, as in
 * for_each_spaced_kmer_ascii_checked_rolling_scalar() for ASCII input. As in
 * for_each_kmer_packed_checked_bytewise_scalar(), the loop tests a block of 32 positions at once,
 * and the blocks whose nucleotides are all valid run without a test per spaced k-mer.
 */
template <
    std::size_t MaxCompiledPieces = kDefaultMaxCompiledPieces,
    Encoding E, Layout L, typename Mask, typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<Mask> && (Mask::layout == L)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_checked_bytewise_scalar(
    PackedSequenceWithValidity<E, L> const& seq,
    Mask const& mask,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    for_each_spaced_kmer_packed_bytewise_scalar_<MaxCompiledPieces, true>(
        seq, check_spaced_kmer_span_<Mask::max_span>(mask.span_k), mask, bit_ext, callback
    );
}

/**
 * @brief Iterate a PackedSequenceWithValidity, extract its valid spaced k-mers, and call a callback
 * on each, across a set of masks.
 *
 * Same as the single-mask overload above, but for a set of masks, which all need to have the same
 * span, as a std::span of compile-time size, see the multi-mask overload of
 * for_each_spaced_kmer_packed_unchecked_bytewise_scalar(). The validity of each spaced k-mer is
 * that of the positions that its own mask keeps.
 */
template <
    std::size_t MaxCompiledPieces = kDefaultMaxCompiledPieces,
    Encoding E, Layout L, typename Mask, std::size_t N, typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<std::remove_const_t<Mask>> && (Mask::layout == L)
        && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_checked_bytewise_scalar(
    PackedSequenceWithValidity<E, L> const& seq,
    std::span<Mask, N> masks,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    static_assert(N > 0, "Need at least one mask.");
    for_each_spaced_kmer_packed_bytewise_scalar_<MaxCompiledPieces, false>(
        seq, common_spaced_kmer_span_<Mask::max_span>(masks), masks, bit_ext, callback
    );
}

} // namespace fisk
