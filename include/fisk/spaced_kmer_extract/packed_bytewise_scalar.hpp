#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <span>

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
// Each byte of the sequence holds four bases, and a span starts at each of them. Each span is taken
// from the word of the 8 bytes loaded at that byte, plus a ninth byte for spans of 30 to 32, as a
// window whose lowest bits hold the span, as for the rolling window of
// for_each_spaced_kmer_ascii_checked_rolling_scalar(). The masks are hence used as they are. Above
// the span, the window still holds the neighboring bases. They need no clearing, as the bit
// extraction only takes the bits of the mask, see `bit_ext` below.

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// Extract and emit the spaced k-mers of all masks from the `window` of the span starting at `pos`.
template <
    bool SingleMask, Encoding E, Layout L,
    typename MaskOrMasks, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE
inline void emit_spaced_kmers_packed_unchecked_bytewise_scalar_(
    MaskOrMasks const& masks, std::size_t pos, std::uint64_t window,
    BitExtract& bit_ext, Callback& callback
) {
    std::size_t mask_idx = 0;
    for_each_mask_(masks, [&](auto const& mask) {
        auto const word = bit_ext(window, mask);
        assert_fits_width_(word, spaced_kmer_weight(mask));
        invoke_spaced_kmer_callback<E, L, SingleMask>(callback, mask_idx, pos, word);
        ++mask_idx;
    });
}

// As emit_spaced_kmers_packed_unchecked_bytewise_scalar_(), but only for the masks whose spaced
// k-mer is valid by the `validity` of the span, see run_packed_bytewise_blocks_()
// (kmer_extract/internal.hpp): either AllValid, or the validity window at `pos`, as
// PackedValidity::kNucleotides, which is tested per mask against the positions that the mask keeps,
// its validity_mask, see SpacedKmerMask.
template <
    bool SingleMask, Encoding E, Layout L,
    typename MaskOrMasks, typename Validity, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE
inline void emit_spaced_kmers_packed_checked_bytewise_scalar_(
    MaskOrMasks const& masks, std::size_t pos, std::uint64_t window, Validity validity,
    BitExtract& bit_ext, Callback& callback
) {
    if constexpr (std::is_same_v<Validity, AllValid>) {
        emit_spaced_kmers_packed_unchecked_bytewise_scalar_<SingleMask, E, L>(
            masks, pos, window, bit_ext, callback
        );
    } else {
        std::size_t mask_idx = 0;
        for_each_mask_(masks, [&](auto const& mask) {
            if ((validity & mask.validity_mask) == mask.validity_mask) {
                auto const word = bit_ext(window, mask);
                assert_fits_width_(word, spaced_kmer_weight(mask));
                invoke_spaced_kmer_callback<E, L, SingleMask>(callback, mask_idx, pos, word);
            }
            ++mask_idx;
        });
    }
}

// -------------------------------------------------------------------------------------------------
//     Narrow Width
// -------------------------------------------------------------------------------------------------

// Spans up to 29 are within the 8 bytes loaded at their starting byte. The span is a runtime value
// here, as the shifts per position do not depend on it. With validity bits, see
// PackedSequenceWithValidityType, only the valid spaced k-mers are emitted.
template <
    bool SingleMask, PackedSequenceType Seq,
    typename MaskOrMasks, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_bytewise_scalar_narrow_width_(
    Seq const& seq, std::size_t span_k,
    MaskOrMasks const& masks, BitExtract& bit_ext, Callback& callback
) {
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Seq::layout;
    constexpr bool Checked = PackedSequenceWithValidityType<Seq>;
    assert(span_k >= 1 && span_k <= 29);
    std::uint8_t const* const codes = seq.codes().data();

    auto const emit_ = [&](std::size_t pos, std::uint64_t window) FISK_ALWAYS_INLINE_LAMBDA {
        emit_spaced_kmers_packed_unchecked_bytewise_scalar_<SingleMask, E, L>(
            masks, pos, window, bit_ext, callback
        );
    };

    // One step of run_packed_bytewise_blocks_() (kmer_extract/internal.hpp): the spans that start
    // in byte `b`, of which those that are valid by `validity_at` are emitted. Unchecked, that is a
    // test of the whole span, for all masks at once, and checked, a test per mask.
    auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
        auto const emit_if_valid_ = [&](
            std::size_t pos, std::uint64_t window
        ) FISK_ALWAYS_INLINE_LAMBDA {
            if constexpr (Checked) {
                emit_spaced_kmers_packed_checked_bytewise_scalar_<SingleMask, E, L>(
                    masks, pos, window, validity_at(pos), bit_ext, callback
                );
            } else if (packed_kmer_is_valid_(validity_at(pos))) {
                emit_(pos, window);
            }
        };
        std::uint64_t word = load_packed_word_<L>(codes + b);
        if constexpr (L == Layout::kMSB) {
            // Move the span at position 3 to the lowest bits once, so that the shifts per
            // position are constants, as in for_each_kmer_packed_bytewise_scalar_narrow_width_().
            word >>= 58 - 2 * span_k;
            emit_if_valid_(4 * b + 0, word >> 6);
            emit_if_valid_(4 * b + 1, word >> 4);
            emit_if_valid_(4 * b + 2, word >> 2);
            emit_if_valid_(4 * b + 3, word);
        } else if constexpr (L == Layout::kLSB) {
            emit_if_valid_(4 * b + 0, word);
            emit_if_valid_(4 * b + 1, word >> 2);
            emit_if_valid_(4 * b + 2, word >> 4);
            emit_if_valid_(4 * b + 3, word >> 6);
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_spaced_kmer_packed_bytewise_scalar_narrow_width_()"
            );
        }
    };
    run_packed_bytewise_blocks_<1, PackedValidity::kNucleotides>(seq, span_k, step_);
}

// -------------------------------------------------------------------------------------------------
//     Full Width
// -------------------------------------------------------------------------------------------------

// The window of a span of 30 to 32 starting at base `Local` (0 to 3) of its byte, from the
// `word` loaded at that byte, and the `ninth` byte after it, as described above. Where the span
// reaches into the ninth byte, the bases from it are shifted in at the bottom of the shifted word,
// for kMSB in a way that compilers do not turn into a double shift (shld), which on Intel competes
// with PEXT for the same port.
template <Layout L, unsigned K, unsigned Local>
FISK_ALWAYS_INLINE
inline std::uint64_t spaced_kmer_packed_bytewise_full_width_window_(
    std::uint64_t word, std::uint64_t ninth
) {
    constexpr bool in_word = 2 * K + 2 * Local <= 64;
    if constexpr (L == Layout::kMSB) {
        if constexpr (in_word) {
            return word >> (64 - 2 * K - 2 * Local);
        } else {
            constexpr unsigned shift = 2 * K + 2 * Local - 64;
            return (word << shift) | (ninth >> (8 - shift));
        }
    } else if constexpr (L == Layout::kLSB) {
        if constexpr (in_word) {
            return word >> (2 * Local);
        } else {
            return (word >> (2 * Local)) | (ninth << (64 - 2 * Local));
        }
    } else {
        static_assert(
            dependent_false_v<L>,
            "Unhandled Layout in spaced_kmer_packed_bytewise_full_width_window_()"
        );
    }
}

// Spans of 30 to 32 can reach into a ninth byte. K is a template parameter, so that it is known at
// compile time which positions need it.
template <
    unsigned K, bool SingleMask, PackedSequenceType Seq,
    typename MaskOrMasks, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_bytewise_scalar_full_width_(
    Seq const& seq,
    MaskOrMasks const& masks, BitExtract& bit_ext, Callback& callback
) {
    static_assert(K >= 30 && K <= 32, "K must be in [30, 32]");
    constexpr Encoding E = Seq::encoding;
    constexpr Layout L = Seq::layout;
    constexpr bool Checked = PackedSequenceWithValidityType<Seq>;
    std::uint8_t const* const codes = seq.codes().data();

    // As in the narrow-width loop.
    auto const emit_ = [&](std::size_t pos, std::uint64_t window) FISK_ALWAYS_INLINE_LAMBDA {
        emit_spaced_kmers_packed_unchecked_bytewise_scalar_<SingleMask, E, L>(
            masks, pos, window, bit_ext, callback
        );
    };
    auto const step_ = [&](std::size_t b, auto const& validity_at) FISK_ALWAYS_INLINE_LAMBDA {
        auto const emit_if_valid_ = [&](
            std::size_t pos, std::uint64_t window
        ) FISK_ALWAYS_INLINE_LAMBDA {
            if constexpr (Checked) {
                emit_spaced_kmers_packed_checked_bytewise_scalar_<SingleMask, E, L>(
                    masks, pos, window, validity_at(pos), bit_ext, callback
                );
            } else if (packed_kmer_is_valid_(validity_at(pos))) {
                emit_(pos, window);
            }
        };
        std::uint64_t const word = load_packed_word_<L>(codes + b);
        std::uint64_t const ninth = codes[b + 8];
        emit_if_valid_(
            4 * b + 0, spaced_kmer_packed_bytewise_full_width_window_<L, K, 0>(word, ninth)
        );
        emit_if_valid_(
            4 * b + 1, spaced_kmer_packed_bytewise_full_width_window_<L, K, 1>(word, ninth)
        );
        emit_if_valid_(
            4 * b + 2, spaced_kmer_packed_bytewise_full_width_window_<L, K, 2>(word, ninth)
        );
        emit_if_valid_(
            4 * b + 3, spaced_kmer_packed_bytewise_full_width_window_<L, K, 3>(word, ninth)
        );
    };
    run_packed_bytewise_blocks_<1, PackedValidity::kNucleotides>(seq, K, step_);
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
 * masks, as checked by the overloads. With validity bits, see PackedSequenceWithValidityType, only
 * the valid spaced k-mers are emitted.
 */
template <
    bool SingleMask, PackedSequenceType Seq,
    typename MaskOrMasks, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_bytewise_scalar_(
    Seq const& seq, std::size_t span_k,
    MaskOrMasks const& masks, BitExtract& bit_ext, Callback& callback
) {
    for_each_mask_(masks, [&]([[maybe_unused]] auto const& mask) {
        assert(is_valid_spaced_kmer_mask(mask));
        assert(bit_ext(~std::uint64_t{0}, mask) == bit_ext(spaced_kmer_mask_bits_(mask), mask));
    });
    if (seq.length() < span_k) {
        return;
    }

    switch (span_k) {
        case 30: {
            for_each_spaced_kmer_packed_bytewise_scalar_full_width_<30, SingleMask>(
                seq, masks, bit_ext, callback
            );
            break;
        }
        case 31: {
            for_each_spaced_kmer_packed_bytewise_scalar_full_width_<31, SingleMask>(
                seq, masks, bit_ext, callback
            );
            break;
        }
        case 32: {
            for_each_spaced_kmer_packed_bytewise_scalar_full_width_<32, SingleMask>(
                seq, masks, bit_ext, callback
            );
            break;
        }
        default: {
            for_each_spaced_kmer_packed_bytewise_scalar_narrow_width_<SingleMask>(
                seq, span_k, masks, bit_ext, callback
            );
            break;
        }
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
 * @tparam Mask     Spaced k-mer mask, see SpacedKmerMask and make_spaced_kmer_mask(), whose Layout
 *                  has to be the one of the sequence.
 * @tparam Callback Callback function called with each extracted spaced k-mer.
 *
 * @param seq       Input sequence.
 * @param mask      Spaced k-mer mask, in the representation that `bit_ext` expects, such as
 *                  BitExtractMask, BitExtractBlockShiftMask, or BitExtractButterflyMask.
 *                  Its span must be in [1, 32].
 * @param bit_ext   Bit extraction functor, called as bit_ext(word, mask). It must ignore all bits
 *                  of `word` outside of the mask, as PEXT and all fisk bit extraction functions do.
 * @param callback  Callback functor, called as callback(pos, mask_idx, spaced_kmer), or as
 *                  callback(pos, spaced_kmer). `pos` is the start position of the spaced k-mer's
 *                  span in `seq`. See invoke_spaced_kmer_callback().
 */
template <PackedSequenceType Seq, typename Mask, typename BitExtract, typename Callback>
    requires SpacedKmerMaskType<Mask> && (Mask::layout == Seq::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
    Seq const& seq,
    Mask const& mask,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    for_each_spaced_kmer_packed_bytewise_scalar_<true>(
        without_validity_(seq), check_spaced_kmer_span_(mask.span_k), mask, bit_ext, callback
    );
}

/**
 * @brief Iterate a PackedSequence, extract all spaced k-mers from it, and call a callback on each
 * spaced k-mer, across a set of masks.
 *
 * Same as the single-mask overload above, but for a set of masks, which all need to have the same
 * span, as a std::span of compile-time size, possibly of const masks; see dispatch_static_span()
 * for a std::vector of masks. At each position, the spaced k-mers of all masks are emitted, in the
 * order of the masks. The callback must be callable as callback(pos, mask_idx, spaced_kmer); the
 * shorthand callback(pos, spaced_kmer) is only available for a single mask.
 */
template <
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
    for_each_spaced_kmer_packed_bytewise_scalar_<false>(
        without_validity_(seq), common_spaced_kmer_span_(masks), masks, bit_ext, callback
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
template <Encoding E, Layout L, typename Mask, typename BitExtract, typename Callback>
    requires SpacedKmerMaskType<Mask> && (Mask::layout == L)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_packed_checked_bytewise_scalar(
    PackedSequenceWithValidity<E, L> const& seq,
    Mask const& mask,
    BitExtract&& bit_ext,
    Callback&& callback
) {
    for_each_spaced_kmer_packed_bytewise_scalar_<true>(
        seq, check_spaced_kmer_span_(mask.span_k), mask, bit_ext, callback
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
    for_each_spaced_kmer_packed_bytewise_scalar_<false>(
        seq, common_spaced_kmer_span_(masks), masks, bit_ext, callback
    );
}

} // namespace fisk
