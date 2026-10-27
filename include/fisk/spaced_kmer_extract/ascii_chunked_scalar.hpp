#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_chunked.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/spaced_kmer_extract/internal.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Extraction from ASCII, Chunked, Scalar
// =================================================================================================

// The spaced counterparts of the chunked extraction of k-mers from ASCII, see
// kmer_extract/ascii_chunked_scalar.hpp: each chunk is packed into the scratch, and its spaced
// k-mers are extracted with the extraction from packed sequences, see packed_bytewise_scalar.hpp.
// The masks and the bit extraction are those of that extraction, and the masks need to have the
// Layout of the encoder. The callback forms are those of
// for_each_spaced_kmer_packed_unchecked_bytewise_scalar(), with `pos` the start position of the
// spaced k-mer's span in the whole sequence.
//
// The scratch is reusable temporary packed storage; its contents after the call are unspecified.
// Overlapping, nested, or concurrent calls must use distinct scratch buffers. The overloads without
// a scratch use a thread-local buffer, which is shared by all chunked functions on the same thread
// that use the same type of scratch, see ascii_chunked_buffer_(); their callback hence must not
// call any of them again with the thread-local buffer.
//
// `ChunkSize` is the number of bases processed per chunk, and must be > 0. A multiple of 32, such
// as a power of two, keeps each chunk in whole blocks of the packed loops, see
// run_packed_bytewise_blocks_() (kmer_extract/internal.hpp).

// -------------------------------------------------------------------------------------------------
//     Internal Helpers
// -------------------------------------------------------------------------------------------------

// The shared implementation of the functions below, for a single mask or a span of masks: with a
// PackedSequenceWithValidity as `scratch`, the chunks are packed with their validity, and only
// their valid spaced k-mers are emitted, see for_each_ascii_chunk_().
template <
    std::size_t ChunkSize, WordEncoder Encoder, PackedSequenceType PackedSequenceChunk,
    typename MaskOrMasks, typename BitExtract, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_chunked_scalar_(
    std::string_view seq, std::size_t span_k, MaskOrMasks const& masks, Encoder const& encoder,
    BitExtract& bit_ext, PackedSequenceChunk& scratch, Callback& callback
) {
    for_each_ascii_chunk_<ChunkSize>(
        seq, span_k, encoder, scratch, callback,
        [&](auto const& chunk, auto const& chunk_func) FISK_ALWAYS_INLINE_LAMBDA {
            if constexpr (PackedSequenceWithValidityType<PackedSequenceChunk>) {
                for_each_spaced_kmer_packed_checked_bytewise_scalar(
                    chunk, masks, bit_ext, chunk_func
                );
            } else {
                for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
                    chunk, masks, bit_ext, chunk_func
                );
            }
        }
    );
}

// -------------------------------------------------------------------------------------------------
//     for_each_spaced_kmer_ascii_unchecked_chunked_scalar()
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract all spaced k-mers from an ASCII sequence that is known to contain only valid
 * nucleotides, and call a callback on each, for a single mask, using caller-provided temporary
 * storage.
 *
 * Each chunk is packed into `scratch`, and its spaced k-mers are extracted with
 * for_each_spaced_kmer_packed_unchecked_bytewise_scalar(). As for
 * for_each_kmer_ascii_unchecked_chunked_scalar(), the input should only consist of the characters
 * `ACGT` (upper or lower case); any other character is silently encoded as one of the four
 * nucleotides. See above for the scratch and the chunk size.
 *
 * @param seq       Input sequence.
 * @param mask      Spaced k-mer mask, in the representation that `bit_ext` expects, see
 *                  for_each_spaced_kmer_packed_unchecked_bytewise_scalar(). Its span must be in
 *                  [1, 32].
 * @param encoder   Word encoder selecting the Encoding and Layout, e.g. WordEncoderButterfly
 *                  or WordEncoderPext (sequence_pack/word_encoder.hpp).
 * @param bit_ext   Bit extraction functor, called as bit_ext(word, mask).
 * @param scratch   Reusable temporary packed storage for this call.
 * @param callback  Callback functor, called as callback(pos, mask_idx, spaced_kmer), or as
 *                  callback(pos, spaced_kmer) or callback(spaced_kmer). See
 *                  invoke_spaced_kmer_callback().
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Mask,
    typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<Mask> && (Mask::layout == Encoder::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_unchecked_chunked_scalar(
    std::string_view seq, Mask const& mask, Encoder const& encoder, BitExtract&& bit_ext,
    PackedSequence<Encoder::encoding, Encoder::layout>& scratch, Callback&& callback
) {
    for_each_spaced_kmer_ascii_chunked_scalar_<ChunkSize>(
        seq, check_spaced_kmer_span_(mask.span_k), mask, encoder, bit_ext, scratch, callback
    );
}

/**
 * @brief Same as above, for a set of masks, which all need to have the same span, as a std::span
 * of compile-time size, see the multi-mask overload of
 * for_each_spaced_kmer_packed_unchecked_bytewise_scalar(). The callback must be callable as
 * callback(pos, mask_idx, spaced_kmer).
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Mask,
    std::size_t N, typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<std::remove_const_t<Mask>> && (Mask::layout == Encoder::layout)
        && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_unchecked_chunked_scalar(
    std::string_view seq, std::span<Mask, N> masks, Encoder const& encoder, BitExtract&& bit_ext,
    PackedSequence<Encoder::encoding, Encoder::layout>& scratch, Callback&& callback
) {
    static_assert(N > 0, "Need at least one mask.");
    for_each_spaced_kmer_ascii_chunked_scalar_<ChunkSize>(
        seq, common_spaced_kmer_span_(masks), masks, encoder, bit_ext, scratch, callback
    );
}

/**
 * @brief Same as the single-mask overload above, using a thread-local buffer.
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Mask,
    typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<Mask> && (Mask::layout == Encoder::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_unchecked_chunked_scalar(
    std::string_view seq, Mask const& mask, Encoder const& encoder, BitExtract&& bit_ext,
    Callback&& callback
) {
    for_each_spaced_kmer_ascii_unchecked_chunked_scalar<ChunkSize>(
        seq, mask, encoder, std::forward<BitExtract>(bit_ext),
        ascii_chunked_buffer_<PackedSequence<Encoder::encoding, Encoder::layout>>(),
        std::forward<Callback>(callback)
    );
}

/**
 * @brief Same as the multi-mask overload above, using a thread-local buffer.
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Mask,
    std::size_t N, typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<std::remove_const_t<Mask>> && (Mask::layout == Encoder::layout)
        && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_unchecked_chunked_scalar(
    std::string_view seq, std::span<Mask, N> masks, Encoder const& encoder, BitExtract&& bit_ext,
    Callback&& callback
) {
    for_each_spaced_kmer_ascii_unchecked_chunked_scalar<ChunkSize>(
        seq, masks, encoder, std::forward<BitExtract>(bit_ext),
        ascii_chunked_buffer_<PackedSequence<Encoder::encoding, Encoder::layout>>(),
        std::forward<Callback>(callback)
    );
}

// -------------------------------------------------------------------------------------------------
//     for_each_spaced_kmer_ascii_checked_chunked_scalar()
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract the valid spaced k-mers from an ASCII sequence, and call a callback on each, for a
 * single mask, using caller-provided temporary storage.
 *
 * As for_each_spaced_kmer_ascii_unchecked_chunked_scalar(), packing each chunk with its validity,
 * see pack_sequence_checked(), and emitting only the spaced k-mers whose kept positions are valid
 * nucleotides, with for_each_spaced_kmer_packed_checked_bytewise_scalar(). Invalid characters at
 * the positions that the mask skips are allowed, as in
 * for_each_spaced_kmer_ascii_checked_rolling_scalar().
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Mask,
    typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<Mask> && (Mask::layout == Encoder::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_chunked_scalar(
    std::string_view seq, Mask const& mask, Encoder const& encoder, BitExtract&& bit_ext,
    PackedSequenceWithValidity<Encoder::encoding, Encoder::layout>& scratch, Callback&& callback
) {
    for_each_spaced_kmer_ascii_chunked_scalar_<ChunkSize>(
        seq, check_spaced_kmer_span_(mask.span_k), mask, encoder, bit_ext, scratch, callback
    );
}

/**
 * @brief Same as above, for a set of masks, as for the multi-mask overload of
 * for_each_spaced_kmer_ascii_unchecked_chunked_scalar(). The validity of each spaced k-mer is that
 * of the positions that its own mask keeps.
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Mask,
    std::size_t N, typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<std::remove_const_t<Mask>> && (Mask::layout == Encoder::layout)
        && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_chunked_scalar(
    std::string_view seq, std::span<Mask, N> masks, Encoder const& encoder, BitExtract&& bit_ext,
    PackedSequenceWithValidity<Encoder::encoding, Encoder::layout>& scratch, Callback&& callback
) {
    static_assert(N > 0, "Need at least one mask.");
    for_each_spaced_kmer_ascii_chunked_scalar_<ChunkSize>(
        seq, common_spaced_kmer_span_(masks), masks, encoder, bit_ext, scratch, callback
    );
}

/**
 * @brief Same as the single-mask overload above, using a thread-local buffer.
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Mask,
    typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<Mask> && (Mask::layout == Encoder::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_chunked_scalar(
    std::string_view seq, Mask const& mask, Encoder const& encoder, BitExtract&& bit_ext,
    Callback&& callback
) {
    for_each_spaced_kmer_ascii_checked_chunked_scalar<ChunkSize>(
        seq, mask, encoder, std::forward<BitExtract>(bit_ext),
        ascii_chunked_buffer_<PackedSequenceWithValidity<Encoder::encoding, Encoder::layout>>(),
        std::forward<Callback>(callback)
    );
}

/**
 * @brief Same as the multi-mask overload above, using a thread-local buffer.
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Mask,
    std::size_t N, typename BitExtract, typename Callback
>
    requires SpacedKmerMaskType<std::remove_const_t<Mask>> && (Mask::layout == Encoder::layout)
        && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_chunked_scalar(
    std::string_view seq, std::span<Mask, N> masks, Encoder const& encoder, BitExtract&& bit_ext,
    Callback&& callback
) {
    for_each_spaced_kmer_ascii_checked_chunked_scalar<ChunkSize>(
        seq, masks, encoder, std::forward<BitExtract>(bit_ext),
        ascii_chunked_buffer_<PackedSequenceWithValidity<Encoder::encoding, Encoder::layout>>(),
        std::forward<Callback>(callback)
    );
}

} // namespace fisk
