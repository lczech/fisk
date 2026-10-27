#pragma once

#include <cstddef>
#include <string_view>
#include <utility>

#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_chunked.hpp"
#include "fisk/kmer_extract/internal.hpp"
#include "fisk/kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from ASCII, Chunked, Scalar
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Internal Helpers
// -------------------------------------------------------------------------------------------------

// The shared implementation of the unchecked and checked functions below: with a
// PackedSequenceWithValidity as `scratch`, the chunks are packed with their validity, and the
// packed loops emit their valid k-mers only, see for_each_ascii_chunk_().
template <
    std::size_t ChunkSize, WordEncoder Encoder, PackedSequenceType PackedSequenceChunk,
    typename Func
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_chunked_scalar_(
    std::string_view seq, std::size_t k, Encoder const& encoder, PackedSequenceChunk& scratch,
    Func& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }

    // The dispatch on k is outside of the chunk loop, so that each k gets a chunk loop of its own,
    // compiled for that k, instead of one chunk loop that dispatches on k for each chunk.
    #define FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(K_) \
        case K_: \
            for_each_ascii_chunk_<ChunkSize>( \
                seq, K_, encoder, scratch, func, \
                [](auto const& chunk, auto const& chunk_func) FISK_ALWAYS_INLINE_LAMBDA { \
                    if constexpr (K_ <= 29) { \
                        for_each_kmer_packed_bytewise_scalar_narrow_width_<K_>(chunk, chunk_func); \
                    } else { \
                        for_each_kmer_packed_bytewise_scalar_full_width_<K_>(chunk, chunk_func); \
                    } \
                } \
            ); \
            break;

    switch (k) {
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(1)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(2)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(3)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(4)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(5)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(6)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(7)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(8)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(9)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(10)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(11)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(12)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(13)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(14)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(15)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(16)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(17)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(18)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(19)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(20)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(21)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(22)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(23)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(24)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(25)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(26)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(27)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(28)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(29)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(30)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(31)
        FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_(32)
    }

    #undef FISK_KMER_ASCII_CHUNKED_SCALAR_CASE_
}

// -------------------------------------------------------------------------------------------------
//     for_each_kmer_ascii_unchecked_chunked_scalar()
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract all k-mers for k in [1, 32] from an ASCII sequence that is known to contain only
 * valid nucleotides, and call a callback on each, using caller-provided temporary storage.
 *
 * Faster than the checked extraction from text, such as
 * for_each_kmer_ascii_checked_rolling_scalar(), as it does no validity checking, and encodes the
 * input a whole word at a time. The sequence is processed in chunks of `ChunkSize` bases, so that
 * memory use stays small and constant regardless of sequence length. Each chunk is packed into
 * `scratch`, and its k-mers are extracted with for_each_kmer_packed_unchecked_bytewise_scalar().
 *
 * The input should only consist of the characters `ACGT` (upper or lower case). Any other character
 * is silently encoded as one of the four nucleotides, changing every k-mer that overlaps it. If
 * this is acceptable (for instance for quick homology checks), this function is faster than the
 * checked alternatives.
 *
 * The callback receives a `Kmer<Encoder::encoding, Encoder::layout>`, in sequence order. It may be
 * called either as `func(pos, kmer)` or as `func(kmer)`, where `pos` is the start position of the
 * k-mer in `seq`; see invoke_kmer_callback(). `scratch` is reusable temporary packed storage; its
 * contents after the call are unspecified. Overlapping, nested, or concurrent calls must use
 * distinct scratch buffers.
 *
 * @tparam ChunkSize Number of bases processed per chunk; must be > 0. A multiple of 32, such as a
 *                   power of two, keeps each chunk in whole blocks of the packed loops, see
 *                   run_packed_bytewise_blocks_().
 *
 * @param seq        Input sequence.
 * @param k          K-mer size, in [1, 32].
 * @param encoder    Word encoder selecting the Encoding and Layout, e.g. WordEncoderButterfly
 *                   or WordEncoderPext (sequence_pack/word_encoder.hpp).
 * @param scratch    Reusable temporary packed storage for this call.
 * @param func       Callback function to be called for each k-mer.
 */
template <std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_unchecked_chunked_scalar(
    std::string_view seq, std::size_t k, Encoder const& encoder,
    PackedSequence<Encoder::encoding, Encoder::layout>& scratch,
    Func&& func
) {
    // `func` is called from every chunk, so it is passed on as an lvalue, never forwarded.
    for_each_kmer_ascii_chunked_scalar_<ChunkSize>(seq, k, encoder, scratch, func);
}

/**
 * @brief Same as above, using a thread-local buffer.
 *
 * The buffer is shared by all chunked functions on the same thread that use the same type of
 * scratch, see ascii_chunked_buffer_(). The callback hence must not call any of them again with
 * the thread-local buffer. Use the overload taking `scratch` for nested calls.
 */
template <std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_unchecked_chunked_scalar(
    std::string_view seq, std::size_t k, Encoder const& encoder, Func&& func
) {
    for_each_kmer_ascii_unchecked_chunked_scalar<ChunkSize>(
        seq, k, encoder,
        ascii_chunked_buffer_<PackedSequence<Encoder::encoding, Encoder::layout>>(),
        std::forward<Func>(func)
    );
}

} // namespace fisk
