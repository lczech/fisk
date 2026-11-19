#pragma once

#include <cstddef>
#include <string_view>
#include <utility>

#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/simd/simd.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_chunked.hpp"
#include "fisk/kmer_extract/internal.hpp"
#include "fisk/kmer_extract/packed_bytewise_simd.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from ASCII, Chunked, SIMD
// =================================================================================================

// The SIMD counterparts of the functions in ascii_chunked_scalar.hpp: each chunk is packed into the
// scratch, and its k-mers are extracted with the extraction from packed sequences of a SIMD arch,
// see packed_bytewise_simd.hpp, which calls back with vectors of consecutive k-mers. As each chunk
// ends a run of that extraction, the last vector of every chunk, not only of the sequence, can
// have fewer valid lanes than the arch has lanes.

// -------------------------------------------------------------------------------------------------
//     Internal Helpers
// -------------------------------------------------------------------------------------------------

namespace detail {

// The shared implementation of the unchecked and checked functions below: with a
// PackedSequenceWithValidity as `scratch`, the chunks are packed with their validity, and the
// packed loop calls back with the valid lanes of its vectors, see for_each_ascii_chunk().
template <
    SimdArch Simd, std::size_t ChunkSize, WordEncoder Encoder,
    PackedSequenceType PackedSequenceChunk, typename Func
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_chunked_simd(
    std::string_view seq, std::size_t k, Encoder const& encoder, PackedSequenceChunk& scratch,
    Func& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k(32);
    }
    for_each_ascii_chunk<ChunkSize>(
        seq, k, encoder, scratch, func,
        [k](auto const& chunk, auto const& chunk_func) FISK_ALWAYS_INLINE_LAMBDA {
            if constexpr (PackedSequenceWithValidityType<PackedSequenceChunk>) {
                for_each_kmer_packed_checked_bytewise_simd<Simd>(chunk, k, chunk_func);
            } else {
                for_each_kmer_packed_unchecked_bytewise_simd<Simd>(chunk, k, chunk_func);
            }
        }
    );
}

} // namespace detail

// -------------------------------------------------------------------------------------------------
//     for_each_kmer_ascii_unchecked_chunked_simd()
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract all k-mers for k in [1, 32] from an ASCII sequence that is known to contain only
 * valid nucleotides with the given SIMD arch, yielding consecutive k-mers in the vectors of that
 * arch, using caller-provided temporary storage.
 *
 * As for_each_kmer_ascii_unchecked_chunked_scalar(), with the k-mers of each chunk extracted with
 * for_each_kmer_packed_unchecked_bytewise_simd(). Func is called as `func(pos, kmers,
 * valid_count)`, with a KmerVector<Encoder::encoding, Encoder::layout, Simd>: lane `j` of `kmers`
 * holds the k-mer starting at `pos + j` in `seq`, for `j < valid_count`. The single k-mers can be
 * taken out with for_each_valid_lane() or store_valid_lanes().
 *
 * The last vector of each chunk may be partial. With a ChunkSize that is a multiple of 32, each
 * chunk ends on a whole vector, so that the vectors are those of the packed loop on the whole
 * sequence: in order, and full except the last.
 *
 * The encoder that packs the chunks is independent of `Simd`: any word encoder works with any
 * arch.
 *
 * @tparam Simd      SIMD arch, such as SimdAVX2, to be given explicitly.
 * @tparam ChunkSize Number of nucleotides processed per chunk; must be > 0. A multiple of 32, such
 *                   as a power of two, keeps each chunk in whole blocks of the packed loops, see
 *                   run_packed_bytewise_blocks().
 *
 * @param seq        Input sequence.
 * @param k          K-mer size, in [1, 32].
 * @param encoder    Word encoder selecting the Encoding and Layout, e.g. WordEncoderDirectDouble
 *                   or WordEncoderPext (sequence_pack/word_encoder.hpp).
 * @param scratch    Reusable temporary packed storage for this call.
 * @param func       Callback function to be called for each vector of k-mers.
 */
template <
    SimdArch Simd, std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder,
    typename Func
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_unchecked_chunked_simd(
    std::string_view seq, std::size_t k, Encoder const& encoder,
    PackedSequence<Encoder::encoding, Encoder::layout>& scratch,
    Func&& func
) {
    // `func` is called from every chunk, so it is passed on as an lvalue, never forwarded.
    detail::for_each_kmer_ascii_chunked_simd<Simd, ChunkSize>(seq, k, encoder, scratch, func);
}

/**
 * @brief Same as above, using a thread-local buffer.
 *
 * The buffer is shared by all chunked functions on the same thread that use the same type of
 * scratch, see ascii_chunked_buffer(). The callback hence must not call any of them again with
 * the thread-local buffer. Use the overload taking `scratch` for nested calls.
 */
template <
    SimdArch Simd, std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder,
    typename Func
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_unchecked_chunked_simd(
    std::string_view seq, std::size_t k, Encoder const& encoder, Func&& func
) {
    for_each_kmer_ascii_unchecked_chunked_simd<Simd, ChunkSize>(
        seq, k, encoder,
        detail::ascii_chunked_buffer<PackedSequence<Encoder::encoding, Encoder::layout>>(),
        std::forward<Func>(func)
    );
}

// -------------------------------------------------------------------------------------------------
//     for_each_kmer_ascii_checked_chunked_simd()
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract the valid k-mers for k in [1, 32] from an ASCII sequence with the given SIMD arch,
 * yielding consecutive k-mers in the vectors of that arch, with their valid lanes, using
 * caller-provided temporary storage.
 *
 * As for_each_kmer_ascii_unchecked_chunked_simd(), packing each chunk with its validity, see
 * pack_sequence_checked(), and extracting its k-mers with
 * for_each_kmer_packed_checked_bytewise_simd(). Func is called as `func(pos, kmers, valid_lanes)`,
 * with a KmerVector<Encoder::encoding, Encoder::layout, Simd>: lane `j` of `kmers` holds the k-mer
 * starting at `pos + j` in `seq`, and is valid if it is set in `valid_lanes`, see
 * SimdVectorValidLanes, that is, if it consists of valid nucleotides, as for
 * for_each_kmer_ascii_checked_chunked_scalar(). The lanes beyond the last k-mer of a chunk are
 * never valid. The valid k-mers can be taken out with for_each_valid_lane() or
 * store_valid_lanes(). With a ChunkSize that is a multiple of 32, the vectors follow each other
 * without gaps, as for the unchecked function.
 */
template <
    SimdArch Simd, std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder,
    typename Func
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_checked_chunked_simd(
    std::string_view seq, std::size_t k, Encoder const& encoder,
    PackedSequenceWithValidity<Encoder::encoding, Encoder::layout>& scratch,
    Func&& func
) {
    // `func` is called from every chunk, so it is passed on as an lvalue, never forwarded.
    detail::for_each_kmer_ascii_chunked_simd<Simd, ChunkSize>(seq, k, encoder, scratch, func);
}

/**
 * @brief Same as above, using a thread-local buffer, see the thread-local overload of
 * for_each_kmer_ascii_unchecked_chunked_simd().
 */
template <
    SimdArch Simd, std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder,
    typename Func
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_checked_chunked_simd(
    std::string_view seq, std::size_t k, Encoder const& encoder, Func&& func
) {
    for_each_kmer_ascii_checked_chunked_simd<Simd, ChunkSize>(
        seq, k, encoder,
        detail::ascii_chunked_buffer<
            PackedSequenceWithValidity<Encoder::encoding, Encoder::layout>
        >(),
        std::forward<Func>(func)
    );
}

} // namespace fisk
