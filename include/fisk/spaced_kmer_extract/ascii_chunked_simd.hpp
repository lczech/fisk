#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "fisk/util/intrinsics.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_chunked.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/spaced_kmer_extract/internal.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Extraction from ASCII, Chunked, SIMD
// =================================================================================================

// The SIMD counterparts of the functions in ascii_chunked_scalar.hpp: each chunk is packed into the
// scratch, and its spaced k-mers are extracted with the SIMD bit extraction kernels of
// packed_bytewise_simd.hpp, whose arch is that of the kernels. The kernels need to have the Layout
// of the encoder, and the encoder is independent of their arch. The callback forms are those of
// for_each_spaced_kmer_packed_unchecked_bytewise_simd() and
// for_each_spaced_kmer_packed_checked_bytewise_simd(), with `pos` the start position of the
// spaced k-mers' spans in the whole sequence. As each chunk ends a run of that extraction, the
// last vector of every chunk, not only of the sequence, can have fewer valid lanes than the arch
// has lanes. See ascii_chunked_scalar.hpp for the scratch and the chunk size.

// -------------------------------------------------------------------------------------------------
//     Internal Helpers
// -------------------------------------------------------------------------------------------------

// The shared implementation of the functions below, see
// for_each_spaced_kmer_ascii_chunked_scalar_() (ascii_chunked_scalar.hpp).
template <
    std::size_t ChunkSize, WordEncoder Encoder, PackedSequenceType PackedSequenceChunk,
    typename Kernel, std::size_t N, typename Callback
>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_chunked_simd_(
    std::string_view seq, std::span<Kernel, N> kernels, Encoder const& encoder,
    PackedSequenceChunk& scratch, Callback& callback
) {
    static_assert(N > 0, "Need at least one kernel.");
    for_each_ascii_chunk_<ChunkSize>(
        seq, common_spaced_kmer_span_(kernels), encoder, scratch, callback,
        [&](auto const& chunk, auto const& chunk_func) FISK_ALWAYS_INLINE_LAMBDA {
            if constexpr (PackedSequenceWithValidityType<PackedSequenceChunk>) {
                for_each_spaced_kmer_packed_checked_bytewise_simd(chunk, kernels, chunk_func);
            } else {
                for_each_spaced_kmer_packed_unchecked_bytewise_simd(chunk, kernels, chunk_func);
            }
        }
    );
}

// -------------------------------------------------------------------------------------------------
//     for_each_spaced_kmer_ascii_unchecked_chunked_simd()
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract all spaced k-mers from an ASCII sequence that is known to contain only valid
 * nucleotides with SIMD bit extraction, and call a callback on them, for a set of kernels, using
 * caller-provided temporary storage.
 *
 * Each chunk is packed into `scratch`, and its spaced k-mers are extracted with
 * for_each_spaced_kmer_packed_unchecked_bytewise_simd(), which calls back as
 * `callback(pos, mask_idx, spaced_kmers, valid_count)`. As for
 * for_each_kmer_ascii_unchecked_chunked_scalar(), the input should only consist of the characters
 * `ACGT` (upper or lower case); any other character is silently encoded as one of the four
 * nucleotides.
 *
 * @tparam Kernel   Bit extraction kernel of a SIMD arch, tagged as spaced k-mer mask with the
 *                  Layout of the encoder, see make_spaced_kmer_kernel(), and possibly const. The
 *                  spans of all kernels must be the same, and in [1, 32].
 * @tparam N        Number of kernels, fixed at compile time; see dispatch_static_span() for a
 *                  std::vector of kernels.
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Kernel,
    std::size_t N, typename Callback
>
    requires SpacedKmerMaskType<std::remove_const_t<Kernel>>
        && BitExtractKernel<typename Kernel::bit_extract_type>
        && (Kernel::layout == Encoder::layout) && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_unchecked_chunked_simd(
    std::string_view seq, std::span<Kernel, N> kernels, Encoder const& encoder,
    PackedSequence<Encoder::encoding, Encoder::layout>& scratch, Callback&& callback
) {
    for_each_spaced_kmer_ascii_chunked_simd_<ChunkSize>(seq, kernels, encoder, scratch, callback);
}

/**
 * @brief Same as above, for a single kernel, with `mask_idx` always 0. The callback can also be
 * called without it, as callback(pos, spaced_kmers, valid_count).
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, Layout L,
    typename Kernel, typename Callback
>
    requires BitExtractKernel<Kernel> && (L == Encoder::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_unchecked_chunked_simd(
    std::string_view seq, SpacedKmerMask<L, Kernel> const& kernel, Encoder const& encoder,
    PackedSequence<Encoder::encoding, Encoder::layout>& scratch, Callback&& callback
) {
    // A copy of the kernel of the caller, so that the loop can keep its values in registers: with
    // a reference, it would have to reload them after each store, which might change them.
    std::array<SpacedKmerMask<L, Kernel>, 1> const kernels{{kernel}};
    for_each_spaced_kmer_ascii_chunked_simd_<ChunkSize>(
        seq, std::span{kernels}, encoder, scratch, callback
    );
}

/**
 * @brief Same as the overload for a set of kernels above, using a thread-local buffer.
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Kernel,
    std::size_t N, typename Callback
>
    requires SpacedKmerMaskType<std::remove_const_t<Kernel>>
        && BitExtractKernel<typename Kernel::bit_extract_type>
        && (Kernel::layout == Encoder::layout) && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_unchecked_chunked_simd(
    std::string_view seq, std::span<Kernel, N> kernels, Encoder const& encoder,
    Callback&& callback
) {
    for_each_spaced_kmer_ascii_unchecked_chunked_simd<ChunkSize>(
        seq, kernels, encoder,
        ascii_chunked_buffer_<PackedSequence<Encoder::encoding, Encoder::layout>>(),
        std::forward<Callback>(callback)
    );
}

/**
 * @brief Same as the overload for a single kernel above, using a thread-local buffer.
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, Layout L,
    typename Kernel, typename Callback
>
    requires BitExtractKernel<Kernel> && (L == Encoder::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_unchecked_chunked_simd(
    std::string_view seq, SpacedKmerMask<L, Kernel> const& kernel, Encoder const& encoder,
    Callback&& callback
) {
    for_each_spaced_kmer_ascii_unchecked_chunked_simd<ChunkSize>(
        seq, kernel, encoder,
        ascii_chunked_buffer_<PackedSequence<Encoder::encoding, Encoder::layout>>(),
        std::forward<Callback>(callback)
    );
}

// -------------------------------------------------------------------------------------------------
//     for_each_spaced_kmer_ascii_checked_chunked_simd()
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract the valid spaced k-mers from an ASCII sequence with SIMD bit extraction, and call
 * a callback on them, for a set of kernels, using caller-provided temporary storage.
 *
 * As for_each_spaced_kmer_ascii_unchecked_chunked_simd(), packing each chunk with its validity,
 * see pack_sequence_checked(), and extracting its spaced k-mers with
 * for_each_spaced_kmer_packed_checked_bytewise_simd(), which calls back with the valid lanes of
 * each vector, as `callback(pos, mask_idx, spaced_kmers, valid_lanes)`: lane `j` is valid if all
 * positions that mask `mask_idx` keeps are valid nucleotides in the span at `pos + j`.
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Kernel,
    std::size_t N, typename Callback
>
    requires SpacedKmerMaskType<std::remove_const_t<Kernel>>
        && BitExtractKernel<typename Kernel::bit_extract_type>
        && (Kernel::layout == Encoder::layout) && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_chunked_simd(
    std::string_view seq, std::span<Kernel, N> kernels, Encoder const& encoder,
    PackedSequenceWithValidity<Encoder::encoding, Encoder::layout>& scratch, Callback&& callback
) {
    for_each_spaced_kmer_ascii_chunked_simd_<ChunkSize>(seq, kernels, encoder, scratch, callback);
}

/**
 * @brief Same as above, for a single kernel, with `mask_idx` always 0. The callback can also be
 * called without it, as callback(pos, spaced_kmers, valid_lanes).
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, Layout L,
    typename Kernel, typename Callback
>
    requires BitExtractKernel<Kernel> && (L == Encoder::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_chunked_simd(
    std::string_view seq, SpacedKmerMask<L, Kernel> const& kernel, Encoder const& encoder,
    PackedSequenceWithValidity<Encoder::encoding, Encoder::layout>& scratch, Callback&& callback
) {
    // A copy of the kernel of the caller, so that the loop can keep its values in registers: with
    // a reference, it would have to reload them after each store, which might change them.
    std::array<SpacedKmerMask<L, Kernel>, 1> const kernels{{kernel}};
    for_each_spaced_kmer_ascii_chunked_simd_<ChunkSize>(
        seq, std::span{kernels}, encoder, scratch, callback
    );
}

/**
 * @brief Same as the overload for a set of kernels above, using a thread-local buffer.
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, typename Kernel,
    std::size_t N, typename Callback
>
    requires SpacedKmerMaskType<std::remove_const_t<Kernel>>
        && BitExtractKernel<typename Kernel::bit_extract_type>
        && (Kernel::layout == Encoder::layout) && (N != std::dynamic_extent)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_chunked_simd(
    std::string_view seq, std::span<Kernel, N> kernels, Encoder const& encoder,
    Callback&& callback
) {
    for_each_spaced_kmer_ascii_checked_chunked_simd<ChunkSize>(
        seq, kernels, encoder,
        ascii_chunked_buffer_<PackedSequenceWithValidity<Encoder::encoding, Encoder::layout>>(),
        std::forward<Callback>(callback)
    );
}

/**
 * @brief Same as the overload for a single kernel above, using a thread-local buffer.
 */
template <
    std::size_t ChunkSize = kDefaultAsciiChunkSize, WordEncoder Encoder, Layout L,
    typename Kernel, typename Callback
>
    requires BitExtractKernel<Kernel> && (L == Encoder::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_chunked_simd(
    std::string_view seq, SpacedKmerMask<L, Kernel> const& kernel, Encoder const& encoder,
    Callback&& callback
) {
    for_each_spaced_kmer_ascii_checked_chunked_simd<ChunkSize>(
        seq, kernel, encoder,
        ascii_chunked_buffer_<PackedSequenceWithValidity<Encoder::encoding, Encoder::layout>>(),
        std::forward<Callback>(callback)
    );
}

} // namespace fisk
