#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "fisk/kmer_extract/packed.hpp"
#include "fisk/seq_pack/seq_pack.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from ASCII, Assuming Valid Input
// =================================================================================================

/**
 * @brief Default chunk size, in bases, for for_each_kmer_ascii_assume_valid().
 *
 * Packs down to a 16 KiB buffer, which fits comfortably in L1 cache.
 */
inline constexpr std::size_t kDefaultAsciiChunkSize = 65536;

// -------------------------------------------------------------------------------------------------
//     Internal Helpers
// -------------------------------------------------------------------------------------------------

FISK_NOINLINE_COLD
inline void throw_invalid_chunk_size_()
{
    throw std::invalid_argument(
        "Invalid call to for_each_kmer_ascii_assume_valid() with chunk_size == 0"
    );
}

// One reused buffer per thread and Encoder, shared across all callback types.
template <WordEncoder Encoder>
inline PackedSequence<Encoder::encoding, Encoder::layout>& ascii_assume_valid_buffer_()
{
    thread_local PackedSequence<Encoder::encoding, Encoder::layout> buffer;
    return buffer;
}

// Each chunk packs its chunk_size new bases plus the next K-1 bases, so that it emits exactly the
// k-mers starting within its own new bases: across chunks, every k-mer is emitted once, in order.
// The impl functions do not check for sequences shorter than K themselves, hence the check here.
template <WordEncoder Encoder, unsigned K, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_assume_valid_chunked_(
    std::string_view seq, Encoder const& encoder, std::size_t chunk_size, Func& func
) {
    constexpr Encoding E = Encoder::encoding;
    constexpr Layout L = Encoder::layout;

    // The impl functions report positions relative to the chunk they are given; this shifts them
    // to positions in the whole sequence, and forwards to `func` in whichever form it takes.
    struct ChunkCallback
    {
        Func& func;
        std::size_t offset;

        FISK_ALWAYS_INLINE
        void operator()(std::size_t pos, Kmer<E, L> const& kmer) const
        {
            invoke_kmer_callback(func, offset + pos, kmer);
        }
    };

    auto& buffer = ascii_assume_valid_buffer_<Encoder>();

    std::size_t const seq_len = seq.size();
    std::size_t const margin  = K - 1;

    for (std::size_t chunk_start = 0; chunk_start < seq_len; chunk_start += chunk_size) {
        std::size_t const chunk_end = std::min(chunk_start + chunk_size, seq_len);
        std::size_t const view_end  = std::min(chunk_end + margin, seq_len);

        pack_sequence(seq.substr(chunk_start, view_end - chunk_start), encoder, buffer);
        if (buffer.length < K) {
            continue;
        }
        ChunkCallback chunk_func{func, chunk_start};
        if constexpr (K >= 30) {
            for_each_kmer_packed_aligned_wide_impl_<E, L, K>(buffer, chunk_func);
        } else {
            for_each_kmer_packed_aligned_narrow_impl_<E, L, K>(buffer, chunk_func);
        }
    }
}

// -------------------------------------------------------------------------------------------------
//     for_each_kmer_ascii_assume_valid()
// -------------------------------------------------------------------------------------------------

/**
 * @brief Extract all k-mers for k in [1, 32] from an ASCII sequence that is known to contain only
 * valid nucleotides, and call a callback on each.
 *
 * Faster than for_each_kmer(), as it does no validity checking, and encodes the input a whole word
 * at a time. The sequence is processed in chunks of `chunk_size` bases, so that memory use stays
 * small and constant regardless of sequence length.
 *
 * The input should only consist of the characters `ACGT` (upper or lower case). Any other character
 * is silently encoded as one of the four nucleotides, changing every k-mer that overlaps it. If
 * this is accetable (for instance for quick homology checks), this function is faster than the
 * checked alternatives, such as for_each_kmer().
 *
 * The callback receives a `Kmer<Encoder::encoding, Encoder::layout>`, in sequence order. It may be
 * called either as `func(pos, kmer)` or as `func(kmer)`, where `pos` is the start position of the
 * k-mer in `seq`; see invoke_kmer_callback(). The callback must not call this function again with
 * the same encoder type on the same thread, as they would share an internal buffer.
 *
 * @param seq        Input sequence.
 * @param k          K-mer size, in [1, 32].
 * @param encoder    Word encoder selecting the Encoding and Layout, e.g. WordEncoderButterfly
 *                   or WordEncoderPext (seq_pack.hpp).
 * @param chunk_size Number of bases processed per chunk; must be > 0.
 * @param func       Callback function to be called for each k-mer.
 */
template <WordEncoder Encoder, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_assume_valid(
    std::string_view seq, std::size_t k, Encoder const& encoder,
    std::size_t chunk_size, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (chunk_size == 0) {
        throw_invalid_chunk_size_();
    }
    if (seq.size() < k) {
        return;
    }

    // `func` is called from every chunk, so it is passed on as an lvalue, never forwarded.
    #define FISK_ASCII_ASSUME_VALID_CASE_(K_) \
        case K_: \
            for_each_kmer_ascii_assume_valid_chunked_<Encoder, K_>(seq, encoder, chunk_size, func); \
            break;

    switch (k) {
        FISK_ASCII_ASSUME_VALID_CASE_(1)
        FISK_ASCII_ASSUME_VALID_CASE_(2)
        FISK_ASCII_ASSUME_VALID_CASE_(3)
        FISK_ASCII_ASSUME_VALID_CASE_(4)
        FISK_ASCII_ASSUME_VALID_CASE_(5)
        FISK_ASCII_ASSUME_VALID_CASE_(6)
        FISK_ASCII_ASSUME_VALID_CASE_(7)
        FISK_ASCII_ASSUME_VALID_CASE_(8)
        FISK_ASCII_ASSUME_VALID_CASE_(9)
        FISK_ASCII_ASSUME_VALID_CASE_(10)
        FISK_ASCII_ASSUME_VALID_CASE_(11)
        FISK_ASCII_ASSUME_VALID_CASE_(12)
        FISK_ASCII_ASSUME_VALID_CASE_(13)
        FISK_ASCII_ASSUME_VALID_CASE_(14)
        FISK_ASCII_ASSUME_VALID_CASE_(15)
        FISK_ASCII_ASSUME_VALID_CASE_(16)
        FISK_ASCII_ASSUME_VALID_CASE_(17)
        FISK_ASCII_ASSUME_VALID_CASE_(18)
        FISK_ASCII_ASSUME_VALID_CASE_(19)
        FISK_ASCII_ASSUME_VALID_CASE_(20)
        FISK_ASCII_ASSUME_VALID_CASE_(21)
        FISK_ASCII_ASSUME_VALID_CASE_(22)
        FISK_ASCII_ASSUME_VALID_CASE_(23)
        FISK_ASCII_ASSUME_VALID_CASE_(24)
        FISK_ASCII_ASSUME_VALID_CASE_(25)
        FISK_ASCII_ASSUME_VALID_CASE_(26)
        FISK_ASCII_ASSUME_VALID_CASE_(27)
        FISK_ASCII_ASSUME_VALID_CASE_(28)
        FISK_ASCII_ASSUME_VALID_CASE_(29)
        FISK_ASCII_ASSUME_VALID_CASE_(30)
        FISK_ASCII_ASSUME_VALID_CASE_(31)
        FISK_ASCII_ASSUME_VALID_CASE_(32)
    }

    #undef FISK_ASCII_ASSUME_VALID_CASE_
}

/**
 * @brief Same as above, using kDefaultAsciiChunkSize.
 */
template <WordEncoder Encoder, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_assume_valid(
    std::string_view seq, std::size_t k, Encoder const& encoder, Func&& func
) {
    for_each_kmer_ascii_assume_valid(
        seq, k, encoder, kDefaultAsciiChunkSize, std::forward<Func>(func)
    );
}

} // namespace fisk
