#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

#include "fisk/util/intrinsics.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from ASCII, Chunked
// =================================================================================================

// The chunked extraction from ASCII packs a sequence chunk by chunk into a reused PackedSequence,
// and runs the extraction from packed sequences on each chunk. The chunked functions of the
// contiguous and the spaced k-mers share the driver below, and differ only in the packed loop that
// they run on each chunk.

/**
 * @brief Default chunk size, in bases, of the chunked k-mer extraction from ASCII.
 *
 * Packs down to a 16 KiB buffer, which fits comfortably in L1 cache.
 */
inline constexpr std::size_t kDefaultAsciiChunkSize = 65536;

// -------------------------------------------------------------------------------------------------
//     Internal Helpers
// -------------------------------------------------------------------------------------------------

// One reused buffer per thread and type of scratch, shared by all chunked functions and callback
// types that use it.
template <PackedSequenceType PackedSequenceChunk>
inline PackedSequenceChunk& ascii_chunked_buffer_()
{
    thread_local PackedSequenceChunk buffer;
    return buffer;
}

// The callback that the packed loops get for a chunk starting at `offset`: it shifts the positions
// that they report, relative to the chunk, to positions in the whole sequence, and forwards
// everything else as it is. It is constrained to the forms that `func` accepts, so that the packed
// loops pick the same callback form for it as for `func` itself. In every form, a first argument
// of type std::size_t is the position; the forms without it start with a (spaced) k-mer. Made here
// rather than in the driver, so that its type depends on `Func` only, not on the chunk size, and
// the packed loops are compiled once per callback.
template <typename Func>
FISK_ALWAYS_INLINE
inline auto ascii_chunk_callback_(Func& func, std::size_t offset)
{
    return [&func, offset]<typename First, typename... Args>(
        First&& first, Args&&... args
    ) FISK_ALWAYS_INLINE_LAMBDA requires std::invocable<Func&, First, Args...> {
        if constexpr (std::is_same_v<std::remove_cvref_t<First>, std::size_t>) {
            func(offset + first, std::forward<Args>(args)...);
        } else {
            func(std::forward<First>(first), std::forward<Args>(args)...);
        }
    };
}

// -------------------------------------------------------------------------------------------------
//     Driver
// -------------------------------------------------------------------------------------------------

/**
 * @brief Pack `seq` chunk by chunk into `scratch`, and call `run(scratch, chunk_func)` for each
 * chunk, where `chunk_func` forwards to `func` with positions in `seq`.
 *
 * Each chunk packs its `ChunkSize` new bases plus the next `span - 1`, so that the packed loop that
 * `run` calls emits exactly the k-mers (or spaced k-mers of span `span`) that start within its own
 * new bases: across chunks, every k-mer is emitted once, in order. To the packed loop, the end of
 * the scratch is the end of the sequence, so that it emits nothing that starts after the new bases
 * of the chunk; with validity, the positions after them have no valid lanes. The chunks after the
 * last k-mer start are not packed.
 *
 * With a PackedSequenceWithValidity as `scratch`, the chunks are packed with
 * pack_sequence_checked(), and otherwise with pack_sequence_unchecked().
 */
template <
    std::size_t ChunkSize, WordEncoder Encoder, PackedSequenceType PackedSequenceChunk,
    typename Func, typename Run
>
    requires (PackedSequenceChunk::encoding == Encoder::encoding)
        && (PackedSequenceChunk::layout == Encoder::layout)
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_ascii_chunk_(
    std::string_view seq, std::size_t span, Encoder const& encoder, PackedSequenceChunk& scratch,
    Func& func, Run&& run
) {
    static_assert(ChunkSize > 0, "The chunk size must be > 0.");
    std::size_t const seq_len = seq.size();
    for (std::size_t chunk_start = 0; seq_len - chunk_start >= span; chunk_start += ChunkSize) {
        std::size_t const chunk_end = std::min(chunk_start + ChunkSize, seq_len);
        std::size_t const view_end  = std::min(chunk_end + span - 1, seq_len);
        std::string_view const view = seq.substr(chunk_start, view_end - chunk_start);

        if constexpr (PackedSequenceWithValidityType<PackedSequenceChunk>) {
            pack_sequence_checked(view, encoder, scratch);
        } else {
            pack_sequence_unchecked(view, encoder, scratch);
        }
        run(scratch, ascii_chunk_callback_(func, chunk_start));

        if (chunk_end == seq_len) {
            break;
        }
    }
}

} // namespace fisk
