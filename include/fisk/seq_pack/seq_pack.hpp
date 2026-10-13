#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/types.hpp"
#include "fisk/seq_pack/word_encoder.hpp"

namespace fisk {

// =================================================================================================
//     Pack Sequence
// =================================================================================================

// -----------------------------------------------------------------------------
//     Chunk Write Helper
// -----------------------------------------------------------------------------

/**
 * @brief Write the packed bases of one encoder call to `dest`, respecting `L`.
 */
template <Layout L, int Bytes = 2>
inline void write_two_bit_chunk_(char* dest, std::uint64_t value) noexcept
{
    // Each word encoder packs 8 bases (16 bits) into the low bits of its return value. A byte is
    // the smallest addressable unit of PackedSequence's storage, so those 16 bits become 2
    // sequential output bytes. For `kLSB`, the encoder already produces those 2 bytes in the
    // right relative order for free (the first of the 8 bases ends up in the low bits of the first
    // byte, and so on); for `kMSB`, the two bytes come out address-reversed relative to base order
    // (the byte holding the later 4 bases lands first), so they need a fix-up swap before the
    // sequential write.
    //
    // `Bytes` (1 or 2) is a template parameter rather than a runtime one so that the hot full-chunk
    // call sites below (always 2) keep a fixed-width store with no added branch; only the ragged
    // tail chunk (<=4 real bases) ever instantiates the 1-byte write, and byte 0 -- the first of
    // the two output bytes post-swap -- is the one holding those bases in both layouts.

    static_assert(Bytes == 1 || Bytes == 2, "write_two_bit_chunk_() only supports 1 or 2 bytes");
    assert(dest != nullptr);
    assert(value <= 0xFFFF);
    std::uint16_t v16 = static_cast<std::uint16_t>(value);
    if constexpr (L == Layout::kMSB) {
        v16 = byte_swap_16(v16);
    } else if constexpr (L == Layout::kLSB) {
        // The bytes are already in output order.
    } else {
        static_assert(dependent_false_v<L>, "Unhandled Layout in write_two_bit_chunk_()");
    }
    std::memcpy(dest, &v16, Bytes);
}

// -----------------------------------------------------------------------------
//     Pack Sequence
// -----------------------------------------------------------------------------

/**
 * @brief Pack a whole ASCII sequence into a PackedSequence, reusing existing storage.
 *
 * Processes the sequence in 8-byte chunks via `encoder` (see word_encoder.hpp), writing
 * each chunk's 16-bit result to its 2 sequential output bytes at `out.data[i/4 .. i/4+1]`, where
 * `i` is the chunk's starting byte offset into `seq` (see write_two_bit_chunk_() for the per-byte
 * ordering). Encoders with more than one lane process `lanes` chunks at once, and only the chunks
 * at the end that do not fill a whole vector one at a time.
 *
 * `Encoder::encoding` and `Encoder::layout` (see Encoding and Layout in core/types.hpp) select
 * the out type. Passing an encoder whose encoding or layout differs from `out`'s is a compile
 * error.
 *
 * Forced inline, as the compilers otherwise tend to keep this out of line, where they optimize
 * the loops less well for the given encoder.
 */
template <WordEncoder Encoder>
FISK_ALWAYS_INLINE
inline void pack_sequence(
    std::string_view seq,
    Encoder const& encoder,
    PackedSequence<Encoder::encoding, Encoder::layout>& out
) {
    constexpr Layout layout = Encoder::layout;

    std::size_t const seq_len   = seq.size();
    std::size_t const num_bytes = (seq_len + 3) / 4;

    out.length = seq_len;

    // Every byte in the resized output is written below, including the partial final byte. Avoid
    // clearing retained storage first: callers commonly reuse `out` across many sequences/chunks.
    out.data.resize(num_bytes);
    char* const out_bytes = reinterpret_cast<char*>(out.data.data());
    char const* const data = seq.data();

    // Read the 8-byte chunk starting at seq byte offset `off`.
    auto read_chunk = [data](std::size_t off) -> std::uint64_t {
        std::uint64_t word;
        std::memcpy(&word, data + off, 8);
        return word;
    };

    // byte offset into seq
    std::size_t i = 0;

    // Full vectors of `lanes` chunks.
    if constexpr (Encoder::lanes > 1) {
        constexpr std::size_t vec_bytes = Encoder::lanes * 8;
        for (; i + vec_bytes <= seq_len; i += vec_bytes) {
            auto const x = encoder(
                Encoder::load_unaligned(reinterpret_cast<std::uint64_t const*>(data + i))
            );
            Encoder::store_packed(x, out_bytes + i / 4);
        }
    }

    // Full 8-base chunks.
    for (; i + 8 <= seq_len; i += 8) {
        write_two_bit_chunk_<layout>(out_bytes + i / 4, encoder(read_chunk(i)));
    }

    // Remaining < 8 bases: zero-pad into a local word before encoding. `out` holds no trailing
    // bytes beyond its real content, so a <=4-base remainder (one real output byte, not two) must
    // write only that one byte -- writing the usual 2 would spill past the allocation.
    if (i < seq_len) {
        std::size_t const remaining = seq_len - i;
        std::uint64_t word = 0;
        std::memcpy(&word, data + i, remaining);
        if (remaining <= 4) {
            write_two_bit_chunk_<layout, 1>(out_bytes + i / 4, encoder(word));
        } else {
            write_two_bit_chunk_<layout, 2>(out_bytes + i / 4, encoder(word));
        }
    }
}

/**
 * @brief Pack a whole ASCII sequence into a freshly allocated PackedSequence.
 *
 * Convenience wrapper around the in-place overload above. Prefer that overload directly when
 * packing many sequences in a loop (e.g. in a benchmark), reusing one PackedSequence across
 * calls, to avoid attributing repeated heap allocation to whatever is being measured.
 */
template <WordEncoder Encoder>
inline PackedSequence<Encoder::encoding, Encoder::layout> pack_sequence(
    std::string_view seq, Encoder const& encoder
) {
    PackedSequence<Encoder::encoding, Encoder::layout> out;
    pack_sequence(seq, encoder, out);
    return out;
}

} // namespace fisk
