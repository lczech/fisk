#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"

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
//     Pack Loop
// -----------------------------------------------------------------------------

/**
 * @brief Pack `seq` into the codes at `codes`, and with `WithValidity`, also its validity bits
 * into `valid`.
 *
 * The loop shared by pack_sequence_unchecked() and pack_sequence_checked(). `codes` has to hold
 * `ceil(seq.size() / 4)` bytes, all of which are written, including the unused bits of the last
 * one. Without `WithValidity`, `valid` is unused and may be null. With it, `valid` has to hold
 * `ceil(seq.size() / 8)` bytes, all of which are written as well.
 */
template <bool WithValidity, WordEncoder Encoder>
FISK_ALWAYS_INLINE
inline void pack_sequence_(
    std::string_view seq,
    Encoder const& encoder,
    std::uint8_t* codes,
    std::uint8_t* valid
) {
    constexpr Layout layout = Encoder::layout;

    std::size_t const seq_len = seq.size();
    char* const out_bytes = reinterpret_cast<char*>(codes);
    char const* const data = seq.data();

    // Read the 8-byte chunk starting at seq byte offset `off`.
    auto read_chunk = [data](std::size_t off) -> std::uint64_t {
        std::uint64_t word;
        std::memcpy(&word, data + off, 8);
        return word;
    };

    // byte offset into seq. Each 8 bytes of it give one byte of validity bits, at offset `i / 8`.
    std::size_t i = 0;

    // End of the full 8-base chunks.
    std::size_t const words_end = seq_len - (seq_len % 8);

    // The main work for encoders with more than one lane: all full 8-base chunks of sequences of
    // at least one vector, `lanes` of them at a time. What they leave is at most the last bases
    // that do not fill a chunk.
    if constexpr (Encoder::lanes > 1) {
        constexpr std::size_t vec_bytes = Encoder::lanes * 8;
        auto pack_vector = [&](std::size_t off) {
            auto const x = Encoder::load_unaligned(data + off);
            Encoder::store_packed(encoder(x), out_bytes + off / 4);
            if constexpr (WithValidity) {
                Encoder::store_valid_bits(x, valid + off / 8);
            }
        };

        // Full vectors of `lanes` chunks.
        for (; i + vec_bytes <= seq_len; i += vec_bytes) {
            pack_vector(i);
        }

        // The remaining full chunks as one more vector that ends at the last of them. It overlaps
        // the previous one, whose chunks it packs again into the same bytes.
        if (i < words_end && words_end >= vec_bytes) {
            pack_vector(words_end - vec_bytes);
            i = words_end;
        }
    }

    // Full 8-base chunks one at a time: all of them for encoders on SimdScalar, and those of
    // sequences shorter than one vector for the others. As the remainder below, this is scalar
    // code for the codes as well as the validity.
    for (; i < words_end; i += 8) {
        std::uint64_t const word = read_chunk(i);
        write_two_bit_chunk_<layout>(out_bytes + i / 4, encoder(word));
        if constexpr (WithValidity) {
            valid[i / 8] = static_cast<std::uint8_t>(valid_nucleotide_bits<SimdScalar>(word));
        }
    }

    // Remaining < 8 bases: the last 8 bytes of the sequence, shifted down to the bases that remain,
    // with zeros above them. If the sequence is shorter than that, they are copied instead. A
    // remainder of at most 4 bases writes only its one byte of codes, as the byte after it is not
    // part of `codes`. Zero gets code 0 and is not a valid nucleotide, which keeps the bits beyond
    // the end zero.
    if (i < seq_len) {
        std::size_t const remaining = seq_len - i;
        std::uint64_t word = 0;
        if (seq_len >= 8) {
            word = read_chunk(seq_len - 8) >> (8 * (8 - remaining));
        } else {
            std::memcpy(&word, data + i, remaining);
        }
        if (remaining <= 4) {
            write_two_bit_chunk_<layout, 1>(out_bytes + i / 4, encoder(word));
        } else {
            write_two_bit_chunk_<layout, 2>(out_bytes + i / 4, encoder(word));
        }
        if constexpr (WithValidity) {
            valid[i / 8] = static_cast<std::uint8_t>(valid_nucleotide_bits<SimdScalar>(word));
        }
    }
}

// -----------------------------------------------------------------------------
//     Pack Sequence Unchecked
// -----------------------------------------------------------------------------

/**
 * @brief Pack a whole ASCII sequence into a PackedSequence, reusing existing storage.
 *
 * Processes the sequence in 8-byte chunks via `encoder` (see word_encoder.hpp), writing
 * each chunk's 16-bit result to its 2 sequential output bytes at `out.codes()[i/4 .. i/4+1]`, where
 * `i` is the chunk's starting byte offset into `seq` (see write_two_bit_chunk_() for the per-byte
 * ordering). Encoders with more than one lane process `lanes` chunks at once, including the last
 * full chunks, in a vector that overlaps the previous one.
 *
 * `Encoder::encoding` and `Encoder::layout` (see Encoding and Layout in core/types.hpp) select
 * the out type. Passing an encoder whose encoding or layout differs from `out`'s is a compile
 * error. A PackedSequenceWithValidity is not taken, so that its validity cannot go stale; see
 * pack_sequence_checked() for it.
 *
 * Forced inline, as the compilers otherwise tend to keep this out of line, where they optimize
 * the loops less well for the given encoder.
 */
template <WordEncoder Encoder>
FISK_ALWAYS_INLINE
inline void pack_sequence_unchecked(
    std::string_view seq,
    Encoder const& encoder,
    PackedSequence<Encoder::encoding, Encoder::layout>& out
) {
    // The storage of `out` is reused, without clearing the codes, which are all written below.
    out.resize(seq.size());
    pack_sequence_<false>(seq, encoder, out.codes().data(), nullptr);
}

/**
 * @brief Pack a whole ASCII sequence into a freshly allocated PackedSequence.
 *
 * Convenience wrapper around the in-place overload above. Prefer that overload directly when
 * packing many sequences in a loop (e.g. in a benchmark), reusing one PackedSequence across
 * calls, to avoid attributing repeated heap allocation to whatever is being measured.
 */
template <WordEncoder Encoder>
inline PackedSequence<Encoder::encoding, Encoder::layout> pack_sequence_unchecked(
    std::string_view seq,
    Encoder const& encoder
) {
    PackedSequence<Encoder::encoding, Encoder::layout> out;
    pack_sequence_unchecked(seq, encoder, out);
    return out;
}

// -----------------------------------------------------------------------------
//     Pack Sequence Checked
// -----------------------------------------------------------------------------

/**
 * @brief Pack a whole ASCII sequence into a PackedSequenceWithValidity, reusing existing storage.
 *
 * As pack_sequence_unchecked(), and also records which characters are valid nucleotides, from the
 * same loads of the input, see `store_valid_bits()` of the word encoders.
 */
template <WordEncoder Encoder>
FISK_ALWAYS_INLINE
inline void pack_sequence_checked(
    std::string_view seq,
    Encoder const& encoder,
    PackedSequenceWithValidity<Encoder::encoding, Encoder::layout>& out
) {
    out.resize(seq.size());
    pack_sequence_<true>(seq, encoder, out.codes().data(), out.validity().data());
}

/**
 * @brief Pack a whole ASCII sequence into a freshly allocated PackedSequenceWithValidity.
 *
 * Convenience wrapper around the in-place overload above, see pack_sequence_unchecked().
 */
template <WordEncoder Encoder>
inline PackedSequenceWithValidity<Encoder::encoding, Encoder::layout> pack_sequence_checked(
    std::string_view seq,
    Encoder const& encoder
) {
    PackedSequenceWithValidity<Encoder::encoding, Encoder::layout> out;
    pack_sequence_checked(seq, encoder, out);
    return out;
}

} // namespace fisk
