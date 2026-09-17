#pragma once

#include <concepts>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/types.hpp"
#include "fisk/bit_extract/bit_extract.hpp"

namespace fisk {

// =================================================================================================
//     Word Encoders
// =================================================================================================

// Each of these encodes 8 ASCII characters (one uint64_t word, little-endian-loaded via memcpy)
// into their packed 2-bit codes, and is usable directly as the `Encoder` argument to
// pack_sequence() below. They are the word-at-a-time counterparts of the per-character encoders in
// core/char_encoder.hpp, and assume valid input: unlike those, they have no way to report an
// invalid character. Both are templated on the two conventions they produce, and expose them as
// `encoding` and `layout` static members right next to the logic that implements them, so that the
// tags and the bits cannot silently drift apart. What distinguishes the two is only how the
// selected bits are gathered: hardware PEXT, or the portable software butterfly network.
//
// Both conventions enter the same way, and only in one place each. The encoding picks which bits
// of every ASCII byte hold the code, and hence the extraction mask; ACGT additionally needs a SWAR
// pre-transform to fold its two bits into place (see ascii_bits_to_code_() in core/char_encoder.hpp for
// the scalar version of the same trick). ACTG needs no such fold: its two code bits already sit at
// bits 1-2 of every raw (lowercased) ASCII byte, so pack_mask_v alone selects them, and the bit
// extraction (PEXT or the butterfly network) shifts them down to bits 0-1 of the result as part of
// gathering them - no separate shift is needed either way. The layout picks whether the chunk is
// byte-swapped first, so that its first character lands in the high bits of the result instead of
// the low bits; see Layout in core/types.hpp.

/**
 * @brief Concept for anything usable as a word encoder for pack_sequence().
 *
 * Requires the encoder to state both conventions its output is in, from which pack_sequence()
 * derives the type of the PackedSequence it fills, and to encode one 8-character word at a time.
 * The SIMD word encoders (seq_pack/simd.hpp) satisfy this too, through their scalar call operator.
 */
template <typename T>
concept WordEncoder = requires(T const encoder, std::uint64_t word) {
    { T::encoding } -> std::convertible_to<Encoding>;
    { T::layout }   -> std::convertible_to<Layout>;
    { encoder(word) } -> std::convertible_to<std::uint64_t>;
};

/**
 * @brief Mask selecting the two code-bearing bits of every ASCII byte, for a given encoding.
 *
 * This is where the two bits end up after pack_fold_codes_(): bits 0-1 of every byte for ACGT,
 * which its SWAR fold computes explicitly; bits 1-2 for ACTG, which sit there in the raw byte
 * already, without needing pack_fold_codes_ to do anything.
 */
template <Encoding E>
inline constexpr std::uint64_t pack_mask_() noexcept
{
    if constexpr (E == Encoding::kACGT) {
        return 0x0303030303030303ULL;
    } else if constexpr (E == Encoding::kACTG) {
        return 0x0606060606060606ULL;
    } else {
        static_assert(dependent_false_v<E>, "Unhandled Encoding in pack_mask_()");
        return 0;
    }
}

template <Encoding E>
inline constexpr std::uint64_t pack_mask_v = pack_mask_<E>();

/**
 * @brief Apply the per-encoding pre-transform that puts the 2-bit codes under pack_mask_v.
 *
 * A no-op for ACTG, whose codes already sit under its mask; the SWAR fold for ACGT. ACTG needs no
 * shift down to bits 0-1 either: the extraction step (PEXT or the butterfly network) that gathers
 * the bits under pack_mask_v does that implicitly, as part of packing the gathered bits together.
 */
template <Encoding E>
inline constexpr std::uint64_t pack_fold_codes_(std::uint64_t word) noexcept
{
    if constexpr (E == Encoding::kACGT) {
        return ((word >> 1) ^ (word >> 2)) & pack_mask_v<E>;
    } else if constexpr (E == Encoding::kACTG) {
        return word;
    } else {
        static_assert(dependent_false_v<E>, "Unhandled Encoding in pack_fold_codes_()");
        return 0;
    }
}

/**
 * @brief Prepare a chunk for extraction under a given layout.
 *
 * Byte-swaps for MSB, so that the chunk's first character ends up in the high bits; a no-op for
 * LSB, which is the order a little-endian load already gives.
 */
template <Layout L>
inline constexpr std::uint64_t pack_orient_chunk_(std::uint64_t word) noexcept
{
    if constexpr (L == Layout::kMSB) {
        return byte_swap_64(word);
    } else if constexpr (L == Layout::kLSB) {
        return word;
    } else {
        static_assert(dependent_false_v<L>, "Unhandled Layout in pack_orient_chunk_()");
        return 0;
    }
}

#if defined(FISK_HAS_BMI2)

/**
 * @brief Encode 8 ASCII characters into packed 2-bit codes via a single hardware PEXT call.
 */
template <Encoding E, Layout L>
struct WordEncoderPext
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        word = pack_fold_codes_<E>(pack_orient_chunk_<L>(word));
        return bit_extract_pext(word, BitExtractMask(pack_mask_v<E>));
    }
};

#endif // FISK_HAS_BMI2

/**
 * @brief Same as WordEncoderPext, via the software butterfly network instead of PEXT.
 *
 * The fallback tier for hardware without fast PEXT (e.g. pre-Zen3 AMD, where PEXT/PDEP are
 * microcoded and slow), or without PEXT at all. The mask is fixed for a given encoding, so the
 * corresponding BitExtractButterflyTable is precomputed once via a function-local static.
 */
template <Encoding E, Layout L>
struct WordEncoderButterfly
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        static const BitExtractButterflyTable table =
            bit_extract_butterfly_table_preprocess(pack_mask_v<E>);
        word = pack_fold_codes_<E>(pack_orient_chunk_<L>(word));
        return bit_extract_butterfly_table(word, table);
    }
};

// =================================================================================================
//     Chunk Write Helper
// =================================================================================================

/**
 * @brief Write the packed bases of one encoder call to `dest`, respecting `L`.
 */
template <Layout L, int Bytes = 2>
inline void write_two_bit_chunk_(char* dest, std::uint64_t value) noexcept
{
    // Each encoder above packs 8 bases (16 bits) into the low bits of its return value. A byte is
    // the smallest addressable unit of PackedSequence's storage, so those 16 bits become 2
    // sequential output bytes. For `kLSB`, the encoder already produces those 2 bytes in the
    // right relative order for free (the first of the 8 bases ends up in the low bits of the first
    // byte, and so on); for `kMSB`, the two bytes come out address-reversed relative to base order
    // (the byte holding the later 4 bases lands first), so they need a fix-up swap before the
    // sequential write.
    //
    // `Bytes` (1 or 2) is a template parameter rather than a runtime one so that the hot full-chunk
    // call sites below (always 2) keep a fixed-width store with no added branch; only the ragged tail
    // chunk (<=4 real bases) ever instantiates the 1-byte write, and byte 0 -- the first of the two
    // output bytes post-swap -- is the one holding those bases in both layouts.

    static_assert(Bytes == 1 || Bytes == 2, "write_two_bit_chunk_() only supports 1 or 2 bytes");
    assert(dest != nullptr);
    assert(value <= 0xFFFF);
    std::uint16_t v16 = static_cast<std::uint16_t>(value);
    if constexpr (L == Layout::kMSB) {
        v16 = byte_swap_16(v16);
    }
    std::memcpy(dest, &v16, Bytes);
}

// =================================================================================================
//     pack_sequence()
// =================================================================================================

/**
 * @brief Pack a whole ASCII sequence into a PackedSequence, reusing existing storage.
 *
 * Processes the sequence in 8-byte chunks via `encoder` (see the word encoders above), writing
 * each chunk's 16-bit result to its 2 sequential output bytes at `out.data[i/4 .. i/4+1]`, where
 * `i` is the chunk's starting byte offset into `seq` (see write_two_bit_chunk_() for the per-byte
 * ordering).
 *
 * `Encoder::encoding` and `Encoder::layout` (see Encoding and Layout in core/types.hpp) select
 * the out type. Passing an encoder whose encoding or layout differs from `out`'s is a compile
 * error.
 */
template <WordEncoder Encoder>
inline void pack_sequence(
    std::string_view seq,
    Encoder const& encoder,
    PackedSequence<Encoder::encoding, Encoder::layout>& out
) {
    constexpr Layout layout = Encoder::layout;

    std::size_t const seq_len   = seq.size();
    std::size_t const num_bytes = (seq_len + 3) / 4;

    out.length = seq_len;
    out.data.assign(num_bytes, 0);
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
