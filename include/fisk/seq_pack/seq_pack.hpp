#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/simd.hpp"
#include "fisk/core/types.hpp"
#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"

namespace fisk {

// =================================================================================================
//     Word Encoders
// =================================================================================================

// Each of these encodes 8 ASCII characters, loaded as one little-endian word, into their packed
// 2-bit codes, for use with pack_sequence(). They assume valid input: an invalid character is
// packed as an arbitrary code at its own position, without affecting any other position.
//
// WordEncoderPext and WordEncoderButterfly gather the code bits under a mask, via hardware PEXT
// or the software butterfly network. WordEncoderDirect instead packs the codes with instructions
// specific to the SIMD arch, and is the fastest of them. The latter two are templated on a SIMD
// arch (core/simd.hpp), with which they encode `lanes` words at once.

// -----------------------------------------------------------------------------
//     Helper Functions
// -----------------------------------------------------------------------------

/**
 * @brief Concept for anything usable as a word encoder for pack_sequence().
 *
 * Requires the encoder to state both conventions its output is in, from which pack_sequence()
 * derives the type of the PackedSequence it fills, and to encode one 8-character word at a time.
 *
 * It also states the number of `lanes`, that is, words that it encodes at once. With more than
 * one, pack_sequence() additionally uses `simd_vector`, `load_unaligned()`, `store_packed()`, and
 * the call operator on a whole vector, as provided by the encoders that take a SIMD arch.
 */
template <typename T>
concept WordEncoder = requires(T const encoder, std::uint64_t word) {
    { T::encoding } -> std::convertible_to<Encoding>;
    { T::layout }   -> std::convertible_to<Layout>;
    { T::lanes }    -> std::convertible_to<std::size_t>;
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

// -----------------------------------------------------------------------------
//     Word Encoder Pext
// -----------------------------------------------------------------------------

#if defined(FISK_HAS_BMI2)

/**
 * @brief Encode 8 ASCII characters into packed 2-bit codes via a single hardware PEXT call.
 */
template <Encoding E, Layout L>
struct WordEncoderPext
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t lanes = 1;

    inline std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        word = pack_fold_codes_<E>(pack_orient_chunk_<L>(word));
        return bit_extract_pext(word, BitExtractMask(pack_mask_v<E>));
    }
};

#endif // FISK_HAS_BMI2

// -----------------------------------------------------------------------------
//     Word Encoder Butterfly
// -----------------------------------------------------------------------------

/**
 * @brief Same as WordEncoderPext, via the software butterfly network instead of PEXT.
 *
 * The fallback tier for hardware without fast PEXT (e.g. pre-Zen3 AMD, where PEXT/PDEP are
 * microcoded and slow), or without PEXT at all.
 *
 * With a SIMD arch other than SimdScalar (see core/simd.hpp), the call operator on a whole
 * `simd_vector` encodes `lanes` words at once, applying the same steps to each lane. The call
 * operator on a single word remains, for the words at the end that do not fill a whole vector.
 *
 * The encoder is stateless: its butterfly table is a compile-time constant, so that the compiler
 * can keep it in registers. A table in the encoder object instead would have to be reloaded after
 * each store of packed bytes, which might alias it. For the same reason, the call operators are
 * forced inline, as otherwise the constants would be rebuilt on each call.
 */
template <Encoding E, Layout L, SimdArch Simd = SimdScalar>
struct WordEncoderButterfly
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t lanes = Simd::lanes;
    using simd_vector = typename Simd::simd_vector;

    static constexpr BitExtractButterflyTable table =
        make_bit_extract_butterfly_table(pack_mask_v<E>);

    static simd_vector load_unaligned(std::uint64_t const* a) noexcept
    {
        return Simd::load_unaligned(a);
    }

    // Each lane holds the 8 bases of its chunk in its low 16 bits, see write_two_bit_chunk_().
    static void store_packed(simd_vector const& v, char* out) noexcept
    {
        if constexpr (L == Layout::kMSB) {
            Simd::template store_truncated_u16<true>(v, out);
        } else if constexpr (L == Layout::kLSB) {
            Simd::template store_truncated_u16<false>(v, out);
        } else {
            static_assert(
                dependent_false_v<L>, "Unhandled Layout in WordEncoderButterfly::store_packed()"
            );
        }
    }

    // For SimdScalar, the vector is a single word, which is encoded by the call operator below.
    FISK_ALWAYS_INLINE
    simd_vector operator()(simd_vector x) const noexcept
    requires (lanes > 1)
    {
        if constexpr (L == Layout::kMSB) {
            x = Simd::byte_swap_lanes(x);
        } else if constexpr (L == Layout::kLSB) {
            // A little-endian load already has the bases in order.
        } else {
            static_assert(
                dependent_false_v<L>, "Unhandled Layout in WordEncoderButterfly::operator()"
            );
        }
        if constexpr (E == Encoding::kACGT) {
            simd_vector const s1 = Simd::template shift_right<1>(x);
            simd_vector const s2 = Simd::template shift_right<2>(x);
            simd_vector const acgt_mask = Simd::broadcast(pack_mask_v<Encoding::kACGT>);
            x = Simd::bitwise_and(Simd::bitwise_xor(s1, s2), acgt_mask);
        } else if constexpr (E == Encoding::kACTG) {
            // The codes already sit in bits 1-2 of each byte, see pack_fold_codes_().
        } else {
            static_assert(
                dependent_false_v<E>, "Unhandled Encoding in WordEncoderButterfly::operator()"
            );
        }
        return BitExtractKernelButterfly<Simd>::bit_extract(x, table);
    }

    FISK_ALWAYS_INLINE
    std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        return bit_extract_butterfly_table(pack_fold_codes_<E>(pack_orient_chunk_<L>(word)), table);
    }
};

// -----------------------------------------------------------------------------
//     Word Encoder Direct
// -----------------------------------------------------------------------------

/**
 * @brief Encode ASCII characters into packed 2-bit codes via the direct code packing of the arch.
 *
 * Instead of gathering bits under a mask, this reduces each character to its 2-bit code, and then
 * packs each four codes into a byte with a few instructions that are specific to the SIMD arch,
 * see `store_two_bit_codes()` in core/simd.hpp. This is the fastest word encoder on every arch.
 *
 * As for WordEncoderButterfly, the call operator on a whole `simd_vector` processes `lanes` words
 * at once, and the one on a single word the words at the end that do not fill a whole vector.
 */
template <Encoding E, Layout L, SimdArch Simd = SimdScalar>
struct WordEncoderDirect
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t lanes = Simd::lanes;
    using simd_vector = typename Simd::simd_vector;

    static simd_vector load_unaligned(std::uint64_t const* a) noexcept
    {
        return Simd::load_unaligned(a);
    }

    // Each byte holds the code of its base, see the call operator below.
    static void store_packed(simd_vector const& v, char* out) noexcept
    {
        if constexpr (L == Layout::kMSB) {
            Simd::template store_two_bit_codes<true>(v, out);
        } else if constexpr (L == Layout::kLSB) {
            Simd::template store_two_bit_codes<false>(v, out);
        } else {
            static_assert(
                dependent_false_v<L>, "Unhandled Layout in WordEncoderDirect::store_packed()"
            );
        }
    }

    // For SimdScalar, the vector is a single word, which is encoded by the call operator below.
    FISK_ALWAYS_INLINE
    simd_vector operator()(simd_vector x) const noexcept
    requires (lanes > 1)
    {
        return codes_<Simd>(x);
    }

    FISK_ALWAYS_INLINE
    std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        return SimdScalar::pack_two_bit_codes_(codes_<SimdScalar>(pack_orient_chunk_<L>(word)));
    }

private:

    // Reduce each byte to its 2-bit code, in its two lowest bits. The shifts are per lane, but
    // what they move across byte boundaries is masked out.
    template <SimdArch S>
    FISK_ALWAYS_INLINE
    static typename S::simd_vector codes_(typename S::simd_vector x) noexcept
    {
        if constexpr (E == Encoding::kACGT) {
            x = S::bitwise_xor(S::template shift_right<1>(x), S::template shift_right<2>(x));
        } else if constexpr (E == Encoding::kACTG) {
            x = S::template shift_right<1>(x);
        } else {
            static_assert(dependent_false_v<E>, "Unhandled Encoding in WordEncoderDirect");
        }
        return S::bitwise_and(x, S::broadcast(0x0303030303030303ULL));
    }
};

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
    // Each encoder above packs 8 bases (16 bits) into the low bits of its return value. A byte is
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
 * Processes the sequence in 8-byte chunks via `encoder` (see the word encoders above), writing
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
