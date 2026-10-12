#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/simd.hpp"
#include "fisk/core/types.hpp"
#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/seq_pack/seq_pack.hpp"

namespace fisk {

// =================================================================================================
//     SIMD Butterfly Word Encoder
// =================================================================================================

/**
 * @brief Encode ASCII chars into 2-bit packed nucleotide data using SIMD instructions.
 *
 * Encodes `Simd::lanes` 8-byte chunks at once (one `Simd::simd_vector`), for a given ASCII
 * encoding and layout, usable as the `Encoder` argument to pack_sequence_simd() below.
 *
 * `Simd` is one of the SIMD archs of core/simd.hpp, such as SimdAVX2. Mirrors the scalar
 * WordEncoderButterfly<Encoding, Layout> struct in seq_pack.hpp: same ACGT SWAR pre-transform, same
 * MSB byte-swap-before-extracting, just applied to a whole SIMD vector of chunks instead of one
 * uint64_t chunk. Provides both a vector call operator (for the main SIMD loop) and a scalar one
 * (for the head/tail loops in pack_sequence_simd, via the kernel's bit_extract_scalar()), so a
 * single encoder instance drives all three loop phases.
 */
template <SimdArch Simd, Encoding E, Layout L>
struct WordEncoderButterflySimd
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t lanes = Simd::lanes;
    using simd_vector = typename Simd::simd_vector;

    BitExtractKernelButterfly<Simd> kernel;
    simd_vector acgt_mask_simd;

    WordEncoderButterflySimd()
        : kernel(pack_mask_v<E>)
        , acgt_mask_simd(Simd::broadcast(pack_mask_v<Encoding::kACGT>))
    {}

    static simd_vector loadu(std::uint64_t const* a) noexcept
    {
        return Simd::load_unaligned(a);
    }

    static void store(simd_vector const& v, std::uint64_t* out) noexcept
    {
        Simd::store_aligned(v, out);
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
                dependent_false_v<L>, "Unhandled Layout in WordEncoderButterflySimd::store_packed()"
            );
        }
    }

    simd_vector operator()(simd_vector x) const noexcept
    {
        if constexpr (L == Layout::kMSB) {
            x = Simd::byte_swap_lanes(x);
        } else if constexpr (L == Layout::kLSB) {
            // A little-endian load already has the bases in order.
        } else {
            static_assert(
                dependent_false_v<L>, "Unhandled Layout in WordEncoderButterflySimd::operator()"
            );
        }
        if constexpr (E == Encoding::kACGT) {
            simd_vector const s1 = Simd::template shift_right<1>(x);
            simd_vector const s2 = Simd::template shift_right<2>(x);
            x = Simd::bitwise_and(Simd::bitwise_xor(s1, s2), acgt_mask_simd);
        } else if constexpr (E == Encoding::kACTG) {
            // The codes already sit in bits 1-2 of each byte, see pack_fold_codes_().
        } else {
            static_assert(
                dependent_false_v<E>, "Unhandled Encoding in WordEncoderButterflySimd::operator()"
            );
        }
        return kernel.bit_extract(x);
    }

    std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        return kernel.bit_extract_scalar(pack_fold_codes_<E>(pack_orient_chunk_<L>(word)));
    }
};

// =================================================================================================
//     pack_sequence_simd()
// =================================================================================================

// Output addressing matches scalar pack_sequence: the chunk starting at seq byte offset `off`
// writes its 2 bytes at out.data[off/4 .. off/4+1], via write_two_bit_chunk_() (seq_pack.hpp).

/**
 * @brief Pack a whole ASCII sequence into a PackedSequence via a SIMD word encoder (see
 * WordEncoderButterflySimd above), reusing existing storage.
 *
 * Beyond the WordEncoder requirements, `Encoder` has to provide `lanes`, `simd_vector`, `loadu()`,
 * `store_packed()` and a call operator on a whole vector, as WordEncoderButterflySimd does.
 */
template <WordEncoder Encoder>
inline void pack_sequence_simd(
    std::string_view seq,
    Encoder const& encoder,
    PackedSequence<Encoder::encoding, Encoder::layout>& out
) {
    constexpr Layout layout = Encoder::layout;
    constexpr std::size_t lanes = Encoder::lanes;
    using simd_vector = typename Encoder::simd_vector;
    constexpr std::size_t vec_bytes = lanes * 8;

    std::size_t const seq_len   = seq.size();
    std::size_t const num_bytes = (seq_len + 3) / 4;

    out.length = seq_len;

    // Every output byte is filled below, including the final partial byte; preserve reusable
    // storage instead of clearing it before the SIMD and scalar-tail stores overwrite it.
    out.data.resize(num_bytes);
    char* const out_bytes = reinterpret_cast<char*>(out.data.data());
    char const* const data = seq.data();

    std::size_t i = 0;
    for (; i + vec_bytes <= seq_len; i += vec_bytes) {
        simd_vector const x = encoder(
            Encoder::loadu(reinterpret_cast<std::uint64_t const*>(data + i))
        );
        encoder.store_packed(x, out_bytes + i / 4);
    }
    for (; i < seq_len; i += 8) {
        std::size_t const remaining = seq_len - i;
        std::size_t const take = remaining < 8 ? remaining : std::size_t{8};
        std::uint64_t word = 0;
        std::memcpy(&word, data + i, take);
        // out holds no trailing bytes beyond its real content (see pack_sequence(), seq_pack.hpp,
        // for why the <=4-base remainder needs a 1-byte write instead of write_chunk()'s usual 2).
        if (remaining <= 4) {
            write_two_bit_chunk_<layout, 1>(out_bytes + i / 4, encoder(word));
        } else {
            write_two_bit_chunk_<layout, 2>(out_bytes + i / 4, encoder(word));
        }
    }
}

/**
 * @brief Pack a whole ASCII sequence into a freshly allocated PackedSequence via a SIMD word
 * encoder. Convenience wrapper around pack_sequence_simd() above; see its docs for the caveat about
 * reusing one PackedSequence across repeated calls (e.g. in a benchmark) instead.
 */
template <WordEncoder Encoder>
inline PackedSequence<Encoder::encoding, Encoder::layout> pack_sequence_simd(
    std::string_view seq, Encoder const& encoder
) {
    PackedSequence<Encoder::encoding, Encoder::layout> out;
    pack_sequence_simd(seq, encoder, out);
    return out;
}

} // namespace fisk
