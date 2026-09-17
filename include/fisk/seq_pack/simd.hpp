#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/types.hpp"
#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/seq_pack/seq_pack.hpp"

namespace fisk {

// =================================================================================================
//     SIMD Kernel Extensions
// =================================================================================================

// These extend the existing BitExtractKernelButterfly<SIMD> structs (bit_extract/simd.hpp) with
// extra vector ops needed: an unaligned load, a per-lane broadcast, and-or/xor for the ACGT SWAR
// pre-transform, a shift for the same, and a per-64-bit-lane byte reversal for the MSB layout.
//
// byte_reverse_lanes() uses a single shuffle (_mm256/_mm512_shuffle_epi8) on AVX2 and AVX512,
// where that's just baseline AVX2 / already-required AVX512BW (see CMakeLists.txt) respectively.
// SSE2 has no such instruction available, so it falls back to the portable byte_swap_64().

#if defined(FISK_HAS_SSE2)

/**
 * @brief SSE2 vector ops needed for whole-sequence packing, on top of BitExtractKernelButterflySSE2.
 */
struct SeqPackButterflyKernelSSE2 : public BitExtractKernelButterflySSE2
{
    using Base = BitExtractKernelButterflySSE2;
    using Base::Base;

    static simd_vector loadu(std::uint64_t const* a) noexcept
    {
        return _mm_loadu_si128(reinterpret_cast<__m128i const*>(a));
    }

    static simd_vector splat(std::uint64_t value) noexcept
    {
        return _mm_set1_epi64x(static_cast<long long>(value));
    }

    static simd_vector and_(simd_vector a, simd_vector b) noexcept
    {
        return _mm_and_si128(a, b);
    }

    static simd_vector xor_(simd_vector a, simd_vector b) noexcept
    {
        return _mm_xor_si128(a, b);
    }

    template <int Shift>
    static simd_vector shr(simd_vector a) noexcept
    {
        return _mm_srli_epi64(a, Shift);
    }

    static simd_vector byte_reverse_lanes(simd_vector x) noexcept
    {
        simd_vector const m32 = splat(0x00000000FFFFFFFFULL);
        x = _mm_or_si128(
            _mm_slli_epi64(_mm_and_si128(x, m32), 32), _mm_srli_epi64(_mm_andnot_si128(m32, x), 32)
        );
        simd_vector const m16 = splat(0x0000FFFF0000FFFFULL);
        x = _mm_or_si128(
            _mm_slli_epi64(_mm_and_si128(x, m16), 16), _mm_srli_epi64(_mm_andnot_si128(m16, x), 16)
        );
        simd_vector const m8 = splat(0x00FF00FF00FF00FFULL);
        x = _mm_or_si128(
            _mm_slli_epi64(_mm_and_si128(x, m8), 8), _mm_srli_epi64(_mm_andnot_si128(m8, x), 8)
        );
        return x;
    }
};

#endif // FISK_HAS_SSE2

#if defined(FISK_HAS_AVX2)

/**
 * @brief AVX2 vector ops needed for whole-sequence packing, on top of BitExtractKernelButterflyAVX2.
 */
struct SeqPackButterflyKernelAVX2 : public BitExtractKernelButterflyAVX2
{
    using Base = BitExtractKernelButterflyAVX2;
    using Base::Base;

    static simd_vector loadu(std::uint64_t const* a) noexcept
    {
        return _mm256_loadu_si256(reinterpret_cast<__m256i const*>(a));
    }

    static simd_vector splat(std::uint64_t value) noexcept
    {
        return _mm256_set1_epi64x(static_cast<long long>(value));
    }

    static simd_vector and_(simd_vector a, simd_vector b) noexcept
    {
        return _mm256_and_si256(a, b);
    }

    static simd_vector xor_(simd_vector a, simd_vector b) noexcept
    {
        return _mm256_xor_si256(a, b);
    }

    template <int Shift>
    static simd_vector shr(simd_vector a) noexcept
    {
        return _mm256_srli_epi64(a, Shift);
    }

    // _mm256_shuffle_epi8 shuffles within each 128-bit half independently (no cross-half
    // indexing), so the 8-byte-reversal pattern is written once per half and repeated.
    static simd_vector byte_reverse_lanes(simd_vector x) noexcept
    {
        static constexpr std::uint8_t pattern[32] = {
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8
        };
        simd_vector const ctrl = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(pattern));
        return _mm256_shuffle_epi8(x, ctrl);
    }
};

#endif // FISK_HAS_AVX2

#if defined(FISK_HAS_AVX512)

/**
 * @brief AVX512 vector ops needed for whole-sequence packing, on BitExtractKernelButterflyAVX512.
 */
struct SeqPackButterflyKernelAVX512 : public BitExtractKernelButterflyAVX512
{
    using Base = BitExtractKernelButterflyAVX512;
    using Base::Base;

    static simd_vector loadu(std::uint64_t const* a) noexcept
    {
        return _mm512_loadu_si512(reinterpret_cast<__m512i const*>(a));
    }

    static simd_vector splat(std::uint64_t value) noexcept
    {
        return _mm512_set1_epi64(static_cast<long long>(value));
    }

    static simd_vector and_(simd_vector a, simd_vector b) noexcept
    {
        return _mm512_and_si512(a, b);
    }

    static simd_vector xor_(simd_vector a, simd_vector b) noexcept
    {
        return _mm512_xor_si512(a, b);
    }

    template <int Shift>
    static simd_vector shr(simd_vector a) noexcept
    {
        return _mm512_srli_epi64(a, Shift);
    }

    // _mm512_shuffle_epi8 (AVX512BW) shuffles within each 128-bit quarter independently (no
    // cross-quarter indexing), so the 8-byte-reversal pattern is written once and repeated 4x.
    static simd_vector byte_reverse_lanes(simd_vector x) noexcept
    {
        static constexpr std::uint8_t pattern[64] = {
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8
        };
        simd_vector const ctrl = _mm512_loadu_si512(reinterpret_cast<__m512i const*>(pattern));
        return _mm512_shuffle_epi8(x, ctrl);
    }
};

#endif // FISK_HAS_AVX512

#if defined(FISK_HAS_NEON)

/**
 * @brief NEON vector ops needed for whole-sequence packing, on top of BitExtractKernelButterflyNEON.
 */
struct SeqPackButterflyKernelNEON : public BitExtractKernelButterflyNEON
{
    using Base = BitExtractKernelButterflyNEON;
    using Base::Base;

    static simd_vector loadu(std::uint64_t const* a) noexcept
    {
        return vld1q_u64(a);
    }

    static simd_vector splat(std::uint64_t value) noexcept
    {
        return vdupq_n_u64(value);
    }

    static simd_vector and_(simd_vector a, simd_vector b) noexcept
    {
        return vandq_u64(a, b);
    }

    static simd_vector xor_(simd_vector a, simd_vector b) noexcept
    {
        return veorq_u64(a, b);
    }

    template <int Shift>
    static simd_vector shr(simd_vector a) noexcept
    {
        return vshrq_n_u64(a, Shift);
    }

    // NEON has a direct per-byte-lane reversal-within-64-bit-groups intrinsic, so this doesn't
    // need the shift-mask-or staging the x86 variants use.
    static simd_vector byte_reverse_lanes(simd_vector x) noexcept
    {
        return vreinterpretq_u64_u8(vrev64q_u8(vreinterpretq_u8_u64(x)));
    }
};

#endif // FISK_HAS_NEON

// =================================================================================================
//     SIMD Butterfly Word Encoder
// =================================================================================================

/**
 * @brief Encode ASCII chars into 2-bit packed nucleotide data using SIMD instructions.
 *
 * Encodes `Kernel::lanes` 8-byte chunks at once (one `Kernel::simd_vector`), for a given ASCII
 * encoding and layout, usable as the `Encoder` argument to pack_sequence_simd() below.
 *
 * `Kernel` is one of the SeqPackButterflyKernel<SIMD> structs above. Mirrors the scalar
 * WordEncoderButterfly<Encoding, Layout> struct in seq_pack.hpp: same ACGT SWAR pre-transform, same
 * MSB byte-swap-before-extracting, just applied to a whole SIMD vector of chunks instead of one
 * uint64_t chunk. Provides both a vector call operator (for the main SIMD loop) and a scalar one
 * (for the head/tail loops in pack_sequence_simd, reusing the Kernel's own inherited scalar
 * bit_extract()), so a single encoder instance drives all three loop phases.
 */
template <typename Kernel, Encoding E, Layout L>
struct WordEncoderButterflySimd
{
    static constexpr Encoding encoding = E;
    static constexpr Layout layout = L;
    static constexpr std::size_t lanes = Kernel::lanes;
    using simd_vector = typename Kernel::simd_vector;

    Kernel kernel;
    simd_vector acgt_mask_simd;

    WordEncoderButterflySimd()
        : kernel(pack_mask_v<E>)
        , acgt_mask_simd(Kernel::splat(pack_mask_v<Encoding::kACGT>))
    {}

    static simd_vector loadu(std::uint64_t const* a) noexcept
    {
        return Kernel::loadu(a);
    }

    static void store(simd_vector const& v, std::uint64_t* out) noexcept
    {
        Kernel::store(v, out);
    }

    simd_vector operator()(simd_vector x) const noexcept
    {
        if constexpr (L == Layout::kMSB) {
            x = Kernel::byte_reverse_lanes(x);
        }
        if constexpr (E == Encoding::kACGT) {
            simd_vector const s1 = Kernel::template shr<1>(x);
            simd_vector const s2 = Kernel::template shr<2>(x);
            x = Kernel::and_(Kernel::xor_(s1, s2), acgt_mask_simd);
        }
        return kernel.bit_extract(x);
    }

    std::uint64_t operator()(std::uint64_t word) const noexcept
    {
        if constexpr (L == Layout::kMSB) {
            word = byte_swap_64(word);
        }
        if constexpr (E == Encoding::kACGT) {
            word = pack_fold_codes_<E>(word);
        }
        return kernel.bit_extract(word);
    }
};

// One alias per available SIMD, each still templated on the two conventions: the ISA is what
// distinguishes these from each other, while the encoding and layout stay template parameters,
// selected the same way as everywhere else in fisk. Mirrors the scalar WordEncoderButterfly.

#if defined(FISK_HAS_SSE2)
template <Encoding E, Layout L>
using WordEncoderButterflySSE2 = WordEncoderButterflySimd<SeqPackButterflyKernelSSE2, E, L>;
#endif

#if defined(FISK_HAS_AVX2)
template <Encoding E, Layout L>
using WordEncoderButterflyAVX2 = WordEncoderButterflySimd<SeqPackButterflyKernelAVX2, E, L>;
#endif

#if defined(FISK_HAS_AVX512)
template <Encoding E, Layout L>
using WordEncoderButterflyAVX512 = WordEncoderButterflySimd<SeqPackButterflyKernelAVX512, E, L>;
#endif

#if defined(FISK_HAS_NEON)
template <Encoding E, Layout L>
using WordEncoderButterflyNEON = WordEncoderButterflySimd<SeqPackButterflyKernelNEON, E, L>;
#endif

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
 * `store()` and a call operator on a whole vector, as WordEncoderButterflySimd does.
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
    out.data.assign(num_bytes, 0);
    char* const out_bytes = reinterpret_cast<char*>(out.data.data());
    char const* const data = seq.data();

    auto write_chunk = [&](std::size_t off, std::uint64_t value) {
        write_two_bit_chunk_<layout>(out_bytes + off / 4, value);
    };

    std::size_t i = 0;
    for (; i + vec_bytes <= seq_len; i += vec_bytes) {
        simd_vector const x = encoder(
            Encoder::loadu(reinterpret_cast<std::uint64_t const*>(data + i))
        );
        alignas(alignof(simd_vector)) std::uint64_t lane_buf[lanes];
        Encoder::store(x, lane_buf);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            write_chunk(i + lane * 8, lane_buf[lane]);
        }
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
 * @brief Pack a whole ASCII sequence into a freshly allocated PackedSequence via a SIMD word encoder.
 * Convenience wrapper around pack_sequence_simd() above; see its docs for the caveat about
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
