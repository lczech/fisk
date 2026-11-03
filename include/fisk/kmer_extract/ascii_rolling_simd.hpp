#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/simd/simd.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_rolling_scalar.hpp"
#include "fisk/kmer_extract/internal.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from ASCII, Rolling, SIMD
// =================================================================================================

// The SIMD arch encodes and checks a block of 8 characters per lane at once, with
// nucleotide_code_bytes() and valid_nucleotide_bits(); the k-mers themselves are still rolled one
// character at a time, as each one only differs from the previous one in its first and last
// character, so there is not much room for vectorizing that part. With SimdScalar, the block is a
// single word, encoded and checked via SWAR (SIMD within a register).

/**
 * @brief Iterate over all valid k-mers in a sequence, with blocks of characters encoded and checked
 * by the SIMD arch `Simd`.
 *
 * Encodes and checks the characters in blocks of `8 * Simd::lanes`, then rolls the k-mers over
 * each block, with a tight inner loop for fully valid blocks, see
 * for_each_kmer_ascii_checked_rolling_block_(). The characters after the last whole block take the
 * scalar path, see for_each_kmer_ascii_checked_rolling_tail_().
 *
 * Invalid characters reset the valid-window state and suppress any overlapping k-mers.
 *
 * The callback receives a Kmer<E, L>, ACGT/MSB by default:
 *
 *     // Kmer<kACGT, kMSB>
 *     for_each_kmer_ascii_checked_rolling_simd<SimdAVX2>(seq, k, func);
 *     // Kmer<kACTG, kLSB>
 *     for_each_kmer_ascii_checked_rolling_simd<SimdAVX2, Encoding::kACTG, Layout::kLSB>(
 *         seq, k, func
 *     );
 *
 * Func may be called either as `func(pos, kmer)` or as `func(kmer)`, see
 * for_each_kmer_ascii_checked_rolling_scalar().
 */
template<SimdArch Simd, Encoding E = Encoding::kACGT, Layout L = Layout::kMSB, typename Func>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_checked_rolling_simd(
    std::string_view seq, std::size_t k, Func&& func
) {
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.size() < k) {
        return;
    }

    std::size_t const seq_len = seq.size();
    char const* data = seq.data();

    // Rolling constants for MSB and LSB, see for_each_kmer_ascii_checked_rolling_scalar().
    std::uint64_t const mask = (k == 32)
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << (2 * k)) - 1u);
    unsigned const top_shift = 2 * (static_cast<unsigned>(k) - 1);

    std::uint64_t kmer = 0;
    std::size_t valid = 0;
    std::size_t i = 0;

    // The codes of a block, one byte per character, in the order of the characters, as stored by
    // the little-endian lanes.
    constexpr unsigned width = 8 * Simd::lanes;
    alignas(64) std::array<std::uint64_t, Simd::lanes> codes{};
    auto const* code_bytes = reinterpret_cast<std::uint8_t const*>(codes.data());

    for (; i + width <= seq_len; i += width) {
        typename Simd::simd_vector const chars = Simd::load_unaligned(data + i);
        Simd::store_aligned(nucleotide_code_bytes<E, Simd>(chars), codes.data());
        for_each_kmer_ascii_checked_rolling_block_<E, L, width>(
            code_bytes, valid_nucleotide_bits<Simd>(chars), i, k, mask, top_shift, kmer, valid,
            std::forward<Func>(func)
        );
    }

    // Scalar tail.
    for_each_kmer_ascii_checked_rolling_tail_<E, L>(
        data, i, seq_len, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
    );
}

} // namespace fisk
