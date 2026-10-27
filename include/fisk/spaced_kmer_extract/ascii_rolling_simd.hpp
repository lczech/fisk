#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/kmer_spaced.hpp"
#include "fisk/core/types.hpp"
#include "fisk/spaced_kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     SIMD Helper Functions
// =================================================================================================

// Encourage the compiler to fully unroll the lane and mask loops below.
#if defined(__clang__)
    #define FISK_PRAGMA_UNROLL_16 _Pragma("unroll 16")
#elif defined(__GNUC__)
    #define FISK_PRAGMA_UNROLL_16 _Pragma("GCC unroll 16")
#else
    #define FISK_PRAGMA_UNROLL_16
#endif

// Shift the next character into the rolling k-mer, and its validity into the rolling valid bits,
// in Layout `L`. MSB shifts in at the bottom, and keeps the old bases above the span, as the bit
// extraction and the validity check only look at the bits of the mask. LSB shifts in at
// `top_shift`, i.e., the top of the span.
template<Layout L, typename Enc>
FISK_ALWAYS_INLINE
inline void spaced_kmer_ascii_rolling_simd_shift_in_(
    char c,
    Enc&& enc,
    std::size_t top_shift,
    std::uint64_t& rolling_kmer,
    std::uint64_t& rolling_valid_pos
) {
    std::uint8_t const code = static_cast<std::uint8_t>(enc(c));

    // Shift in the next base, and 11 for valid or 00 for invalid into the valid bits. For invalid
    // bases, the low 2 bits are irrelevant, because validity is checked separately before
    // emission. This ensures that spaced k-mers which contain an invalid base in them (which was
    // not masked out) will be skipped. GCC is sensitive to the exact form here: computing `valid`
    // inline in the expression below made the SIMD extraction up to 30% slower with GCC 14.
    std::uint64_t const valid = static_cast<std::uint64_t>(code < 4) * 0x03u;
    if constexpr (L == Layout::kMSB) {
        rolling_kmer      = (rolling_kmer      << 2) | (code & 0x03u);
        rolling_valid_pos = (rolling_valid_pos << 2) | valid;
    } else if constexpr (L == Layout::kLSB) {
        rolling_kmer      = (rolling_kmer      >> 2) | (std::uint64_t{code & 0x03u} << top_shift);
        // rolling_valid_pos = (rolling_valid_pos >> 2) | (valid << top_shift);
        // Using alternative implementation here: faster with Clang, slightly slower with GCC.
        rolling_valid_pos = (rolling_valid_pos >> 2)
            | (static_cast<std::uint64_t>(code < 4) * (std::uint64_t{0x03u} << top_shift));
    } else {
        static_assert(
            dependent_false_v<L>, "Unhandled Layout in spaced_kmer_ascii_rolling_simd_shift_in_()"
        );
    }
}

// Shift the next `count` characters of `data`, at most `Kernel::lanes`, into the rolling k-mer and
// its valid bits, and keep each intermediate state in its lane, for the SIMD extraction below. The
// lanes from `count` on hold zero, and are not valid.
template<typename Kernel, typename Enc>
FISK_ALWAYS_INLINE
inline void spaced_kmer_ascii_rolling_simd_fill_lanes_(
    char const* data,
    std::size_t count,
    Enc&& enc,
    std::size_t top_shift,
    std::uint64_t& rolling_kmer,
    std::uint64_t& rolling_valid_pos,
    std::array<std::uint64_t, Kernel::lanes>& lane_kmers,
    std::uint64_t (&lane_valids)[Kernel::lanes]
) {
    FISK_PRAGMA_UNROLL_16
    for (std::size_t lane = 0; lane < Kernel::lanes; ++lane) {
        if (lane < count) {
            spaced_kmer_ascii_rolling_simd_shift_in_<Kernel::layout>(
                data[lane], enc, top_shift, rolling_kmer, rolling_valid_pos
            );
            lane_kmers[lane]  = rolling_kmer;
            lane_valids[lane] = rolling_valid_pos;
        } else {
            lane_kmers[lane]  = 0;
            lane_valids[lane] = 0;
        }
    }
}

// Call `f` for each lane in [0, L), with the lane index as a compile-time constant, so that the
// lane loops of the SIMD extraction functions below are fully unrolled.
template <std::size_t L, typename F>
FISK_ALWAYS_INLINE
inline void for_each_lane_ct_(F&& f)
{
    if constexpr (L >= 1) { f(std::integral_constant<std::size_t, 0>{}); }
    if constexpr (L >= 2) { f(std::integral_constant<std::size_t, 1>{}); }
    if constexpr (L >= 3) { f(std::integral_constant<std::size_t, 2>{}); }
    if constexpr (L >= 4) { f(std::integral_constant<std::size_t, 3>{}); }
    if constexpr (L >= 5) { f(std::integral_constant<std::size_t, 4>{}); }
    if constexpr (L >= 6) { f(std::integral_constant<std::size_t, 5>{}); }
    if constexpr (L >= 7) { f(std::integral_constant<std::size_t, 6>{}); }
    if constexpr (L >= 8) { f(std::integral_constant<std::size_t, 7>{}); }
}

// =================================================================================================
//     Spaced K-mer Extraction from ASCII, Rolling, SIMD
// =================================================================================================

/**
 * @brief Iterate a sequence, extract all spaced k-mers from it with SIMD bit extraction, and call a
 * callback on vectors of them, for an array of kernels.
 *
 * The callback is called as `callback(pos, mask_idx, spaced_kmers, valid_lanes)`, or, for a single
 * kernel, also as `callback(pos, spaced_kmers, valid_lanes)`: lane `j` of the SpacedKmerVector
 * `spaced_kmers` holds the spaced k-mer of mask `mask_idx` at position `pos + j`, which is valid if
 * bit `j` of `valid_lanes.bits` is set, that is, if all positions that the mask keeps are valid
 * nucleotides. The invalid lanes are unspecified. The vectors cover all positions of the sequence,
 * in steps of the number of lanes; the lanes of the last one past the last position are not valid.
 * Per vector of positions, the masks are emitted in their order, so that `pos` is non-decreasing
 * per mask, but not across masks. For single spaced k-mers, use for_each_valid_lane() in the
 * callback.
 *
 * @tparam Kernel    Bit extraction kernel of a SIMD arch, see BitExtractKernel, tagged as spaced
 *                   k-mer mask, see make_spaced_kmer_kernel(). Its Layout is the Layout of the
 *                   emitted spaced k-mers. The spans of all kernels must be the same, and in [1, 32].
 * @tparam NMasks    Number of kernels in the array.
 * @tparam Enc       Encoder functor, returns 0..3 for valid bases, >=4 for invalid.
 * @tparam Callback  Callback function, called with vectors of spaced k-mers, see above.
 */
template<typename Kernel, std::size_t NMasks, typename Enc, typename Callback>
    requires SpacedKmerMaskType<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_rolling_simd(
    std::string_view seq,
    std::array<Kernel, NMasks> const& kernels,
    Enc&& enc,
    Callback&& callback
) {
    static_assert(NMasks > 0, "Need at least one kernel.");
    using Simd = typename Kernel::simd_arch;
    constexpr Encoding E = std::remove_cvref_t<Enc>::encoding;
    constexpr Layout L = Kernel::layout;
    constexpr std::size_t lanes = Kernel::lanes;

    // Input boundary checks. The masks must not have bits above the span, see
    // spaced_kmer_ascii_rolling_simd_shift_in_().
    std::size_t const span_k = common_spaced_kmer_span_(kernels);
    for ([[maybe_unused]] auto const& kernel : kernels) {
        assert(is_valid_spaced_kmer_mask(kernel));
    }
    std::size_t const top_shift = 2 * (span_k - 1);
    if (seq.size() < span_k) {
        return;
    }
    char const*       data    = seq.data();
    std::size_t const seq_len = seq.size();

    // Sliding window k-mer along the sequence, and its valid positions. All but the last character
    // of the first span are shifted in up front, so that lane 0 of the first vector holds the
    // spaced k-mers at position 0.
    std::uint64_t rolling_kmer      = 0;
    std::uint64_t rolling_valid_pos = 0;
    std::size_t i = 0;
    for (; i + 1 < span_k; ++i) {
        spaced_kmer_ascii_rolling_simd_shift_in_<L>(
            data[i], enc, top_shift, rolling_kmer, rolling_valid_pos
        );
    }

    // The lane k-mers and their valid bits. The lane k-mers are held in registers, see
    // Kernel::from_values().
    std::array<std::uint64_t, lanes> lane_kmers;
    std::uint64_t lane_valids[lanes];

    // Emit the spaced k-mers of all masks for the lanes of the `count` characters at `i`. A lane
    // is valid for a mask if its valid bits cover all bits of the mask.
    auto const emit = [&](std::size_t count) FISK_ALWAYS_INLINE_LAMBDA {
        spaced_kmer_ascii_rolling_simd_fill_lanes_<Kernel>(
            data + i, count, enc, top_shift, rolling_kmer, rolling_valid_pos, lane_kmers,
            lane_valids
        );
        auto const x = Kernel::from_values(lane_kmers);
        std::size_t const pos = i + 1 - span_k;
        FISK_PRAGMA_UNROLL_16
        for (std::size_t m = 0; m < NMasks; ++m) {
            std::uint64_t const mm = kernels[m].mask.mask;
            std::uint64_t bits = 0;
            for_each_lane_ct_<lanes>([&](auto lane_ic) FISK_ALWAYS_INLINE_LAMBDA {
                constexpr std::size_t lane = lane_ic;
                bits |= static_cast<std::uint64_t>((lane_valids[lane] & mm) == mm) << lane;
            });
            invoke_spaced_kmer_vector_valid_lanes_callback<E, L, Simd, NMasks == 1>(
                callback, m, pos,
                spaced_kmer_vector_cast_unchecked<E, L, Simd>(kernels[m].bit_extract(x)),
                SimdVectorValidLanes<Simd>{bits}
            );
        }
    };

    // Whole vectors, and the remaining positions as a last vector with fewer valid lanes.
    for (; i + lanes <= seq_len; i += lanes) {
        emit(lanes);
    }
    if (i < seq_len) {
        emit(seq_len - i);
    }
}

/**
 * @brief Iterate a sequence, extract all spaced k-mers from it with SIMD bit extraction, and call a
 * callback on vectors of them, for a single kernel.
 *
 * Same as the overload above, for a single kernel, with `mask_idx` always 0. The callback can also
 * be called without it, as callback(pos, spaced_kmers, valid_lanes).
 */
template<typename Kernel, typename Enc, typename Callback>
    requires SpacedKmerMaskType<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_ascii_checked_rolling_simd(
    std::string_view seq,
    Kernel const& kernel,
    Enc&& enc,
    Callback&& callback
) {
    std::array<Kernel, 1> const kernels{{kernel}};
    for_each_spaced_kmer_ascii_checked_rolling_simd(
        seq, kernels, std::forward<Enc>(enc), std::forward<Callback>(callback)
    );
}

#undef FISK_PRAGMA_UNROLL_16

} // namespace fisk
