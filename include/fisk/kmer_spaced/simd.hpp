#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/core/intrinsics.hpp"

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
inline void spaced_kmer_simd_shift_in_(
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
    // inline in the expression below made the by_mask extraction up to 30% slower with GCC 14.
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
        static_assert(dependent_false_v<L>, "Unhandled Layout in spaced_kmer_simd_shift_in_()");
    }
}

// Process the remaining characters of `seq`, starting at `i`, that do not fill a whole SIMD
// block, one position at a time. This is position-major by construction, and hence used for
// both the by_mask and the by_position variants.
template<typename Kernel, std::size_t NMasks, typename Enc, typename Callback>
FISK_ALWAYS_INLINE
inline void spaced_kmer_simd_tail_(
    std::string_view seq,
    std::size_t i,
    std::size_t span_k,
    std::size_t top_shift,
    std::array<Kernel, NMasks> const& kernels,
    Enc&& enc,
    std::uint64_t rolling_kmer,
    std::uint64_t rolling_valid_pos,
    Callback&& callback
) {
    for (; i < seq.size(); ++i) {
        spaced_kmer_simd_shift_in_<Kernel::layout>(
            seq[i], enc, top_shift, rolling_kmer, rolling_valid_pos
        );

        // Apply the callback for all masks that are satisfied at this position.
        for (std::size_t m = 0; m < NMasks; ++m) {
            Kernel const& kernel = kernels[m];
            if ((rolling_valid_pos & kernel.mask.mask) == kernel.mask.mask) {
                auto const raw = kernel.bit_extract(rolling_kmer);
                assert_fits_width_(raw, spaced_kmer_weight(kernel.mask.mask));
                invoke_spaced_kmer_callback<
                    std::remove_cvref_t<Enc>::encoding, Kernel::layout, NMasks == 1
                >(callback, m, i + 1 - span_k, raw);
            }
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
//     SIMD Spaced k-mer Masks
// =================================================================================================

/**
 * @brief A bit extraction kernel of bit_extract/simd.hpp, as opposed to a mask or table.
 */
template <typename Kernel>
concept SpacedKmerKernel_ = requires {
    Kernel::lanes;
    typename Kernel::simd_vector;
} && std::constructible_from<Kernel, std::uint64_t>;

/**
 * @brief Create a bit extraction kernel of a spaced k-mer mask, such as
 * BitExtractKernelButterflyAVX2, for the SIMD spaced k-mer extraction functions below.
 */
template <typename Kernel, Layout L>
    requires SpacedKmerKernel_<Kernel>
inline SpacedKmerMask<L, Kernel> make_spaced_kmer_kernel(
    SpacedKmerMask<L, BitExtractMask> const& mask
) {
    return convert_spaced_kmer_mask_<Kernel>(mask, [](std::uint64_t bits) {
        return Kernel(bits);
    });
}

/**
 * @brief Create a bit extraction kernel of a spaced k-mer mask given as a string, see
 * make_spaced_kmer_mask().
 */
template <typename Kernel, Layout L = Layout::kMSB>
    requires SpacedKmerKernel_<Kernel>
inline SpacedKmerMask<L, Kernel> make_spaced_kmer_kernel(std::string const& mask)
{
    return make_spaced_kmer_kernel<Kernel>(make_spaced_kmer_mask<L>(mask));
}

/**
 * @brief Create the bit extraction kernels of a set of spaced k-mer masks, for instance to be used
 * with BitExtractKernelDispatcher.
 */
template <typename Kernel, Layout L>
    requires SpacedKmerKernel_<Kernel>
inline std::vector<SpacedKmerMask<L, Kernel>> make_spaced_kmer_kernels(
    std::vector<SpacedKmerMask<L, BitExtractMask>> const& masks
) {
    return convert_spaced_kmer_masks_<Kernel>(masks, [](std::uint64_t bits) {
        return Kernel(bits);
    });
}

// =================================================================================================
//     SIMD Spaced k-mer Extraction: by_mask
// =================================================================================================

/**
 * @brief Iterate a sequence and extract spaced k-mers using SIMD bit extraction, mask-major,
 * for an array of kernels.
 *
 * Emission order: within each SIMD block of consecutive positions, the spaced k-mers of mask 0 are
 * emitted first, then those of mask 1, and so on. Hence, `pos` is non-decreasing per mask, but not
 * across the whole output. Use for_each_spaced_kmer_simd_by_position() if the output needs to be
 * ordered by position, as in the scalar for_each_spaced_kmer().
 *
 * @tparam Kernel    SIMD/scalar kernel, tagged as spaced k-mer mask, see make_spaced_kmer_kernel().
 *                   Its Layout is the Layout of the emitted spaced k-mers.
 * @tparam NMasks    Number of kernels in the array.
 * @tparam Enc       Encoder functor, returns 0..3 for valid bases, >=4 for invalid.
 * @tparam Callback  Callback functor, called as callback(pos, mask_idx, spaced_kmer), or, when
 *                   NMasks == 1, as callback(pos, spaced_kmer). See invoke_spaced_kmer_callback().
 */
template<typename Kernel, std::size_t NMasks, typename Enc, typename Callback>
    requires SpacedKmerMaskType<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_simd_by_mask(
    std::string_view seq,
    std::array<Kernel, NMasks> const& kernels,
    Enc&& enc,
    Callback&& callback
) {
    static_assert(NMasks > 0, "Need at least one kernel.");

    // Input boundary checks. The masks must not have bits above the span, see
    // spaced_kmer_simd_shift_in_().
    std::size_t const span_k = common_spaced_kmer_span_(kernels);
    for ([[maybe_unused]] auto const& kernel : kernels) {
        assert(is_valid_spaced_kmer_mask(kernel));
    }
    std::size_t const top_shift = 2 * (span_k - 1);
    if (seq.size() < span_k) {
        return;
    }

    // Shorthands
    using simd_vector = typename Kernel::simd_vector;
    char const*       data    = seq.data();
    std::size_t const seq_len = seq.size();

    // Set up input and output buffers to transfer to and from the simd kernel.
    constexpr std::size_t L = Kernel::lanes;
    alignas(64) std::uint64_t simd_buffer[L];
    alignas(64) std::uint64_t simd_valids[L];

    // Sliding window kmer along the sequence, and its valid positions
    std::uint64_t rolling_kmer      = 0;
    std::uint64_t rolling_valid_pos = 0;

    // Iterate the sequence. Each kmer is only constructed once. Per iteration of
    // this outer loop, we do L many increments along the sequence,
    // and call all mask kernels to produce the spaced k-mers.
    std::size_t i = 0;
    for (; i + L <= seq_len; i += L ) {

        // Build one rolling kmer per lane, from consecutive sequence positions.
        // That is, `rolling_kmer` is our rolling k-mer, and in each iteration here
        // its current state (corresponding to one k-mer along the input sequence)
        // gets copied into one of the lanes, until all lanes are filled. Same for the valid bits.
        // Note: This loop is kept inline on purpose. Moving it into a separate helper function
        // made GCC produce about 14% slower code for single masks with SSE2.
        FISK_PRAGMA_UNROLL_16
        for (std::size_t lane = 0; lane < L; ++lane) {
            spaced_kmer_simd_shift_in_<Kernel::layout>(
                data[i + lane], enc, top_shift, rolling_kmer, rolling_valid_pos
            );
            simd_buffer[lane] = rolling_kmer;
            simd_valids[lane] = rolling_valid_pos;
        }

        // Load vector lanes once from all stored kmers.
        simd_vector const x = Kernel::load(simd_buffer);

        // Start position of the spaced k-mer in lane 0. This underflows in the first block(s)
        // while i < span_k - 1. That is fine: a lane can only be valid once span_k characters
        // have been shifted in, and for those lanes, adding the lane index wraps back around to
        // the correct position, as unsigned arithmetic is modular.
        std::size_t const start_pos = i - (span_k - 1);

        // Process all masks/kernels. This loop is compile-time unrolled for speed.
        // We apply each mask to all k-mers stored in the lanes, and emit the valid ones
        // to the callback function.
        FISK_PRAGMA_UNROLL_16
        for (std::size_t m = 0; m < NMasks; ++m) {
            Kernel const& kernel = kernels[m];
            std::uint64_t const mm = kernel.mask.mask;

            // Extract the bits across all lanes. Then, emit the lanes where all positions kept
            // by the mask are valid characters.
            Kernel::store(kernel.bit_extract(x), simd_buffer);
            for_each_lane_ct_<L>([&](auto lane_ic) {
                constexpr std::size_t lane = lane_ic;
                if ((simd_valids[lane] & mm) == mm) {
                    assert_fits_width_(simd_buffer[lane], spaced_kmer_weight(kernel.mask.mask));
                    invoke_spaced_kmer_callback<
                        std::remove_cvref_t<Enc>::encoding, Kernel::layout, NMasks == 1
                    >(callback, m, start_pos + lane, simd_buffer[lane]);
                }
            });
        }
    }

    // Tail loop for the final scalar remainder.
    spaced_kmer_simd_tail_(
        seq, i, span_k, top_shift, kernels, enc, rolling_kmer, rolling_valid_pos, callback
    );
}

/**
 * @brief Iterate a sequence and extract spaced k-mers using SIMD bit extraction, mask-major,
 * for a single kernel.
 *
 * This is just a thin wrapper that copies the kernel into std::array<Kernel,1>
 * and forwards to the array implementation above.
 *
 * @tparam Kernel    SIMD/scalar kernel, tagged as spaced k-mer mask, see make_spaced_kmer_kernel().
 *                   Its Layout is the Layout of the emitted spaced k-mers.
 * @tparam Enc       Encoder functor, returns 0..3 for valid bases, >=4 for invalid.
 * @tparam Callback  Callback functor, called as callback(pos, mask_idx, spaced_kmer), or as
 *                   callback(pos, spaced_kmer), with mask_idx always 0 here. See
 *                   invoke_spaced_kmer_callback().
 */
template<typename Kernel, typename Enc, typename Callback>
    requires SpacedKmerMaskType<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_simd_by_mask(
    std::string_view seq,
    Kernel const& kernel,
    Enc&& enc,
    Callback&& callback
) {
    std::array<Kernel, 1> kernels{{kernel}};
    for_each_spaced_kmer_simd_by_mask(
        seq, kernels, std::forward<Enc>(enc), std::forward<Callback>(callback)
    );
}

// =================================================================================================
//     SIMD Spaced k-mer Extraction: by_position
// =================================================================================================

/**
 * @brief Iterate a sequence and extract spaced k-mers using SIMD bit extraction, position-major,
 * for an array of kernels.
 *
 * Emission order: `pos` is non-decreasing across the whole call; at each `pos`, masks are emitted
 * in mask-array order. This is the same order as in the scalar for_each_spaced_kmer(). If the
 * output does not need to be ordered by position, for_each_spaced_kmer_simd_by_mask() is faster
 * on some CPUs and compilers.
 *
 * @tparam Kernel    SIMD/scalar kernel, tagged as spaced k-mer mask, see make_spaced_kmer_kernel().
 *                   Its Layout is the Layout of the emitted spaced k-mers.
 * @tparam NMasks    Number of kernels in the array.
 * @tparam Enc       Encoder functor, returns 0..3 for valid bases, >=4 for invalid.
 * @tparam Callback  Callback functor, called as callback(pos, mask_idx, spaced_kmer), or, when
 *                   NMasks == 1, as callback(pos, spaced_kmer). See invoke_spaced_kmer_callback().
 */
template<typename Kernel, std::size_t NMasks, typename Enc, typename Callback>
    requires SpacedKmerMaskType<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_simd_by_position(
    std::string_view seq,
    std::array<Kernel, NMasks> const& kernels,
    Enc&& enc,
    Callback&& callback
) {
    // Mechanism: per SIMD block, validity across all (mask, lane) pairs is packed into a presence
    // bitset (bit index `lane * NMasks + m`), and position-major emission walks only the set bits
    // via std::countr_zero(), skipping runs of invalid entries entirely rather than branching over
    // each one. Ascending bit index visits lane ascending (major), then m ascending within a lane
    // (minor), which is exactly position-major order. The bitset uses one uint64_t word per 64
    // (mask, lane) pairs, which is a single word for all ISAs up to AVX2, and for AVX512 with up
    // to 8 masks.

    constexpr std::size_t L = Kernel::lanes;
    static_assert(NMasks > 0, "Need at least one kernel.");

    std::size_t const span_k = common_spaced_kmer_span_(kernels);
    for ([[maybe_unused]] auto const& kernel : kernels) {
        assert(is_valid_spaced_kmer_mask(kernel));
    }
    std::size_t const top_shift = 2 * (span_k - 1);
    if (seq.size() < span_k) {
        return;
    }

    using simd_vector = typename Kernel::simd_vector;
    char const*       data    = seq.data();
    std::size_t const seq_len = seq.size();

    alignas(64) std::uint64_t simd_buffer[L];
    alignas(64) std::uint64_t simd_valids[L];

    std::uint64_t rolling_kmer      = 0;
    std::uint64_t rolling_valid_pos = 0;

    std::size_t i = 0;
    for (; i + L <= seq_len; i += L ) {
        // Build one rolling kmer per lane, as in for_each_spaced_kmer_simd_by_mask() above.
        FISK_PRAGMA_UNROLL_16
        for (std::size_t lane = 0; lane < L; ++lane) {
            spaced_kmer_simd_shift_in_<Kernel::layout>(
                data[i + lane], enc, top_shift, rolling_kmer, rolling_valid_pos
            );
            simd_buffer[lane] = rolling_kmer;
            simd_valids[lane] = rolling_valid_pos;
        }
        simd_vector const x = Kernel::load(simd_buffer);

        // See for_each_spaced_kmer_simd_by_mask() above for why this wraps safely.
        std::size_t const start_pos = i - (span_k - 1);

        // Extract the bits for all masks, and set the presence bits of valid (mask, lane) pairs.
        constexpr std::size_t n_words = (NMasks * L + 63) / 64;
        alignas(64) std::uint64_t mask_kmers[NMasks][L];
        std::uint64_t bits[n_words] = {};
        FISK_PRAGMA_UNROLL_16
        for (std::size_t m = 0; m < NMasks; ++m) {
            Kernel::store(kernels[m].bit_extract(x), mask_kmers[m]);
            std::uint64_t const mm = kernels[m].mask.mask;
            for_each_lane_ct_<L>([&](auto lane_ic) {
                constexpr std::size_t lane = lane_ic;
                if ((simd_valids[lane] & mm) == mm) {
                    std::size_t const idx = lane * NMasks + m;
                    bits[idx / 64] |= (std::uint64_t{1} << (idx % 64));
                }
            });
        }

        // Emit the valid pairs in position-major order, by walking the set bits.
        for (std::size_t w = 0; w < n_words; ++w) {
            std::uint64_t word = bits[w];
            while (word != 0) {
                unsigned const idx
                    = static_cast<unsigned>(w * 64)
                    + static_cast<unsigned>(std::countr_zero(word))
                ;
                word &= (word - 1);
                unsigned const lane = idx / NMasks;
                unsigned const m    = idx % NMasks;
                assert_fits_width_(mask_kmers[m][lane], spaced_kmer_weight(kernels[m].mask.mask));
                invoke_spaced_kmer_callback<
                    std::remove_cvref_t<Enc>::encoding, Kernel::layout, NMasks == 1
                >(callback, m, start_pos + lane, mask_kmers[m][lane]);
            }
        }
    }

    // Tail loop for the final scalar remainder, which is position-major by construction.
    spaced_kmer_simd_tail_(
        seq, i, span_k, top_shift, kernels, enc, rolling_kmer, rolling_valid_pos, callback
    );
}

/**
 * @brief Iterate a sequence and extract spaced k-mers using SIMD bit extraction, position-major,
 * for a single kernel.
 *
 * This is just a thin wrapper that copies the kernel into std::array<Kernel,1>
 * and forwards to the array implementation above.
 *
 * @tparam Kernel    SIMD/scalar kernel, tagged as spaced k-mer mask, see make_spaced_kmer_kernel().
 *                   Its Layout is the Layout of the emitted spaced k-mers.
 * @tparam Enc       Encoder functor, returns 0..3 for valid bases, >=4 for invalid.
 * @tparam Callback  Callback functor, called as callback(pos, mask_idx, spaced_kmer), or as
 *                   callback(pos, spaced_kmer), with mask_idx always 0 here. See
 *                   invoke_spaced_kmer_callback().
 */
template<typename Kernel, typename Enc, typename Callback>
    requires SpacedKmerMaskType<Kernel>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_spaced_kmer_simd_by_position(
    std::string_view seq,
    Kernel const& kernel,
    Enc&& enc,
    Callback&& callback
) {
    std::array<Kernel, 1> kernels{{kernel}};
    for_each_spaced_kmer_simd_by_position(
        seq, kernels, std::forward<Enc>(enc), std::forward<Callback>(callback)
    );
}

#undef FISK_PRAGMA_UNROLL_16

} // namespace fisk
