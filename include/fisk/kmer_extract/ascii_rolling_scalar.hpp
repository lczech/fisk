#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

#include "fisk/core/char_encoder.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from ASCII, Rolling, Scalar
// =================================================================================================

/**
 * @brief Get the k-mer to emit from a rolling `kmer` that is only masked on emission.
 *
 * The rolling extractors shift each MSB base in without masking, so `kmer` keeps the bases that
 * have left the window in its upper bits, and `mask` is only applied here, to the emitted k-mers.
 * That keeps the mask off the dependency chain through `kmer` from one character to the next.
 * For LSB, nothing ever reaches the bits above the window, so it is emitted as is.
 */
template <Layout L>
FISK_ALWAYS_INLINE
inline std::uint64_t kmer_ascii_rolling_window_bits_(
    std::uint64_t kmer, std::uint64_t mask
) noexcept {
    if constexpr (L == Layout::kMSB) {
        return kmer & mask;
    } else if constexpr (L == Layout::kLSB) {
        return kmer;
    } else {
        static_assert(
            dependent_false_v<L>, "Unhandled Layout in kmer_ascii_rolling_window_bits_()"
        );
        return 0;
    }
}

/**
 * @brief Iterate a sequence, extract all valid k-mers from it (using bit shifts),
 * and call a callback function on each k-mer.
 *
 * The encoder must return:
 *   - 0,1,2,3 for valid A/C/G/T-like symbols
 *   - any value >= 4 for invalid symbols
 *
 * Any k-mer overlapping an invalid symbol is skipped.
 *
 * The encoder can be any CharEncoder, such as the lookup table encoder CharEncoderTable (see
 * core/char_encoder.hpp).
 *
 * The callback receives a Kmer (core/kmer.hpp) tagged with the encoder's own Encoding, and with
 * Layout `L`. The encoder therefore has to state its encoding, which is what the CharEncoder
 * concept requires: a bare function carries no such statement, and so cannot be passed here.
 * The layout is chosen at compile time, so either costs the same:
 *
 *     CharEncoderTable<Encoding::kACTG> const encoder;
 *     // Kmer<kACTG, kMSB>
 *     for_each_kmer_ascii_checked_rolling_scalar(seq, k, encoder, func);
 *     // Kmer<kACTG, kLSB>
 *     for_each_kmer_ascii_checked_rolling_scalar<Layout::kLSB>(seq, k, encoder, func);
 *
 * Use kmer_decode() to turn a k-mer back into a string, and kmer_convert() to move it to different
 * conventions.
 *
 * Func may be called either as `func(pos, kmer)` or as `func(kmer)`, where `pos` is the start
 * position of the k-mer in `seq`. See invoke_kmer_callback().
 *
 * @tparam L       Layout of the emitted k-mers, see Layout.
 * @tparam Encoder Encoder turning characters into two-bit codes, see CharEncoder.
 * @tparam Func    Callback function to be called for each valid k-mer.
 */
template<Layout L = Layout::kMSB, typename Encoder, typename Func>
    requires CharEncoder<std::remove_cvref_t<Encoder>>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_checked_rolling_scalar(
    std::string_view seq, std::size_t k, Encoder&& encoder, Func&& func
) {
    // Iterate all k-mers of a sequence, encoded as 2-bit packed uint64_t.
    // For a sequence of length n and k <= 32, this function visits
    // (n - k + 1) overlapping k-mers. Each k-mer is encoded into the
    // lowest 2*k bits of a uint64_t, using the two-bit encoding provided by `encoder`.

    // Boundary checks
    if (k == 0 || k > 32) {
        throw_invalid_kmer_k_(32);
    }
    if (seq.size() < k) {
        return;
    }

    // Shorthands for data access
    std::size_t const seq_len = seq.size();
    char const*       data    = seq.data();

    // MSB shifts each base in at the bottom, and needs a mask to drop the bases that have left the
    // window at the top, applied on emission, see kmer_ascii_rolling_window_bits_(). This works for
    // all k in [1, 32]. LSB shifts each base in at the top, at `top_shift`, and the base leaving at
    // the bottom simply falls off, so it needs no mask.
    std::uint64_t const mask = (k == 32)
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << (2 * k)) - 1u);
    unsigned const top_shift = 2 * (static_cast<unsigned>(k) - 1);

    std::uint64_t kmer = 0;
    std::size_t valid = 0;
    for( std::size_t i = 0; i < seq_len; ++i ) {
        std::uint8_t const code = encoder(data[i]);

        // Always shift in the low 2 bits. For invalid symbols this value is ignored,
        // because all overlapping k-mers will be skipped until the bad position
        // has slid out of the window.
        if constexpr (L == Layout::kMSB) {
            kmer = (kmer << 2) | (code & 0x03u);
        } else if constexpr (L == Layout::kLSB) {
            kmer = (kmer >> 2) | (std::uint64_t{code & 0x03u} << top_shift);
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_kmer_ascii_checked_rolling_scalar()"
            );
        }
        valid = (code < 4) ? (valid + 1) : 0;

        // We can emit once we have seen at least k characters, and the current
        // k-mer window does not overlap the most recent invalid character.
        if( valid >= k ) {
            // valid >= k implies i >= k - 1, so the start position `i + 1 - k` does not underflow.
            assert_fits_width_(kmer_ascii_rolling_window_bits_<L>(kmer, mask), k);
            invoke_kmer_callback<std::remove_cvref_t<Encoder>::encoding, L>(
                func, i + 1 - k, kmer_ascii_rolling_window_bits_<L>(kmer, mask)
            );
        }
    }
}

// =================================================================================================
//     Helpers of the Block-wise Rolling Loops
// =================================================================================================

// The block-wise loops of ascii_rolling_avx2.hpp and ascii_rolling_swar.hpp encode and check
// a block of characters at once, then roll the k-mers over the block with the helpers below.

/**
 * @brief Increment a counter of consecutive valid characters, or reset it to zero, without a
 * branch.
 *
 * Some compilers turn `keep ? count + 1 : 0` into a branch, which mispredicts on scattered invalid
 * characters. Instead, `count + 1` is masked with all zeros or all ones, which relies on the
 * modular arithmetic of unsigned types that are not promoted to int.
 */
template <typename T>
FISK_ALWAYS_INLINE
constexpr T increment_or_reset_(T count, bool keep) noexcept
{
    static_assert(
        std::is_unsigned_v<T> && sizeof(T) >= sizeof(unsigned),
        "increment_or_reset_() needs an unsigned type that is not promoted to int"
    );
    return (count + 1) & (T{0} - static_cast<T>(keep));
}

/**
 * @brief Consume a contiguous run of valid two-bit codes.
 *
 * `base` is the index into the original sequence of `codes[0]`, used to report the start position
 * of each emitted k-mer to the callback. `mask` and `top_shift` are as in
 * for_each_kmer_ascii_checked_rolling_scalar(), for Layout::kMSB and Layout::kLSB, respectively.
 * For MSB, `mask` is only applied to the emitted k-mers, see kmer_ascii_rolling_window_bits_().
 */
template<Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE
inline void for_each_kmer_ascii_checked_rolling_valid_run_(
    std::uint8_t const* codes,
    std::size_t len,
    std::size_t base,
    std::size_t k,
    std::uint64_t mask,
    unsigned top_shift,
    std::uint64_t& kmer,
    std::size_t& valid,
    Func&& func
) {
    // The codes of a valid run are all within 0-3, so they need no masking.
    std::size_t idx = 0;

    // Warm-up phase until we have accumulated k valid characters.
    if (valid < k) {
        std::size_t const need = k - valid;
        std::size_t const warm = std::min(len, need);

        for (; idx < warm; ++idx) {
            if constexpr (L == Layout::kMSB) {
                kmer = (kmer << 2) | codes[idx];
            } else if constexpr (L == Layout::kLSB) {
                kmer = (kmer >> 2) | (std::uint64_t{codes[idx]} << top_shift);
            } else {
                static_assert(
                    dependent_false_v<L>,
                    "Unhandled Layout in for_each_kmer_ascii_checked_rolling_valid_run_()"
                );
            }
        }

        valid += warm;

        // Still not enough valid characters to emit anything.
        if (valid < k) {
            return;
        }

        // The last character processed above, at index base + idx - 1, just completed the first
        // valid k-mer, which hence starts at base + idx - k.
        assert_fits_width_(kmer_ascii_rolling_window_bits_<L>(kmer, mask), k);
        invoke_kmer_callback<E, L>(
            func, base + idx - k, kmer_ascii_rolling_window_bits_<L>(kmer, mask)
        );

        // Saturate valid so we do not keep incrementing a counter that is only used
        // as a threshold predicate.
        valid = k;
    }

    // Fully warm: every additional valid code yields one k-mer.
    for (; idx < len; ++idx) {
        if constexpr (L == Layout::kMSB) {
            kmer = (kmer << 2) | codes[idx];
        } else if constexpr (L == Layout::kLSB) {
            kmer = (kmer >> 2) | (std::uint64_t{codes[idx]} << top_shift);
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_kmer_ascii_checked_rolling_valid_run_()"
            );
        }
        assert_fits_width_(kmer_ascii_rolling_window_bits_<L>(kmer, mask), k);
        invoke_kmer_callback<E, L>(
            func, base + idx + 1 - k, kmer_ascii_rolling_window_bits_<L>(kmer, mask)
        );
    }
}

/**
 * @brief Iterate the k-mers ending in one block of `Width` characters, given their codes and their
 * validity.
 *
 * Bit j of `valid_bits` is set iff `codes[j]` is a valid code. Fully valid blocks are handed to
 * for_each_kmer_ascii_checked_rolling_valid_run_(). Other blocks are rolled in one character at a
 * time, with invalid characters resetting the window; this is branch-free apart from the emission,
 * as splitting the block into valid runs instead would mispredict on scattered invalid characters.
 * Codes of invalid characters are hence rolled into the k-mer as well, and must be within 0-3, too.
 * `base` is the index into the original sequence of `codes[0]`. The other parameters are as for
 * for_each_kmer_ascii_checked_rolling_valid_run_().
 */
template<Encoding E, Layout L, unsigned Width, typename Func>
FISK_ALWAYS_INLINE
inline void for_each_kmer_ascii_checked_rolling_block_(
    std::uint8_t const* codes,
    std::uint32_t valid_bits,
    std::size_t base,
    std::size_t k,
    std::uint64_t mask,
    unsigned top_shift,
    std::uint64_t& kmer,
    std::size_t& valid,
    Func&& func
) {
    static_assert(
        Width >= 1 && Width <= 32,
        "for_each_kmer_ascii_checked_rolling_block_() supports 1 to 32 lanes"
    );
    constexpr std::uint32_t all_valid = (Width == 32)
        ? 0xFFFFFFFFu
        : ((std::uint32_t{1} << Width) - 1u);

    // Fast path: whole block valid.
    if (valid_bits == all_valid) {
        for_each_kmer_ascii_checked_rolling_valid_run_<E, L>(
            codes, Width, base, k, mask, top_shift, kmer, valid, std::forward<Func>(func)
        );
        return;
    }

    // Slow path: block contains invalid characters.
    for (unsigned j = 0; j < Width; ++j) {
        if constexpr (L == Layout::kMSB) {
            kmer = (kmer << 2) | codes[j];
        } else if constexpr (L == Layout::kLSB) {
            kmer = (kmer >> 2) | (std::uint64_t{codes[j]} << top_shift);
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_kmer_ascii_checked_rolling_block_()"
            );
        }

        valid = increment_or_reset_(valid, ((valid_bits >> j) & 1u) != 0);

        if (valid >= k) {
            assert_fits_width_(kmer_ascii_rolling_window_bits_<L>(kmer, mask), k);
            invoke_kmer_callback<E, L>(
                func, base + j + 1 - k, kmer_ascii_rolling_window_bits_<L>(kmer, mask)
            );
        }
    }
}

/**
 * @brief Iterate the k-mers ending in `data[begin, end)` one character at a time, continuing from
 * the state that a block-wise main loop left behind.
 *
 * The same loop as in for_each_kmer_ascii_checked_rolling_scalar(), with the lookup table encoder
 * CharEncoderTable. Used as the scalar tail of the block-wise extractors, and as their full
 * fallback where they have no blocks.
 * The other parameters are as for for_each_kmer_ascii_checked_rolling_valid_run_().
 */
template<Encoding E, Layout L, typename Func>
FISK_ALWAYS_INLINE
inline void for_each_kmer_ascii_checked_rolling_tail_(
    char const* data,
    std::size_t begin,
    std::size_t end,
    std::size_t k,
    std::uint64_t mask,
    unsigned top_shift,
    std::uint64_t& kmer,
    std::size_t& valid,
    Func&& func
) {
    CharEncoderTable<E> const encoder;
    for (std::size_t i = begin; i < end; ++i) {
        std::uint8_t const code = encoder(data[i]);

        if constexpr (L == Layout::kMSB) {
            kmer = (kmer << 2) | (code & 0x03u);
        } else if constexpr (L == Layout::kLSB) {
            kmer = (kmer >> 2) | (std::uint64_t{code & 0x03u} << top_shift);
        } else {
            static_assert(
                dependent_false_v<L>,
                "Unhandled Layout in for_each_kmer_ascii_checked_rolling_tail_()"
            );
        }
        valid = (code < 4) ? (valid + 1) : 0;

        if (valid >= k) {
            assert_fits_width_(kmer_ascii_rolling_window_bits_<L>(kmer, mask), k);
            invoke_kmer_callback<E, L>(
                func, i + 1 - k, kmer_ascii_rolling_window_bits_<L>(kmer, mask)
            );
        }
    }
}

} // namespace fisk
