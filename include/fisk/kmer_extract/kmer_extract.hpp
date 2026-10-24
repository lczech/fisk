#pragma once

#include <string>
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "fisk/core/char_encoder.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction
// =================================================================================================

/**
 * @brief Throw for an invalid k-mer extraction precondition.
 *
 * The extracted k-mer length must satisfy `1 <= k <= k_max`.
 * This helper is kept separate from the hot loop so the performance-critical extractor
 * can remain branch-free and avoid the runtime overhead of an inlined exception path.
 */
FISK_NOINLINE_COLD
inline void throw_invalid_kmer_k_(std::size_t k_max)
{
    // Keep the validity check out of the hot path: throwing here avoids a cold branch/landing pad
    // in otherwise branch-free extractors, which could prevent their inlining.
    throw std::invalid_argument(
        "Invalid call to k-mer extraction with k not in [1, " + std::to_string(k_max) + "]"
    );
}

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

/**
 * @brief Iterate a sequence, extract all k-mers from it (using re-extraction each time),
 * and call a callback function on each k-mer.
 *
 * This is the same as for_each_kmer_ascii_checked_rolling_scalar(), but re-extract the k-mer each
 * time from the input characters. This is of course slower, but apparently used in practice. We
 * hence implement it here for benchmarking.
 *
 * The Layout `L`, the encoder, and Func are as for for_each_kmer_ascii_checked_rolling_scalar().
 */
template<Layout L = Layout::kMSB, typename Encoder, typename Func>
    requires CharEncoder<std::remove_cvref_t<Encoder>>
FISK_ALWAYS_INLINE_FOR_EACH
inline void for_each_kmer_ascii_checked_reextract_scalar(
    std::string_view seq, std::size_t k, Encoder&& encoder, Func&& func
) {
    // Same as above, but each k-mer is extracted separately.
    // Not efficient, and worse for larger k. Meant only for benchmarking.

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

    // Slide the window over the sequence.
    std::size_t const stop = seq_len - k;
    for (std::size_t i = 0; i <= stop; ++i) {
        std::uint64_t kmer = 0;
        bool valid = true;

        for (std::size_t x = 0; x < k; ++x) {
            std::uint8_t const code = static_cast<std::uint8_t>(encoder(data[i + x]));
            valid &= (code < 4);
            if constexpr (L == Layout::kMSB) {
                kmer = (kmer << 2) | (code & 0x3u);
            } else if constexpr (L == Layout::kLSB) {
                kmer |= std::uint64_t{code & 0x3u} << (2 * x);
            } else {
                static_assert(
                    dependent_false_v<L>,
                    "Unhandled Layout in for_each_kmer_ascii_checked_reextract_scalar()"
                );
            }
        }

        if (valid) {
            // Here, i is already the start position of the k-mer.
            assert_fits_width_(kmer, k);
            invoke_kmer_callback<std::remove_cvref_t<Encoder>::encoding, L>(func, i, kmer);
        }
    }
}

} // namespace fisk
