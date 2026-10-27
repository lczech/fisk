#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "fisk/core/char_encoder.hpp"
#include "fisk/util/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/internal.hpp"

namespace fisk {

// =================================================================================================
//     K-mer Extraction from ASCII, Reextract, Scalar
// =================================================================================================

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
