#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "bit_extract.hpp"
#include "bit_extract_simd.hpp"
#include "kmer_spaced.hpp"
#include "seq_enc.hpp"

// =================================================================================================
//     SIMD Helper Functions
// =================================================================================================

/**
 * @brief Emit all valid SIMD lanes in order.
 *
 * The function only emit lanes up to `L`, which is the number of lanes for the SIMD architecture.
 * For each lane, we check if the positions that are kept in the mask are valid, meaning that the
 * spaced k-mer only contains valid characters, and only then emits it to the callback.
 */
template<std::size_t L, typename Callback>
inline void emit_simd_lanes_spaced_kmers(
    std::uint64_t const* kmers,
    std::uint64_t const* valid_pos,
    std::uint64_t mask,
    std::size_t start_pos,
    Callback&& cb
) {
    // Compile-time reduction of unused lanes when L < 8 (e.g., for SSE2).
    if constexpr (L >= 1) { if ((valid_pos[0] & mask) == mask) { cb(start_pos + 0, kmers[0]); }}
    if constexpr (L >= 2) { if ((valid_pos[1] & mask) == mask) { cb(start_pos + 1, kmers[1]); }}
    if constexpr (L >= 3) { if ((valid_pos[2] & mask) == mask) { cb(start_pos + 2, kmers[2]); }}
    if constexpr (L >= 4) { if ((valid_pos[3] & mask) == mask) { cb(start_pos + 3, kmers[3]); }}
    if constexpr (L >= 5) { if ((valid_pos[4] & mask) == mask) { cb(start_pos + 4, kmers[4]); }}
    if constexpr (L >= 6) { if ((valid_pos[5] & mask) == mask) { cb(start_pos + 5, kmers[5]); }}
    if constexpr (L >= 7) { if ((valid_pos[6] & mask) == mask) { cb(start_pos + 6, kmers[6]); }}
    if constexpr (L >= 8) { if ((valid_pos[7] & mask) == mask) { cb(start_pos + 7, kmers[7]); }}
}

// =================================================================================================
//     SIMD Spaced k-mer Extraction
// =================================================================================================

/**
 * @brief Iterate a sequence and extract spaced k-mers using SIMD bit extraction,
 * for an array of kernels.
 *
 * This is the implementation used for arrays and single kernels. The single-kernel overload
 * simply forwards into this by wrapping the kernel into a std::array<Kernel,1>. That has a slight
 * overhead for some int copies, but should be negligible. If needed, do this copy only once outside.
 *
 * @tparam Kernel    SIMD/scalar kernel type.
 * @tparam NMasks    Number of kernels in the array.
 * @tparam Enc       Encoder functor, returns 0..3 for valid bases, >=4 for invalid.
 * @tparam Callback  Callback functor, called as callback(mask_idx, pos, spaced_kmer).
 */
template<typename Kernel, std::size_t NMasks, typename Enc, typename Callback>
inline void for_each_spaced_kmer_simd(
    std::string_view seq,
    std::size_t const span_k,
    std::array<Kernel, NMasks> const& kernels,
    Enc&& enc,
    Callback&& callback
) {
    static_assert(NMasks > 0, "Need at least one kernel.");
    static_assert(
        std::is_invocable_v<Callback, std::size_t, std::size_t, std::uint64_t>,
        "Callback must be callable as callback(mask_idx, pos, spaced_kmer)."
    );

    // Input boundary checks
    if (span_k == 0 || span_k > 32) {
        throw std::runtime_error(
            "Invalid call to SIMD spaced k-mer extraction with k not in [1, 32]"
        );
    }
    if (seq.size() < span_k) {
        return;
    }

    // Optional sanity checks on the masks. Left out here for benchmarking speed.
    // for (auto const& kernel : kernels) {
    //     if( !is_valid_spaced_kmer_mask(kernel.mask) ) {
    //         throw std::runtime_error("Invalid spaced k-mer mask");
    //     }
    // }

    // Mask to keep only the lowest 2*k bits, works for all k in [1, 32]
    std::uint64_t const span_mask = (span_k == 32)
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << (2 * span_k)) - 1u)
    ;

    // Shorthands
    using simd_vector = typename Kernel::simd_vector;
    char const*       data    = seq.data();
    std::size_t const seq_len = seq.size();

    // Set up input and output buffers to transfer to and from the simd kernel.
    constexpr std::size_t L = Kernel::lanes;
    alignas(64) std::uint64_t simd_buffer[L];
    alignas(64) std::uint64_t simd_valids[L];

    // Sliding window kmer along the sequence, and current number of valid input chars
    std::uint64_t rolling_kmer      = 0;
    std::uint64_t rolling_valid_pos = 0;

    // Iterate the sequence. Each kmer is only constructed once. Per iteration of
    // this outer loop, the inner loop does L many increments along the sequence,
    // and calls all mask kernels to produce the spaced k-mers.
    std::size_t i = 0;
    for (; i + L <= seq_len; i += L ) {

        // Some preprocessor shenanigans to encourage loop unrolling
        #if defined(__clang__)
            #define FISK_PRAGMA_UNROLL_16 _Pragma("unroll 16")
        #elif defined(__GNUC__)
            #define FISK_PRAGMA_UNROLL_16 _Pragma("GCC unroll 16")
        #else
            #define FISK_PRAGMA_UNROLL_16
        #endif

        // Build one rolling kmer per lane, from consecutive sequence positions.
        // That is, `rolling_kmer` is our rolling k-mer, and in each iteration here
        // its current state (corresponding to one k-mer along the input sequence)
        // gets copied into one of the lanes, until all lanes are filled.
        FISK_PRAGMA_UNROLL_16
        for (std::size_t lane = 0; lane < L; ++lane) {
            std::uint8_t const code = static_cast<std::uint8_t>(enc(data[i + lane]));

            // Shift in the next base. For invalid bases, the low 2 bits are irrelevant,
            // because validity is checked separately before emission.
            rolling_kmer = ((rolling_kmer << 2) & span_mask) | (code & 0x03u);

            // Shift in 11 for valid, 00 for invalid. This ensures that spaced k-mers which
            // contain an invalid base in them (which was not masked out) will be skipped.
            rolling_valid_pos
                = ((rolling_valid_pos << 2) & span_mask)
                | (static_cast<std::uint64_t>(code < 4) * 0x03u)
            ;

            // Store the kmer and its valid bits in the current lane.
            simd_buffer[lane] = rolling_kmer;
            simd_valids[lane] = rolling_valid_pos;
        }

        // Load vector lanes once from all stored kmers.
        simd_vector const x = Kernel::load(simd_buffer);
        std::size_t const start_pos = i - (span_k - 1);

        // Process all masks/kernels. This loop is compile-time unrolled for speed.
        // We apply each mask to all k-mers stored in the lanes, and emit the valid ones
        // to the callback function.
        FISK_PRAGMA_UNROLL_16
        for (std::size_t m = 0; m < NMasks; ++m) {
            Kernel const& kernel = kernels[m];

            // Extract the bits across all lanes, and emit the valid ones.
            simd_vector const y = kernel.bit_extract(x);
            Kernel::store(y, simd_buffer);
            emit_simd_lanes_spaced_kmers<L>(
                simd_buffer,
                simd_valids,
                kernel.mask.mask,
                start_pos,
                [&](std::size_t pos, std::uint64_t value) {
                    callback(m, pos, value);
                }
            );
        }

        #undef FISK_PRAGMA_UNROLL_16
    }

    // Tail loop for the final scalar remainder.
    for (; i < seq_len; ++i) {
        std::uint8_t const code = static_cast<std::uint8_t>(enc(data[i]));

        // Shift in the new character, as before.
        rolling_kmer = ((rolling_kmer << 2) & span_mask) | (code & 0x03u);
        rolling_valid_pos
            = ((rolling_valid_pos << 2) & span_mask)
            | (static_cast<std::uint64_t>(code < 4) * 0x03u)
        ;

        // Apply the callback for all masks that are satisfied at this position.
        for (std::size_t m = 0; m < NMasks; ++m) {
            Kernel const& kernel = kernels[m];
            if ((rolling_valid_pos & kernel.mask.mask) == kernel.mask.mask) {
                callback(m, i + 1 - span_k, kernel.bit_extract(rolling_kmer));
            }
        }
    }
}

/**
 * @brief Iterate a sequence and extract spaced k-mers using SIMD bit extraction,
 * for a single kernel.
 *
 * This is just a thin wrapper that copies the kernel into std::array<Kernel,1>
 * and forwards to the array implementation above.
 *
 * @tparam Kernel    SIMD/scalar kernel type.
 * @tparam Enc       Encoder functor, returns 0..3 for valid bases, >=4 for invalid.
 * @tparam Callback  Callback functor, called as callback(pos, value).
 */
template<typename Kernel, typename Enc, typename Callback>
inline void for_each_spaced_kmer_simd(
    std::string_view seq,
    std::size_t const span_k,
    Kernel const& kernel,
    Enc&& enc,
    Callback&& callback
) {
    static_assert(
        std::is_invocable_v<Callback, std::size_t, std::uint64_t>,
        "Callback must be callable as callback(pos, value)."
    );

    std::array<Kernel, 1> kernels{{kernel}};
    for_each_spaced_kmer_simd(
        seq,
        span_k,
        kernels,
        std::forward<Enc>(enc),
        [&](std::size_t /*mask_idx*/, std::size_t pos, std::uint64_t value) {
            callback(pos, value);
        }
    );
}

// =================================================================================================
//     XOR Hashing
// =================================================================================================

template<class Kernel>
inline std::uint64_t compute_spaced_kmer_hash_simd(
    std::string const& seq, size_t k, Kernel const& kernel
) {
    // Compute all spaced kmers across the sequence, and xor their hashes, for our checking.
    std::uint64_t hash = 0;
    for_each_spaced_kmer_simd(
        std::string_view(seq),
        k,
        kernel,
        char_to_nt_table,
        [&](std::size_t /* start_pos */, std::uint64_t wmer) {
            hash += wmer;
        }
    );
    return hash;
}

template<class Kernel>
inline std::uint64_t compute_spaced_kmer_hash_simd(
    std::string const& seq,
    std::size_t span_k,
    BitExtractKernelDispatcher<Kernel> const& dispatcher
) {
    std::uint64_t hash = 0;

    // Compute all spaced kmers across the sequence, and xor their hashes, for our checking.
    dispatcher.run([&](auto const& kernels_arr)
    {
        // Callback: (mask_idx, start_pos, value)
        for_each_spaced_kmer_simd(
            std::string_view(seq),
            span_k,
            kernels_arr,
            char_to_nt_table,
            [&](std::size_t /*mask_idx*/, std::size_t /*pos*/, std::uint64_t wmer) {
                hash += wmer;
            }
        );
    });

    return hash;
}
