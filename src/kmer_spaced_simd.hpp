#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "bit_extract.hpp"
#include "bit_extract_simd.hpp"
#include "seq_enc.hpp"

// =================================================================================================
//     SIMD Helper Functions
// =================================================================================================

/**
 * @brief Helper function for for_each_spaced_kmer_simd() to emit all k-mers
 * in the lanes during one step of the iteration.
 */
template <int L, class Callback>
inline void emit_spaced_kmer_lanes(
    std::uint64_t const* out,
    unsigned valid_bits,
    std::size_t start_pos,
    Callback&& callback
) {
    // The template param L dictates how many lanes the SIMD architecture has.
    // We use constexpr statements here to unroll the relevant emissions at compile time.
    if constexpr (L >= 1) { if (valid_bits & 0x01u) { callback(start_pos + 0, out[0]); }}
    if constexpr (L >= 2) { if (valid_bits & 0x02u) { callback(start_pos + 1, out[1]); }}
    if constexpr (L >= 3) { if (valid_bits & 0x04u) { callback(start_pos + 2, out[2]); }}
    if constexpr (L >= 4) { if (valid_bits & 0x08u) { callback(start_pos + 3, out[3]); }}
    if constexpr (L >= 5) { if (valid_bits & 0x10u) { callback(start_pos + 4, out[4]); }}
    if constexpr (L >= 6) { if (valid_bits & 0x20u) { callback(start_pos + 5, out[5]); }}
    if constexpr (L >= 7) { if (valid_bits & 0x40u) { callback(start_pos + 6, out[6]); }}
    if constexpr (L >= 8) { if (valid_bits & 0x80u) { callback(start_pos + 7, out[7]); }}
}

// =================================================================================================
//     Spaced k-mer Iteration Single Mask
// =================================================================================================

// Shared streaming body (skip invalid), per-kmer callback (ordered)
template<class Kernel, class Callback>
inline void for_each_spaced_kmer_simd(
    std::string_view seq,
    size_t const span_k,
    Kernel const& kernel,
    Callback&& callback
) {
    // Assertions and checks
    static_assert(
        std::is_invocable_v<Callback, std::size_t, std::uint64_t>,
        "Callback must be callable as callback(start_pos, value)."
    );
    if( span_k < 1 || span_k > 32 ) {
        throw std::invalid_argument("k must be in [1,32]");
    }

    // Mask to keep only the lowest 2*k bits, works for all k in [1, 32]
    std::uint64_t const span_mask = (span_k == 32)
        ?  ~std::uint64_t{0}                    // all 64 bits
        : ((std::uint64_t{1} << (2 * span_k)) - 1u)  // lower 2*k bits set
    ;

    // Set up input and output buffers to transfer to and from the simd kernel.
    constexpr std::size_t L = Kernel::lanes;
    alignas(64) std::uint64_t in[L];
    alignas(64) std::uint64_t out[L];

    // Sliding window kmer along the sequence, and current number of valid input chars
    std::uint64_t kmer = 0;
    std::size_t valid_run = 0;

    // Iterate the sequence. Each kmer is only constructed once. Per iteration of
    // this outer loop, the inner loop does L many increments along the sequence.
    std::size_t i = 0;
    for (; i + L <= seq.size(); i += L ) {
        unsigned valid_bits = 0;

        // Inner loop to fill all SIMD lanes with consecutive kmers.
        #pragma unroll
        for( std::size_t lane = 0; lane < L; ++lane ) {
            std::uint8_t const code = char_to_nt_table(seq[i + lane]);

            // Check for input char validity, and reset the kmer if not.
            if( code == SEQ_NT4_INVALID ) {
                kmer = 0;
                valid_run = 0;
                in[lane] = 0;
                continue;
            }
            ++valid_run;

            // Update kmer and input lane buffer, and set the valid bit for the lane.
            kmer = ((kmer << 2) | std::uint64_t(code & 0x03u)) & span_mask;
            in[lane] = kmer;
            valid_bits |= (unsigned(valid_run >= span_k) << lane);
        }

        // Shortcut if no lane was valid. No need to run the simd kernel.
        if( !valid_bits) {
            continue;
        }

        // Run bit extraction in parallel across the simd lanes.
        typename Kernel::simd_vector X = kernel.load(in);
        typename Kernel::simd_vector Y = kernel.bit_extract(X);
        kernel.store( Y, out );

        // Emit all lanes in order as spaced kmers to the callback function.
        std::size_t const start_pos = i - (span_k - 1);
        emit_spaced_kmer_lanes<L>( out, valid_bits, start_pos, std::forward<Callback>( callback ));
    }

    // Tail, same as above, but scalar
    for(; i < seq.size(); ++i) {
        // Process the next input char
        std::uint8_t const code = char_to_nt_table(seq[i]);
        if( code == SEQ_NT4_INVALID ) {
            kmer = 0;
            valid_run = 0;
            continue;
        }
        ++valid_run;

        // Update the k-mer and emit the callback
        kmer = ((kmer << 2) | std::uint64_t(code & 0x03u)) & span_mask;
        if (valid_run < span_k) {
            continue;
        }
        std::uint64_t const y = kernel.bit_extract( kmer );
        callback(i - (span_k - 1), y);
    }
}

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
        [&](std::size_t /* start_pos */, std::uint64_t wmer) {
            hash ^= wmer;
        }
    );
    return hash;
}

// =================================================================================================
//     Spaced k-mer Iteration Multi Masks
// =================================================================================================

template <class Kernel, std::size_t NMasks, class Callback>
inline void for_each_spaced_kmer_simd(
    std::string_view seq,
    size_t const span_k,
    std::array<Kernel, NMasks> const& kernels,
    Callback&& callback
) {
    static_assert(NMasks > 0, "Need at least one mask/kernel.");
    static_assert(
        std::is_invocable_v<Callback, std::size_t, std::size_t, std::uint64_t>,
        "Callback must be callable as callback(mask_idx, start_pos, value)."
    );
    if( span_k < 1 || span_k > 32 ) {
        throw std::invalid_argument("k must be in [1,32]");
    }

    // Mask to keep only the lowest 2*k bits, works for all k in [1, 32]
    std::uint64_t const span_mask = (span_k == 32)
        ?  ~std::uint64_t{0}                    // all 64 bits
        : ((std::uint64_t{1} << (2 * span_k)) - 1u)  // lower 2*k bits set
    ;

    // Set up input and output buffers to transfer to and from the simd kernel.
    constexpr int L = Kernel::lanes;
    alignas(64) std::uint64_t in[L];
    alignas(64) std::uint64_t out[L];

    // Sliding window kmer along the sequence, and current number of valid input chars
    std::uint64_t kmer = 0;
    std::size_t valid_run = 0;

    // Iterate the sequence. Each kmer is only constructed once. Per iteration of
    // this outer loop, the inner loop does L many increments along the sequence,
    // and calls all mask kernels to produce the spaced k-mers.
    std::size_t i = 0;
    for (; i + (std::size_t)L <= seq.size(); i += (std::size_t)L) {
        unsigned valid_bits = 0;

        // Inner loop to fill all SIMD lanes with consecutive kmers.
        #pragma unroll
        for( std::size_t lane = 0; lane < L; ++lane ) {
            std::uint8_t const code = char_to_nt_table(seq[i + lane]);

            // Check for input char validity, and reset the kmer if not.
            if( code == SEQ_NT4_INVALID ) {
                kmer = 0;
                valid_run = 0;
                in[lane] = 0;
                continue;
            }
            ++valid_run;

            // Update kmer and input lane buffer, and set the valid bit for the lane.
            kmer = ((kmer << 2) | std::uint64_t(code & 0x03u)) & span_mask;
            in[lane] = kmer;
            valid_bits |= (unsigned(valid_run >= span_k) << lane);
        }

        // Shortcut if no lane was valid. No need to run the simd kernel.
        if( !valid_bits) {
            continue;
        }

        // Load windows vector once
        typename Kernel::simd_vector X = kernels[0].load(in);
        std::size_t const start_pos = i - (span_k - 1);

        // Loop over masks (kernels)
        #pragma unroll
        for (std::size_t mid = 0; mid < NMasks; ++mid) {
            typename Kernel::simd_vector Y = kernels[mid].bit_extract(X);
            kernels[mid].store( Y, out );

            emit_spaced_kmer_lanes<L>( out, valid_bits, start_pos,
                [&]( std::size_t pos, std::uint64_t val ) {
                    callback( mid, pos, val );
                }
            );
        }
    }

    // Tail, same as above, but scalar
    for (; i < seq.size(); ++i) {
        // Process the next input char
        std::uint8_t const code = char_to_nt_table(seq[i]);
        if( code == SEQ_NT4_INVALID ) {
            kmer = 0;
            valid_run = 0;
            continue;
        }
        ++valid_run;

        // Update the k-mer
        kmer = ((kmer << 2) | std::uint64_t(code & 0x03u)) & span_mask;
        if (valid_run < span_k) {
            continue;
        }

        // Emit the callback for every mask
        std::size_t const pos = i - (span_k - 1);
        #pragma unroll
        for (std::size_t mid = 0; mid < NMasks; ++mid) {
            std::uint64_t const y = kernels[mid].bit_extract( kmer );
            callback(mid, pos, y);
        }
    }
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
            [&](std::size_t /*mask_idx*/, std::size_t /*pos*/, std::uint64_t val) {
                hash ^= val;
            }
        );
    });

    return hash;
}
