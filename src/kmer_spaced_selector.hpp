#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bit_extract.hpp"
#include "bit_extract_simd.hpp"
#include "kmer_spaced.hpp"
#include "kmer_spaced_simd.hpp"
#include "seq_enc.hpp"
#include "sys_info.hpp"
#include "utils.hpp"

// =================================================================================================
//     Spaced K-mer Algorithms Enum
// =================================================================================================

/**
 * @brief Spaced k-mer extraction modes.
 *
 * For simplicity, we currently only include the implementations that were most performant in our
 * benchmarks, i.e., hardware PEXT as well as the Butterfly table and its SIMD accelerations.
 * If needed, this can be trivially expanded to test for additional implementations.
 */
enum class SpacedKmerMode : int
{
    /** @brief Use hardware PEXT */
    kPext,

    /** @brief Use scalar butterfly table algorithm */
    kButterflyTable,

    /** @brief Use the SSE2 butterfly table algorithm */
    kButterflyTableSSE2,

    /** @brief Use the AVX2 butterfly table algorithm */
    kButterflyTableAVX2,

    /** @brief Use the AVX512 butterfly table algorithm */
    kButterflyTableAVX512,

    /** @brief Use the Neon butterfly table algorithm */
    kButterflyTableNeon,
};

inline std::string spaced_kmer_mode_name(SpacedKmerMode mode)
{
    switch (mode) {
        case SpacedKmerMode::kPext:                 return "PEXT";
        case SpacedKmerMode::kButterflyTable:       return "ButterflyTable";
        case SpacedKmerMode::kButterflyTableSSE2:   return "ButterflyTableSSE2";
        case SpacedKmerMode::kButterflyTableAVX2:   return "ButterflyTableAVX2";
        case SpacedKmerMode::kButterflyTableAVX512: return "ButterflyTableAVX512";
        case SpacedKmerMode::kButterflyTableNeon:   return "ButterflyTableNeon";
        default: {
            throw std::invalid_argument(
                "Invalid SpacedKmerMode in spaced_kmer_mode_name(): " +
                std::to_string(static_cast<int>(mode))
            );
        }
    }
}

// =================================================================================================
//     Spaced K-mer Algorithm Selector
// =================================================================================================

/**
 * @brief Benchmark the full spaced-k-mer extraction loop and return the fastest mode.
 *
 * This benchmarks the complete extraction path, from input sequence to extracted spaced k-mers.
 * A random ACGT-only input sequence is generated once and then reused for all candidates.
 *
 * The selector returns only the mode. The caller is expected to do a single outer `switch`
 * and then instantiate the corresponding hot loop without any further runtime dispatch inside.
 *
 * @param mask       Two-bit spaced-k-mer mask, with first and last positions set.
 * @param span_k     Full span of the spaced seed, in `[1, 32]`.
 * @param seq_len    Length of the random benchmark sequence.
 * @return           The fastest spaced-k-mer extraction mode for this mask/span on this build.
 */
inline SpacedKmerMode spaced_kmer_selector(
    BitExtractMask const mask,
    std::size_t const span_k,
    std::size_t const seq_len = (1 << 16)
) {
    using clock = std::chrono::steady_clock;

    if (span_k == 0 || span_k > 32) {
        throw std::invalid_argument("spaced_kmer_selector(): span_k must be in [1, 32]");
    }
    if (!is_valid_spaced_kmer_mask(mask.mask, span_k)) {
        throw std::invalid_argument("spaced_kmer_selector(): invalid spaced k-mer mask");
    }
    if (seq_len < span_k) {
        throw std::invalid_argument("spaced_kmer_selector(): seq_len must be >= span_k");
    }

    // ------------------------------------------------------------
    //     Preparations
    // ------------------------------------------------------------

    // Prepare random input data once, outside the benchmark.
    // ACGT-only input keeps the benchmark focused on the extraction itself.
    std::string seq;
    seq.resize(seq_len);
    {
        Splitmix64 rng{};
        static constexpr char nts[4] = {'A', 'C', 'G', 'T'};
        for (std::size_t i = 0; i < seq_len; ++i) {
            seq[i] = nts[rng.get_uint64() & 0x3u];
        }
    }

    // Precompute scalar butterfly table once.
    // SIMD kernel constructors also preprocess internally, once per candidate.
    auto const butterfly_table = bit_extract_butterfly_table_preprocess(mask.mask);

    // ------------------------------------------------------------
    //     Benchmark helper
    // ------------------------------------------------------------

    struct CandidateResult
    {
        SpacedKmerMode mode;
        std::chrono::nanoseconds time;
        std::uint64_t result;
    };

    std::vector<CandidateResult> results;
    auto benchmark_candidate_ = [&](SpacedKmerMode mode, auto&& func)
    {
        // Warmup
        volatile std::uint64_t sink = 0;
        sink = func();

        // Timed runs; keep the result so we can verify equality across implementations.
        constexpr std::size_t repeats = 3;
        auto best = std::chrono::nanoseconds::max();
        for (std::size_t r = 0; r < repeats; ++r) {
            auto const start = clock::now();
            std::uint64_t const sum = func();
            auto const stop = clock::now();
            sink = sum;

            auto const dt = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            if (dt < best) {
                best = dt;
            }
        }

        if (!results.empty() && results.front().result != sink) {
            throw std::runtime_error(
                "spaced_kmer_selector(): candidate result mismatch for mode " +
                spaced_kmer_mode_name(mode)
            );
        }

        results.push_back({mode, best, sink});
    };

    // ------------------------------------------------------------
    // Benchmark candidates
    // ------------------------------------------------------------

    #if defined(FISK_HAS_BMI2)
    if (bmi2_enabled()) {
        benchmark_candidate_(
            SpacedKmerMode::kPext,
            [&]() -> std::uint64_t {
                return compute_spaced_kmer_hash(
                    seq,
                    span_k,
                    mask,
                    char_to_nt_table,
                    [&](std::uint64_t x, BitExtractMask const& m) noexcept {
                        return bit_extract_pext(x, m);
                    }
                );
            }
        );
    }
    #endif

    benchmark_candidate_(
        SpacedKmerMode::kButterflyTable,
        [&]() -> std::uint64_t {
            return compute_spaced_kmer_hash(
                seq,
                span_k,
                mask,
                char_to_nt_table,
                [&](std::uint64_t x, BitExtractMask const&) noexcept {
                    return bit_extract_butterfly_table(x, butterfly_table);
                }
            );
        }
    );

    #if defined(FISK_HAS_SSE2)
    benchmark_candidate_(
        SpacedKmerMode::kButterflyTableSSE2,
        [&]() -> std::uint64_t {
            return compute_spaced_kmer_hash_simd(
                seq,
                span_k,
                BitExtractKernelButterflySSE2(mask.mask)
            );
        }
    );
    #endif

    #if defined(FISK_HAS_AVX2)
    benchmark_candidate_(
        SpacedKmerMode::kButterflyTableAVX2,
        [&]() -> std::uint64_t {
            return compute_spaced_kmer_hash_simd(
                seq,
                span_k,
                BitExtractKernelButterflyAVX2(mask.mask)
            );
        }
    );
    #endif

    #if defined(FISK_HAS_AVX512)
    benchmark_candidate_(
        SpacedKmerMode::kButterflyTableAVX512,
        [&]() -> std::uint64_t {
            return compute_spaced_kmer_hash_simd(
                seq,
                span_k,
                BitExtractKernelButterflyAVX512(mask.mask)
            );
        }
    );
    #endif

    #if defined(FISK_HAS_NEON)
    benchmark_candidate_(
        SpacedKmerMode::kButterflyTableNeon,
        [&]() -> std::uint64_t {
            return compute_spaced_kmer_hash_simd(
                seq,
                span_k,
                BitExtractKernelButterflyNEON(mask.mask)
            );
        }
    );
    #endif

    // ------------------------------------------------------------
    // Return the fastest candidate
    // ------------------------------------------------------------

    if (results.empty()) {
        throw std::runtime_error("spaced_kmer_selector(): no candidate implementations available");
    }
    auto const best = std::min_element(
        results.begin(),
        results.end(),
        [](auto const& a, auto const& b) {
            return a.time < b.time;
        }
    );

    return best->mode;
}

/**
 * @brief Overload taking the spaced-k-mer mask as raw uint64_t.
 */
inline SpacedKmerMode spaced_kmer_selector(
    std::uint64_t const mask,
    std::size_t const span_k,
    std::size_t const seq_len = (1 << 16)
) {
    return spaced_kmer_selector(BitExtractMask(mask), span_k, seq_len);
}
