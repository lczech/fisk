#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "fisk/kmer_spaced/simd.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/core/cpu_runtime.hpp"
#include "fisk/core/intrinsics.hpp"
#include "fisk/core/random.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"

namespace fisk {

// =================================================================================================
//     Spaced K-mer Algorithms Enum
// =================================================================================================

/**
 * @brief Spaced k-mer extraction modes.
 *
 * For simplicity, we currently only include the implementations that were most performant in our
 * benchmarks, i.e., hardware PEXT, the (scalar) Butterfly table, and its SIMD accelerations, see
 * for_each_spaced_kmer_ascii_checked_simd(). If needed, this can be trivially expanded to test for
 * additional implementations.
 */
enum class SpacedKmerMode : int
{
    /** @brief Use hardware PEXT (scalar). */
    kPext,

    /** @brief Use scalar butterfly table algorithm. */
    kButterflyTable,

    /** @brief SSE2 butterfly table. */
    kButterflyTableSSE2,
    /** @brief AVX2 butterfly table. */
    kButterflyTableAVX2,
    /** @brief AVX512 butterfly table. */
    kButterflyTableAVX512,
    /** @brief Neon butterfly table. */
    kButterflyTableNeon,
};

inline std::string spaced_kmer_mode_name(SpacedKmerMode mode)
{
    using Mode = SpacedKmerMode;
    switch (mode) {
        // Scalar
        case Mode::kPext:                  return "PEXT";
        case Mode::kButterflyTable:        return "ButterflyTable";

        // SIMD
        case Mode::kButterflyTableSSE2:    return "ButterflyTableSSE2";
        case Mode::kButterflyTableAVX2:    return "ButterflyTableAVX2";
        case Mode::kButterflyTableAVX512:  return "ButterflyTableAVX512";
        case Mode::kButterflyTableNeon:    return "ButterflyTableNeon";

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
 * This benchmarks the complete extraction path, from input sequence to extracted spaced k-mers,
 * using the set of masks that the caller intends to run together, as the performance of the SIMD
 * implementations depends on the number of masks.
 *
 * The selector returns only the mode. The caller is expected to do a single outer `switch`
 * and then instantiate the corresponding hot loop without any further runtime dispatch inside.
 *
 * The SIMD candidates hand out vectors, see for_each_spaced_kmer_ascii_checked_simd(). They are
 * timed with their spaced k-mers taken out one at a time, with for_each_valid_lane(), as the scalar
 * candidates hand them out.
 *
 * @param masks      Spaced k-mer masks, see make_spaced_kmer_masks(), all with the same span.
 *                   Their Layout is used for the benchmark.
 * @param seq_len    Length of the random benchmark sequence.
 * @return           The fastest spaced-k-mer extraction mode for this mask set on this build.
 */
template <Layout L>
inline SpacedKmerMode spaced_kmer_selector(
    std::vector<SpacedKmerMask<L, BitExtractMask>> const& masks,
    std::size_t const seq_len = (1 << 16)
) {
    if (masks.empty()) {
        throw std::invalid_argument("spaced_kmer_selector(): need at least one mask");
    }
    std::size_t const span_k = common_spaced_kmer_span_(masks);
    for (auto const& mask : masks) {
        if (!is_valid_spaced_kmer_mask(mask)) {
            throw std::invalid_argument("spaced_kmer_selector(): invalid spaced k-mer mask");
        }
    }
    if (seq_len < span_k) {
        throw std::invalid_argument("spaced_kmer_selector(): seq_len must be >= span_k");
    }

    using clock = std::chrono::steady_clock;

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

    // Precompute scalar butterfly tables once, one per mask.
    // SIMD kernels are created once per candidate below.
    auto const butterfly_tables = make_spaced_kmer_butterfly_tables(masks);

    struct CandidateResult
    {
        SpacedKmerMode mode;
        std::chrono::nanoseconds time;
        std::uint64_t checksum;
    };
    std::vector<CandidateResult> results;

    // Each candidate's run_fn runs its whole extraction loop once, with the given callback.
    auto benchmark_candidate_ = [&](SpacedKmerMode mode, auto&& run_fn)
    {
        // Warmup, which also sums all extracted values, to verify equality across implementations.
        std::uint64_t checksum = 0;
        run_fn([&](std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& value) {
            checksum += kmer_value(value);
        });

        // Timed runs. The callback is stateless on purpose: an accumulator captured by reference
        // escapes into the out-of-line std::visit bodies of BitExtractKernelDispatcher::run(),
        // where it gets loaded, added to, and stored back per value, which slows some candidates
        // down (e.g. the SIMD ones with GCC) far more than others and skews the ranking.
        constexpr std::size_t repeats = 3;
        auto best = std::chrono::nanoseconds::max();
        for (std::size_t r = 0; r < repeats; ++r) {
            auto const start = clock::now();
            run_fn([](std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& value) {
                do_not_optimize(kmer_value(value));
            });
            auto const stop = clock::now();

            auto const dt = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            if (dt < best) {
                best = dt;
            }
        }

        if (!results.empty() && results.front().checksum != checksum) {
            throw std::runtime_error(
                "spaced_kmer_selector(): candidate result mismatch for mode " +
                spaced_kmer_mode_name(mode)
            );
        }

        results.push_back({mode, best, checksum});
    };

    // ------------------------------------------------------------
    // Scalar candidates
    // ------------------------------------------------------------

    #if defined(FISK_HAS_BMI2)
    if (bmi2_enabled()) {
        benchmark_candidate_(SpacedKmerMode::kPext, [&](auto&& cb) {
            for_each_spaced_kmer(
                std::string_view(seq), masks, CharEncoderTable<Encoding::kACGT>{},
                [](std::uint64_t x, BitExtractMask const& m) noexcept {
                    return bit_extract_pext(x, m);
                },
                cb
            );
        });
    }
    #endif

    benchmark_candidate_(SpacedKmerMode::kButterflyTable, [&](auto&& cb) {
        for_each_spaced_kmer(
            std::string_view(seq), butterfly_tables, CharEncoderTable<Encoding::kACGT>{},
            [](std::uint64_t x, BitExtractButterflyTable const& bf) noexcept {
                return bit_extract_butterfly_table(x, bf);
            },
            cb
        );
    });

    // ------------------------------------------------------------
    // SIMD candidates, per ISA
    // ------------------------------------------------------------

    // Benchmark the candidate of one kernel type, with its spaced k-mers taken out of the vectors.
    auto benchmark_simd_candidate_ = [&]<typename Kernel>(
        std::type_identity<Kernel>, SpacedKmerMode mode
    ) {
        BitExtractKernelDispatcher<SpacedKmerMask<L, Kernel>> disp(
            make_spaced_kmer_kernels<Kernel>(masks)
        );
        benchmark_candidate_(mode, [&](auto&& cb) {
            disp.run([&](auto const& kernels) {
                for_each_spaced_kmer_ascii_checked_simd(
                    std::string_view(seq), kernels, CharEncoderTable<Encoding::kACGT>{},
                    [&](std::size_t pos, std::size_t mask_idx, auto vec, auto valid_lanes) {
                        for_each_valid_lane(
                            pos, vec, valid_lanes,
                            [&](std::size_t p, auto const& value) { cb(p, mask_idx, value); }
                        );
                    }
                );
            });
        });
    };

    #if defined(FISK_HAS_SSE2)
    benchmark_simd_candidate_(
        std::type_identity<BitExtractKernelButterfly<SimdSSE2>>{},
        SpacedKmerMode::kButterflyTableSSE2
    );
    #endif

    #if defined(FISK_HAS_AVX2)
    benchmark_simd_candidate_(
        std::type_identity<BitExtractKernelButterfly<SimdAVX2>>{},
        SpacedKmerMode::kButterflyTableAVX2
    );
    #endif

    #if defined(FISK_HAS_AVX512)
    benchmark_simd_candidate_(
        std::type_identity<BitExtractKernelButterfly<SimdAVX512>>{},
        SpacedKmerMode::kButterflyTableAVX512
    );
    #endif

    #if defined(FISK_HAS_NEON)
    benchmark_simd_candidate_(
        std::type_identity<BitExtractKernelButterfly<SimdNEON>>{},
        SpacedKmerMode::kButterflyTableNeon
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
 * @brief Overload for a single mask: forwards to the multi-mask overload with a 1-element vector.
 */
template <Layout L>
inline SpacedKmerMode spaced_kmer_selector(
    SpacedKmerMask<L, BitExtractMask> const& mask,
    std::size_t const seq_len = (1 << 16)
) {
    return spaced_kmer_selector(std::vector<SpacedKmerMask<L, BitExtractMask>>{mask}, seq_len);
}

} // namespace fisk
