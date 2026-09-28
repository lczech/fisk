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

namespace fisk {

// =================================================================================================
//     Spaced K-mer Algorithms Enum
// =================================================================================================

/**
 * @brief Which emission order to restrict the selector to.
 *
 * The emission orders of for_each_spaced_kmer_simd_by_mask() and
 * for_each_spaced_kmer_simd_by_position() differ in observable behavior, so the selector does not
 * pick between them unless asked to via kEither.
 */
enum class SpacedKmerAxis : int
{
    /** @brief pos is non-decreasing per mask, but not across the whole output. Often fastest. */
    kByMask,

    /** @brief pos is non-decreasing across the whole output, as in for_each_spaced_kmer(). */
    kByPosition,

    /** @brief Benchmark both emission orders, and return the fastest mode of either. */
    kEither,
};

/**
 * @brief Spaced k-mer extraction modes.
 *
 * For simplicity, we currently only include the implementations that were most performant in our
 * benchmarks, i.e., hardware PEXT, the (scalar) Butterfly table, and its by_mask/by_position SIMD
 * accelerations. If needed, this can be trivially expanded to test for additional implementations.
 *
 * kPext and kButterflyTable are scalar, and always emit by position, so they are candidates for
 * every SpacedKmerAxis.
 */
enum class SpacedKmerMode : int
{
    /** @brief Use hardware PEXT (scalar, always position-major). */
    kPext,

    /** @brief Use scalar butterfly table algorithm (scalar, always position-major). */
    kButterflyTable,

    /** @brief SSE2 butterfly table, by_mask emission order. */
    kButterflyTableSSE2ByMask,
    /** @brief AVX2 butterfly table, by_mask emission order. */
    kButterflyTableAVX2ByMask,
    /** @brief AVX512 butterfly table, by_mask emission order. */
    kButterflyTableAVX512ByMask,
    /** @brief Neon butterfly table, by_mask emission order. */
    kButterflyTableNeonByMask,

    /** @brief SSE2 butterfly table, by_position emission order. */
    kButterflyTableSSE2ByPosition,
    /** @brief AVX2 butterfly table, by_position emission order. */
    kButterflyTableAVX2ByPosition,
    /** @brief AVX512 butterfly table, by_position emission order. */
    kButterflyTableAVX512ByPosition,
    /** @brief Neon butterfly table, by_position emission order. */
    kButterflyTableNeonByPosition,
};

inline std::string spaced_kmer_mode_name(SpacedKmerMode mode)
{
    using Mode = SpacedKmerMode;
    switch (mode) {
        // Scalar
        case Mode::kPext:                           return "PEXT";
        case Mode::kButterflyTable:                 return "ButterflyTable";

        // SIMD by mask
        case Mode::kButterflyTableSSE2ByMask:       return "ButterflyTableSSE2ByMask";
        case Mode::kButterflyTableAVX2ByMask:       return "ButterflyTableAVX2ByMask";
        case Mode::kButterflyTableAVX512ByMask:     return "ButterflyTableAVX512ByMask";
        case Mode::kButterflyTableNeonByMask:       return "ButterflyTableNeonByMask";

        // SIMD by position
        case Mode::kButterflyTableSSE2ByPosition:   return "ButterflyTableSSE2ByPosition";
        case Mode::kButterflyTableAVX2ByPosition:   return "ButterflyTableAVX2ByPosition";
        case Mode::kButterflyTableAVX512ByPosition: return "ButterflyTableAVX512ByPosition";
        case Mode::kButterflyTableNeonByPosition:   return "ButterflyTableNeonByPosition";

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
 * @param masks      Two-bit spaced-k-mer masks, each with first and last positions set.
 * @param span_k     Full span of the spaced seed, in `[1, 32]`.
 * @param axis       Which emission order to restrict candidates to; see SpacedKmerAxis.
 * @param seq_len    Length of the random benchmark sequence.
 * @return           The fastest spaced-k-mer extraction mode for this mask set on this build.
 */
inline SpacedKmerMode spaced_kmer_selector(
    std::vector<BitExtractMask> const& masks,
    std::size_t const span_k,
    SpacedKmerAxis const axis = SpacedKmerAxis::kByPosition,
    std::size_t const seq_len = (1 << 16)
) {
    if (span_k == 0 || span_k > 32) {
        throw std::invalid_argument("spaced_kmer_selector(): span_k must be in [1, 32]");
    }
    if (masks.empty()) {
        throw std::invalid_argument("spaced_kmer_selector(): need at least one mask");
    }
    for (auto const& mask : masks) {
        if (!is_valid_spaced_kmer_mask(mask.mask, span_k)) {
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
    // SIMD kernel dispatchers also preprocess internally, once per candidate.
    std::vector<BitExtractButterflyTable> butterfly_tables;
    butterfly_tables.reserve(masks.size());
    for (auto const& mask : masks) {
        butterfly_tables.push_back(bit_extract_butterfly_table_preprocess(mask.mask));
    }

    // BitExtractKernelDispatcher takes raw mask values, not BitExtractMask.
    std::vector<std::uint64_t> raw_masks;
    raw_masks.reserve(masks.size());
    for (auto const& mask : masks) {
        raw_masks.push_back(mask.mask);
    }

    bool const want_by_mask     = (
        (axis == SpacedKmerAxis::kByMask)     || (axis == SpacedKmerAxis::kEither)
    );
    bool const want_by_position = (
        (axis == SpacedKmerAxis::kByPosition) || (axis == SpacedKmerAxis::kEither)
    );

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
        run_fn([&](std::size_t /*pos*/, std::size_t /*mask_idx*/, std::uint64_t value) {
            checksum += value;
        });

        // Timed runs. The callback is stateless on purpose: an accumulator captured by reference
        // escapes into the out-of-line std::visit bodies of BitExtractKernelDispatcher::run(),
        // where it gets loaded, added to, and stored back per value, which slows some candidates
        // down (e.g. GCC by_mask) far more than others and skews the ranking.
        constexpr std::size_t repeats = 3;
        auto best = std::chrono::nanoseconds::max();
        for (std::size_t r = 0; r < repeats; ++r) {
            auto const start = clock::now();
            run_fn([](std::size_t /*pos*/, std::size_t /*mask_idx*/, std::uint64_t value) {
                do_not_optimize(value);
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
    // Scalar candidates: always position-major, valid under either axis.
    // ------------------------------------------------------------

    #if defined(FISK_HAS_BMI2)
    if (bmi2_enabled()) {
        benchmark_candidate_(SpacedKmerMode::kPext, [&](auto&& cb) {
            for_each_spaced_kmer(
                std::string_view(seq), span_k, masks, CharEncoderTable<Encoding::kACGT>{},
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
            std::string_view(seq), span_k, butterfly_tables, CharEncoderTable<Encoding::kACGT>{},
            [](std::uint64_t x, BitExtractButterflyTable const& bf) noexcept {
                return bit_extract_butterfly_table(x, bf);
            },
            cb
        );
    });

    // ------------------------------------------------------------
    // SIMD candidates, per ISA, per requested axis.
    // ------------------------------------------------------------

    // Benchmark the by_mask and/or by_position candidates for one kernel type.
    auto benchmark_simd_candidates_ = [&]<typename Kernel>(
        std::type_identity<Kernel>, SpacedKmerMode mode_by_mask, SpacedKmerMode mode_by_position
    ) {
        BitExtractKernelDispatcher<Kernel> disp(raw_masks);
        if (want_by_mask) {
            benchmark_candidate_(mode_by_mask, [&](auto&& cb) {
                disp.run([&](auto const& kernels) {
                    for_each_spaced_kmer_simd_by_mask(
                        std::string_view(seq), span_k, kernels,
                        CharEncoderTable<Encoding::kACGT>{}, cb
                    );
                });
            });
        }
        if (want_by_position) {
            benchmark_candidate_(mode_by_position, [&](auto&& cb) {
                disp.run([&](auto const& kernels) {
                    for_each_spaced_kmer_simd_by_position(
                        std::string_view(seq), span_k, kernels,
                        CharEncoderTable<Encoding::kACGT>{}, cb
                    );
                });
            });
        }
    };

    #if defined(FISK_HAS_SSE2)
    benchmark_simd_candidates_(
        std::type_identity<BitExtractKernelButterflySSE2>{},
        SpacedKmerMode::kButterflyTableSSE2ByMask, SpacedKmerMode::kButterflyTableSSE2ByPosition
    );
    #endif

    #if defined(FISK_HAS_AVX2)
    benchmark_simd_candidates_(
        std::type_identity<BitExtractKernelButterflyAVX2>{},
        SpacedKmerMode::kButterflyTableAVX2ByMask, SpacedKmerMode::kButterflyTableAVX2ByPosition
    );
    #endif

    #if defined(FISK_HAS_AVX512)
    benchmark_simd_candidates_(
        std::type_identity<BitExtractKernelButterflyAVX512>{},
        SpacedKmerMode::kButterflyTableAVX512ByMask, SpacedKmerMode::kButterflyTableAVX512ByPosition
    );
    #endif

    #if defined(FISK_HAS_NEON)
    benchmark_simd_candidates_(
        std::type_identity<BitExtractKernelButterflyNEON>{},
        SpacedKmerMode::kButterflyTableNeonByMask, SpacedKmerMode::kButterflyTableNeonByPosition
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
inline SpacedKmerMode spaced_kmer_selector(
    BitExtractMask const mask,
    std::size_t const span_k,
    SpacedKmerAxis const axis = SpacedKmerAxis::kByPosition,
    std::size_t const seq_len = (1 << 16)
) {
    return spaced_kmer_selector(std::vector<BitExtractMask>{mask}, span_k, axis, seq_len);
}

/**
 * @brief Overload taking a single spaced-k-mer mask as raw uint64_t.
 */
inline SpacedKmerMode spaced_kmer_selector(
    std::uint64_t const mask,
    std::size_t const span_k,
    SpacedKmerAxis const axis = SpacedKmerAxis::kByPosition,
    std::size_t const seq_len = (1 << 16)
) {
    return spaced_kmer_selector(BitExtractMask(mask), span_k, axis, seq_len);
}

/**
 * @brief Overload taking spaced-k-mer masks as raw uint64_t values.
 */
inline SpacedKmerMode spaced_kmer_selector(
    std::vector<std::uint64_t> const& masks,
    std::size_t const span_k,
    SpacedKmerAxis const axis = SpacedKmerAxis::kByPosition,
    std::size_t const seq_len = (1 << 16)
) {
    std::vector<BitExtractMask> converted;
    converted.reserve(masks.size());
    for (auto const& mask : masks) {
        converted.emplace_back(mask);
    }
    return spaced_kmer_selector(converted, span_k, axis, seq_len);
}

} // namespace fisk
