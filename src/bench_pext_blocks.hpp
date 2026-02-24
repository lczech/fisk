#pragma once

#include <bit> // std::popcount (C++20)
#include <cstdint>
#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <limits>
#include <random>

#include "utils.hpp"
#include "microbench.hpp"
#include "pext.hpp"
#include "pext_zp7.hpp"
#include "pext_instlatx64.hpp"
#include "pext_adaptive.hpp"
#include "bench_pext_weights.hpp"
#include "sys_info.hpp"

// Already defined in `bench_pext_weights.hpp`
// struct PextInput
// {
//     std::uint64_t value;
//     std::uint64_t mask;

//     // For the preprocessed implementation, we also pre-compute the block tables
//     PextBlockTable block_table;
// };

/**
 * @brief Generate a random bitmask of width W (<= 64) with exactly `runs` runs of consecutive 1s.
 *
 * Runs are maximal contiguous segments of 1 bits. For example:
 *   0011100111000  has 2 runs of 1s.
 *
 * Distribution notes:
 * - This produces a reasonably random mask subject to the constraint, but is not
 *   a perfectly uniform distribution over all masks with `runs` runs.
 * - It randomizes: (a) total number of 1s, (b) run lengths, (c) zero gaps.
 *
 * Boundary handling:
 * - runs == 0 -> returns 0
 * - runs must satisfy 0 <= runs <= ceil(W/2). For W=64, max is 32.
 * - Works for any W in [1, 64]; default W=64.
 */
template <std::size_t W = 64, class URBG>
std::uint64_t random_mask_with_runs(std::size_t runs, URBG& rng)
{
    static_assert(W >= 1 && W <= 64, "W must be in [1, 64]");

    const std::size_t max_runs = (W + 1) / 2; // ceil(W/2)
    if (runs > max_runs) {
        throw std::invalid_argument("runs too large for given width");
    }
    if (runs == 0) {
        return 0ULL;
    }

    // Internal gaps between runs must be >= 1 when runs >= 2
    const std::size_t min_internal_zeros = (runs >= 2) ? (runs - 1) : 0;

    // Choose total ones K: must be >= runs (each run at least 1),
    // and must leave room for the required internal zeros.
    const std::size_t min_ones = runs;
    const std::size_t max_ones = W - min_internal_zeros;
    if (min_ones > max_ones) {
        // Should not happen if runs <= ceil(W/2), but keep it airtight.
        throw std::logic_error("No feasible mask for given runs/width");
    }

    std::uniform_int_distribution<std::size_t> dist_ones(min_ones, max_ones);
    const std::size_t K = dist_ones(rng);

    // Helper: random composition of N into k positive integers.
    auto random_positive_composition = [&](std::size_t N, std::size_t k) {
        // If k == 1, only [N]
        std::vector<std::size_t> parts(k, 1);
        N -= k; // remaining to distribute as nonnegative increments
        if (k == 0) return parts;

        // Distribute N as nonnegative into k buckets via stars-and-bars with random cut points.
        // Choose (k-1) cut points in [0..N+k-1]? We can do simpler:
        // Generate k-1 integers in [0..N], sort them, take differences.
        std::vector<std::size_t> cuts;
        cuts.reserve(k > 0 ? k - 1 : 0);
        std::uniform_int_distribution<std::size_t> d(0, N);
        for (std::size_t i = 0; i + 1 < k; ++i) cuts.push_back(d(rng));
        std::sort(cuts.begin(), cuts.end());

        std::size_t prev = 0;
        for (std::size_t i = 0; i + 1 < k; ++i) {
            const std::size_t val = cuts[i] - prev;
            parts[i] += val;
            prev = cuts[i];
        }
        parts[k - 1] += (N - prev);
        return parts;
    };

    // 1-run lengths: composition of K into `runs` positive parts
    std::vector<std::size_t> one_runs = random_positive_composition(K, runs);

    // Now handle zeros. Total zeros:
    const std::size_t Z = W - K;

    // We have (runs+1) zero gaps: [prefix, between..., suffix]
    // Internal gaps (runs-1 of them) must be >= 1 if runs >= 2.
    const std::size_t gaps = runs + 1;
    std::vector<std::size_t> zero_gaps(gaps, 0);

    std::size_t remaining_zeros = Z;

    if (runs >= 2) {
        // Assign mandatory 1 zero to each internal gap
        for (std::size_t i = 1; i <= runs - 1; ++i) {
            zero_gaps[i] = 1;
        }
        remaining_zeros -= (runs - 1);
    }

    // Distribute remaining_zeros freely across all gaps (including internal ones)
    // as a nonnegative composition into `gaps` parts.
    {
        // Nonnegative composition by random cut points in [0..remaining_zeros]
        std::vector<std::size_t> cuts;
        cuts.reserve(gaps > 1 ? gaps - 1 : 0);
        std::uniform_int_distribution<std::size_t> d(0, remaining_zeros);
        for (std::size_t i = 0; i + 1 < gaps; ++i) cuts.push_back(d(rng));
        std::sort(cuts.begin(), cuts.end());

        std::size_t prev = 0;
        for (std::size_t i = 0; i + 1 < gaps; ++i) {
            zero_gaps[i] += cuts[i] - prev;
            prev = cuts[i];
        }
        zero_gaps[gaps - 1] += remaining_zeros - prev;
    }

    // Pack into a uint64_t, MSB..LSB or LSB..MSB. Choose one and be consistent.
    // Here: bit 0 is the least-significant bit, we fill from LSB upward.
    std::uint64_t mask = 0;
    std::size_t pos = 0;

    auto put_zeros = [&](std::size_t n) {
        pos += n;
        if (pos > W) throw std::logic_error("packing overflow (zeros)");
    };
    auto put_ones = [&](std::size_t n) {
        if (n == 0) return;
        if (pos + n > W) throw std::logic_error("packing overflow (ones)");
        // Set n bits starting at pos
        // Special case n == 64 to avoid shift UB (but W<=64 and pos+n<=W).
        if (n == 64) {
            mask = std::numeric_limits<std::uint64_t>::max();
        } else {
            const std::uint64_t ones = (n == 64) ? ~0ULL : ((1ULL << n) - 1ULL);
            mask |= (ones << pos);
        }
        pos += n;
    };

    // prefix zeros
    put_zeros(zero_gaps[0]);

    // alternating 1-run and zero gap
    for (std::size_t i = 0; i < runs; ++i) {
        put_ones(one_runs[i]);
        put_zeros(zero_gaps[i + 1]);
    }

    // pos should be exactly W
    if (pos != W) {
        throw std::logic_error("packing did not fill width exactly");
    }

    // If W < 64, ensure upper bits are 0
    if constexpr (W < 64) {
        const std::uint64_t high_mask = (1ULL << W) - 1ULL;
        mask &= high_mask;
    }

    return mask;
}

inline std::size_t count_runs(std::uint64_t x)
{
    // Bits that start a run of 1s (as seen from the direction of the LSB):
    // - bit is 1
    // - previous bit is 0
    const std::uint64_t run_starts = x & ~(x << 1);
    return static_cast<std::size_t>( std::popcount( run_starts ));
}

inline void print_bits(std::uint64_t x, std::ostream& os)
{
    for (int i = 63; i >= 0; --i) {
        os << ((x >> i) & 1ULL);
        if (i % 8 == 0 && i != 0) {
            os << ' ';
        }
    }
}

inline std::vector<PextInput> make_input_blocks(
    std::size_t n, std::size_t runs, std::uint64_t seed,
    std::vector<size_t>& adaptive_counts
) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint64_t> dist_u64;
    std::vector<PextInput> v;
    v.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t value = dist_u64(rng);
        std::uint64_t mask  = random_mask_with_runs<64>(runs, rng);

        // sanity check that we got what we expected
        if( runs != count_runs(mask) ) {
            throw std::runtime_error( "issue in random_mask_with_runs()" );
        }

        // debug printing
        // print_bits(mask, std::cout);
        // std::cout << " runs=" << runs << "\n";

        // add to the list of inputs
        v.push_back( PextInput{
            value,
            mask,
            pext_sw_block_table_preprocess_u64( mask ),
            AdaptivePext( mask )
        });
        ++adaptive_counts[static_cast<size_t>( v.back().adaptive_pext.mode())];
        // std::cout << v.back().adaptive_pext.mode_name() << "\n";
    }
    return v;
}

inline void bench_pext_blocks(std::ostream& csv_os)
{
    std::size_t const n = 100;
    std::size_t const rounds = (1u << 8);
    // std::size_t const n = (1u << 20);
    // std::size_t const rounds = 10;
    // std::size_t const repeats = 64;
    std::size_t const repeats = 16;

    // User output
    std::cout << "\n=== PEXT Blocks ===\n";
    std::cout << "n=" << n << ", rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // Collect which adaptive mode was chosen how often.
    auto adaptive_counts = std::vector<size_t>( 7, 0 );

    // Run a benchmark for each weight of the mask.
    // Most of our software implementations of PEXT have a runtime depending on that,
    // so we want to test the effects of different masks on the implementations.
    for( size_t runs = 0; runs <= 32; ++runs ) {
        std::string case_label = "popcount=" + std::to_string(runs);
        if( stdout_is_terminal() ) {
            std::cout << "\rmask popcount "
                << std::setw(2) << runs << " / 32"
                << std::flush;
            // std::cout << case_label << "\n";
        }

        // Helper to generate fresh input for each repetition
        auto make_inputs_rep = [runs, &adaptive_counts]()
        {
            auto seed = static_cast<std::uint64_t>(0xC0FFEEULL) ^ static_cast<std::uint64_t>(runs);
            return make_input_blocks( n, runs, seed, adaptive_counts );
        };

        Microbench<PextInput> suite("PEXT_blocks");
        suite.rounds(rounds).repeats(repeats);

        auto results = suite.run(
            make_inputs_rep,
            #ifdef HAVE_BMI2
            bench(
                "pext_hw_bmi2",
                [](PextInput const& in){ return pext_hw_bmi2_u64(in.value, in.mask);
            }),
            #endif
            bench(
                "pext_sw_bitloop",
                [](PextInput const& in){ return pext_sw_bitloop_u64(in.value, in.mask);
            }),
            bench(
                "pext_sw_split32",
                [](PextInput const& in){ return pext_sw_split32_u64(in.value, in.mask);
            }),
            bench(
                "pext_sw_table8",
                [](PextInput const& in){ return pext_sw_table8_u64(in.value, in.mask);
            }),
            bench(
                "pext_sw_block_table",
                [](PextInput const& in){ return pext_sw_block_table_u64(in.value, in.block_table);
            }),
            bench(
                "pext_sw_block_table_unrolled2",
                [](PextInput const& in){ return pext_sw_block_table_u64_unrolled2(in.value, in.block_table);
            }),
            bench(
                "pext_sw_block_table_unrolled4",
                [](PextInput const& in){ return pext_sw_block_table_u64_unrolled4(in.value, in.block_table);
            }),
            bench(
                "pext_sw_block_table_unrolled8",
                [](PextInput const& in){ return pext_sw_block_table_u64_unrolled8(in.value, in.block_table);
            }),
            bench(
                "pext_sw_adaptive",
                [](PextInput const& in){ return in.adaptive_pext(in.value);
            }),
            #ifdef PLATFORM_X86_64
            bench(
                "pext_sw_instlatx",
                [](PextInput const& in){ return pext64_emu(in.value, in.mask);
            }),
            #endif
            bench(
                "pext_sw_zp7",
                [](PextInput const& in){ return zp7_pext_64(in.value, in.mask);
            })
        );

        write_csv_rows(csv_os, "PEXT_blocks", case_label, results);
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }

    // Print adative pext counts
    std::cout << "Adaptive Pext counts:\n";
    for( size_t i = 0; i < adaptive_counts.size(); ++i ) {
        std::cout << "  " << adaptive_counts[i] << " <== " << AdaptivePext::mode_name(static_cast<AdaptivePext::ExtractMode>(i)) << "\n";
    }
    std::cout << "\n";
}
