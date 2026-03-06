#pragma once

#include <bit> // std::popcount (C++20)
#include <cstdint>
#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "utils.hpp"
#include "microbench.hpp"
#include "bit_extract.hpp"
#include "bit_extract_zp7.hpp"
#include "bit_extract_instlatx64.hpp"
#include "bit_extract_adaptive.hpp"
#include "sys_info.hpp"

/**
 * @brief Helper to store a value and a mask (plus its software helpers) for testing.
 *
 * This is simply an entry in our test data, which is then used to benchmark how fast
 * the implementions are in extracting the bits from the value using the mask.
 */
struct BitExtractInput
{
    std::uint64_t value;
    std::uint64_t mask;

    // For the preprocessed implementations, we also pre-compute their tables
    BitExtractBlockTable block_table;
    BitExtractNetworkTable network_table;

    // We also need to store an instance of the adaptive bit extract here, which evaluates
    // the fastest algorithm to use for the given mask - which is mask-dependent.
    AdaptiveBitExtract adaptive_bit_extract;
};

/**
 * @brief Generate a random bit extract mask with a given amount of set bits.
 */
inline std::uint64_t random_mask_with_popcount(std::mt19937_64& rng, int popcnt)
{
    if (popcnt <= 0) return 0ull;
    if (popcnt >= 64) return ~0ull;

    std::uint64_t mask = 0;
    std::uniform_int_distribution<int> dist(0, 63);
    while (static_cast<int>(std::popcount(mask)) < popcnt) {
        mask |= (1ull << dist(rng));
    }
    return mask;
}

/**
 * @brief Generate input for the benchmark here, with @p n entries of @p popcnt weight.
 */
static inline std::vector<BitExtractInput> make_inputs(
    std::size_t n, int popcnt, std::uint64_t seed,
    std::vector<size_t>& adaptive_counts
) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint64_t> dist_u64;

    std::vector<BitExtractInput> v;
    v.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t value = dist_u64(rng);
        std::uint64_t mask  = random_mask_with_popcount(rng, popcnt);
        v.push_back( BitExtractInput{
            value,
            mask,
            bit_extract_block_table_preprocess( mask ),
            bit_extract_network_table_preprocess( mask ),
            AdaptiveBitExtract( mask )
        });
        ++adaptive_counts[static_cast<size_t>( v.back().adaptive_bit_extract.mode())];
        // std::cout << v.back().adaptive_bit_extract.mode_name() << "\n";
    }
    return v;
}

/**
 * @brief Benchmark different bit extract implementations using randomly generated values and masks,
 * for a range of mask weights (number of bits set) from 0 to 64.
 */
inline void bench_bit_extract_weights(std::ostream& csv_os)
{
    std::size_t const n = 16;
    std::size_t const rounds = (1u << 7);
    // std::size_t const n = (1u << 20);
    // std::size_t const rounds = 10;
    std::size_t const repeats = 32;

    // User output
    std::cout << "\n=== bit extract weights ===\n";
    std::cout << "n=" << n << ", rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // Collect which adaptive mode was chosen how often.
    // This is not really important, but we are curious to see this.
    auto adaptive_counts = std::vector<size_t>( AdaptiveBitExtract::mode_count(), 0 );

    // Run a benchmark for each weight of the mask.
    // Most of our bit extract software implementations have a runtime depending on that,
    // so we want to test the effects of different masks on the implementations.
    for (int w = 0; w <= 64; ++w) {
        std::string case_label = "popcount=" + std::to_string(w);
        if( stdout_is_terminal() ) {
            std::cout << "\rmask popcount "
                << std::setw(2) << w << " / 64"
                << std::flush;
            // std::cout << case_label << "\n";
        }

        // Helper to generate fresh input for each repetition
        auto make_inputs_rep = [w, &adaptive_counts]()
        {
            auto seed = static_cast<std::uint64_t>(0xC0FFEEULL) ^ static_cast<std::uint64_t>(w);
            return make_inputs( n, w, seed, adaptive_counts );
        };

        Microbench<BitExtractInput> suite("bit_extract_weights");
        suite.rounds(rounds).repeats(repeats);

        auto results = suite.run(
            make_inputs_rep,
            #ifdef HAVE_BMI2
            bench(
                "pext_hw_bmi2",
                [](BitExtractInput const& in){ return bit_extract_pext(in.value, in.mask);
            }),
            #endif
            bench(
                "bit_extract_bitloop",
                [](BitExtractInput const& in){ return bit_extract_bitloop(in.value, in.mask);
            }),
            bench(
                "bit_extract_split32",
                [](BitExtractInput const& in){ return bit_extract_split32(in.value, in.mask);
            }),
            bench(
                "bit_extract_byte_table",
                [](BitExtractInput const& in){ return bit_extract_byte_table(in.value, in.mask);
            }),
            bench(
                "bit_extract_block_table",
                [](BitExtractInput const& in){ return bit_extract_block_table(in.value, in.block_table);
            }),
            bench(
                "bit_extract_block_table_unrolled1",
                [](BitExtractInput const& in){ return bit_extract_block_table_unrolled<1>(in.value, in.block_table);
            }),
            bench(
                "bit_extract_block_table_unrolled2",
                [](BitExtractInput const& in){ return bit_extract_block_table_unrolled<2>(in.value, in.block_table);
            }),
            bench(
                "bit_extract_block_table_unrolled4",
                [](BitExtractInput const& in){ return bit_extract_block_table_unrolled<4>(in.value, in.block_table);
            }),
            bench(
                "bit_extract_block_table_unrolled8",
                [](BitExtractInput const& in){ return bit_extract_block_table_unrolled<8>(in.value, in.block_table);
            }),
            bench(
                "bit_extract_network_table",
                [](BitExtractInput const& in){ return bit_extract_network_table(in.value, in.network_table);
            }),
            bench(
                "bit_extract_adaptive",
                [](BitExtractInput const& in){ return in.adaptive_bit_extract(in.value);
            }),
            #ifdef PLATFORM_X86_64
            bench(
                "bit_extract_instlatx",
                [](BitExtractInput const& in){ return pext64_emu(in.value, in.mask);
            }),
            #endif
            bench(
                "bit_extract_zp7",
                [](BitExtractInput const& in){ return zp7_pext_64(in.value, in.mask);
            })
        );

        write_csv_rows(csv_os, "bit_extract_weights", case_label, results);
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }

    // Print adative bit extract counts
    std::cout << "adaptive bit extract counts:\n";
    for( size_t i = 0; i < adaptive_counts.size(); ++i ) {
        std::cout << "  " << adaptive_counts[i] << " <== " << AdaptiveBitExtract::mode_name(static_cast<AdaptiveBitExtract::ExtractMode>(i)) << "\n";
    }
    std::cout << "\n";
}
