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
#include "pext.hpp"
#include "pext_zp7.hpp"
#include "pext_instlatx64.hpp"
#include "pext_adaptive.hpp"
#include "sys_info.hpp"

struct PextInput
{
    std::uint64_t value;
    std::uint64_t mask;

    // For the preprocessed implementation, we also pre-compute the block tables
    PextBlockTable block_table;

    // We also need to store an instance of the adaptive pext here, which evaluates
    // the fastest algorithm to use for the given mask - which is mask-dependent.
    AdaptivePext adaptive_pext;
};

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

inline std::vector<PextInput> make_inputs(
    std::size_t n, int popcnt, std::uint64_t seed,
    std::vector<size_t>& adaptive_counts
) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint64_t> dist_u64;

    std::vector<PextInput> v;
    v.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t value = dist_u64(rng);
        std::uint64_t mask  = random_mask_with_popcount(rng, popcnt);
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

inline void bench_pext_weights(std::ostream& csv_os)
{
    std::size_t const n = 16;
    std::size_t const rounds = (1u << 7);
    // std::size_t const n = (1u << 20);
    // std::size_t const rounds = 10;
    std::size_t const repeats = 32;

    // User output
    std::cout << "\n=== PEXT ===\n";
    std::cout << "n=" << n << ", rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // Collect which adaptive mode was chosen how often.
    auto adaptive_counts = std::vector<size_t>( 7, 0 );

    // Run a benchmark for each weight of the mask.
    // Most of our software implementations of PEXT have a runtime depending on that,
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

        Microbench<PextInput> suite("PEXT");
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

        write_csv_rows(csv_os, "PEXT", case_label, results);
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
