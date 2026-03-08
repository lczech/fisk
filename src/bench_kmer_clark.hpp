#pragma once

#include <cstdint>
#include <iostream>
#include <functional>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

#include "utils.hpp"
#include "kmer_clark.hpp"
#include "kmer_spaced.hpp"
#include "seq_enc.hpp"
#include "microbench.hpp"
#include "sys_info.hpp"

/**
 * @brief Benchmark the original CLARK implementation vs our improved one.
 */
inline void bench_kmer_clark(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    std::size_t const rounds = 4;
    std::size_t const repeats = 16;
    size_t const k = 31;

    // User output
    std::string const suite_title = "kmer_clark";
    std::cout << "\n=== spaced k-mer clark ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // We store the masks here, in a simplified form with just the names.
    // The actual masks are hard-coded in clark.
    std::vector<std::string> mask_names = {{ "T295", "T38570", "T58570" }};
    std::vector<std::string> mask_strings = {{
        "1111011101110010111001011011111",
        "1111101011100101101110011011111",
        "1111101001110101101100111011111"
    }};
    std::vector<BitExtractMask> masks;
    for (auto const& mask : mask_strings) {
        masks.push_back( BitExtractMask( prepare_spaced_kmer_bit_extract_mask( mask )));
    }

    // Prepare a benchmark with repititions
    Microbench<std::string> suite(suite_title);
    suite
        .rounds(rounds)
        .repeats(repeats)
        .units_fn([](std::string const& seq) {
            // 1 unit per original (unspaced) k-mer
            return static_cast<double>(seq.size() - k + 1);
        });

    // Run the benchmark for all algorithms
    auto results = suite.run(
        sequences, // vector<std::string>

        // clark
        bench(
            "clark_original",
            [&](std::string const& seq){
                return clark_getObjectsDataComputeFull( seq, mask_names );
            }
        ),
        bench(
            "clark_improved",
            [&](std::string const& seq){
                return clark_improved( seq, masks );
            }
        )
    );

    std::string case_label = "n/a";
    write_csv_rows(csv_os, suite_title, case_label, results);
}
