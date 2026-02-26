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

inline void bench_kmer_clark(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    std::size_t const rounds = 4;
    std::size_t const repeats = 16;
    size_t const k = 31;

    // User output
    std::cout << "\n=== spaced k-mer clark ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // We store the masks here, in a simplified form with just the names.
    // The actual masks are hard-coded in clark.
    std::vector<std::string> masks = {{ "T295", "T38570", "T58570" }};

    // Prepare a benchmark with repititions
    Microbench<std::string> suite("kmer_clark");
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
                return clark_getObjectsDataComputeFull( seq, masks );
            }
        ),
        bench(
            "clark_improved",
            [&](std::string const& seq){
                return clark_improved( seq );
            }
        )
    );

    std::string case_label = "n/a";
    write_csv_rows(csv_os, "kmer_clark", case_label, results);
}
