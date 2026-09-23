#include <iostream>
#include <string>
#include <vector>

#include "kmer_clark/bench.hpp"
#include "microbench.hpp"

void bench_kmer_clark(
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
        bench(
            "clark_original",
            run_var_original
        ),
        bench(
            "clark_improved",
            run_var_improved
        )
    );

    std::string case_label = "n/a";
    write_csv_rows(csv_os, suite_title, case_label, results);
}
