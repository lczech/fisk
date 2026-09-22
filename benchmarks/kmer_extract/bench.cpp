#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kmer_extract/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

void bench_kmer_extract(
    std::vector<std::string> const& sequences,
    std::size_t k_min,
    std::size_t k_max,
    std::ostream& csv_os
) {
    // Boundary checks
    if( k_min < 1 || k_min > 32 || k_max < 1 || k_max > 32 ) {
        throw std::runtime_error( "Invalid k outside of [1, 32]" );
    }
    if( k_min > k_max ) {
        throw std::runtime_error( "Invalid k_min > k_max" );
    }

    std::size_t const rounds = 2;
    std::size_t const repeats = 8;

    // User output
    std::string const suite_title = "kmer_extract";
    std::cout << "\n=== k-mer extract ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    // Run a benchmark for each valid k.
    for( std::size_t k = k_min; k <= k_max; ++k) {
        if( stdout_is_terminal() ) {
            std::cout << "\rk " << std::setw(2) << k << std::flush;
        }

        Microbench<std::string> suite(suite_title);
        suite
            .rounds(rounds)
            .repeats(repeats)
            .units_fn([k](std::string const& seq) {
                // 1 unit per k-mer
                return static_cast<double>(seq.size() - k + 1);
            });

        auto results = suite.run(
            sequences, // vector<std::string>

            // Full re-extract
            bench(
                "ifs_re",
                [k](std::string const& seq){
                    return run_var_ifs_re(seq, k);
                }
            ),
            bench(
                "switch_re",
                [k](std::string const& seq){
                    return run_var_switch_re(seq, k);
                }
            ),
            bench(
                "table_re",
                [k](std::string const& seq){
                    return run_var_table_re(seq, k);
                }
            ),
            bench(
                "ascii_re",
                [k](std::string const& seq){
                    return run_var_ascii_re(seq, k);
                }
            ),

            // Shift bits
            bench(
                "ifs_shift",
                [k](std::string const& seq){
                    return run_var_ifs_shift(seq, k);
                }
            ),
            bench(
                "switch_shift",
                [k](std::string const& seq){
                    return run_var_switch_shift(seq, k);
                }
            ),
            bench(
                "table_shift",
                [k](std::string const& seq){
                    return run_var_table_shift(seq, k);
                }
            ),
            bench(
                "ascii_shift",
                [k](std::string const& seq){
                    return run_var_ascii_shift(seq, k);
                }
            ),

            // SIMD, currently only AVX2 and scalar, for testing
            bench(
                "simd_avx2",
                [k](std::string const& seq){
                    return run_var_simd_avx2(seq, k);
                }
            ),
            bench(
                "simd_scalar",
                [k](std::string const& seq){
                    return run_var_simd_scalar(seq, k);
                }
            )
        );

        std::string case_label = "k=" + std::to_string(k);
        write_csv_rows(csv_os, suite_title, case_label, results);
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }
}

void bench_kmer_extract(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    // Test all valid k-mer sizes
    bench_kmer_extract( sequences, 1, 32, csv_os );
}
