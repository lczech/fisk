#include <algorithm>
#include <bit>
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

    // Backing storage for the Write sink (see sink.hpp), sized to the longest sequence actually
    // benchmarked (rounded up to a power of two) so that no call here ever wraps it. Different
    // techniques here could in principle differ in how many k-mers they actually emit (e.g. one
    // skipping windows a differently-behaved one still emits for), which would make a wrapped
    // buffer's leftover contents depend on which technique last touched each slot; never wrapping
    // sidesteps that regardless of whether it would in fact occur.
    std::size_t max_seq_len = 0;
    for (auto const& seq : sequences) {
        max_seq_len = std::max(max_seq_len, seq.size());
    }
    std::vector<std::uint64_t> sink_buffer(
        std::bit_ceil(std::max<std::size_t>(max_seq_len, 1)), 0
    );

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
                [k, &sink_buffer](std::string const& seq){
                    return run_var_ifs_re(seq, k, sink_buffer);
                }
            ),
            bench(
                "switch_re",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_switch_re(seq, k, sink_buffer);
                }
            ),
            bench(
                "table_re",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_table_re(seq, k, sink_buffer);
                }
            ),
            bench(
                "ascii_re",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_ascii_re(seq, k, sink_buffer);
                }
            ),

            // Shift bits
            bench(
                "ifs_shift",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_ifs_shift(seq, k, sink_buffer);
                }
            ),
            bench(
                "switch_shift",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_switch_shift(seq, k, sink_buffer);
                }
            ),
            bench(
                "table_shift",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_table_shift(seq, k, sink_buffer);
                }
            ),
            bench(
                "ascii_shift",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_ascii_shift(seq, k, sink_buffer);
                }
            ),

            // SIMD, currently only AVX2 and scalar, for testing
            bench(
                "simd_avx2",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_simd_avx2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_scalar",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_simd_scalar(seq, k, sink_buffer);
                }
            )
        );

        std::string case_label = "k=" + std::to_string(k);
        write_csv_rows(csv_os, suite_title, case_label, results, kSinkName);
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
