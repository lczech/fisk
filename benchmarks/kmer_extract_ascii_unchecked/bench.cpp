#include <algorithm>
#include <bit>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kmer_extract_ascii_unchecked/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

void bench_kmer_extract_ascii_unchecked(
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
    std::string const suite_title = "kmer_extract_ascii_unchecked";
    std::cout << "\n=== k-mer extract from ASCII, unchecked ===\n";
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

    // Each scratch buffer stays dedicated to one variant, so retained capacity from one benchmark
    // cannot affect another variant's setup.
    PackedAcgtMsb chunked_scratch_acgt_msb;
    PackedAcgtLsb chunked_scratch_acgt_lsb;
    PackedActgMsb chunked_scratch_actg_msb;
    PackedActgLsb chunked_scratch_actg_lsb;
    PackedAcgtMsb pack_scratch_acgt_msb;
    PackedAcgtLsb pack_scratch_acgt_lsb;
    PackedActgMsb pack_scratch_actg_msb;
    PackedActgLsb pack_scratch_actg_lsb;

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
        std::string const k_label = "k=" + std::to_string(k);

        // The variants without checks encode invalid characters (see --n-prob in main.cpp) as
        // some valid base, so they emit other k-mers than the checked ones, and run in a suite of
        // their own. One suite.run() call per Encoding and Layout, as Microbench::run()
        // cross-validates the sinks of all benches in one call, and those produce different
        // k-mer values; within each call, all variants use the same word encoder, so they still
        // agree exactly.
        auto const results_acgt_msb = suite.run(
            sequences,
            bench(
                "chunked_scalar_threadlocal",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_chunked_scalar_threadlocal_acgt_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "chunked_scalar_scratch",
                [k, &chunked_scratch_acgt_msb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_scalar_scratch_acgt_msb(
                        seq, k, chunked_scratch_acgt_msb, sink_buffer
                    );
                }
            ),
            bench(
                "pack_then_bytewise_scalar",
                [k, &pack_scratch_acgt_msb, &sink_buffer](std::string const& seq){
                    return run_var_pack_then_bytewise_scalar_acgt_msb(
                        seq, k, pack_scratch_acgt_msb, sink_buffer
                    );
                }
            )
        );
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=msb;" + k_label, results_acgt_msb, kSinkName
        );

        auto const results_acgt_lsb = suite.run(
            sequences,
            bench(
                "chunked_scalar_threadlocal",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_chunked_scalar_threadlocal_acgt_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "chunked_scalar_scratch",
                [k, &chunked_scratch_acgt_lsb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_scalar_scratch_acgt_lsb(
                        seq, k, chunked_scratch_acgt_lsb, sink_buffer
                    );
                }
            ),
            bench(
                "pack_then_bytewise_scalar",
                [k, &pack_scratch_acgt_lsb, &sink_buffer](std::string const& seq){
                    return run_var_pack_then_bytewise_scalar_acgt_lsb(
                        seq, k, pack_scratch_acgt_lsb, sink_buffer
                    );
                }
            )
        );
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=lsb;" + k_label, results_acgt_lsb, kSinkName
        );

        auto const results_actg_msb = suite.run(
            sequences,
            bench(
                "chunked_scalar_threadlocal",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_chunked_scalar_threadlocal_actg_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "chunked_scalar_scratch",
                [k, &chunked_scratch_actg_msb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_scalar_scratch_actg_msb(
                        seq, k, chunked_scratch_actg_msb, sink_buffer
                    );
                }
            ),
            bench(
                "pack_then_bytewise_scalar",
                [k, &pack_scratch_actg_msb, &sink_buffer](std::string const& seq){
                    return run_var_pack_then_bytewise_scalar_actg_msb(
                        seq, k, pack_scratch_actg_msb, sink_buffer
                    );
                }
            )
        );
        write_csv_rows(
            csv_os, suite_title, "encoding=actg;layout=msb;" + k_label, results_actg_msb, kSinkName
        );

        auto const results_actg_lsb = suite.run(
            sequences,
            bench(
                "chunked_scalar_threadlocal",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_chunked_scalar_threadlocal_actg_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "chunked_scalar_scratch",
                [k, &chunked_scratch_actg_lsb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_scalar_scratch_actg_lsb(
                        seq, k, chunked_scratch_actg_lsb, sink_buffer
                    );
                }
            ),
            bench(
                "pack_then_bytewise_scalar",
                [k, &pack_scratch_actg_lsb, &sink_buffer](std::string const& seq){
                    return run_var_pack_then_bytewise_scalar_actg_lsb(
                        seq, k, pack_scratch_actg_lsb, sink_buffer
                    );
                }
            )
        );
        write_csv_rows(
            csv_os, suite_title, "encoding=actg;layout=lsb;" + k_label, results_actg_lsb, kSinkName
        );
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }
}

void bench_kmer_extract_ascii_unchecked(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    // Test all valid k-mer sizes
    bench_kmer_extract_ascii_unchecked( sequences, 1, 32, csv_os );
}
