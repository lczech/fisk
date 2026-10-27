#include <algorithm>
#include <bit>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kmer_extract_ascii_checked/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

void bench_kmer_extract_ascii_checked(
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
    std::string const suite_title = "kmer_extract_ascii_checked";
    std::cout << "\n=== k-mer extract from ASCII, checked ===\n";
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
    // The chunked vector variants also write the lanes that pad their last vector, at most 8.
    std::vector<std::uint64_t> sink_buffer(std::bit_ceil(max_seq_len + 8), 0);

    // Each scratch buffer stays dedicated to one variant, so retained capacity from one
    // benchmark cannot affect another variant's setup.
    CheckedAcgtMsb scratch_scalar_acgt_msb;
    CheckedAcgtMsb scratch_simd_scalar_scalars_acgt_msb;
    #if defined(FISK_HAS_AVX2)
    CheckedAcgtMsb scratch_simd_avx2_scalars_acgt_msb;
    #endif // FISK_HAS_AVX2
    #if defined(FISK_HAS_AVX512)
    CheckedAcgtMsb scratch_simd_avx512_scalars_acgt_msb;
    #endif // FISK_HAS_AVX512
    #if defined(FISK_HAS_NEON)
    CheckedAcgtMsb scratch_simd_neon_scalars_acgt_msb;
    #endif // FISK_HAS_NEON
    #if defined(FISK_HAS_SSE2)
    CheckedAcgtMsb scratch_simd_sse2_scalars_acgt_msb;
    #endif // FISK_HAS_SSE2
    CheckedAcgtMsb scratch_simd_scalar_acgt_msb;
    #if defined(FISK_HAS_SSE2)
    CheckedAcgtMsb scratch_simd_sse2_acgt_msb;
    #endif // FISK_HAS_SSE2
    #if defined(FISK_HAS_AVX2)
    CheckedAcgtMsb scratch_simd_avx2_acgt_msb;
    #endif // FISK_HAS_AVX2
    #if defined(FISK_HAS_AVX512)
    CheckedAcgtMsb scratch_simd_avx512_acgt_msb;
    #endif // FISK_HAS_AVX512
    #if defined(FISK_HAS_NEON)
    CheckedAcgtMsb scratch_simd_neon_acgt_msb;
    #endif // FISK_HAS_NEON
    CheckedAcgtLsb scratch_scalar_acgt_lsb;
    CheckedAcgtLsb scratch_simd_scalar_scalars_acgt_lsb;
    #if defined(FISK_HAS_AVX2)
    CheckedAcgtLsb scratch_simd_avx2_scalars_acgt_lsb;
    #endif // FISK_HAS_AVX2
    #if defined(FISK_HAS_AVX512)
    CheckedAcgtLsb scratch_simd_avx512_scalars_acgt_lsb;
    #endif // FISK_HAS_AVX512
    #if defined(FISK_HAS_NEON)
    CheckedAcgtLsb scratch_simd_neon_scalars_acgt_lsb;
    #endif // FISK_HAS_NEON
    #if defined(FISK_HAS_SSE2)
    CheckedAcgtLsb scratch_simd_sse2_scalars_acgt_lsb;
    #endif // FISK_HAS_SSE2
    CheckedAcgtLsb scratch_simd_scalar_acgt_lsb;
    #if defined(FISK_HAS_SSE2)
    CheckedAcgtLsb scratch_simd_sse2_acgt_lsb;
    #endif // FISK_HAS_SSE2
    #if defined(FISK_HAS_AVX2)
    CheckedAcgtLsb scratch_simd_avx2_acgt_lsb;
    #endif // FISK_HAS_AVX2
    #if defined(FISK_HAS_AVX512)
    CheckedAcgtLsb scratch_simd_avx512_acgt_lsb;
    #endif // FISK_HAS_AVX512
    #if defined(FISK_HAS_NEON)
    CheckedAcgtLsb scratch_simd_neon_acgt_lsb;
    #endif // FISK_HAS_NEON

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
                "reextract_scalar_ifs",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_reextract_scalar_ifs_acgt_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "reextract_scalar_switch",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_reextract_scalar_switch_acgt_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "reextract_scalar_table",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_reextract_scalar_table_acgt_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "reextract_scalar_ascii",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_reextract_scalar_ascii_acgt_msb(seq, k, sink_buffer);
                }
            ),

            // Shift bits
            bench(
                "rolling_scalar_ifs",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_scalar_ifs_acgt_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_scalar_switch",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_scalar_switch_acgt_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_scalar_table",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_scalar_table_acgt_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_scalar_ascii",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_scalar_ascii_acgt_msb(seq, k, sink_buffer);
                }
            ),

            bench(
                "chunked_scalar",
                [k, &scratch_scalar_acgt_msb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_scalar_acgt_msb(
                        seq, k, scratch_scalar_acgt_msb, sink_buffer
                    );
                }
            ),
            bench(
                "chunked_simd_scalar_scalars",
                [k, &scratch_simd_scalar_scalars_acgt_msb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_simd_scalar_scalars_acgt_msb(
                        seq, k, scratch_simd_scalar_scalars_acgt_msb, sink_buffer
                    );
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "chunked_simd_sse2_scalars",
                [k, &scratch_simd_sse2_scalars_acgt_msb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_simd_sse2_scalars_acgt_msb(
                        seq, k, scratch_simd_sse2_scalars_acgt_msb, sink_buffer
                    );
                }
            ),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench(
                "chunked_simd_avx2_scalars",
                [k, &scratch_simd_avx2_scalars_acgt_msb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_simd_avx2_scalars_acgt_msb(
                        seq, k, scratch_simd_avx2_scalars_acgt_msb, sink_buffer
                    );
                }
            ),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench(
                "chunked_simd_avx512_scalars",
                [k, &scratch_simd_avx512_scalars_acgt_msb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_simd_avx512_scalars_acgt_msb(
                        seq, k, scratch_simd_avx512_scalars_acgt_msb, sink_buffer
                    );
                }
            ),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench(
                "chunked_simd_neon_scalars",
                [k, &scratch_simd_neon_scalars_acgt_msb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_simd_neon_scalars_acgt_msb(
                        seq, k, scratch_simd_neon_scalars_acgt_msb, sink_buffer
                    );
                }
            ),
            #endif // FISK_HAS_NEON
            // SIMD, currently only AVX2 and scalar, for testing
            bench(
                "rolling_avx2",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_avx2_acgt_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_swar",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_swar_acgt_msb(seq, k, sink_buffer);
                }
            )
        );

        std::string const k_label = "k=" + std::to_string(k);
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=msb;" + k_label, results, kSinkName
        );

        // LSB emits different k-mer values than MSB, so it needs its own suite.run() call, as
        // Microbench::run() cross-validates the sinks of all benches within one call.
        auto const results_lsb = suite.run(
            sequences,
            bench(
                "rolling_scalar_table",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_scalar_table_acgt_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_scalar_ascii",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_scalar_ascii_acgt_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "chunked_scalar",
                [k, &scratch_scalar_acgt_lsb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_scalar_acgt_lsb(
                        seq, k, scratch_scalar_acgt_lsb, sink_buffer
                    );
                }
            ),
            bench(
                "chunked_simd_scalar_scalars",
                [k, &scratch_simd_scalar_scalars_acgt_lsb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_simd_scalar_scalars_acgt_lsb(
                        seq, k, scratch_simd_scalar_scalars_acgt_lsb, sink_buffer
                    );
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "chunked_simd_sse2_scalars",
                [k, &scratch_simd_sse2_scalars_acgt_lsb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_simd_sse2_scalars_acgt_lsb(
                        seq, k, scratch_simd_sse2_scalars_acgt_lsb, sink_buffer
                    );
                }
            ),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench(
                "chunked_simd_avx2_scalars",
                [k, &scratch_simd_avx2_scalars_acgt_lsb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_simd_avx2_scalars_acgt_lsb(
                        seq, k, scratch_simd_avx2_scalars_acgt_lsb, sink_buffer
                    );
                }
            ),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench(
                "chunked_simd_avx512_scalars",
                [k, &scratch_simd_avx512_scalars_acgt_lsb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_simd_avx512_scalars_acgt_lsb(
                        seq, k, scratch_simd_avx512_scalars_acgt_lsb, sink_buffer
                    );
                }
            ),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench(
                "chunked_simd_neon_scalars",
                [k, &scratch_simd_neon_scalars_acgt_lsb, &sink_buffer](std::string const& seq){
                    return run_var_chunked_simd_neon_scalars_acgt_lsb(
                        seq, k, scratch_simd_neon_scalars_acgt_lsb, sink_buffer
                    );
                }
            ),
            #endif // FISK_HAS_NEON
            bench(
                "rolling_avx2",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_avx2_acgt_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_swar",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_swar_acgt_lsb(seq, k, sink_buffer);
                }
            )
        );
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=lsb;" + k_label, results_lsb, kSinkName
        );

        // ACTG emits different k-mer values than ACGT, so it gets its own suite.run() call too.
        auto const results_actg = suite.run(
            sequences,
            bench(
                "rolling_scalar_ascii",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_scalar_ascii_actg_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_avx2",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_avx2_actg_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_swar",
                [k, &sink_buffer](std::string const& seq){
                    return run_var_rolling_swar_actg_msb(seq, k, sink_buffer);
                }
            )
        );
        write_csv_rows(
            csv_os, suite_title, "encoding=actg;layout=msb;" + k_label, results_actg, kSinkName
        );

        // The chunked SIMD variants that take the vectors whole sum all lanes, valid or not,
        // see bench.hpp, and the lanes past the last k-mer depend on the arch, so that each
        // arch and Layout has its own suite.run().
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=msb;" + k_label,
            suite.run(
                sequences,
                bench(
                    "chunked_simd_scalar",
                    [k, &scratch_simd_scalar_acgt_msb, &sink_buffer](std::string const& seq){
                        return run_var_chunked_simd_scalar_acgt_msb(
                            seq, k, scratch_simd_scalar_acgt_msb, sink_buffer
                        );
                    }
                )
            ),
            kSinkName
        );
        #if defined(FISK_HAS_SSE2)
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=msb;" + k_label,
            suite.run(
                sequences,
                bench(
                    "chunked_simd_sse2",
                    [k, &scratch_simd_sse2_acgt_msb, &sink_buffer](std::string const& seq){
                        return run_var_chunked_simd_sse2_acgt_msb(
                            seq, k, scratch_simd_sse2_acgt_msb, sink_buffer
                        );
                    }
                )
            ),
            kSinkName
        );
        #endif // FISK_HAS_SSE2
        #if defined(FISK_HAS_AVX2)
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=msb;" + k_label,
            suite.run(
                sequences,
                bench(
                    "chunked_simd_avx2",
                    [k, &scratch_simd_avx2_acgt_msb, &sink_buffer](std::string const& seq){
                        return run_var_chunked_simd_avx2_acgt_msb(
                            seq, k, scratch_simd_avx2_acgt_msb, sink_buffer
                        );
                    }
                )
            ),
            kSinkName
        );
        #endif // FISK_HAS_AVX2
        #if defined(FISK_HAS_AVX512)
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=msb;" + k_label,
            suite.run(
                sequences,
                bench(
                    "chunked_simd_avx512",
                    [k, &scratch_simd_avx512_acgt_msb, &sink_buffer](std::string const& seq){
                        return run_var_chunked_simd_avx512_acgt_msb(
                            seq, k, scratch_simd_avx512_acgt_msb, sink_buffer
                        );
                    }
                )
            ),
            kSinkName
        );
        #endif // FISK_HAS_AVX512
        #if defined(FISK_HAS_NEON)
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=msb;" + k_label,
            suite.run(
                sequences,
                bench(
                    "chunked_simd_neon",
                    [k, &scratch_simd_neon_acgt_msb, &sink_buffer](std::string const& seq){
                        return run_var_chunked_simd_neon_acgt_msb(
                            seq, k, scratch_simd_neon_acgt_msb, sink_buffer
                        );
                    }
                )
            ),
            kSinkName
        );
        #endif // FISK_HAS_NEON
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=lsb;" + k_label,
            suite.run(
                sequences,
                bench(
                    "chunked_simd_scalar",
                    [k, &scratch_simd_scalar_acgt_lsb, &sink_buffer](std::string const& seq){
                        return run_var_chunked_simd_scalar_acgt_lsb(
                            seq, k, scratch_simd_scalar_acgt_lsb, sink_buffer
                        );
                    }
                )
            ),
            kSinkName
        );
        #if defined(FISK_HAS_SSE2)
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=lsb;" + k_label,
            suite.run(
                sequences,
                bench(
                    "chunked_simd_sse2",
                    [k, &scratch_simd_sse2_acgt_lsb, &sink_buffer](std::string const& seq){
                        return run_var_chunked_simd_sse2_acgt_lsb(
                            seq, k, scratch_simd_sse2_acgt_lsb, sink_buffer
                        );
                    }
                )
            ),
            kSinkName
        );
        #endif // FISK_HAS_SSE2
        #if defined(FISK_HAS_AVX2)
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=lsb;" + k_label,
            suite.run(
                sequences,
                bench(
                    "chunked_simd_avx2",
                    [k, &scratch_simd_avx2_acgt_lsb, &sink_buffer](std::string const& seq){
                        return run_var_chunked_simd_avx2_acgt_lsb(
                            seq, k, scratch_simd_avx2_acgt_lsb, sink_buffer
                        );
                    }
                )
            ),
            kSinkName
        );
        #endif // FISK_HAS_AVX2
        #if defined(FISK_HAS_AVX512)
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=lsb;" + k_label,
            suite.run(
                sequences,
                bench(
                    "chunked_simd_avx512",
                    [k, &scratch_simd_avx512_acgt_lsb, &sink_buffer](std::string const& seq){
                        return run_var_chunked_simd_avx512_acgt_lsb(
                            seq, k, scratch_simd_avx512_acgt_lsb, sink_buffer
                        );
                    }
                )
            ),
            kSinkName
        );
        #endif // FISK_HAS_AVX512
        #if defined(FISK_HAS_NEON)
        write_csv_rows(
            csv_os, suite_title, "encoding=acgt;layout=lsb;" + k_label,
            suite.run(
                sequences,
                bench(
                    "chunked_simd_neon",
                    [k, &scratch_simd_neon_acgt_lsb, &sink_buffer](std::string const& seq){
                        return run_var_chunked_simd_neon_acgt_lsb(
                            seq, k, scratch_simd_neon_acgt_lsb, sink_buffer
                        );
                    }
                )
            ),
            kSinkName
        );
        #endif // FISK_HAS_NEON
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }
}

void bench_kmer_extract_ascii_checked(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    // Test all valid k-mer sizes
    bench_kmer_extract_ascii_checked( sequences, 1, 32, csv_os );
}
