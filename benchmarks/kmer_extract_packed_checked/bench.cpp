#include <algorithm>
#include <bit>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kmer_extract_packed_checked/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

void bench_kmer_extract_packed_checked(
    std::vector<std::string> const& sequences,
    std::size_t k_min,
    std::size_t k_max,
    std::ostream& csv_os
) {
    if (k_min < 1 || k_min > 32 || k_max < 1 || k_max > 32) {
        throw std::runtime_error("Invalid k outside of [1, 32]");
    }
    if (k_min > k_max) {
        throw std::runtime_error("Invalid k_min > k_max");
    }

    std::size_t const rounds = 8;
    std::size_t const repeats = 8;

    std::string const suite_title = "kmer_extract_packed_checked";
    std::cout << "\n=== k-mer extract (packed, checked) ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    std::vector<CheckedMsb> packed_msb;
    std::vector<CheckedLsb> packed_lsb;
    packed_msb.reserve(sequences.size());
    packed_lsb.reserve(sequences.size());
    for (auto const& seq : sequences) {
        packed_msb.push_back(
            pack_sequence_with_validity(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{})
        );
        packed_lsb.push_back(
            pack_sequence_with_validity(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{})
        );
    }

    // Backing storage for the Write sink (see sink.hpp), sized to the longest sequence actually
    // benchmarked (rounded up to a power of two) so that no call here ever wraps it. That matters
    // more here than in most suites: the SIMD variants write whole registers at a time (2/4/8
    // k-mers per call) while aligned/rolling write one, and a wrap's residual contents depend on
    // that stride -- a register-width write and a scalar write that touched the same k-mers in a
    // wrapped buffer can legitimately leave different values behind in the slots that were
    // overwritten more than once, which would make finalize() disagree between them despite both
    // being correct. Never wrapping sidesteps that entirely, rather than trying to reconcile it.
    std::size_t max_seq_len = 0;
    for (auto const& seq : sequences) {
        max_seq_len = std::max(max_seq_len, seq.size());
    }
    std::vector<std::uint64_t> sink_buffer(
        std::bit_ceil(std::max<std::size_t>(max_seq_len, 1)), 0
    );

    std::size_t const narrow_width_k_max = std::min<std::size_t>(k_max, 29);

    // -----------------------------------------------------------------------
    //     layout=msb, k in [k_min, min(k_max, 29)]: narrow-width and full-width
    // -----------------------------------------------------------------------
    for (std::size_t k = k_min; k <= narrow_width_k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rlayout=msb k " << std::setw(2) << k << std::flush;
        }
        Microbench<CheckedMsb> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](CheckedMsb const& seq) { return static_cast<double>(seq.length - k + 1); }
        );
        auto results = suite.run(
            packed_msb,
            #if defined(FISK_HAS_SSE2)
            bench(
                "simd_narrow_width_sse2",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_narrow_width_sse2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_full_width_sse2",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_full_width_sse2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_sse2",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_sse2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_sse2_scalars",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_sse2_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench(
                "simd_narrow_width_avx2",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_narrow_width_avx2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_full_width_avx2",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_full_width_avx2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx2",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_avx2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx2_scalars",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_avx2_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench(
                "simd_narrow_width_avx512",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_narrow_width_avx512(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_full_width_avx512",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_full_width_avx512(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx512",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_avx512(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx512_scalars",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_avx512_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench(
                "simd_narrow_width_neon",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_narrow_width_neon(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_full_width_neon",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_full_width_neon(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_neon",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_neon(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_neon_scalars",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_neon_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_NEON
            bench(
                "aligned",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_aligned(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_rolling(seq, k, sink_buffer);
                }
            )
        );
        write_csv_rows(
            csv_os, suite_title, "layout=msb;k=" + std::to_string(k), results, kSinkName
        );
    }

    // -----------------------------------------------------------------------
    //     layout=msb, k in [30, k_max]: full-width only
    // -----------------------------------------------------------------------
    for (std::size_t k = std::max<std::size_t>(k_min, 30); k <= k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rlayout=msb k " << std::setw(2) << k << std::flush;
        }
        Microbench<CheckedMsb> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](CheckedMsb const& seq) { return static_cast<double>(seq.length - k + 1); }
        );
        auto results = suite.run(
            packed_msb,
            #if defined(FISK_HAS_SSE2)
            bench(
                "simd_full_width_sse2",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_full_width_sse2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_sse2",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_sse2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_sse2_scalars",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_sse2_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench(
                "simd_full_width_avx2",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_full_width_avx2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx2",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_avx2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx2_scalars",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_avx2_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench(
                "simd_full_width_avx512",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_full_width_avx512(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx512",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_avx512(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx512_scalars",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_avx512_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench(
                "simd_full_width_neon",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_full_width_neon(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_neon",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_neon(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_neon_scalars",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_simd_neon_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_NEON
            bench(
                "aligned",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_aligned(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling",
                [k, &sink_buffer](CheckedMsb const& seq) {
                    return run_var_checked_msb_rolling(seq, k, sink_buffer);
                }
            )
        );
        write_csv_rows(
            csv_os, suite_title, "layout=msb;k=" + std::to_string(k), results, kSinkName
        );
    }

    // -----------------------------------------------------------------------
    //     layout=lsb, k in [k_min, min(k_max, 29)]: narrow-width and full-width
    // -----------------------------------------------------------------------
    for (std::size_t k = k_min; k <= narrow_width_k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rlayout=lsb k=" << std::setw(2) << k << std::flush;
        }
        Microbench<CheckedLsb> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](CheckedLsb const& seq) { return static_cast<double>(seq.length - k + 1); }
        );
        auto results = suite.run(
            packed_lsb,
            #if defined(FISK_HAS_SSE2)
            bench(
                "simd_narrow_width_sse2",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_narrow_width_sse2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_full_width_sse2",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_full_width_sse2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_sse2",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_sse2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_sse2_scalars",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_sse2_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench(
                "simd_narrow_width_avx2",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_narrow_width_avx2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_full_width_avx2",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_full_width_avx2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx2",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_avx2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx2_scalars",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_avx2_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench(
                "simd_narrow_width_avx512",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_narrow_width_avx512(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_full_width_avx512",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_full_width_avx512(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx512",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_avx512(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx512_scalars",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_avx512_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench(
                "simd_narrow_width_neon",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_narrow_width_neon(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_full_width_neon",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_full_width_neon(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_neon",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_neon(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_neon_scalars",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_neon_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_NEON
            bench(
                "aligned",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_aligned(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_rolling(seq, k, sink_buffer);
                }
            )
        );
        write_csv_rows(
            csv_os, suite_title, "layout=lsb;k=" + std::to_string(k), results, kSinkName
        );
    }

    // -----------------------------------------------------------------------
    //     layout=lsb, k in [30, k_max]: full-width only
    // -----------------------------------------------------------------------
    for (std::size_t k = std::max<std::size_t>(k_min, 30); k <= k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rlayout=lsb k=" << std::setw(2) << k << std::flush;
        }
        Microbench<CheckedLsb> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](CheckedLsb const& seq) { return static_cast<double>(seq.length - k + 1); }
        );
        auto results = suite.run(
            packed_lsb,
            #if defined(FISK_HAS_SSE2)
            bench(
                "simd_full_width_sse2",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_full_width_sse2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_sse2",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_sse2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_sse2_scalars",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_sse2_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench(
                "simd_full_width_avx2",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_full_width_avx2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx2",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_avx2(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx2_scalars",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_avx2_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench(
                "simd_full_width_avx512",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_full_width_avx512(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx512",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_avx512(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_avx512_scalars",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_avx512_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench(
                "simd_full_width_neon",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_full_width_neon(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_neon",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_neon(seq, k, sink_buffer);
                }
            ),
            bench(
                "simd_neon_scalars",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_simd_neon_scalars(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_NEON
            bench(
                "aligned",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_aligned(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling",
                [k, &sink_buffer](CheckedLsb const& seq) {
                    return run_var_checked_lsb_rolling(seq, k, sink_buffer);
                }
            )
        );
        write_csv_rows(
            csv_os, suite_title, "layout=lsb;k=" + std::to_string(k), results, kSinkName
        );
    }

    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}

void bench_kmer_extract_packed_checked(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    bench_kmer_extract_packed_checked(sequences, 1, 32, csv_os);
}
