#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kmer_extract_packed_unchecked/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

void bench_kmer_extract_packed_unchecked(
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

    std::string const suite_title = "kmer_extract_packed_unchecked";
    std::cout << "\n=== k-mer extract (packed) ===\n";

    write_csv_header(csv_os);

    std::vector<PackedMsb> packed_msb;
    std::vector<PackedLsb> packed_lsb;
    packed_msb.reserve(sequences.size());
    packed_lsb.reserve(sequences.size());
    for (auto const& seq : sequences) {
        packed_msb.push_back(
            pack_sequence_unchecked(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{})
        );
        packed_lsb.push_back(
            pack_sequence_unchecked(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{})
        );
    }

    // The buffer of the Write sink, see sink.hpp; Sum and Barrier ignore it.
    auto sink_buffer = make_sink_buffer();

    std::size_t const narrow_width_k_max = std::min<std::size_t>(k_max, 29);

    // -----------------------------------------------------------------------
    //     layout=msb, k in [k_min, min(k_max, 29)]: narrow-width and full-width
    // -----------------------------------------------------------------------
    for (std::size_t k = k_min; k <= narrow_width_k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rlayout=msb k " << std::setw(2) << k << std::flush;
        }
        Microbench<PackedMsb> suite(suite_title);
        suite.units_fn(
            [k](PackedMsb const& seq) { return static_cast<double>(seq.length() - k + 1); }
        );
        auto results = suite.run(
            packed_msb,
            #if defined(FISK_HAS_SSE2)
            bench(
                "bytewise_simd_narrow_width_sse2",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_narrow_width_sse2_msb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_full_width_sse2",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_sse2_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_sse2",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_sse2_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_sse2_scalars",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_sse2_scalars_msb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench(
                "bytewise_simd_narrow_width_avx2",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_narrow_width_avx2_msb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_full_width_avx2",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_avx2_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx2",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx2_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx2_scalars",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx2_scalars_msb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench(
                "bytewise_simd_narrow_width_avx512",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_narrow_width_avx512_msb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_full_width_avx512",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_avx512_msb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_avx512",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx512_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx512_scalars",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx512_scalars_msb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench(
                "bytewise_simd_narrow_width_neon",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_narrow_width_neon_msb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_full_width_neon",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_neon_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_neon",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_neon_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_neon_scalars",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_neon_scalars_msb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_NEON
            bench(
                "bytewise_simd_narrow_width_scalar",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_narrow_width_scalar_msb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_full_width_scalar",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_scalar_msb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_scalar",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_scalar_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_scalar_scalars",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_scalar_scalars_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_scalar",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_scalar_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_scalar",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_rolling_scalar_msb(seq, k, sink_buffer);
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
        Microbench<PackedMsb> suite(suite_title);
        suite.units_fn(
            [k](PackedMsb const& seq) { return static_cast<double>(seq.length() - k + 1); }
        );
        auto results = suite.run(
            packed_msb,
            #if defined(FISK_HAS_SSE2)
            bench(
                "bytewise_simd_full_width_sse2",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_sse2_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_sse2",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_sse2_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_sse2_scalars",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_sse2_scalars_msb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench(
                "bytewise_simd_full_width_avx2",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_avx2_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx2",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx2_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx2_scalars",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx2_scalars_msb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench(
                "bytewise_simd_full_width_avx512",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_avx512_msb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_avx512",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx512_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx512_scalars",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx512_scalars_msb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench(
                "bytewise_simd_full_width_neon",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_neon_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_neon",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_neon_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_neon_scalars",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_neon_scalars_msb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_NEON
            bench(
                "bytewise_simd_full_width_scalar",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_scalar_msb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_scalar",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_scalar_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_scalar_scalars",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_simd_scalar_scalars_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_scalar",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_bytewise_scalar_msb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_scalar",
                [k, &sink_buffer](PackedMsb const& seq) {
                    return run_var_unchecked_rolling_scalar_msb(seq, k, sink_buffer);
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
        Microbench<PackedLsb> suite(suite_title);
        suite.units_fn(
            [k](PackedLsb const& seq) { return static_cast<double>(seq.length() - k + 1); }
        );
        auto results = suite.run(
            packed_lsb,
            #if defined(FISK_HAS_SSE2)
            bench(
                "bytewise_simd_narrow_width_sse2",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_narrow_width_sse2_lsb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_full_width_sse2",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_sse2_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_sse2",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_sse2_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_sse2_scalars",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_sse2_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench(
                "bytewise_simd_narrow_width_avx2",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_narrow_width_avx2_lsb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_full_width_avx2",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_avx2_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx2",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx2_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx2_scalars",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx2_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench(
                "bytewise_simd_narrow_width_avx512",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_narrow_width_avx512_lsb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_full_width_avx512",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_avx512_lsb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_avx512",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx512_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx512_scalars",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx512_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench(
                "bytewise_simd_narrow_width_neon",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_narrow_width_neon_lsb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_full_width_neon",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_neon_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_neon",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_neon_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_neon_scalars",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_neon_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_NEON
            bench(
                "bytewise_simd_narrow_width_scalar",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_narrow_width_scalar_lsb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_full_width_scalar",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_scalar_lsb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_scalar",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_scalar_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_scalar_scalars",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_scalar_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_scalar",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_scalar_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_scalar",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_rolling_scalar_lsb(seq, k, sink_buffer);
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
        Microbench<PackedLsb> suite(suite_title);
        suite.units_fn(
            [k](PackedLsb const& seq) { return static_cast<double>(seq.length() - k + 1); }
        );
        auto results = suite.run(
            packed_lsb,
            #if defined(FISK_HAS_SSE2)
            bench(
                "bytewise_simd_full_width_sse2",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_sse2_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_sse2",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_sse2_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_sse2_scalars",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_sse2_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_SSE2
            #if defined(FISK_HAS_AVX2)
            bench(
                "bytewise_simd_full_width_avx2",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_avx2_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx2",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx2_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx2_scalars",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx2_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX2
            #if defined(FISK_HAS_AVX512)
            bench(
                "bytewise_simd_full_width_avx512",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_avx512_lsb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_avx512",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx512_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_avx512_scalars",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_avx512_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_AVX512
            #if defined(FISK_HAS_NEON)
            bench(
                "bytewise_simd_full_width_neon",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_neon_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_neon",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_neon_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_neon_scalars",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_neon_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            #endif // FISK_HAS_NEON
            bench(
                "bytewise_simd_full_width_scalar",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_full_width_scalar_lsb(
                        seq, k, sink_buffer
                    );
                }
            ),
            bench(
                "bytewise_simd_scalar",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_scalar_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_simd_scalar_scalars",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_simd_scalar_scalars_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "bytewise_scalar",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_bytewise_scalar_lsb(seq, k, sink_buffer);
                }
            ),
            bench(
                "rolling_scalar",
                [k, &sink_buffer](PackedLsb const& seq) {
                    return run_var_unchecked_rolling_scalar_lsb(seq, k, sink_buffer);
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

void bench_kmer_extract_packed_unchecked(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    bench_kmer_extract_packed_unchecked(sequences, 1, 32, csv_os);
}
