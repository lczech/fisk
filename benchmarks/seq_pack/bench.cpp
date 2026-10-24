#include <iostream>
#include <string>
#include <vector>

#include "microbench.hpp"
#include "seq_pack/bench.hpp"
#include "seq_pack/validate_external.hpp"

using namespace fisk;

namespace {

// Order-insensitive checksum over a PackedSequence: sufficient for cross-validation within one
// encoding/layout group (bit match expected), and to keep the compiler from optimizing
// pack_sequence_unchecked() calls away, without adding an order-sensitive (and more expensive)
// reduction to the timed region.
template <Encoding E, Layout L>
std::uint64_t pack_sequence_sink(PackedSequence<E, L> const& s)
{
    std::uint64_t h = 0;
    for (auto const w : s.codes()) {
        h += w;
    }
    return h;
}

} // namespace

void bench_seq_pack(std::vector<std::string> const& sequences, std::ostream& csv_os)
{
    std::size_t const rounds  = 8;
    std::size_t const repeats = 16;

    std::string const suite_title = "seq_pack";
    std::cout << "\n=== sequence pack ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    external_validation::validate(sequences);

    write_csv_header(csv_os);

    // -----------------------------------------------------------------------
    //     encoding=acgt;layout=lsb
    // -----------------------------------------------------------------------
    {
        PackedSequence<Encoding::kACGT, Layout::kLSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats)
            .units_fn([](std::string const& in) { return static_cast<double>(in.size()); })
            .finalize([&out]() { return pack_sequence_sink(out); });
        auto results = suite.run(
            sequences,
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_pext(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "pext_sse2",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_pext_sse2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "pext_avx2",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_pext_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "pext_avx512",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_pext_avx512(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_SSE2)
            bench(
                "butterfly_sse2",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_butterfly_sse2(seq, out);
                }
            ),
            bench(
                "fold_sse2",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_fold_sse2(seq, out);
                }
            ),
            bench(
                "direct_sse2",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_direct_sse2(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSSE3)
            bench(
                "direct_double_sse2",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_direct_double_sse2(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "butterfly_avx2",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_butterfly_avx2(seq, out);
                }
            ),
            bench(
                "fold_avx2",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_fold_avx2(seq, out);
                }
            ),
            bench(
                "direct_avx2",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_direct_avx2(seq, out);
                }
            ),
            bench(
                "direct_double_avx2",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_direct_double_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "butterfly_avx512",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_butterfly_avx512(seq, out);
                }
            ),
            bench(
                "fold_avx512",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_fold_avx512(seq, out);
                }
            ),
            bench(
                "direct_avx512",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_direct_avx512(seq, out);
                }
            ),
            bench(
                "direct_double_avx512",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_direct_double_avx512(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "butterfly_neon",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_butterfly_neon(seq, out);
                }
            ),
            bench(
                "fold_neon",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_fold_neon(seq, out);
                }
            ),
            bench(
                "direct_neon",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_direct_neon(seq, out);
                }
            ),
            bench(
                "direct_double_neon",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_direct_double_neon(seq, out);
                }
            ),
            #endif
            bench(
                "fold_scalar",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_fold_scalar(seq, out);
                }
            ),
            bench(
                "direct_scalar",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_direct_scalar(seq, out);
                }
            ),
            bench(
                "butterfly_scalar",
                [&](std::string const& seq) {
                    return run_var_acgt_lsb_butterfly_scalar(seq, out);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt;layout=lsb", results);
    }

    // -----------------------------------------------------------------------
    //     encoding=acgt;layout=msb
    // -----------------------------------------------------------------------
    {
        PackedSequence<Encoding::kACGT, Layout::kMSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats)
            .units_fn([](std::string const& in) { return static_cast<double>(in.size()); })
            .finalize([&out]() { return pack_sequence_sink(out); });
        auto results = suite.run(
            sequences,
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_pext(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "pext_sse2",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_pext_sse2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "pext_avx2",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_pext_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "pext_avx512",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_pext_avx512(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_SSE2)
            bench(
                "butterfly_sse2",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_butterfly_sse2(seq, out);
                }
            ),
            bench(
                "fold_sse2",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_fold_sse2(seq, out);
                }
            ),
            bench(
                "direct_sse2",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_direct_sse2(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSSE3)
            bench(
                "direct_double_sse2",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_direct_double_sse2(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "butterfly_avx2",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_butterfly_avx2(seq, out);
                }
            ),
            bench(
                "fold_avx2",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_fold_avx2(seq, out);
                }
            ),
            bench(
                "direct_avx2",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_direct_avx2(seq, out);
                }
            ),
            bench(
                "direct_double_avx2",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_direct_double_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "butterfly_avx512",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_butterfly_avx512(seq, out);
                }
            ),
            bench(
                "fold_avx512",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_fold_avx512(seq, out);
                }
            ),
            bench(
                "direct_avx512",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_direct_avx512(seq, out);
                }
            ),
            bench(
                "direct_double_avx512",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_direct_double_avx512(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "butterfly_neon",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_butterfly_neon(seq, out);
                }
            ),
            bench(
                "fold_neon",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_fold_neon(seq, out);
                }
            ),
            bench(
                "direct_neon",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_direct_neon(seq, out);
                }
            ),
            bench(
                "direct_double_neon",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_direct_double_neon(seq, out);
                }
            ),
            #endif
            bench(
                "fold_scalar",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_fold_scalar(seq, out);
                }
            ),
            bench(
                "direct_scalar",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_direct_scalar(seq, out);
                }
            ),
            bench(
                "butterfly_scalar",
                [&](std::string const& seq) {
                    return run_var_acgt_msb_butterfly_scalar(seq, out);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt;layout=msb", results);
    }

    // -----------------------------------------------------------------------
    //     encoding=actg;layout=lsb
    // -----------------------------------------------------------------------
    {
        PackedSequence<Encoding::kACTG, Layout::kLSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats)
            .units_fn([](std::string const& in) { return static_cast<double>(in.size()); })
            .finalize([&out]() { return pack_sequence_sink(out); });
        auto results = suite.run(
            sequences,
            #if defined(FISK_HAS_AVX2)
            bench(
                "cute_mul",
                [&](std::string const& seq) {
                    return run_var_cute_mul(seq, out);
                }
            ),
            bench(
                "cute_movemask",
                [&](std::string const& seq) {
                    return run_var_cute_movemask(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_BMI2)
            bench(
                "fastdna_pext_unordered",
                [&](std::string const& seq) {
                    return run_var_fastdna_pext_unordered(seq, out);
                }
            ),
            bench(
                "fastdna_pext_unordered_unroll2",
                [&](std::string const& seq) {
                    return run_var_fastdna_pext_unordered_unroll2(seq, out);
                }
            ),
            bench(
                "fastdna_pext_unordered_unroll4",
                [&](std::string const& seq) {
                    return run_var_fastdna_pext_unordered_unroll4(seq, out);
                }
            ),
            bench(
                "pext",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_pext(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "pext_sse2",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_pext_sse2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "pext_avx2",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_pext_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "pext_avx512",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_pext_avx512(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_SSE2)
            bench(
                "butterfly_sse2",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_butterfly_sse2(seq, out);
                }
            ),
            bench(
                "fold_sse2",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_fold_sse2(seq, out);
                }
            ),
            bench(
                "direct_sse2",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_direct_sse2(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSSE3)
            bench(
                "direct_double_sse2",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_direct_double_sse2(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "butterfly_avx2",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_butterfly_avx2(seq, out);
                }
            ),
            bench(
                "fold_avx2",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_fold_avx2(seq, out);
                }
            ),
            bench(
                "direct_avx2",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_direct_avx2(seq, out);
                }
            ),
            bench(
                "direct_double_avx2",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_direct_double_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "butterfly_avx512",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_butterfly_avx512(seq, out);
                }
            ),
            bench(
                "fold_avx512",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_fold_avx512(seq, out);
                }
            ),
            bench(
                "direct_avx512",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_direct_avx512(seq, out);
                }
            ),
            bench(
                "direct_double_avx512",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_direct_double_avx512(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "butterfly_neon",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_butterfly_neon(seq, out);
                }
            ),
            bench(
                "fold_neon",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_fold_neon(seq, out);
                }
            ),
            bench(
                "direct_neon",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_direct_neon(seq, out);
                }
            ),
            bench(
                "direct_double_neon",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_direct_double_neon(seq, out);
                }
            ),
            #endif
            bench(
                "fold_scalar",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_fold_scalar(seq, out);
                }
            ),
            bench(
                "direct_scalar",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_direct_scalar(seq, out);
                }
            ),
            bench(
                "butterfly_scalar",
                [&](std::string const& seq) {
                    return run_var_actg_lsb_butterfly_scalar(seq, out);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg;layout=lsb", results);
    }

    // -----------------------------------------------------------------------
    //     encoding=actg;layout=msb
    // -----------------------------------------------------------------------
    {
        PackedSequence<Encoding::kACTG, Layout::kMSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats)
            .units_fn([](std::string const& in) { return static_cast<double>(in.size()); })
            .finalize([&out]() { return pack_sequence_sink(out); });
        auto results = suite.run(
            sequences,
            bench(
                "fastdna_lut",
                [&](std::string const& seq) {
                    return run_var_fastdna_lut(seq, out);
                }
            ),
            #if defined(FISK_HAS_BMI2)
            bench(
                "fastdna_pext_ordered",
                [&](std::string const& seq) {
                    return run_var_fastdna_pext_ordered(seq, out);
                }
            ),
            bench(
                "pext",
                [&](std::string const& seq) {
                    return run_var_actg_msb_pext(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "pext_sse2",
                [&](std::string const& seq) {
                    return run_var_actg_msb_pext_sse2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "pext_avx2",
                [&](std::string const& seq) {
                    return run_var_actg_msb_pext_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "pext_avx512",
                [&](std::string const& seq) {
                    return run_var_actg_msb_pext_avx512(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_SSE2)
            bench(
                "butterfly_sse2",
                [&](std::string const& seq) {
                    return run_var_actg_msb_butterfly_sse2(seq, out);
                }
            ),
            bench(
                "fold_sse2",
                [&](std::string const& seq) {
                    return run_var_actg_msb_fold_sse2(seq, out);
                }
            ),
            bench(
                "direct_sse2",
                [&](std::string const& seq) {
                    return run_var_actg_msb_direct_sse2(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSSE3)
            bench(
                "direct_double_sse2",
                [&](std::string const& seq) {
                    return run_var_actg_msb_direct_double_sse2(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "butterfly_avx2",
                [&](std::string const& seq) {
                    return run_var_actg_msb_butterfly_avx2(seq, out);
                }
            ),
            bench(
                "fold_avx2",
                [&](std::string const& seq) {
                    return run_var_actg_msb_fold_avx2(seq, out);
                }
            ),
            bench(
                "direct_avx2",
                [&](std::string const& seq) {
                    return run_var_actg_msb_direct_avx2(seq, out);
                }
            ),
            bench(
                "direct_double_avx2",
                [&](std::string const& seq) {
                    return run_var_actg_msb_direct_double_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "butterfly_avx512",
                [&](std::string const& seq) {
                    return run_var_actg_msb_butterfly_avx512(seq, out);
                }
            ),
            bench(
                "fold_avx512",
                [&](std::string const& seq) {
                    return run_var_actg_msb_fold_avx512(seq, out);
                }
            ),
            bench(
                "direct_avx512",
                [&](std::string const& seq) {
                    return run_var_actg_msb_direct_avx512(seq, out);
                }
            ),
            bench(
                "direct_double_avx512",
                [&](std::string const& seq) {
                    return run_var_actg_msb_direct_double_avx512(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "butterfly_neon",
                [&](std::string const& seq) {
                    return run_var_actg_msb_butterfly_neon(seq, out);
                }
            ),
            bench(
                "fold_neon",
                [&](std::string const& seq) {
                    return run_var_actg_msb_fold_neon(seq, out);
                }
            ),
            bench(
                "direct_neon",
                [&](std::string const& seq) {
                    return run_var_actg_msb_direct_neon(seq, out);
                }
            ),
            bench(
                "direct_double_neon",
                [&](std::string const& seq) {
                    return run_var_actg_msb_direct_double_neon(seq, out);
                }
            ),
            #endif
            bench(
                "fold_scalar",
                [&](std::string const& seq) {
                    return run_var_actg_msb_fold_scalar(seq, out);
                }
            ),
            bench(
                "direct_scalar",
                [&](std::string const& seq) {
                    return run_var_actg_msb_direct_scalar(seq, out);
                }
            ),
            bench(
                "butterfly_scalar",
                [&](std::string const& seq) {
                    return run_var_actg_msb_butterfly_scalar(seq, out);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg;layout=msb", results);
    }
}
