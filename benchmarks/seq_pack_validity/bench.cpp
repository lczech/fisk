#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "microbench.hpp"
#include "seq_pack_validity/bench.hpp"

using namespace fisk;

namespace {

// Order-insensitive checksum over a PackedSequenceWithValidity, as in seq_pack/bench.cpp, with
// the validity bytes weighted apart from the code bytes, so that a mismatch in either shows.
template <Encoding E, Layout L>
std::uint64_t pack_sequence_validity_sink(PackedSequenceWithValidity<E, L> const& s)
{
    std::uint64_t h = 0;
    for (auto const w : s.codes()) {
        h += w;
    }
    for (auto const w : s.validity()) {
        h += std::uint64_t{w} << 32;
    }
    return h;
}

} // namespace

void bench_seq_pack_validity(std::vector<std::string> const& sequences, std::ostream& csv_os)
{
    std::size_t const rounds  = 8;
    std::size_t const repeats = 16;

    std::string const suite_title = "seq_pack_validity";
    std::cout << "\n=== sequence pack with validity ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    // -----------------------------------------------------------------------
    //     encoding=acgt;layout=lsb
    // -----------------------------------------------------------------------
    {
        PackedSequenceWithValidity<Encoding::kACGT, Layout::kLSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats)
            .units_fn([](std::string const& in) { return static_cast<double>(in.size()); })
            .finalize([&out]() { return pack_sequence_validity_sink(out); });
        auto results = suite.run(
            sequences,
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_pext(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "pext_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_pext_sse2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "pext_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_pext_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "pext_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_pext_avx512(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_SSE2)
            bench(
                "butterfly_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_butterfly_sse2(seq, out);
                }
            ),
            bench(
                "fold_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_fold_sse2(seq, out);
                }
            ),
            bench(
                "direct_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_direct_sse2(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSSE3)
            bench(
                "direct_double_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_direct_double_sse2(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "butterfly_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_butterfly_avx2(seq, out);
                }
            ),
            bench(
                "fold_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_fold_avx2(seq, out);
                }
            ),
            bench(
                "direct_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_direct_avx2(seq, out);
                }
            ),
            bench(
                "direct_double_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_direct_double_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "butterfly_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_butterfly_avx512(seq, out);
                }
            ),
            bench(
                "fold_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_fold_avx512(seq, out);
                }
            ),
            bench(
                "direct_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_direct_avx512(seq, out);
                }
            ),
            bench(
                "direct_double_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_direct_double_avx512(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "butterfly_neon",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_butterfly_neon(seq, out);
                }
            ),
            bench(
                "fold_neon",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_fold_neon(seq, out);
                }
            ),
            bench(
                "direct_neon",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_direct_neon(seq, out);
                }
            ),
            bench(
                "direct_double_neon",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_direct_double_neon(seq, out);
                }
            ),
            #endif
            bench(
                "fold_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_fold_scalar(seq, out);
                }
            ),
            bench(
                "direct_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_direct_scalar(seq, out);
                }
            ),
            bench(
                "butterfly_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_lsb_butterfly_scalar(seq, out);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt;layout=lsb", results);
    }

    // -----------------------------------------------------------------------
    //     encoding=acgt;layout=msb
    // -----------------------------------------------------------------------
    {
        PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats)
            .units_fn([](std::string const& in) { return static_cast<double>(in.size()); })
            .finalize([&out]() { return pack_sequence_validity_sink(out); });
        auto results = suite.run(
            sequences,
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_pext(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "pext_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_pext_sse2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "pext_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_pext_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "pext_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_pext_avx512(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_SSE2)
            bench(
                "butterfly_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_butterfly_sse2(seq, out);
                }
            ),
            bench(
                "fold_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_fold_sse2(seq, out);
                }
            ),
            bench(
                "direct_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_direct_sse2(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSSE3)
            bench(
                "direct_double_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_direct_double_sse2(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "butterfly_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_butterfly_avx2(seq, out);
                }
            ),
            bench(
                "fold_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_fold_avx2(seq, out);
                }
            ),
            bench(
                "direct_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_direct_avx2(seq, out);
                }
            ),
            bench(
                "direct_double_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_direct_double_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "butterfly_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_butterfly_avx512(seq, out);
                }
            ),
            bench(
                "fold_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_fold_avx512(seq, out);
                }
            ),
            bench(
                "direct_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_direct_avx512(seq, out);
                }
            ),
            bench(
                "direct_double_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_direct_double_avx512(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "butterfly_neon",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_butterfly_neon(seq, out);
                }
            ),
            bench(
                "fold_neon",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_fold_neon(seq, out);
                }
            ),
            bench(
                "direct_neon",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_direct_neon(seq, out);
                }
            ),
            bench(
                "direct_double_neon",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_direct_double_neon(seq, out);
                }
            ),
            #endif
            bench(
                "fold_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_fold_scalar(seq, out);
                }
            ),
            bench(
                "direct_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_direct_scalar(seq, out);
                }
            ),
            bench(
                "butterfly_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_acgt_msb_butterfly_scalar(seq, out);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt;layout=msb", results);
    }

    // -----------------------------------------------------------------------
    //     encoding=actg;layout=lsb
    // -----------------------------------------------------------------------
    {
        PackedSequenceWithValidity<Encoding::kACTG, Layout::kLSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats)
            .units_fn([](std::string const& in) { return static_cast<double>(in.size()); })
            .finalize([&out]() { return pack_sequence_validity_sink(out); });
        auto results = suite.run(
            sequences,
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_pext(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "pext_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_pext_sse2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "pext_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_pext_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "pext_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_pext_avx512(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_SSE2)
            bench(
                "butterfly_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_butterfly_sse2(seq, out);
                }
            ),
            bench(
                "fold_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_fold_sse2(seq, out);
                }
            ),
            bench(
                "direct_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_direct_sse2(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSSE3)
            bench(
                "direct_double_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_direct_double_sse2(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "butterfly_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_butterfly_avx2(seq, out);
                }
            ),
            bench(
                "fold_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_fold_avx2(seq, out);
                }
            ),
            bench(
                "direct_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_direct_avx2(seq, out);
                }
            ),
            bench(
                "direct_double_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_direct_double_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "butterfly_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_butterfly_avx512(seq, out);
                }
            ),
            bench(
                "fold_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_fold_avx512(seq, out);
                }
            ),
            bench(
                "direct_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_direct_avx512(seq, out);
                }
            ),
            bench(
                "direct_double_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_direct_double_avx512(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "butterfly_neon",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_butterfly_neon(seq, out);
                }
            ),
            bench(
                "fold_neon",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_fold_neon(seq, out);
                }
            ),
            bench(
                "direct_neon",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_direct_neon(seq, out);
                }
            ),
            bench(
                "direct_double_neon",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_direct_double_neon(seq, out);
                }
            ),
            #endif
            bench(
                "fold_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_fold_scalar(seq, out);
                }
            ),
            bench(
                "direct_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_direct_scalar(seq, out);
                }
            ),
            bench(
                "butterfly_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_actg_lsb_butterfly_scalar(seq, out);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg;layout=lsb", results);
    }

    // -----------------------------------------------------------------------
    //     encoding=actg;layout=msb
    // -----------------------------------------------------------------------
    {
        PackedSequenceWithValidity<Encoding::kACTG, Layout::kMSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats)
            .units_fn([](std::string const& in) { return static_cast<double>(in.size()); })
            .finalize([&out]() { return pack_sequence_validity_sink(out); });
        auto results = suite.run(
            sequences,
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_pext(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSE2)
            bench(
                "pext_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_pext_sse2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "pext_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_pext_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "pext_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_pext_avx512(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_SSE2)
            bench(
                "butterfly_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_butterfly_sse2(seq, out);
                }
            ),
            bench(
                "fold_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_fold_sse2(seq, out);
                }
            ),
            bench(
                "direct_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_direct_sse2(seq, out);
                }
            ),
            #if defined(FISK_HAS_SSSE3)
            bench(
                "direct_double_sse2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_direct_double_sse2(seq, out);
                }
            ),
            #endif
            #endif
            #if defined(FISK_HAS_AVX2)
            bench(
                "butterfly_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_butterfly_avx2(seq, out);
                }
            ),
            bench(
                "fold_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_fold_avx2(seq, out);
                }
            ),
            bench(
                "direct_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_direct_avx2(seq, out);
                }
            ),
            bench(
                "direct_double_avx2",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_direct_double_avx2(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_AVX512)
            bench(
                "butterfly_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_butterfly_avx512(seq, out);
                }
            ),
            bench(
                "fold_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_fold_avx512(seq, out);
                }
            ),
            bench(
                "direct_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_direct_avx512(seq, out);
                }
            ),
            bench(
                "direct_double_avx512",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_direct_double_avx512(seq, out);
                }
            ),
            #endif
            #if defined(FISK_HAS_NEON)
            bench(
                "butterfly_neon",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_butterfly_neon(seq, out);
                }
            ),
            bench(
                "fold_neon",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_fold_neon(seq, out);
                }
            ),
            bench(
                "direct_neon",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_direct_neon(seq, out);
                }
            ),
            bench(
                "direct_double_neon",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_direct_double_neon(seq, out);
                }
            ),
            #endif
            bench(
                "fold_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_fold_scalar(seq, out);
                }
            ),
            bench(
                "direct_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_direct_scalar(seq, out);
                }
            ),
            bench(
                "butterfly_scalar",
                [&](std::string const& seq) {
                    return run_var_validity_actg_msb_butterfly_scalar(seq, out);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg;layout=msb", results);
    }
}
