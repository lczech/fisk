#pragma once

#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/core/types.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "fisk/seq_pack/simd.hpp"
#include "microbench.hpp"

/**
 * @brief Simple order-insensitive checksum over a PackedSequence.
 *
 * Sufficient for cross-validation within one encoding/bit-order group (bit match expected), and to
 * keep the compiler from optimizing calls away. This is deliberately not order-sensitive:
 * that would mean adding a multiply or the like into the hot loop; too expensive.
 */
template <Encoding E, Layout L>
inline std::uint64_t pack_sequence_sink(PackedSequence<E, L> const& s)
{
    std::uint64_t h = 0;
    for (auto const w : s.data) {
        h += w;
    }
    return h;
}

/**
 * @brief Benchmark pack_sequence() across all four encoding x bit-order combinations.
 *
 * Each combination (encoding=actg/acgt, layout=lsb/msb) runs as its own Microbench suite, never
 * mixed with another: within one suite, PEXT and butterfly-table candidates are supposed to agree
 * bit-for-bit (same encoding, same bit order), which is exactly what the sink cross-validation
 * checks.
 *
 * One PackedSequence per group is constructed outside the timed calls and reused across all
 * rounds/repeats, avoiding repeated heap allocation during measured.
 */
inline void bench_seq_pack(std::vector<std::string> const& sequences, std::ostream& csv_os)
{
    std::size_t const rounds  = 8;
    std::size_t const repeats = 16;

    std::string const suite_title = "seq_pack";
    std::cout << "\n=== sequence pack ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    // -----------------------------------------------------------------------
    //     encoding=acgt;layout=lsb
    // -----------------------------------------------------------------------
    {
        PackedSequence<Encoding::kACGT, Layout::kLSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [](std::string const& in) { return static_cast<double>(in.size()); }
        );
        auto results = suite.run(
            sequences,
            #if defined(FISK_HAS_BMI2)
            bench("pext", [&](std::string const& seq) {
                pack_sequence(seq, EncodeAcgt8PextLsb{}, out);
                return pack_sequence_sink(out);
            }),
            #endif
            bench("butterfly", [&](std::string const& seq) {
                pack_sequence(seq, EncodeAcgt8ButterflyLsb{}, out);
                return pack_sequence_sink(out);
            })
            #if defined(FISK_HAS_SSE2)
            ,
            bench("butterfly_sse2", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeAcgtButterflySse2Lsb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_AVX2)
            ,
            bench("butterfly_avx2", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeAcgtButterflyAvx2Lsb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_AVX512)
            ,
            bench("butterfly_avx512", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeAcgtButterflyAvx512Lsb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_NEON)
            ,
            bench("butterfly_neon", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeAcgtButterflyNeonLsb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt;layout=lsb", results);
    }

    // -----------------------------------------------------------------------
    //     encoding=acgt;layout=msb
    // -----------------------------------------------------------------------
    {
        PackedSequence<Encoding::kACGT, Layout::kMSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [](std::string const& in) { return static_cast<double>(in.size()); }
        );
        auto results = suite.run(
            sequences,
            #if defined(FISK_HAS_BMI2)
            bench("pext", [&](std::string const& seq) {
                pack_sequence(seq, EncodeAcgt8PextMsb{}, out);
                return pack_sequence_sink(out);
            }),
            #endif
            bench("butterfly", [&](std::string const& seq) {
                pack_sequence(seq, EncodeAcgt8ButterflyMsb{}, out);
                return pack_sequence_sink(out);
            })
            #if defined(FISK_HAS_SSE2)
            ,
            bench("butterfly_sse2", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeAcgtButterflySse2Msb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_AVX2)
            ,
            bench("butterfly_avx2", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeAcgtButterflyAvx2Msb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_AVX512)
            ,
            bench("butterfly_avx512", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeAcgtButterflyAvx512Msb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_NEON)
            ,
            bench("butterfly_neon", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeAcgtButterflyNeonMsb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt;layout=msb", results);
    }

    // -----------------------------------------------------------------------
    //     encoding=actg;layout=lsb
    // -----------------------------------------------------------------------
    {
        PackedSequence<Encoding::kACTG, Layout::kLSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [](std::string const& in) { return static_cast<double>(in.size()); }
        );
        auto results = suite.run(
            sequences,
            #if defined(FISK_HAS_BMI2)
            bench("pext", [&](std::string const& seq) {
                pack_sequence(seq, EncodeActg8PextLsb{}, out);
                return pack_sequence_sink(out);
            }),
            #endif
            bench("butterfly", [&](std::string const& seq) {
                pack_sequence(seq, EncodeActg8ButterflyLsb{}, out);
                return pack_sequence_sink(out);
            })
            #if defined(FISK_HAS_SSE2)
            ,
            bench("butterfly_sse2", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeActgButterflySse2Lsb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_AVX2)
            ,
            bench("butterfly_avx2", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeActgButterflyAvx2Lsb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_AVX512)
            ,
            bench("butterfly_avx512", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeActgButterflyAvx512Lsb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_NEON)
            ,
            bench("butterfly_neon", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeActgButterflyNeonLsb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg;layout=lsb", results);
    }

    // -----------------------------------------------------------------------
    //     encoding=actg;layout=msb
    // -----------------------------------------------------------------------
    {
        PackedSequence<Encoding::kACTG, Layout::kMSB> out;
        Microbench<std::string> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [](std::string const& in) { return static_cast<double>(in.size()); }
        );
        auto results = suite.run(
            sequences,
            #if defined(FISK_HAS_BMI2)
            bench("pext", [&](std::string const& seq) {
                pack_sequence(seq, EncodeActg8PextMsb{}, out);
                return pack_sequence_sink(out);
            }),
            #endif
            bench("butterfly", [&](std::string const& seq) {
                pack_sequence(seq, EncodeActg8ButterflyMsb{}, out);
                return pack_sequence_sink(out);
            })
            #if defined(FISK_HAS_SSE2)
            ,
            bench("butterfly_sse2", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeActgButterflySse2Msb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_AVX2)
            ,
            bench("butterfly_avx2", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeActgButterflyAvx2Msb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_AVX512)
            ,
            bench("butterfly_avx512", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeActgButterflyAvx512Msb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
            #if defined(FISK_HAS_NEON)
            ,
            bench("butterfly_neon", [&](std::string const& seq) {
                pack_sequence_simd(seq, EncodeActgButterflyNeonMsb{}, out);
                return pack_sequence_sink(out);
            })
            #endif
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg;layout=msb", results);
    }
}
