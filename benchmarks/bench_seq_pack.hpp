#pragma once

#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/core/seq_pack.hpp"
#include "microbench.hpp"

/**
 * @brief Simple order-insensitive checksum over a packed TwoBitSequence.
 *
 * Sufficient for cross-validation within one convention/order group (bit match expected), and to
 * keep the compiler from optimizing calls away. This is deliberately not order-sensitive:
 * that would mean adding a multiply or the like into the hot loop; too expensive.
 */
template <BitOrder Order>
inline std::uint64_t pack_sequence_sink(TwoBitSequence<Order> const& s)
{
    std::uint64_t h = 0;
    for (auto const w : s.data) {
        h += w;
    }
    return h;
}

/**
 * @brief Benchmark pack_sequence() across all four convention x bit-order combinations.
 *
 * Each combination (conv=actg/acgt, order=lsb/msb) runs as its own Microbench suite, never mixed
 * with another: within one suite, PEXT and butterfly-table candidates are supposed to agree
 * bit-for-bit (same convention, same order), which is exactly what the sink cross-validation
 * checks.
 *
 * One TwoBitSequence per group is constructed outside the timed calls and reused across all
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
    //     conv=acgt;order=lsb
    // -----------------------------------------------------------------------
    {
        TwoBitSequence<BitOrder::Lsb> out;
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
        );
        write_csv_rows(csv_os, suite_title, "conv=acgt;order=lsb", results);
    }

    // -----------------------------------------------------------------------
    //     conv=acgt;order=msb
    // -----------------------------------------------------------------------
    {
        TwoBitSequence<BitOrder::Msb> out;
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
        );
        write_csv_rows(csv_os, suite_title, "conv=acgt;order=msb", results);
    }

    // -----------------------------------------------------------------------
    //     conv=actg;order=lsb
    // -----------------------------------------------------------------------
    {
        TwoBitSequence<BitOrder::Lsb> out;
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
        );
        write_csv_rows(csv_os, suite_title, "conv=actg;order=lsb", results);
    }

    // -----------------------------------------------------------------------
    //     conv=actg;order=msb
    // -----------------------------------------------------------------------
    {
        TwoBitSequence<BitOrder::Msb> out;
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
        );
        write_csv_rows(csv_os, suite_title, "conv=actg;order=msb", results);
    }
}
