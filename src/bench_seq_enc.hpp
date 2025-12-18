#pragma once

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "seq_enc.hpp"
#include "microbench.hpp"

inline void bench_seq_enc(std::vector<std::string> const& sequences, std::ostream& csv_os)
{
    std::size_t const rounds = 8;
    std::size_t const repeats = 16;

    // User output
    std::cout << "\n=== sequence encode ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    Microbench<std::string> suite("encode_2bit");
    suite
        .rounds(rounds)
        .repeats(repeats)
        .units_fn([](std::string const& in) {
            // 1 unit per base
            return static_cast<double>(in.size());
        });

    auto results = suite.run(
        sequences, // vector<std::string>
        bench(
            "char_to_nt_switch",
            [&](std::string const& seq){ return sequence_encode(seq, char_to_nt_switch);
        }),
        bench(
            "char_to_nt_table",
            [&](std::string const& seq){ return sequence_encode(seq, char_to_nt_table);
        }),
        bench(
            "char_to_nt_ascii",
            [&](std::string const& seq){ return sequence_encode(seq, char_to_nt_ascii);
        })
    );

    std::string const case_label = "n/a";
    write_csv_header(csv_os);
    write_csv_rows(csv_os, "seq_encode", case_label, results);
}
