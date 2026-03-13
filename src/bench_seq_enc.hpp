#pragma once

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "seq_enc.hpp"
#include "microbench.hpp"

/**
 * @brief Benchmark different implementations for encoding ASCII chars into the two bit encoding.
 *
 * This tests both variants of the implementations, those that check that the character is valid
 * in `ACGT`, and those that assume it is. The former will usually be more important in practice
 * on input data, while the latter might be used internally after parsing has already been done.
 *
 * The idea to test both is that the extra check as well as the exception thrown might cause the
 * compiler to emit different code, and in particular not be able to inline those functions.
 * Hence, we benchmark them all here, to see the effects of this.
 */
inline void bench_seq_enc(std::vector<std::string> const& sequences, std::ostream& csv_os)
{
    std::size_t const rounds = 8;
    std::size_t const repeats = 16;

    // User output
    std::string const suite_title = "seq_encode";
    std::cout << "\n=== sequence encode ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Quick verification check of the ascii exploit:
    test_char_to_nt_ascii();

    Microbench<std::string> suite(suite_title);
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
            "char_to_nt_ifs",
            [&](std::string const& seq){ return sequence_encode(seq, char_to_nt_ifs);
        }),
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

        // The unchecked ascii encoder is the fastest, but only valid if it is guaranteed
        // that the input only consists of ACGT characters.
        // bench(
        //     "char_to_nt_ascii_unchecked",
        //     [&](std::string const& seq){ return sequence_encode(seq, char_to_nt_ascii_unchecked);
        // })
    );

    std::string const case_label = "n/a";
    write_csv_header(csv_os);
    write_csv_rows(csv_os, suite_title, case_label, results);
}
