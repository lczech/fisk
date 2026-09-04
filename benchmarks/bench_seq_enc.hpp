#pragma once

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "fisk/core/seq_enc.hpp"
#include "microbench.hpp"

/**
 * @brief Scan a sequence and encode each character, combining them to get a final "hash".
 *
 * The hash obtained here is not a good one, as it is simply the sum of all two-bit encodings
 * of the characters. But it is enough to check that all the above functions give the same result,
 * and sufficient to force the compiler to actually run the encoding.
 */
template <typename EncodeFunc>
inline std::uint64_t sequence_encode_hash(std::string_view seq, EncodeFunc&& encode)
{
    std::uint64_t h = 0;
    for (char c : seq) {
        // h = (h << 2) | encode(c);
        h += encode(c);
    }
    return h;
}

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
            "char_to_nt_ifs_acgt",
            [&](std::string const& seq){ return sequence_encode_hash(seq, char_to_nt_ifs_acgt);
        }),
        bench(
            "char_to_nt_switch_acgt",
            [&](std::string const& seq){ return sequence_encode_hash(seq, char_to_nt_switch_acgt);
        }),
        bench(
            "char_to_nt_table_acgt",
            [&](std::string const& seq){ return sequence_encode_hash(seq, char_to_nt_table_acgt);
        }),
        bench(
            "char_to_nt_ascii_acgt",
            [&](std::string const& seq){ return sequence_encode_hash(seq, char_to_nt_ascii_acgt);
        })

        // The unchecked ascii encoder is the fastest, but only valid if it is guaranteed
        // that the input only consists of ACGT characters.
        // bench(
        //     "char_to_nt_ascii_unchecked_acgt",
        //     [&](std::string const& seq){ return sequence_encode_hash(seq, char_to_nt_ascii_unchecked_acgt);
        // })
    );

    std::string const case_label = "n/a";
    write_csv_header(csv_os);
    write_csv_rows(csv_os, suite_title, case_label, results);
}
