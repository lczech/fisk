#pragma once

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/core/intrinsics.hpp"
#include "microbench.hpp"

using namespace fisk;

/**
 * @brief Scan a sequence and encode each character, combining them to get a final "hash".
 *
 * The hash obtained here is not a good one, as it is simply the sum of all two-bit encodings
 * of the characters. But it is enough to check that all the above functions give the same result,
 * and sufficient to force the compiler to actually run the loop.
 *
 * With `Barrier == false` (the default), the compiler is still free to vectorize the loop, which
 * might or might not reflect actual usage. With `Barrier == true`, each character's freshly
 * encoded value is forced into a register via do_not_optimize() before being folded into the
 * hash, blocking that vectorization and isolating the true per-character scalar cost of `encoder`.
 */
template <bool Barrier = false, typename Encoder>
inline std::uint64_t sequence_encode_hash(std::string_view seq, Encoder&& encoder)
{
    std::uint64_t h = 0;
    for (char c : seq) {
        auto const code = encoder(c);
        if constexpr (Barrier) {
            do_not_optimize(code);
        }
        // h = (h << 2) | code;
        h += code;
    }
    return h;
}

/**
 * @brief Benchmark different implementations for encoding ASCII chars into the two bit encoding.
 *
 * This tests both variants of the implementations, those that check that the character is valid
 * in the encoding, and those that assume it is. The former will usually be more important in
 * practice on input data, while the latter might be used internally after parsing has already
 * been done.
 *
 * The idea to test both is that the extra check as well as the exception thrown might cause the
 * compiler to emit different code, and in particular not be able to inline those functions.
 * Hence, we benchmark them all here, to see the effects of this. Each technique is also
 * benchmarked with a do_not_optimize barrier (see sequence_encode_hash()) to isolate its true
 * scalar per-character cost from whatever the compiler manages to auto-vectorize away, and for
 * both encodings (ACGT and ACTG), one suite each.
 */
inline void bench_char_encoder(std::vector<std::string> const& sequences, std::ostream& csv_os)
{
    std::size_t const rounds = 8;
    std::size_t const repeats = 16;

    // User output
    std::string const suite_title = "char_encoder";
    std::cout << "\n=== char encoder ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    // -----------------------------------------------------------------------
    //     encoding=acgt
    // -----------------------------------------------------------------------
    {
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

            bench("ifs", [&](std::string const& seq) {
                return sequence_encode_hash(seq, CharEncoderIfs<Encoding::kACGT>{});
            }),
            bench("ifs_barrier", [&](std::string const& seq) {
                return sequence_encode_hash<true>(seq, CharEncoderIfs<Encoding::kACGT>{});
            }),
            bench("switch", [&](std::string const& seq) {
                return sequence_encode_hash(seq, CharEncoderSwitch<Encoding::kACGT>{});
            }),
            bench("switch_barrier", [&](std::string const& seq) {
                return sequence_encode_hash<true>(seq, CharEncoderSwitch<Encoding::kACGT>{});
            }),
            bench("table", [&](std::string const& seq) {
                return sequence_encode_hash(seq, CharEncoderTable<Encoding::kACGT>{});
            }),
            bench("table_barrier", [&](std::string const& seq) {
                return sequence_encode_hash<true>(seq, CharEncoderTable<Encoding::kACGT>{});
            }),
            bench("ascii", [&](std::string const& seq) {
                return sequence_encode_hash(seq, CharEncoderAscii<Encoding::kACGT>{});
            }),
            bench("ascii_barrier", [&](std::string const& seq) {
                return sequence_encode_hash<true>(seq, CharEncoderAscii<Encoding::kACGT>{});
            })
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt", results);

        // ascii_assume_valid runs in its own suite.run() call, not alongside the checked encoders
        // above: Microbench::run() cross-validates that every bench in the same call produces
        // the same sink, but ascii_assume_valid has no validity check by design (that's its whole
        // speed advantage) and so does not agree with the others on invalid characters -- and
        // `sequences` here may contain a small fraction of those (see --n-prob in main.cpp).
        // Splitting it off still benchmarks it and still cross-validates barrier vs. non-barrier
        // against each other, just not against the checked techniques' different handling of
        // invalid input.
        auto const results_assume_valid = suite.run(
            sequences,
            bench("ascii_assume_valid", [&](std::string const& seq) {
                return sequence_encode_hash(
                    seq, CharEncoderAscii<Encoding::kACGT, InputValidity::kAssumeValid>{}
                );
            }),
            bench("ascii_assume_valid_barrier", [&](std::string const& seq) {
                return sequence_encode_hash<true>(
                    seq, CharEncoderAscii<Encoding::kACGT, InputValidity::kAssumeValid>{}
                );
            })
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt", results_assume_valid);
    }

    // -----------------------------------------------------------------------
    //     encoding=actg
    // -----------------------------------------------------------------------
    {
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

            bench("ifs", [&](std::string const& seq) {
                return sequence_encode_hash(seq, CharEncoderIfs<Encoding::kACTG>{});
            }),
            bench("ifs_barrier", [&](std::string const& seq) {
                return sequence_encode_hash<true>(seq, CharEncoderIfs<Encoding::kACTG>{});
            }),
            bench("switch", [&](std::string const& seq) {
                return sequence_encode_hash(seq, CharEncoderSwitch<Encoding::kACTG>{});
            }),
            bench("switch_barrier", [&](std::string const& seq) {
                return sequence_encode_hash<true>(seq, CharEncoderSwitch<Encoding::kACTG>{});
            }),
            bench("table", [&](std::string const& seq) {
                return sequence_encode_hash(seq, CharEncoderTable<Encoding::kACTG>{});
            }),
            bench("table_barrier", [&](std::string const& seq) {
                return sequence_encode_hash<true>(seq, CharEncoderTable<Encoding::kACTG>{});
            }),
            bench("ascii", [&](std::string const& seq) {
                return sequence_encode_hash(seq, CharEncoderAscii<Encoding::kACTG>{});
            }),
            bench("ascii_barrier", [&](std::string const& seq) {
                return sequence_encode_hash<true>(seq, CharEncoderAscii<Encoding::kACTG>{});
            })
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg", results);

        // See the matching comment in the encoding=acgt block above for why ascii_assume_valid
        // runs in its own suite.run() call rather than alongside the checked encoders.
        auto const results_assume_valid = suite.run(
            sequences,
            bench("ascii_assume_valid", [&](std::string const& seq) {
                return sequence_encode_hash(
                    seq, CharEncoderAscii<Encoding::kACTG, InputValidity::kAssumeValid>{}
                );
            }),
            bench("ascii_assume_valid_barrier", [&](std::string const& seq) {
                return sequence_encode_hash<true>(
                    seq, CharEncoderAscii<Encoding::kACTG, InputValidity::kAssumeValid>{}
                );
            })
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg", results_assume_valid);
    }
}
