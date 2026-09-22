#include <iostream>
#include <string>
#include <vector>

#include "char_encoder/bench.hpp"
#include "microbench.hpp"

void bench_char_encoder(std::vector<std::string> const& sequences, std::ostream& csv_os)
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
            bench("ifs", run_var_acgt_ifs),
            bench("switch", run_var_acgt_switch),
            bench("table", run_var_acgt_table),
            bench("ascii_validate", run_var_acgt_ascii_validate)
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt", results);

        // ascii_assume_valid runs in its own suite.run() call, not alongside the checked encoders
        // above: Microbench::run() cross-validates that every bench in the same call produces
        // the same sink, but ascii_assume_valid has no validity check by design (that's its whole
        // speed advantage) and so does not agree with the others on invalid characters -- and
        // `sequences` here may contain a small fraction of those (see --n-prob in main.cpp).
        auto const results_assume_valid = suite.run(
            sequences,
            bench("ascii_assume_valid", run_var_acgt_ascii_assume_valid)
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
            bench("ifs", run_var_actg_ifs),
            bench("switch", run_var_actg_switch),
            bench("table", run_var_actg_table),
            bench("ascii_validate", run_var_actg_ascii_validate)
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg", results);

        // See the matching comment in the encoding=acgt block above for why ascii_assume_valid
        // runs in its own suite.run() call rather than alongside the checked encoders.
        auto const results_assume_valid = suite.run(
            sequences,
            bench("ascii_assume_valid", run_var_actg_ascii_assume_valid)
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg", results_assume_valid);
    }
}
