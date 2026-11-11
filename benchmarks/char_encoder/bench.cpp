#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "char_encoder/bench.hpp"
#include "microbench.hpp"

void bench_char_encoder(std::vector<InputSequence> const& sequences, std::ostream& csv_os)
{
    // User output
    std::string const suite_title = "char_encoder";
    std::cout << "\n=== char encoder ===\n";

    write_csv_header(csv_os);

    // The buffer of the Write sink, see sink.hpp; Sum and Barrier ignore it.
    auto sink_buffer = make_sink_buffer();

    // -----------------------------------------------------------------------
    //     encoding=acgt
    // -----------------------------------------------------------------------
    {
        Microbench<InputSequence> suite(suite_title);
        suite
            .units_fn([](InputSequence const& in) {
                // 1 unit per base
                return static_cast<double>(in.size());
            });

        auto results = suite.run(
            sequences, // vector<std::string>
            bench(
                "ifs",
                [&](InputSequence const& seq){
                    return run_var_ifs_acgt(seq, sink_buffer);
                }
            ),
            bench(
                "switch",
                [&](InputSequence const& seq){
                    return run_var_switch_acgt(seq, sink_buffer);
                }
            ),
            bench(
                "table",
                [&](InputSequence const& seq){
                    return run_var_table_acgt(seq, sink_buffer);
                }
            ),
            bench(
                "ascii_checked",
                [&](InputSequence const& seq){
                    return run_var_ascii_checked_acgt(seq, sink_buffer);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt", results, kSinkName);

        // ascii_unchecked runs in its own suite.run() call, not alongside the checked encoders
        // above: Microbench::run() cross-validates that every bench in the same call produces
        // the same sink, but ascii_unchecked has no validity check by design (that's its whole
        // speed advantage) and so does not agree with the others on invalid characters -- and
        // `sequences` here may contain a small fraction of those (see --n-prob in main.cpp).
        auto const results_unchecked = suite.run(
            sequences,
            bench(
                "ascii_unchecked",
                [&](InputSequence const& seq){
                    return run_var_ascii_unchecked_acgt(seq, sink_buffer);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt", results_unchecked, kSinkName);
    }

    // -----------------------------------------------------------------------
    //     encoding=actg
    // -----------------------------------------------------------------------
    {
        Microbench<InputSequence> suite(suite_title);
        suite
            .units_fn([](InputSequence const& in) {
                // 1 unit per base
                return static_cast<double>(in.size());
            });

        auto results = suite.run(
            sequences, // vector<std::string>
            bench(
                "ifs",
                [&](InputSequence const& seq){
                    return run_var_ifs_actg(seq, sink_buffer);
                }
            ),
            bench(
                "switch",
                [&](InputSequence const& seq){
                    return run_var_switch_actg(seq, sink_buffer);
                }
            ),
            bench(
                "table",
                [&](InputSequence const& seq){
                    return run_var_table_actg(seq, sink_buffer);
                }
            ),
            bench(
                "ascii_checked",
                [&](InputSequence const& seq){
                    return run_var_ascii_checked_actg(seq, sink_buffer);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg", results, kSinkName);

        // See the matching comment in the encoding=acgt block above for why ascii_unchecked
        // runs in its own suite.run() call rather than alongside the checked encoders.
        auto const results_unchecked = suite.run(
            sequences,
            bench(
                "ascii_unchecked",
                [&](InputSequence const& seq){
                    return run_var_ascii_unchecked_actg(seq, sink_buffer);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg", results_unchecked, kSinkName);
    }
}
