#include <algorithm>
#include <bit>
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

    // Backing storage for the Write sink (see sink.hpp), sized to the longest sequence actually
    // benchmarked (rounded up to a power of two) so that no call here ever wraps it. One call
    // sweeps a whole sequence's worth of bases, and every technique here writes exactly one value
    // per base, so this is also the exact touch count, not just a safe upper bound -- matching
    // bit_extract's own exact-fit buffers in spirit, just sized from the actual input instead of
    // a fixed batch constant.
    std::size_t max_seq_len = 0;
    for (auto const& seq : sequences) {
        max_seq_len = std::max(max_seq_len, seq.size());
    }
    std::vector<std::uint64_t> sink_buffer(
        std::bit_ceil(std::max<std::size_t>(max_seq_len, 1)), 0
    );

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
            bench(
                "ifs",
                [&](std::string const& seq){
                    return run_var_acgt_ifs(seq, sink_buffer);
                }
            ),
            bench(
                "switch",
                [&](std::string const& seq){
                    return run_var_acgt_switch(seq, sink_buffer);
                }
            ),
            bench(
                "table",
                [&](std::string const& seq){
                    return run_var_acgt_table(seq, sink_buffer);
                }
            ),
            bench(
                "ascii_validate",
                [&](std::string const& seq){
                    return run_var_acgt_ascii_validate(seq, sink_buffer);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt", results, kSinkName);

        // ascii_assume_valid runs in its own suite.run() call, not alongside the checked encoders
        // above: Microbench::run() cross-validates that every bench in the same call produces
        // the same sink, but ascii_assume_valid has no validity check by design (that's its whole
        // speed advantage) and so does not agree with the others on invalid characters -- and
        // `sequences` here may contain a small fraction of those (see --n-prob in main.cpp).
        auto const results_assume_valid = suite.run(
            sequences,
            bench(
                "ascii_assume_valid",
                [&](std::string const& seq){
                    return run_var_acgt_ascii_assume_valid(seq, sink_buffer);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=acgt", results_assume_valid, kSinkName);
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
            bench(
                "ifs",
                [&](std::string const& seq){
                    return run_var_actg_ifs(seq, sink_buffer);
                }
            ),
            bench(
                "switch",
                [&](std::string const& seq){
                    return run_var_actg_switch(seq, sink_buffer);
                }
            ),
            bench(
                "table",
                [&](std::string const& seq){
                    return run_var_actg_table(seq, sink_buffer);
                }
            ),
            bench(
                "ascii_validate",
                [&](std::string const& seq){
                    return run_var_actg_ascii_validate(seq, sink_buffer);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg", results, kSinkName);

        // See the matching comment in the encoding=acgt block above for why ascii_assume_valid
        // runs in its own suite.run() call rather than alongside the checked encoders.
        auto const results_assume_valid = suite.run(
            sequences,
            bench(
                "ascii_assume_valid",
                [&](std::string const& seq){
                    return run_var_actg_ascii_assume_valid(seq, sink_buffer);
                }
            )
        );
        write_csv_rows(csv_os, suite_title, "encoding=actg", results_assume_valid, kSinkName);
    }
}
