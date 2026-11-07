// =================================================================================================
//     Benchmark Suite Structure
// =================================================================================================
//
// Each compared implementation variant lives in its own translation unit: benchmarks/<suite>/
// var_<name>.cpp defines exactly one run_var_<name>() and nothing else that could pull unrelated
// code into the same TU. This keeps one variant's compiler codegen (inlining, register
// allocation, layout) from being perturbed by changes to a completely unrelated variant that just
// happens to share a file with it.
//
// This isolation is set up and enforced at the build level, not by convention alone: see
// CMakeLists.txt's `fisk_benchmarks` target, which globs every benchmarks/*.cpp file into one
// executable (so a new var_*.cpp needs no manual registration) and explicitly turns
// INTERPROCEDURAL_OPTIMIZATION off (LTO/IPO would otherwise let the linker re-fuse all those TUs
// back into one optimization unit, silently defeating the point). There is currently no automated
// check that a var_*.cpp only ever defines one run_var_*().
//
// See sink.hpp for the separate (complementary) per-call measurement strategy (Sum/Barrier/Write)
// that every run_var_*() reduces its result through.

#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <stdexcept>
#include <filesystem>

#include "arg_parser.hpp"
#include "extractor_scaling/bench.hpp"
#include "kmer_extract_ascii_checked/bench.hpp"
#include "kmer_extract_ascii_unchecked/bench.hpp"
#include "kmer_extract_packed_unchecked/bench.hpp"
#include "kmer_extract_packed_checked/bench.hpp"
#include "spaced_kmer_extract_ascii_checked_single/bench.hpp"
#include "spaced_kmer_extract_ascii_checked_multi/bench.hpp"
#include "spaced_kmer_extract_ascii_unchecked_single/bench.hpp"
#include "spaced_kmer_extract_ascii_unchecked_multi/bench.hpp"
#include "spaced_kmer_extract_packed_unchecked_single/bench.hpp"
#include "spaced_kmer_extract_packed_unchecked_multi/bench.hpp"
#include "spaced_kmer_extract_packed_checked_single/bench.hpp"
#include "spaced_kmer_extract_packed_checked_multi/bench.hpp"
#include "spaced_kmer_extract_packed_unchecked_spans/bench.hpp"
#include "spaced_kmer_extract_packed_checked_spans/bench.hpp"
#include "spaced_kmer_extract_ascii_unchecked_spans/bench.hpp"
#include "spaced_kmer_extract_ascii_checked_spans/bench.hpp"
#include "kmer_clark/bench.hpp"
#include "bit_extract_blocks/bench.hpp"
#include "bit_extract_weights/bench.hpp"
#include "char_encoder/bench.hpp"
#include "sequence_pack_unchecked/bench.hpp"
#include "sequence_pack_checked/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"
#include "seq_data.hpp"
#include "fisk/util/cpu_runtime.hpp"

using namespace fisk;

struct Options
{
    // Input fasta file with sequence data
    std::string input_file;

    // Input sequence length to randomly generate
    std::string input_length;

    // Probability to randomly generate an N base instead of ACGT
    std::string n_prob = "0.001";

    // Seed of the random input, fixed so that every machine measures the same input
    std::string seed = "42";

    // Timing settings, see MicrobenchSettings
    std::string sample_ms = "10";
    std::string warmup_ms = "1";
    std::string repeats = "5";

    // Value of k for the k-mers
    std::string k;

    // File with masks
    std::string masks_file;

    // Largest input length of the extractor scaling, 4 Gi bases
    std::string scaling_max_length = "4294967296";

    // Output directory for benchmark results
    std::string output_dir = "results";
};

int main(int argc, char **argv)
{
    // Set up command line parsing
    Options opts;
    ArgParser parser(argv[0]);

    // ------------------------------------------------------------------------
    //     Declare command line options
    // ------------------------------------------------------------------------

    parser.add_option(
        "--input-fasta", "-i",
        "Input fasta file with sequence data. Excludes --input-length",
        opts.input_file
    );

    parser.add_option(
        "--input-length", "-l",
        "Input length to randomly geneate a sequence of ACGT. Excludes --input-fasta",
        opts.input_length
    );

    parser.add_option(
        "--n-prob", "-p",
        "Probability to randomly generate an N base instead of ACGT",
        opts.n_prob
    );

    parser.add_option(
        "--seed", "-s",
        "Seed of the random input sequence (default: 42)",
        opts.seed
    );

    parser.add_option(
        "--sample-ms", "",
        "Time of each measurement in milliseconds, as whole passes over the input (default: 10)",
        opts.sample_ms
    );

    parser.add_option(
        "--warmup-ms", "",
        "Minimum warm-up before each measurement in milliseconds (default: 1)",
        opts.warmup_ms
    );

    parser.add_option(
        "--repeats", "",
        "Number of measurements per benchmark; the CSVs report their minimum, median, and mean "
        "(default: 5)",
        opts.repeats
    );

    parser.add_option(
        "--k", "-k",
        "Value of k to use for the k-mers",
        opts.k
    );

    parser.add_option(
        "--masks-file", "-m",
        "File path with masks, given as sequences of [01], one per line",
        opts.masks_file
    );

    parser.add_option(
        "--scaling-max-length", "",
        "Largest input length of the extractor scaling, a random sequence that is in memory as a "
        "whole; 0 to skip it (default: 4294967296, i.e., 4 Gi bases)",
        opts.scaling_max_length
    );

    parser.add_option(
        "--output-dir", "-o",
        "Output directory for benchmark CSV files (default: results)",
        opts.output_dir
    );

    // ------------------------------------------------------------------------
    //     Parse args and setup
    // ------------------------------------------------------------------------

    // Run our simple arg parser.
    try {
        parser.parse(argc, argv);
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Use --help for usage.\n";
        return 1;
    }

    // Timing settings for all suites.
    microbench_settings.sample_seconds = std::stod(opts.sample_ms) / 1000.0;
    microbench_settings.warmup_seconds = std::stod(opts.warmup_ms) / 1000.0;
    microbench_settings.repeats = std::stoul(opts.repeats);
    if(
        !(microbench_settings.sample_seconds > 0.0) || !(microbench_settings.warmup_seconds >= 0.0)
        || microbench_settings.repeats == 0
    ) {
        throw std::invalid_argument(
            "Options --sample-ms and --repeats have to be > 0, and --warmup-ms >= 0."
        );
    }

    // Get the sequence to process, either random, or from fasta.
    // The input description goes into sys_info.txt along with the timing settings.
    std::vector<std::string> sequences;
    std::string input_info;
    if (opts.input_file.size() && opts.input_length.size()) {
        throw std::invalid_argument(
            "Options --input-length and --input-fasta are mutually exclusive."
        );
    }
    if (opts.input_file.size()) {
        std::cout << "Reading input file " << opts.input_file << "\n";
        sequences = load_fasta_clean( opts.input_file );
        size_t sum = 0;
        for( auto const& seq : sequences ) {
            sum += seq.size();
        }
        std::cout << "Input file with " << sequences.size() << " sequences";
        std::cout << " and " << sum << " total nucleotides\n";
        input_info = "file " + opts.input_file + ", " + std::to_string(sequences.size())
            + " sequences, " + std::to_string(sum) + " nucleotides";
    } else {
        // By default 256 Ki bases, which fit the L2 cache of the CPUs we benchmark, so that
        // the measurements show the algorithms rather than memory throughput.
        size_t inp_len = 0;
        if(opts.input_length.size()) {
            inp_len = std::stoul(opts.input_length);
        } else {
            std::cout << "No input provided, using default\n";
            size_t const default_len = (1u << 18);
            inp_len = default_len;
        }
        auto const n_prob = std::stod(opts.n_prob);
        auto const seed = std::stoull(opts.seed);
        std::cout << "Generating input sequence of length " << inp_len;
        std::cout << " with N probability " << n_prob << " and seed " << seed << "\n";
        sequences = std::vector<std::string>{ random_acgt( inp_len, n_prob, seed )};
        input_info = "random, length " + std::to_string(inp_len) + ", N probability "
            + opts.n_prob + ", seed " + std::to_string(seed);
    }
    std::cout << "Measuring " << opts.sample_ms << " ms per sample after at least "
        << opts.warmup_ms << " ms of warm-up, " << opts.repeats << " repeats\n";

    // Get the value for k.
    size_t k = 0;
    if( opts.k.size() ) {
        k = std::stoul(opts.k);
        std::cout << "Using k=" << k << "\n";
        if( k < 1 || k > 32 ) {
            throw std::invalid_argument(
                "Option -k has to be in [1, 32]."
            );
        }
    } else {
        std::cout << "No k provided, testing all values\n";
    }

    // Get the masks file
    std::vector<std::string> single_masks;
    std::vector<std::vector<std::string>> multi_masks;
    if( opts.masks_file.size() ) {
        // Use the same set of masks in both cases
        single_masks = load_lines( opts.masks_file );
        multi_masks.push_back(single_masks);
    } else {
        // Masks to use by default.
        // The default.txt mask is identical to W22L31.txt
        std::string const single_mask_file = "default.txt";
        std::vector<std::string> multi_mask_files = {
            "W10L15.txt",
            "W14L31.txt",
            "W18L31.txt",
            "W22L31.txt",
            "W26L31.txt",
            "W32L45.txt"
        };

        // By default, we test single masks with the `default`, and multi across all mask sets,
        // one of which has a span above 32.
        std::cout << "No masks provided, using defaults\n";
        auto bin_path = parent_directory(argv[0]) / "../masks/";
        single_masks = load_lines( (bin_path / single_mask_file).string() );
        for( auto const& mmf : multi_mask_files ) {
            multi_masks.push_back(load_lines( (bin_path / mmf).string() ));
        }
    }
    std::cout << "Using " << single_masks.size() << " single masks\n";
    std::cout << "Using " << multi_masks.size() << " multi mask sets with {";
    for( auto const& masks : multi_masks ) {
        std::cout << " " << masks.size();
    }
    std::cout << " } masks\n";

    // Prepare output directory
    std::filesystem::path out_dir;
    try {
        out_dir = ensure_output_dir(opts.output_dir);
    } catch (std::exception const& e) {
        std::cerr << "Output directory error: " << e.what() << "\n";
        return 2;
    }

    // ------------------------------------------------------------------------
    //     Main
    // ------------------------------------------------------------------------

    // User output of system information
    {
        auto os_info = get_ofstream(out_dir, "sys_info.txt" );
        info_print_system(os_info);
        info_print_benchmark_build(os_info);
        info_print_power(os_info);
        os_info << "Benchmark run:\n";
        os_info << "  input      : " << input_info << "\n";
        os_info << "  sample     : " << opts.sample_ms << " ms\n";
        os_info << "  warm-up    : " << opts.warmup_ms << " ms\n";
        os_info << "  repeats    : " << opts.repeats << "\n";
        os_info << "  scaling    : up to " << opts.scaling_max_length << " bases\n";
    }

    // Run the benchmarks

    // Bit Extract Weights
    {
        auto os = get_ofstream(out_dir, "bit_extract_weights.csv" );
        bench_bit_extract_weights( os );
    }

    // Bit Extract Blocks
    {
        auto os = get_ofstream(out_dir, "bit_extract_blocks.csv" );
        bench_bit_extract_blocks( os );
    }

    // Per-character encoding
    {
        auto os = get_ofstream(out_dir, "char_encoder.csv" );
        bench_char_encoder( sequences, os );
    }

    // Sequence packing (whole-sequence ASCII to 2-bit)
    {
        auto os = get_ofstream(out_dir, "sequence_pack_unchecked.csv" );
        bench_sequence_pack_unchecked( sequences, os );
    }

    // Sequence packing with validity (as above, also recording which characters are valid)
    {
        auto os = get_ofstream(out_dir, "sequence_pack_checked.csv" );
        bench_sequence_pack_checked( sequences, os );
    }

    // K-mer extraction: test either the given size of k, or the full range if no k provided.

    // K-mers from ASCII, checked
    {
        auto os = get_ofstream(out_dir, "kmer_extract_ascii_checked.csv" );
        if( k == 0 ) {
            bench_kmer_extract_ascii_checked( sequences, os );
        } else {
            bench_kmer_extract_ascii_checked( sequences, k, k, os );
        }
    }

    // K-mers from ASCII, unchecked
    {
        auto os = get_ofstream(out_dir, "kmer_extract_ascii_unchecked.csv" );
        if( k == 0 ) {
            bench_kmer_extract_ascii_unchecked( sequences, os );
        } else {
            bench_kmer_extract_ascii_unchecked( sequences, k, k, os );
        }
    }

    // K-mers from a PackedSequence
    {
        auto os = get_ofstream(out_dir, "kmer_extract_packed_unchecked.csv" );
        if( k == 0 ) {
            bench_kmer_extract_packed_unchecked( sequences, os );
        } else {
            bench_kmer_extract_packed_unchecked( sequences, k, k, os );
        }
    }

    // K-mers from a PackedSequenceWithValidity, emitting the valid k-mers only
    {
        auto os = get_ofstream(out_dir, "kmer_extract_packed_checked.csv" );
        if( k == 0 ) {
            bench_kmer_extract_packed_checked( sequences, os );
        } else {
            bench_kmer_extract_packed_checked( sequences, k, k, os );
        }
    }

    // Spaced k-mers from ASCII, single mask
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_ascii_checked_single.csv" );
        bench_spaced_kmer_extract_ascii_checked_single( sequences, single_masks, os );
    }

    // Spaced k-mers from ASCII, multi masks
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_ascii_checked_multi.csv" );
        bench_spaced_kmer_extract_ascii_checked_multi( sequences, multi_masks, os );
    }

    // Spaced k-mers from ASCII, unchecked, single mask
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_ascii_unchecked_single.csv" );
        bench_spaced_kmer_extract_ascii_unchecked_single( sequences, single_masks, os );
    }

    // Spaced k-mers from ASCII, unchecked, multi masks
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_ascii_unchecked_multi.csv" );
        bench_spaced_kmer_extract_ascii_unchecked_multi( sequences, multi_masks, os );
    }

    // Spaced k-mers from a PackedSequence, single mask
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_packed_unchecked_single.csv" );
        bench_spaced_kmer_extract_packed_unchecked_single( sequences, single_masks, os );
    }

    // Spaced k-mers from a PackedSequence, multi masks
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_packed_unchecked_multi.csv" );
        bench_spaced_kmer_extract_packed_unchecked_multi( sequences, multi_masks, os );
    }

    // Spaced k-mers from a PackedSequenceWithValidity, single mask
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_packed_checked_single.csv" );
        bench_spaced_kmer_extract_packed_checked_single( sequences, single_masks, os );
    }

    // Spaced k-mers from a PackedSequenceWithValidity, multi masks
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_packed_checked_multi.csv" );
        bench_spaced_kmer_extract_packed_checked_multi( sequences, multi_masks, os );
    }

    // Spaced k-mers from a PackedSequence, one mask per span
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_packed_unchecked_spans.csv" );
        bench_spaced_kmer_extract_packed_unchecked_spans( sequences, os );
    }

    // Spaced k-mers from a PackedSequenceWithValidity, one mask per span
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_packed_checked_spans.csv" );
        bench_spaced_kmer_extract_packed_checked_spans( sequences, os );
    }

    // Spaced k-mers from ASCII without checks, chunked, one mask per span
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_ascii_unchecked_spans.csv" );
        bench_spaced_kmer_extract_ascii_unchecked_spans( sequences, os );
    }

    // Spaced k-mers from ASCII with checks, chunked, one mask per span
    {
        auto os = get_ofstream(out_dir, "spaced_kmer_extract_ascii_checked_spans.csv" );
        bench_spaced_kmer_extract_ascii_checked_spans( sequences, os );
    }

    // Clark
    {
        auto os = get_ofstream(out_dir, "kmer_clark.csv" );
        bench_kmer_clark( sequences, os );
    }

    // The extractors over the input length, last, as its input takes the most memory
    {
        auto os = get_ofstream(out_dir, "extractor_scaling.csv" );
        bench_extractor_scaling(
            std::stoull(opts.scaling_max_length), std::stod(opts.n_prob), std::stoull(opts.seed),
            os
        );
    }

    return 0;
}
