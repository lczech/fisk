#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <stdexcept>
#include <filesystem>

#include "arg_parser.hpp"
#include "bench_pext.hpp"
#include "fs_utils.hpp"
#include "sequence.hpp"
#include "sys_info.hpp"

struct Options
{
    // Input sequence length to randomly generate
    std::string input_length;

    // Input fasta file with sequence data
    std::string input_file;

    // Output directory for benchmark results
    std::string output_dir = "benchmarks";
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
        "--input-length", "-l",
        "Input length to randomly geneate a sequence of ACGT. Excludes --input-fasta",
        opts.input_length
    );

    parser.add_option(
        "--input-fasta", "-i",
        "Input fasta file with sequence data. Excludes --input-length",
        opts.input_file
    );

    parser.add_option(
        "--output-dir", "-o",
        "Output directory for benchmark CSV files (default: benchmarks)",
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

    // Get the sequence to process, either random, or from fasta.
    std::string sequence;
    if (opts.input_file.size() && opts.input_length.size()) {
        throw std::invalid_argument(
            "Options --input-length and --input-fasta are mutually exclusive."
        );
    }
    if (opts.input_file.size()) {
        std::cout << "Reading input file " << opts.input_file << "\n";
        sequence = load_fasta_clean( opts.input_file );
    } else {
        size_t inp_len = 0;
        if(opts.input_length.size()) {
            inp_len = std::stoul(opts.input_length);
        } else {
            std::cout << "No input provided\n";
            size_t const default_len = 1000000;
            inp_len = default_len;
        }
        std::cout << "Generating input sequence of length " << inp_len << "\n";
        sequence = random_acgt(inp_len);
    }

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
        info_print_platform(os_info);
        info_print_cpu(os_info);
        info_print_compiler(os_info);
        info_print_intrinsics(os_info);
    }

    // Run the benchmarks
    {
        auto os_pext = get_ofstream(out_dir, "pext.csv" );
        bench_pext( os_pext );
    }

    return 0;
}
