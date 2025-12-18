#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <stdexcept>
#include <filesystem>

#include "arg_parser.hpp"
#include "bench_kmer_extract.hpp"
#include "bench_pext.hpp"
#include "bench_seq_enc.hpp"
#include "fs_utils.hpp"
#include "sequence.hpp"
#include "sys_info.hpp"

struct Options
{
    // Input sequence length to randomly generate
    std::string input_length;

    // Input fasta file with sequence data
    std::string input_file;

    // Value of k for the k-mers
    std::string k;

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
        "--k", "-k",
        "Value of k to use for the k-mers",
        opts.k
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
    std::vector<std::string> sequences;
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
        std::cout << "and " << sum << " total nucleotides\n";
    } else {
        size_t inp_len = 0;
        if(opts.input_length.size()) {
            inp_len = std::stoul(opts.input_length);
        } else {
            std::cout << "No input provided, using default\n";
            size_t const default_len = (1u << 25);
            inp_len = default_len;
        }
        std::cout << "Generating input sequence of length " << inp_len << "\n";
        sequences = std::vector<std::string>{ random_acgt(inp_len) };
    }

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
    // {
    //     auto os_pext = get_ofstream(out_dir, "pext.csv" );
    //     bench_pext( os_pext );
    // }
    // {
    //     auto os_seq_enc = get_ofstream(out_dir, "seq_enc.csv" );
    //     bench_seq_enc( sequences, os_seq_enc );
    // }
    {
        // Test either the given size of k, or the full range if no k provided.
        auto os_kmer_extract = get_ofstream(out_dir, "kmer_extract.csv" );
        if( k == 0 ) {
            bench_kmer_extract( sequences, os_kmer_extract );
        } else {
            bench_kmer_extract( sequences, k, k, os_kmer_extract );
        }
    }

    return 0;
}
