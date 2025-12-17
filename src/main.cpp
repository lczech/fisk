#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <stdexcept>

#include "arg_parser.hpp"
#include "cpu_intrinsics.hpp"

struct Options {
    // Input fasta file with sequence data
    std::string input_file;
};

int main(int argc, char **argv)
{
    // Set up command line parsing
    Options opts;
    ArgParser parser(argv[0]);

    // ------------------------------------------------------------------------
    //     Declare options
    // ------------------------------------------------------------------------

    parser.add_option(
        "--input-fasta", "-i",
        "Input fasta file with sequence data",
        opts.input_file
    );

    // ------------------------------------------------------------------------
    //     Parse
    // ------------------------------------------------------------------------

    try {
        parser.parse(argc, argv);
    } catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Use --help for usage.\n";
        return 1;
    }

    // ------------------------------------------------------------------------
    //     Main
    // ------------------------------------------------------------------------

    // For now, just checking that the CLI works
    if (opts.input_file.size()) {
        std::cout << "Input file: " << opts.input_file << "\n";
    } else {
        std::cout << "No input file provided\n";
    }

    // User output of CPU features
    print_intrinsics_support();

    return 0;
}
