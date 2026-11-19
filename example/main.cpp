// Extract all k-mers and spaced k-mers of the sequences of a FASTA file, such as a human genome,
// with for_each_kmer() and for_each_spaced_kmer() of fisk, and print their number, a checksum, and
// the time it took, as CSV. See README.md.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fisk/extraction/for_each.hpp"
#include "fisk/util/aligned_allocator.hpp"

using namespace fisk;

// The sequences of a FASTA file, without their names.
// No error checking to keep this example simple. The functions take any string, such as
// std::string. AlignedString is a string whose characters start at a cache line (64 bytes), which
// the SIMD algorithms can read faster.
std::vector<AlignedString> read_fasta(std::string const& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open " + path);
    }
    std::vector<AlignedString> sequences;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '>') {
            sequences.emplace_back();
        } else if (!sequences.empty()) {
            sequences.back() += line;
        }
    }
    return sequences;
}

// The user callback, i.e., the function that is executed for each k-mer: here, it simply counts
// the k-mers, and sums their values as a checksum. It is called with each k-mer and its position,
// and with each spaced k-mer, its position, and the index of its mask (all ignored here).
struct MyCounter
{
    std::uint64_t count = 0;
    std::uint64_t checksum = 0;

    // For k-mers, the callback is expected to take the position and the k-mer value.
    template <typename Kmer>
    void operator()(std::size_t /*pos*/, Kmer kmer)
    {
        ++count;
        checksum += kmer_value(kmer);
    }

    // For spaced k-mers, the callback is expected to also take the mask index (to allow for
    // extraction with multiple masks at the same time).
    template <typename Kmer>
    void operator()(std::size_t /*pos*/, std::size_t /*mask_idx*/, Kmer kmer)
    {
        ++count;
        checksum += kmer_value(kmer);
    }
};

// Extract all k-mers with k = 31 from the characters of the sequences, encoded as A = 0, C = 1,
// G = 2, T = 3 (ACGT), with the first nucleotide in the lowest bits (LSB), and print a CSV row.
// Checked: the k-mers that contain other characters than ACGT, in upper or lower case, such as N,
// are skipped. for_each_kmer() runs the algorithm that is likely the fastest on this CPU, which
// for_each_kmer_algorithm_name() names.
void extract_kmers(
    std::vector<AlignedString> const& sequences, std::size_t nucleotides, double read_seconds
) {
    // Get the name of the algorithm that for_each_kmer() will use, for the CSV output.
    std::string const algorithm = for_each_kmer_algorithm_name<
        InputCheck::kChecked, Encoding::kACGT, Layout::kLSB
    >(std::string_view{}, 31);

    // Count the k-mers and their checksum, and measure the time it took.
    MyCounter counter;
    auto const start = std::chrono::steady_clock::now();
    for (auto const& seq : sequences) {
        for_each_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kLSB>(
            std::string_view(seq), 31, counter
        );
    }
    std::chrono::duration<double> const seconds = std::chrono::steady_clock::now() - start;

    // Print a CSV row with the results.
    std::cout << "kmers," << algorithm << "," << sequences.size() << "," << nucleotides << ","
              << counter.count << "," << counter.checksum << "," << read_seconds << ","
              << seconds.count() << "\n";
}

// Extract all spaced k-mers with a mask of span 31 and weight 22, with the same encoding and
// checks, and print a CSV row. The masks are prepared once, for all sequences.
void extract_spaced_kmers(
    std::vector<AlignedString> const& sequences, std::size_t nucleotides, double read_seconds
) {
    // The spaced k-mer mask, here only one, but could be several.
    SpacedKmerMaskForms<Layout::kLSB> const mask_forms(
        std::vector<std::string>{ "1111011101110010111001011011111" }
    );

    // Get the name of the algorithm that for_each_spaced_kmer() will use, for the CSV output.
    std::string const algorithm = for_each_spaced_kmer_algorithm_name<
        InputCheck::kChecked, Encoding::kACGT, Layout::kLSB
    >(std::string_view{}, mask_forms);

    // Count the spaced k-mers and their checksum, and measure the time it took.
    MyCounter counter;
    auto const start = std::chrono::steady_clock::now();
    for (auto const& seq : sequences) {
        for_each_spaced_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kLSB>(
            std::string_view(seq), mask_forms, counter
        );
    }
    std::chrono::duration<double> const seconds = std::chrono::steady_clock::now() - start;

    // Print a CSV row with the results.
    std::cout << "spaced_kmers," << algorithm << "," << sequences.size() << ","
              << nucleotides << "," << counter.count << "," << counter.checksum << ","
              << read_seconds << "," << seconds.count() << "\n";
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <fasta file>\n";
        return 1;
    }

    // Read the sequences of the FASTA file, and measure the time it took.
    auto const start = std::chrono::steady_clock::now();
    auto const sequences = read_fasta(argv[1]);
    std::chrono::duration<double> const read_seconds = std::chrono::steady_clock::now() - start;
    std::size_t nucleotides = 0;
    for (auto const& seq : sequences) {
        nucleotides += seq.size();
    }

    // Print the CSV header, and extract the k-mers and spaced k-mers.
    std::cout << "extraction,algorithm,sequences,nucleotides,kmers,checksum,"
              << "read_seconds,extract_seconds\n";
    extract_kmers(sequences, nucleotides, read_seconds.count());
    extract_spaced_kmers(sequences, nucleotides, read_seconds.count());
}
