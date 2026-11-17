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
// No error checking to keep this example simple.
// The functions take any string, such as std::string. AlignedString is a std::string whose
// characters start at a cache line (64 bytes), which the SIMD algorithms can read faster, mostly
// with AVX-512, see fisk/util/aligned_allocator.hpp.
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

// The callback, i.e., the function that is executed for each k-mer: here, it simply counts
// the k-mers, and sums their values as a checksum. It is called with each k-mer and its position,
// and with each spaced k-mer, its position, and the index of its mask.
struct MyCounter
{
    std::uint64_t count = 0;
    std::uint64_t checksum = 0;

    template <typename Kmer>
    void operator()(std::size_t /*pos*/, Kmer kmer)
    {
        ++count;
        checksum += kmer_value(kmer);
    }

    template <typename Kmer>
    void operator()(std::size_t /*pos*/, std::size_t /*mask_idx*/, Kmer kmer)
    {
        ++count;
        checksum += kmer_value(kmer);
    }
};

// Extract all k-mers with k = 31 from the characters of the sequences, encoded as A = 0, C = 1,
// G = 2, T = 3 (ACGT), with the first base in the lowest bits (LSB), and print a CSV row. Checked:
// the k-mers that contain other characters than ACGT, in upper or lower case, such as N, are
// skipped. for_each_kmer() runs the algorithm that is likely the fastest on this CPU, which
// for_each_kmer_algorithm_name() names. It is called directly in the timed loop: behind a lambda
// that the compiler does not inline, the loop can be slower.
void extract_kmers(
    std::vector<AlignedString> const& sequences, std::size_t bases, double read_seconds
) {
    std::string const algorithm = for_each_kmer_algorithm_name<
        InputCheck::kChecked, Encoding::kACGT, Layout::kLSB
    >(std::string_view{}, 31);

    MyCounter counter;
    auto const start = std::chrono::steady_clock::now();
    for (auto const& seq : sequences) {
        for_each_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kLSB>(
            std::string_view(seq), 31, counter
        );
    }
    std::chrono::duration<double> const seconds = std::chrono::steady_clock::now() - start;

    std::cout << "kmers," << algorithm << "," << sequences.size() << "," << bases << ","
              << counter.count << "," << counter.checksum << "," << read_seconds << ","
              << seconds.count() << "\n";
}

// Extract all spaced k-mers with a mask of span 31 and weight 22, with the same encoding and
// checks, and print a CSV row. The masks are prepared once, for all sequences.
void extract_spaced_kmers(
    std::vector<AlignedString> const& sequences, std::size_t bases, double read_seconds
) {
    SpacedKmerMaskForms<Layout::kLSB> const mask_forms(
        std::vector<std::string>{ "1111011101110010111001011011111" }
    );
    std::string const algorithm = for_each_spaced_kmer_algorithm_name<
        InputCheck::kChecked, Encoding::kACGT, Layout::kLSB
    >(std::string_view{}, mask_forms);

    MyCounter counter;
    auto const start = std::chrono::steady_clock::now();
    for (auto const& seq : sequences) {
        for_each_spaced_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kLSB>(
            std::string_view(seq), mask_forms, counter
        );
    }
    std::chrono::duration<double> const seconds = std::chrono::steady_clock::now() - start;

    std::cout << "spaced_kmers," << algorithm << "," << sequences.size() << "," << bases << ","
              << counter.count << "," << counter.checksum << "," << read_seconds << ","
              << seconds.count() << "\n";
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <fasta file>\n";
        return 1;
    }

    auto const start = std::chrono::steady_clock::now();
    auto const sequences = read_fasta(argv[1]);
    std::chrono::duration<double> const read_seconds = std::chrono::steady_clock::now() - start;
    std::size_t bases = 0;
    for (auto const& seq : sequences) {
        bases += seq.size();
    }

    std::cout << "extraction,algorithm,sequences,bases,kmers,checksum,"
              << "read_seconds,extract_seconds\n";
    extract_kmers(sequences, bases, read_seconds.count());
    extract_spaced_kmers(sequences, bases, read_seconds.count());
}
