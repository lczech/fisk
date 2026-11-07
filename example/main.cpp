// Extract all k-mers and spaced k-mers of the sequences of a FASTA file, such as a human genome,
// with the extractors of fisk, and print their number, a checksum, and the time it took, as CSV.
// See README.md.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fisk/extractor/kmer_extractor.hpp"
#include "fisk/extractor/spaced_kmer_extractor.hpp"

using namespace fisk;

// The sequences of a FASTA file, without their names.
std::vector<std::string> read_fasta(std::string const& path)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open " + path);
    }
    std::vector<std::string> sequences;
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

// The callback: counts the k-mers, and sums their values as a checksum. The extractors call it
// with each k-mer and its position, and with each spaced k-mer, its position, and the index of its
// mask.
struct Counter
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

// Choose the fastest algorithm for the callback on this CPU, extract all k-mers of the sequences,
// and print a CSV row.
template <typename Extractor>
void extract(
    char const* name, Extractor& extractor, std::vector<std::string> const& sequences,
    std::size_t bases, double read_seconds
) {
    extractor.tune(Counter{});

    Counter counter;
    auto const start = std::chrono::steady_clock::now();
    for (auto const& seq : sequences) {
        extractor.for_each(seq, counter);
    }
    std::chrono::duration<double> const seconds = std::chrono::steady_clock::now() - start;

    std::cout << name << "," << extractor.algorithm_name() << "," << sequences.size() << ","
              << bases << "," << counter.count << "," << counter.checksum << ","
              << read_seconds << "," << seconds.count() << "\n";
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

    // k-mers with k = 31, and spaced k-mers with a mask of span 31 and weight 22, from the
    // characters of the sequences, encoded as A = 0, C = 1, G = 2, T = 3 (ACGT), with the first
    // base in the lowest bits (LSB). Checked: the k-mers that contain other characters than ACGT,
    // in upper or lower case, such as N, are skipped.
    KmerExtractor<InputFormat::kAscii, InputCheck::kChecked, Encoding::kACGT, Layout::kLSB> kmers(
        31
    );
    SpacedKmerExtractor<InputFormat::kAscii, InputCheck::kChecked, Encoding::kACGT, Layout::kLSB>
        spaced_kmers(std::vector<std::string>{ "1111011101110010111001011011111" });

    std::cout << "extractor,algorithm,sequences,bases,kmers,checksum,"
              << "read_seconds,extract_seconds\n";
    extract("kmers", kmers, sequences, bases, read_seconds.count());
    extract("spaced_kmers", spaced_kmers, sequences, bases, read_seconds.count());
}
