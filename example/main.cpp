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

#include "fisk/extractor/for_each.hpp"
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

// Extract all k-mers of the sequences with `extract_one(seq, counter)`, and print a CSV row.
template <typename ExtractOne>
void extract(
    char const* name, std::string const& algorithm, ExtractOne extract_one,
    std::vector<AlignedString> const& sequences, std::size_t bases, double read_seconds
) {
    MyCounter counter;
    auto const start = std::chrono::steady_clock::now();
    for (auto const& seq : sequences) {
        extract_one(seq, counter);
    }
    std::chrono::duration<double> const seconds = std::chrono::steady_clock::now() - start;

    std::cout << name << "," << algorithm << "," << sequences.size() << "," << bases << ","
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

    // k-mers with k = 31, and spaced k-mers with a mask of span 31 and weight 22, from the
    // characters of the sequences, encoded as A = 0, C = 1, G = 2, T = 3 (ACGT), with the first
    // base in the lowest bits (LSB). Checked: the k-mers that contain other characters than ACGT,
    // in upper or lower case, such as N, are skipped. The masks are prepared once, for all
    // sequences. Both functions run the algorithm that is likely the fastest on this CPU, which
    // the _algorithm_name() functions name.
    constexpr auto C = InputCheck::kChecked;
    constexpr auto E = Encoding::kACGT;
    constexpr auto L = Layout::kLSB;
    std::size_t const k = 31;
    SpacedKmerMaskForms<L> const mask_forms(
        std::vector<std::string>{ "1111011101110010111001011011111" }
    );

    std::cout << "extraction,algorithm,sequences,bases,kmers,checksum,"
              << "read_seconds,extract_seconds\n";
    extract(
        "kmers", for_each_kmer_algorithm_name<C, E, L>(std::string_view{}, k, MyCounter{}),
        [&](std::string_view seq, MyCounter& counter) {
            for_each_kmer<C, E, L>(seq, k, counter);
        },
        sequences, bases, read_seconds.count()
    );
    extract(
        "spaced_kmers",
        for_each_spaced_kmer_algorithm_name<C, E, L>(std::string_view{}, mask_forms, MyCounter{}),
        [&](std::string_view seq, MyCounter& counter) {
            for_each_spaced_kmer<C, E, L>(seq, mask_forms, counter);
        },
        sequences, bases, read_seconds.count()
    );
}
