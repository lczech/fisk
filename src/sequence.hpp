#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <cctype>
#include <stdexcept>
#include <random>
#include <array>
#include <vector>

inline std::vector<std::string> load_fasta_clean(std::string const& path)
{
    // Read FASTA file, remove all non-ACGT characters, and concatenate all sequences.
    // This is just for a simple test here, so we skip all the complexity of dealing
    // with characters that need to be skipped etc. If needed, see genesis:
    // https://github.com/lczech/genesis/blob/master/lib/genesis/sequence/kmer/extractor.hpp

    // Prepare input
    std::ifstream in(path);
    if(!in) {
        throw std::runtime_error("Could not open FASTA file: " + path);
    }

    // Not the most efficient - see genesis for a way faster set of functions:
    // https://github.com/lczech/genesis/blob/master/lib/genesis/sequence/formats/fasta_reader.hpp
    // But good enough for our simple test purposes here.
    std::vector<std::string> result;
    std::string line;
    while(std::getline(in, line)) {
        // skip headers/labels
        if(line.empty() || line[0] == '>') {
            result.emplace_back();
            continue;
        }
        std::string seq;
        seq.reserve(line.size());
        for(char& c : line) {
            char uc = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            switch(uc) {
                case 'A':
                case 'C':
                case 'G':
                case 'T':
                    // Only use proper nucleotides in our simple test here.
                    // We leave k-mer extraction with
                    seq.push_back(uc);
                    break;
                default:
                    // drop other nucleotides
                    break;
            }
        }
        result.back() += std::move(seq);
    }

    return result;
}

inline std::string random_acgt(std::size_t n)
{
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static constexpr std::array<char, 4> bases{'A','C','G','T'};
    std::uniform_int_distribution<size_t> dist(0, 3);

    std::string s;
    s.resize(n);
    for(std::size_t i = 0; i < n; ++i) {
        s[i] = bases[dist(rng)];
    }
    return s;
}
