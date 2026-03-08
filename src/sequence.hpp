#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <cctype>
#include <stdexcept>
#include <random>
#include <array>
#include <vector>

/**
 * @brief Read FASTA file, remove all non-ACGT characters, and concatenate all sequences.
 */
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

/**
 * @brief Generate a string of random `ACGT` letters of a given length @p len,
 * and a probability @p n_prob in [0.0, 1.0] of generating invalid `N` bases.
 */
inline std::string random_acgt(std::size_t len, double n_prob = 0.0)
{
    // Draw from ACGT
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static constexpr std::array<char, 4> bases{'A','C','G','T'};
    std::uniform_int_distribution<size_t> base_dist(0, 3);

    // Add random invalid N characters
    if( !std::isfinite(n_prob) || n_prob < 0.0 || n_prob > 1.0 ) {
        throw std::invalid_argument("Invalid probability not in [0.0, 1.0]");
    }
    std::bernoulli_distribution n_dist(n_prob);

    // Fill the result string
    std::string s;
    s.resize(len);
    for(std::size_t i = 0; i < len; ++i) {
        if( n_dist(rng) ) {
            s[i] = 'N';
        } else {
            s[i] = bases[base_dist(rng)];
        }
    }
    return s;
}
