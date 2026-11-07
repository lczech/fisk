#pragma once

#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
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
        if(line.empty()) {
            continue;
        }
        if(line[0] == '>') {
            result.emplace_back();
            continue;
        }
        if(result.empty()) {
            throw std::runtime_error(
                "Invalid FASTA file (missing header before sequence data): " + path
            );
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
                    // replace other chancers with N, which we will ignore in k-mer extraction.
                    seq.push_back('N');
                    break;
            }
        }
        result.back() += std::move(seq);
    }

    if(result.empty()) {
        throw std::runtime_error("Invalid FASTA file (no FASTA records found): " + path);
    }

    return result;
}

/**
 * @brief Generate a string of random `ACGT` letters of a given length @p len,
 * and a probability @p n_prob in [0.0, 1.0] of generating invalid `N` bases.
 *
 * The same @p seed gives the same string on every platform: the bases and the `N`s are taken from
 * the raw output of the engine, which the standard specifies, instead of from the std
 * distributions, whose algorithms differ between standard libraries.
 */
inline std::string random_acgt(std::size_t len, double n_prob, std::uint64_t seed)
{
    if( !std::isfinite(n_prob) || n_prob < 0.0 || n_prob > 1.0 ) {
        throw std::invalid_argument("Invalid probability not in [0.0, 1.0]");
    }
    std::mt19937_64 rng{seed};
    static constexpr std::array<char, 4> bases{'A','C','G','T'};

    // An N if the upper 53 bits of a draw are below n_prob * 2^53, which is exact for any double
    // in [0.0, 1.0]; else the base from the lowest two bits.
    auto const n_threshold = static_cast<std::uint64_t>(std::ldexp(n_prob, 53));

    // Fill the result string
    std::string s;
    s.resize(len);
    for(std::size_t i = 0; i < len; ++i) {
        std::uint64_t const r = rng();
        if( (r >> 11) < n_threshold ) {
            s[i] = 'N';
        } else {
            s[i] = bases[static_cast<std::size_t>(r & 3u)];
        }
    }
    return s;
}
