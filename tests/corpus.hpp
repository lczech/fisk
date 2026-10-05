#pragma once

// Shared building blocks for test inputs: random sequences, invalid-character injection, and the
// character sets both draw from. The actual corpora (seeds, length ranges, placements) stay in the
// test files, since each is tuned to the boundaries of the functions it exercises; see oracle.hpp
// for the ground truth the results are checked against.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fisk/core/random.hpp"

// =================================================================================================
//     Character Sets
// =================================================================================================

// Valid nucleotides, both cases.
inline constexpr std::string_view kValidBases = "ACGTacgt";

// Invalid characters for hand-picked placement tests, where readable failures matter: N, the usual
// gap and padding markers, and the IUPAC ambiguity codes, both cases.
inline constexpr std::string_view kHandPickedInvalid = "Nn-.RYSWKMBDHVryswkmbdhv";

// Every byte value that is not a valid nucleotide, for fuzzing: 248 of them, including '\0', and
// the bytes >= 0x80 that are negative as a signed char. Also includes the bytes that differ from a
// valid nucleotide in a single bit, which is what bit-trick encoders are most likely to confuse.
inline std::string const& all_invalid_bytes()
{
    static std::string const bytes = [] {
        std::string out;
        for (int i = 0; i < 256; ++i) {
            auto const c = static_cast<char>(i);
            if (kValidBases.find(c) == std::string_view::npos) {
                out += c;
            }
        }
        return out;
    }();
    return bytes;
}

// =================================================================================================
//     Generators
// =================================================================================================

// Random sequence of `len` characters, drawn uniformly from `alphabet`.
inline std::string random_sequence(
    fisk::Splitmix64& rng, std::size_t len, std::string_view alphabet = kValidBases
) {
    std::string s;
    s.reserve(len);
    for (std::size_t i = 0; i < len; ++i) {
        s += alphabet[rng.get_uint64() % alphabet.size()];
    }
    return s;
}

// Overwrite each position of `seq` with a random character from `invalid`, with probability `rate`.
inline void inject_invalid(
    std::string& seq, fisk::Splitmix64& rng, double rate, std::string_view invalid
) {
    for (auto& c : seq) {
        if (rng.get_double() < rate) {
            c = invalid[rng.get_uint64() % invalid.size()];
        }
    }
}

// Random sequences with invalid characters from the full byte range injected at `rate`, of random
// lengths in [0, max_len]. Used to stress the window-reset logic, or the locality of garbage for
// the assume-valid functions, at different densities.
inline std::vector<std::string> invalid_injected_sequences(
    std::size_t count, std::size_t max_len, double rate, std::uint64_t seed
) {
    fisk::Splitmix64 rng(seed);
    std::vector<std::string> out;
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t const len = static_cast<std::size_t>(rng.get_uint64() % (max_len + 1));
        std::string seq = random_sequence(rng, len);
        inject_invalid(seq, rng, rate, all_invalid_bytes());
        out.push_back(std::move(seq));
    }
    return out;
}

// =================================================================================================
//     k Values
// =================================================================================================

// k values spanning the full supported range [1, 32], concentrated around word and byte boundaries
// and around the narrow (k <= 29) / wide (k in [30, 32]) split of the packed extractors. Functions
// that dispatch every k to its own compile-time specialization need all of 1..32 instead.
inline std::vector<std::size_t> const& test_ks()
{
    static std::vector<std::size_t> const ks = {
        1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 27, 28, 29, 30, 31, 32
    };
    return ks;
}
