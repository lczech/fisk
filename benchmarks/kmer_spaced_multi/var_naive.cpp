#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "kmer_spaced_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_naive(std::string const& seq, std::size_t k, std::vector<std::vector<std::size_t>> const& naive_masks)
{
    std::uint64_t hash = 0;
    std::size_t const stop = seq.size() - k;
    for (std::size_t i = 0; i <= stop; ++i) {
        for (auto const& mask : naive_masks) {
            hash += compute_spaced_kmer_naive(std::string_view(seq), mask, i);
        }
    }
    return hash;
}
