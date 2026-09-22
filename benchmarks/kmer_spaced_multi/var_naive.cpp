#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "kmer_spaced_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_naive(std::string const& seq, std::size_t k, std::vector<std::vector<std::size_t>> const& naive_masks)
{
    return compute_spaced_kmer_hash_naive(seq, k, naive_masks, compute_spaced_kmer_naive);
}
