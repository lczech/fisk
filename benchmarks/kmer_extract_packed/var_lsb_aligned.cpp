#include <cstddef>
#include <cstdint>

#include "fisk/kmer_extract/packed.hpp"
#include "kmer_extract_packed/bench.hpp"

using namespace fisk;

std::uint64_t run_var_lsb_aligned(PackedLsb const& seq, std::size_t k)
{
    std::uint64_t hash = 0;
    for_each_kmer_packed_aligned(seq, k, [&](auto kmer) { hash += kmer_value(kmer); });
    return hash;
}
