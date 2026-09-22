#include <cstdint>
#include <string>

#include "fisk/kmer_extract/simd.hpp"
#include "kmer_extract/bench.hpp"

using namespace fisk;

std::uint64_t run_var_simd_avx2(std::string const& seq, std::size_t k)
{
    std::uint64_t hash = 0;
    for_each_kmer_simd(seq, k, [&](auto kmer) { hash += kmer_value(kmer); });
    return hash;
}
