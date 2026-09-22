#include <cstddef>
#include <cstdint>
#include <string>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_simd_block_table_scalar(std::string const& seq, std::size_t k, BitExtractKernelDispatcher<BitExtractKernelBlockScalar<>> const& kernel)
{
    return compute_spaced_kmer_hash_simd(seq, k, kernel);
}
