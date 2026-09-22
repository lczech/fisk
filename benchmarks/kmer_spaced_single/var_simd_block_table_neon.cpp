#include <cstddef>
#include <cstdint>
#include <string>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_single/bench.hpp"

#if defined(FISK_HAS_NEON)

using namespace fisk;

std::uint64_t run_var_simd_block_table_neon(std::string const& seq, std::size_t k, BitExtractKernelBlockNEON<> const& kernel)
{
    return compute_spaced_kmer_hash_simd(seq, k, kernel);
}

#endif // FISK_HAS_NEON
