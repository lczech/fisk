#include <cstddef>
#include <cstdint>
#include <string>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_multi/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_simd_butterfly_table_avx2(std::string const& seq, std::size_t k, BitExtractKernelDispatcher<BitExtractKernelButterflyAVX2> const& kernel)
{
    return compute_spaced_kmer_hash_simd(seq, k, kernel);
}

#endif // FISK_HAS_AVX2
