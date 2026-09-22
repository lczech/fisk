#include <cstddef>
#include <cstdint>
#include <string>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_single/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_simd_pext(std::string const& seq, std::size_t k, BitExtractKernelPEXT<> const& kernel)
{
    return compute_spaced_kmer_hash_simd(seq, k, kernel);
}

#endif // FISK_HAS_BMI2
