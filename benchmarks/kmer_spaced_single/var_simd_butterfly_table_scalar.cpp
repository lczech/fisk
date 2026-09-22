#include <cstddef>
#include <cstdint>
#include <string>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_single/bench.hpp"

using namespace fisk;

std::uint64_t run_var_simd_butterfly_table_scalar(std::string const& seq, std::size_t k, BitExtractKernelButterflyScalar const& kernel)
{
    std::uint64_t hash = 0;
    for_each_spaced_kmer_simd(
        std::string_view(seq), k, kernel, CharEncoderTable<Encoding::kACGT>{},
        [&](std::size_t /*pos*/, std::uint64_t wmer) {
            hash += wmer;
        }
    );
    return hash;
}
