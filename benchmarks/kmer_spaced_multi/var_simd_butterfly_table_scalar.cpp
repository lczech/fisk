#include <cstddef>
#include <cstdint>
#include <string>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_multi/bench.hpp"

using namespace fisk;

std::uint64_t run_var_simd_butterfly_table_scalar(std::string const& seq, std::size_t k, BitExtractKernelDispatcher<BitExtractKernelButterflyScalar> const& kernel)
{
    std::uint64_t hash = 0;
    kernel.run([&](auto const& kernels_arr) {
        for_each_spaced_kmer_simd(
            std::string_view(seq), k, kernels_arr, CharEncoderTable<Encoding::kACGT>{},
            [&](std::size_t /*mask_idx*/, std::size_t /*pos*/, std::uint64_t wmer) {
                hash += wmer;
            }
        );
    });
    return hash;
}
