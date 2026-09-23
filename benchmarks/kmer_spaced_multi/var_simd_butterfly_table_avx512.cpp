#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_multi/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_simd_butterfly_table_avx512(
    std::string const& seq,
    std::size_t k,
    BitExtractKernelDispatcher<BitExtractKernelButterflyAVX512> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    kernel.run([&](auto const& kernels_arr) {
        for_each_spaced_kmer_simd(
            std::string_view(seq), k, kernels_arr, CharEncoderTable<Encoding::kACGT>{},
            [&](std::size_t /*mask_idx*/, std::size_t /*pos*/, std::uint64_t wmer) {
                sink.consume(wmer);
            }
        );
    });
    return sink.finalize();
}

#endif // FISK_HAS_AVX512
