#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_single/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_simd_block_table_avx2(
    std::string const& seq,
    std::size_t k,
    BitExtractKernelBlockAVX2<> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_simd(
        std::string_view(seq), k, kernel, CharEncoderTable<Encoding::kACGT>{},
        [&](std::size_t /*pos*/, std::uint64_t wmer) {
            sink.consume(wmer);
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_AVX2
