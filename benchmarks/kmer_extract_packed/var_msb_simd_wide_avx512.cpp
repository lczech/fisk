#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_msb_simd_wide_avx512(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_simd_wide_avx512_(seq, k, [&](__m512i v, std::size_t) noexcept {
        sink.consume(v);
    });
    return sink.finalize();
}

#endif // FISK_HAS_AVX512
