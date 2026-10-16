#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed/bench.hpp"

#if defined(FISK_HAS_AVX512)

using namespace fisk;

std::uint64_t run_var_msb_simd_full_width_avx512(
    PackedMsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_packed_simd_full_width_<SimdAVX512, true, false>(
        seq, nullptr, k,
        [&](std::size_t, __m512i v, std::size_t, AllValid_) noexcept { sink.consume(v); }
    );
    return sink.finalize();
}

#endif // FISK_HAS_AVX512
