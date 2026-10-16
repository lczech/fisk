#include <cstddef>
#include <cstdint>
#include <vector>

#include "fisk/kmer_extract/packed_simd.hpp"
#include "kmer_extract_packed_checked/bench.hpp"

#if defined(FISK_HAS_AVX2)

using namespace fisk;

std::uint64_t run_var_checked_lsb_simd_full_width_avx2(
    CheckedLsb const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    auto const consume = [&](std::size_t, __m256i v, SimdValidLanes<SimdAVX2> lanes) noexcept {
        sink.consume(valid_kmers_<SimdAVX2>(v, lanes));
    };
    for_each_kmer_packed_simd_full_width_<SimdAVX2, true, true>(
        seq, seq.valid.data(), k,
        checked_kmer_emit_<SimdAVX2, Encoding::kACGT, Layout::kLSB>(consume, k)
    );
    return sink.finalize();
}

#endif // FISK_HAS_AVX2
