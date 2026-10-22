#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_single/bench.hpp"

using namespace fisk;

std::uint64_t run_var_simd_butterfly_table_scalar_by_position(
    std::string const& seq,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelButterfly<SimdScalar>> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_simd_by_position(
        std::string_view(seq), kernel, CharEncoderTable<Encoding::kACGT>{},
        [&](std::size_t /*pos*/, auto const& wmer) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(wmer));
        }
    );
    return sink.finalize();
}
