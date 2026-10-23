#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/simd.hpp"
#include "kmer_spaced_single/bench.hpp"

#if defined(FISK_HAS_BMI2)

using namespace fisk;

std::uint64_t run_var_simd_pext_vectors(
    std::string const& seq,
    SpacedKmerMask<Layout::kLSB, BitExtractKernelPEXT> const& kernel,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_spaced_kmer_ascii_checked_simd(
        std::string_view(seq), kernel, CharEncoderTable<Encoding::kACGT>{},
        [&](
            std::size_t /*pos*/, auto spaced_kmers, auto valid_lanes
        ) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(spaced_kmers));
            do_not_optimize(valid_lanes.bits);
        }
    );
    return sink.finalize();
}

#endif // FISK_HAS_BMI2
