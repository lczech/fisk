#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_extract/ascii_rolling_avx2.hpp"
#include "kmer_extract_ascii_checked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_rolling_avx2_actg_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_ascii_checked_rolling_avx2<Encoding::kACTG, Layout::kMSB>(
        seq, k, [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(kmer));
        }
    );
    return sink.finalize();
}
