#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "kmer_extract/bench.hpp"

using namespace fisk;

std::uint64_t run_var_acgt_lsb_table_shift(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    CharEncoderTable<Encoding::kACGT> const encoder;
    for_each_kmer_ascii_checked_rolling_scalar<Layout::kLSB>(
        seq, k, encoder, [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(kmer));
        }
    );
    return sink.finalize();
}
