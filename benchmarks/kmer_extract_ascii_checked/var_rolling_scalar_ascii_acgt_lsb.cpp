#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_extract/ascii_rolling_scalar.hpp"
#include "kmer_extract_ascii_checked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_rolling_scalar_ascii_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    CharEncoderAscii<Encoding::kACGT> const encoder;
    for_each_kmer_ascii_checked_rolling_scalar<Layout::kLSB>(
        seq, k, encoder, [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(kmer));
        }
    );
    return sink.finalize();
}
