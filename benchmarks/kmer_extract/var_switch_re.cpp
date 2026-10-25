#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_extract/ascii_reextract_scalar.hpp"
#include "kmer_extract/bench.hpp"

using namespace fisk;

std::uint64_t run_var_switch_re(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_ascii_checked_reextract_scalar(
        seq, k, CharEncoderSwitch<Encoding::kACGT>{}, [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA {
            sink.consume(kmer_value(kmer));
        }
    );
    return sink.finalize();
}
