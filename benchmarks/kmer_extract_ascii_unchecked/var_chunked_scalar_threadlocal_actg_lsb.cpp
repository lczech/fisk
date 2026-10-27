#include <cstdint>
#include <string>
#include <vector>

#include "fisk/kmer_extract/ascii_chunked_scalar.hpp"
#include "chunked.hpp"
#include "kmer_extract_ascii_unchecked/bench.hpp"

using namespace fisk;

std::uint64_t run_var_chunked_scalar_threadlocal_actg_lsb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    for_each_kmer_ascii_unchecked_chunked_scalar(
        seq, k, ChunkedWordEncoder<Encoding::kACTG, Layout::kLSB>{},
        [&](auto kmer) FISK_ALWAYS_INLINE_LAMBDA { sink.consume(kmer_value(kmer)); }
    );
    return sink.finalize();
}
