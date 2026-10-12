#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "kmer_spaced_single/bench.hpp"

using namespace fisk;

namespace {

// The below is a re-implementation of parts of https://github.com/CominLab/MISSH
// where we losely follow their code, in order to get a baseline for comparison.

/**
 * @brief Reimplementation of the MISSH spaced k-mer extraction.
 */
std::uint64_t compute_spaced_kmer_missh(
    std::string_view seq, std::vector<size_t> const& mask, size_t start_pos
) {
    // Compute a single spaced kmer at the given position
    std::uint64_t result = 0;
    bool valid = true;
    for( size_t i = 0; i < mask.size(); ++i ) {
        // Comin et al use a switch statement for the encoding, which is slow.
        auto const c = static_cast<std::uint64_t>(
            CharEncoderSwitch<Encoding::kACGT>{}( seq[start_pos + mask[i]] )
        );
        valid &= (c < 4);

        // The original code builds the kmer backwards, with the last base at the highest bits.
        // result |= (c << (2 * i));

        // We instead keep it in order, so that sorting of kmers etc works as expected.
        // The speed of this is not significantly different from the above, in our tests.
        result <<= 2;
        result |= c;
    }
    return valid ? result : 0;
}

} // namespace

std::uint64_t run_var_missh(
    std::string const& seq,
    std::size_t k,
    std::vector<std::size_t> const& naive_mask,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    std::size_t const stop = seq.size() - k;
    for (std::size_t i = 0; i <= stop; ++i) {
        sink.consume(compute_spaced_kmer_missh(std::string_view(seq), naive_mask, i));
    }
    return sink.finalize();
}
