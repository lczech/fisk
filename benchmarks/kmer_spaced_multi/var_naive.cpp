#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "kmer_spaced_multi/bench.hpp"

using namespace fisk;

namespace {

/**
 * @brief Improvement on the MISSH implementation, by using a faster char encoding function.
 */
std::uint64_t compute_spaced_kmer_naive(
    std::string_view seq, std::vector<size_t> const& mask, size_t start_pos
) {
    // This is the same as compute_spaced_kmer_missh() in kmer_spaced_single/var_missh.cpp, with
    // the only difference being the use of the CharEncoderTable encoder instead of
    // CharEncoderSwitch. As the char encoding is called k times for each k-mer, this is
    // significantly faster.

    // Compute a single spaced kmer at the given position
    std::uint64_t result = 0;
    bool valid = true;
    for( auto p : mask ) {
        auto const c = static_cast<std::uint64_t>(
            CharEncoderTable<Encoding::kACGT>{}( seq[start_pos + p] )
        );
        valid &= (c < 4);
        result <<= 2;
        result |= c;
    }
    return valid ? result : 0;
}

} // namespace

std::uint64_t run_var_naive(
    std::string const& seq,
    std::size_t k,
    std::vector<std::vector<std::size_t>> const& naive_masks,
    std::vector<std::uint64_t>& sink_buffer
) {
    auto sink = make_sink(sink_buffer);
    std::size_t const stop = seq.size() - k;
    for (std::size_t i = 0; i <= stop; ++i) {
        for (auto const& mask : naive_masks) {
            sink.consume(compute_spaced_kmer_naive(std::string_view(seq), mask, i));
        }
    }
    return sink.finalize();
}
