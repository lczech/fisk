#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/cast.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_chunked_scalar.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "kmer_extract/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     for_each_kmer_ascii_unchecked_chunked_scalar()
// =================================================================================================

// The function as an `extract` for the checks of the chunked loops, see kmer_extract/common.hpp.
static constexpr auto unchecked = []<std::size_t ChunkSize>(
    std::string const& seq, std::size_t k, auto const& encoder, auto* scratch,
    auto const& callback
) {
    if (scratch) {
        for_each_kmer_ascii_unchecked_chunked_scalar<ChunkSize>(
            seq, k, encoder, *scratch, callback
        );
    } else {
        for_each_kmer_ascii_unchecked_chunked_scalar<ChunkSize>(seq, k, encoder, callback);
    }
};

TEST(KmerExtractAsciiChunkedScalar, Unchecked)
{
    check_ascii_chunked_all<false>(unchecked);
}

// The default chunk size, kDefaultAsciiChunkSize, must be exactly correct as well, in both
// callback forms.
TEST(KmerExtractAsciiChunkedScalar, UncheckedDefaultChunkSize)
{
    WordEncoderButterfly<Encoding::kACGT, Layout::kMSB> encoder;
    PackedSequence<Encoding::kACGT, Layout::kMSB> scratch;
    for (auto const& seq : chunked_test_sequences()) {
        for (std::size_t k = 1; k <= 32; ++k) {
            auto const expected = oracle_kmers_all_windows<Encoding::kACGT, Layout::kMSB>(seq, k);
            check_kmer_callbacks<Encoding::kACGT, Layout::kMSB>(expected, k, [&](auto const& cb) {
                for_each_kmer_ascii_unchecked_chunked_scalar(seq, k, encoder, cb);
            });
            check_kmer_callbacks<Encoding::kACGT, Layout::kMSB>(expected, k, [&](auto const& cb) {
                for_each_kmer_ascii_unchecked_chunked_scalar(seq, k, encoder, scratch, cb);
            });
        }
    }
}

// Separate caller-owned buffers make same-encoder extraction inside a callback safe.
TEST(KmerExtractAsciiChunkedScalar, ScratchEnablesNestedSameEncoder)
{
    constexpr Encoding E = Encoding::kACGT;
    constexpr Layout L = Layout::kMSB;
    constexpr std::size_t k = 21;
    WordEncoderButterfly<E, L> encoder;
    std::string const& outer = chunked_test_sequences()[100];
    std::string const& inner = chunked_test_sequences()[63];
    auto const outer_expected = oracle_kmers_all_windows<E, L>(outer, k);
    auto const inner_expected = oracle_kmers_all_windows<E, L>(inner, k);
    PackedSequence<E, L> outer_scratch;
    PackedSequence<E, L> inner_scratch;
    std::vector<std::size_t> outer_positions;
    std::vector<std::uint64_t> outer_values;

    for_each_kmer_ascii_unchecked_chunked_scalar(
        outer, k, encoder, outer_scratch,
        [&](std::size_t outer_pos, Kmer<E, L> outer_kmer) {
            std::vector<std::size_t> inner_positions;
            std::vector<std::uint64_t> inner_values;
            for_each_kmer_ascii_unchecked_chunked_scalar(
                inner, k, encoder, inner_scratch,
                [&](std::size_t inner_pos, Kmer<E, L> inner_kmer) {
                    inner_positions.push_back(inner_pos);
                    inner_values.push_back(kmer_value(inner_kmer));
                }
            );
            check_emitted_kmers(inner_expected, k, inner_values, &inner_positions);
            outer_positions.push_back(outer_pos);
            outer_values.push_back(kmer_value(outer_kmer));
        }
    );
    check_emitted_kmers(outer_expected, k, outer_values, &outer_positions);
}
