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
//     for_each_kmer_ascii_checked_chunked_scalar()
// =================================================================================================

// The function as an `extract` for the checks of the chunked loops, see kmer_extract/common.hpp.
static constexpr auto checked = []<std::size_t ChunkSize>(
    std::string const& seq, std::size_t k, auto const& encoder, auto* scratch,
    auto const& callback
) {
    if (scratch) {
        for_each_kmer_ascii_checked_chunked_scalar<ChunkSize>(seq, k, encoder, *scratch, callback);
    } else {
        for_each_kmer_ascii_checked_chunked_scalar<ChunkSize>(seq, k, encoder, callback);
    }
};

TEST(KmerExtractAsciiChunkedScalar, Checked)
{
    check_ascii_chunked_all<true>(checked);
}

// The checks of the checked extraction from ASCII, with chunks of 7 bases.
TEST(KmerExtractAsciiChunkedScalar, CheckedAsciiChecks)
{
    auto const extract = ascii_chunked_as_ascii_extractor(checked);
    check_ascii_oracle(extract);
    check_ascii_length_boundaries(extract);
    check_ascii_invalid_character_placement(extract);
    check_ascii_invalid_injection_fuzz(extract);
    check_ascii_block_boundaries(extract);
    check_ascii_mixed_case(extract);
}
