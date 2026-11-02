#include "spaced_kmer_extract/ascii_chunked_scalar_common.hpp"
#include "testing.hpp"

// =================================================================================================
//     for_each_spaced_kmer_ascii_unchecked_chunked_scalar()
// =================================================================================================

TEST(SpacedKmerExtractAsciiChunkedScalar, Unchecked)
{
    check_chunked<false>();
}

TEST(SpacedKmerExtractAsciiChunkedScalar, UncheckedErrorContract)
{
    check_chunked_error_contract<false, 64>();
    check_chunked_error_contract<false, 128>();
}
