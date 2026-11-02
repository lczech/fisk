#include "spaced_kmer_extract/ascii_chunked_scalar_common.hpp"
#include "testing.hpp"

// =================================================================================================
//     for_each_spaced_kmer_ascii_checked_chunked_scalar()
// =================================================================================================

TEST(SpacedKmerExtractAsciiChunkedScalar, Checked)
{
    check_chunked<true>();
}

TEST(SpacedKmerExtractAsciiChunkedScalar, CheckedErrorContract)
{
    check_chunked_error_contract<true, 64>();
    check_chunked_error_contract<true, 128>();
}
