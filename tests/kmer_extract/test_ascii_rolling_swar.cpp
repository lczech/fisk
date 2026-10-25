#include <cstddef>
#include <string>

#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_rolling_swar.hpp"
#include "kmer_extract/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// for_each_kmer_ascii_checked_rolling_swar() for the checks of kmer_extract/common.hpp.
static constexpr auto extract = []<Encoding E, Layout L>(
    std::string const& seq, std::size_t k, auto const& callback
) {
    for_each_kmer_ascii_checked_rolling_swar<E, L>(seq, k, callback);
};

// =================================================================================================
//     Oracle Correctness
// =================================================================================================

// See check_ascii_oracle(): all four conventions on the valid-only sequences.
TEST(KmerExtractAsciiRollingSwar, Oracle)
{
    check_ascii_oracle(extract);
}

// =================================================================================================
//     k and Length Boundaries
// =================================================================================================

// k must be in [1, 32]; k == 0 or k > 32 is a documented precondition violation.
TEST(KmerExtractAsciiRollingSwar, InvalidKThrows)
{
    check_invalid_k_throws(32, [](std::string const& seq, std::size_t k) {
        extract.template operator()<Encoding::kACGT, Layout::kMSB>(seq, k, [](KmerAcgtMsb) {});
    });
}

TEST(KmerExtractAsciiRollingSwar, LengthBoundaries)
{
    check_ascii_length_boundaries(extract);
}

// =================================================================================================
//     Invalid Characters
// =================================================================================================

TEST(KmerExtractAsciiRollingSwar, InvalidCharacterPlacement)
{
    check_ascii_invalid_character_placement(extract);
}

TEST(KmerExtractAsciiRollingSwar, InvalidInjectionFuzz)
{
    check_ascii_invalid_injection_fuzz(extract);
}

// =================================================================================================
//     Block Boundaries
// =================================================================================================

TEST(KmerExtractAsciiRollingSwar, BlockBoundaries)
{
    check_ascii_block_boundaries(extract);
}

// =================================================================================================
//     Case Insensitivity
// =================================================================================================

TEST(KmerExtractAsciiRollingSwar, MixedCase)
{
    check_ascii_mixed_case(extract);
}

// =================================================================================================
//     Conventions
// =================================================================================================

// Without template arguments, the extractor keeps emitting ACGT/MSB. The callback only accepts
// KmerAcgtMsb, so any other default fails to compile.
TEST(KmerExtractAsciiRollingSwar, DefaultConventions)
{
    std::string const seq = "ACGTA";
    std::size_t count = 0;
    auto const callback = [&](KmerAcgtMsb) { ++count; };
    for_each_kmer_ascii_checked_rolling_swar(seq, 3, callback);
    EXPECT_EQ(count, std::size_t{3});
}
