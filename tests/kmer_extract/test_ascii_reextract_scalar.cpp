#include <cstddef>
#include <string>

#include "fisk/core/char_encoder.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_reextract_scalar.hpp"
#include "kmer_extract/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// for_each_kmer_ascii_checked_reextract_scalar() for the checks of kmer_extract/common.hpp: with
// the lookup table encoder of E, and with any encoder.
static constexpr auto extract = []<Encoding E, Layout L>(
    std::string const& seq, std::size_t k, auto const& callback
) {
    for_each_kmer_ascii_checked_reextract_scalar<L>(seq, k, CharEncoderTable<E>{}, callback);
};
static constexpr auto extract_with = []<Layout L>(
    std::string const& seq, std::size_t k, auto const& enc, auto const& callback
) {
    for_each_kmer_ascii_checked_reextract_scalar<L>(seq, k, enc, callback);
};

// =================================================================================================
//     Oracle Correctness
// =================================================================================================

// See check_ascii_oracle(): all four conventions on the valid-only sequences.
TEST(KmerExtractAsciiReextractScalar, Oracle)
{
    check_ascii_oracle(extract);
}

// =================================================================================================
//     k and Length Boundaries
// =================================================================================================

// k must be in [1, 32]; k == 0 or k > 32 is a documented precondition violation.
TEST(KmerExtractAsciiReextractScalar, InvalidKThrows)
{
    check_invalid_k_throws(32, [](std::string const& seq, std::size_t k) {
        extract.template operator()<Encoding::kACGT, Layout::kMSB>(seq, k, [](KmerAcgtMsb) {});
    });
}

TEST(KmerExtractAsciiReextractScalar, LengthBoundaries)
{
    check_ascii_length_boundaries(extract);
}

// =================================================================================================
//     Invalid Characters
// =================================================================================================

TEST(KmerExtractAsciiReextractScalar, InvalidCharacterPlacement)
{
    check_ascii_invalid_character_placement(extract);
}

// See SentinelNotFourEncoder: an encoder whose invalid sentinel is not exactly 4.
TEST(KmerExtractAsciiReextractScalar, InvalidSentinelNotFour)
{
    check_ascii_invalid_sentinel_not_four(extract_with);
}

TEST(KmerExtractAsciiReextractScalar, InvalidInjectionFuzz)
{
    check_ascii_invalid_injection_fuzz(extract);
}

// =================================================================================================
//     Block Boundaries
// =================================================================================================

TEST(KmerExtractAsciiReextractScalar, BlockBoundaries)
{
    check_ascii_block_boundaries(extract);
}

// =================================================================================================
//     Case Insensitivity
// =================================================================================================

TEST(KmerExtractAsciiReextractScalar, MixedCase)
{
    check_ascii_mixed_case(extract);
}

// =================================================================================================
//     Conventions
// =================================================================================================

// Without template arguments, the extractor keeps emitting ACGT/MSB. The callback only accepts
// KmerAcgtMsb, so any other default fails to compile.
TEST(KmerExtractAsciiReextractScalar, DefaultConventions)
{
    std::string const seq = "ACGTA";
    std::size_t count = 0;
    auto const callback = [&](KmerAcgtMsb) { ++count; };
    CharEncoderTable<Encoding::kACGT> const table;
    for_each_kmer_ascii_checked_reextract_scalar(seq, 3, table, callback);
    EXPECT_EQ(count, std::size_t{3});
}
