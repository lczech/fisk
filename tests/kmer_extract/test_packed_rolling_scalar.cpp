#include <cstddef>

#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/packed_rolling_scalar.hpp"
#include "kmer_extract/common.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// The extractors under test, each wrapped once so that it can be handed to the checks of
// kmer_extract/common.hpp under every convention.
static constexpr auto unchecked = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_unchecked_rolling_scalar(seq, k, func);
};
static constexpr auto checked = [](auto const& seq, std::size_t k, auto const& func) {
    for_each_kmer_packed_checked_rolling_scalar(seq, k, func);
};
// The `check` of the checked extractor, see checked_kmers_check().
static auto const checked_check = checked_kmers_check(checked);
static constexpr auto checked_throws = [](auto const& seq, std::size_t k) {
    for_each_kmer_packed_checked_rolling_scalar(seq, k, [](auto) {});
};

// =================================================================================================
//     Unchecked
// =================================================================================================

TEST(KmerExtractPackedRollingScalar, UncheckedAcgtMsb)
{
    check_packed_variant<Encoding::kACGT, Layout::kMSB>(unchecked);
}

TEST(KmerExtractPackedRollingScalar, UncheckedAcgtLsb)
{
    check_packed_variant<Encoding::kACGT, Layout::kLSB>(unchecked);
}

TEST(KmerExtractPackedRollingScalar, UncheckedActgMsb)
{
    check_packed_variant<Encoding::kACTG, Layout::kMSB>(unchecked);
}

TEST(KmerExtractPackedRollingScalar, UncheckedActgLsb)
{
    check_packed_variant<Encoding::kACTG, Layout::kLSB>(unchecked);
}

// =================================================================================================
//     Checked
// =================================================================================================

TEST(KmerExtractPackedRollingScalar, CheckedRandomAcgtMsb)
{
    check_checked_random_sequences<Encoding::kACGT, Layout::kMSB>(checked_check);
}

TEST(KmerExtractPackedRollingScalar, CheckedRandomAcgtLsb)
{
    check_checked_random_sequences<Encoding::kACGT, Layout::kLSB>(checked_check);
}

TEST(KmerExtractPackedRollingScalar, CheckedRandomActgMsb)
{
    check_checked_random_sequences<Encoding::kACTG, Layout::kMSB>(checked_check);
}

TEST(KmerExtractPackedRollingScalar, CheckedRandomActgLsb)
{
    check_checked_random_sequences<Encoding::kACTG, Layout::kLSB>(checked_check);
}

TEST(KmerExtractPackedRollingScalar, CheckedSingleInvalidMsb)
{
    check_checked_single_invalid<Encoding::kACGT, Layout::kMSB>(checked_check);
}

TEST(KmerExtractPackedRollingScalar, CheckedSingleInvalidLsb)
{
    check_checked_single_invalid<Encoding::kACGT, Layout::kLSB>(checked_check);
}

TEST(KmerExtractPackedRollingScalar, CheckedValidRunsMsb)
{
    check_checked_valid_runs<Encoding::kACGT, Layout::kMSB>(checked_check);
}

TEST(KmerExtractPackedRollingScalar, CheckedValidRunsLsb)
{
    check_checked_valid_runs<Encoding::kACGT, Layout::kLSB>(checked_check);
}

TEST(KmerExtractPackedRollingScalar, CheckedAllValidEqualsUncheckedMsb)
{
    check_checked_all_valid_equals_unchecked<Encoding::kACGT, Layout::kMSB>(checked_check);
}

TEST(KmerExtractPackedRollingScalar, CheckedAllValidEqualsUncheckedLsb)
{
    check_checked_all_valid_equals_unchecked<Encoding::kACGT, Layout::kLSB>(checked_check);
}

TEST(KmerExtractPackedRollingScalar, CheckedInvalidInValidityOnlyMsb)
{
    check_checked_invalid_in_validity_only<Encoding::kACGT, Layout::kMSB>(checked_check, unchecked);
}

TEST(KmerExtractPackedRollingScalar, CheckedInvalidInValidityOnlyLsb)
{
    check_checked_invalid_in_validity_only<Encoding::kACGT, Layout::kLSB>(checked_check, unchecked);
}

TEST(KmerExtractPackedRollingScalar, CheckedThrowsMsb)
{
    check_checked_throws<Encoding::kACGT, Layout::kMSB>(checked_throws);
}

TEST(KmerExtractPackedRollingScalar, CheckedThrowsLsb)
{
    check_checked_throws<Encoding::kACGT, Layout::kLSB>(checked_throws);
}
