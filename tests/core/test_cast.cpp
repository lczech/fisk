#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fisk/core/cast.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_wide.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/types.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "simd_lanes.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     K-mers
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Compile-Time Guarantees
// -------------------------------------------------------------------------------------------------

// kmer_value() of a vector of k-mers gives its vector, also with the single word of SimdScalar.
using KmerVectorScalar = KmerVector<Encoding::kACGT, Layout::kMSB, SimdScalar>;

static_assert(std::is_same_v<decltype(kmer_value(KmerVectorScalar{})), std::uint64_t>);

// The predicate behind the debug check in kmer_cast(): a word fits a width iff it has no bits set
// above it, including at the boundary widths 1 and 32.
static_assert(fits_width_(0x06, 3) && fits_width_(0x3F, 3));
static_assert(!fits_width_(0x40, 3));
static_assert(fits_width_(0x3, 1) && !fits_width_(0x4, 1));
static_assert(fits_width_(std::uint64_t{1} << 61, 31) && !fits_width_(std::uint64_t{1} << 62, 31));
static_assert(fits_width_(~std::uint64_t{0}, 32));

// -------------------------------------------------------------------------------------------------
//     Value Round Trip
// -------------------------------------------------------------------------------------------------

// kmer_value() must be exactly the word the k-mer was cast from, in both directions, under every
// convention, and kmer_cast_unchecked() must give the same k-mer.
template <typename K>
static void check_value_round_trip()
{
    constexpr Encoding E = K::encoding;
    constexpr Layout L = K::layout;
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 131313 + width)) {
            auto const raw = oracle_encode<E, L>(seq).value();
            auto const kmer = kmer_cast<E, L>(raw, width);
            EXPECT_EQ(kmer_value(kmer), raw);
            EXPECT_EQ((kmer_cast<E, L>(kmer_value(kmer), width)), kmer);
            EXPECT_EQ((kmer_cast_unchecked<E, L>(raw)), kmer);
            EXPECT_EQ(kmer_decode(kmer, width), seq);
        }
    }
}

TEST(CoreCast, ValueRoundTrip)
{
    check_value_round_trip<KmerAcgtMsb>();
    check_value_round_trip<KmerAcgtLsb>();
    check_value_round_trip<KmerActgMsb>();
    check_value_round_trip<KmerActgLsb>();
}

// A k-mer must never carry bits above its width, which is what kmer_cast() asserts on and what
// every operation below is entitled to assume.
template <typename K>
static void check_no_high_bits()
{
    for (std::size_t width = 1; width < 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 111111 + width)) {
            auto const kmer = oracle_kmer<K>(seq);
            EXPECT_EQ(kmer_value(kmer) >> (2 * width), std::uint64_t{0});
            EXPECT_EQ(kmer_value(complement(kmer, width)) >> (2 * width), std::uint64_t{0});
            EXPECT_EQ(kmer_value(reverse(kmer, width)) >> (2 * width), std::uint64_t{0});
            EXPECT_EQ(kmer_value(reverse_complement(kmer, width)) >> (2 * width), std::uint64_t{0});
        }
    }
}

TEST(CoreCast, NoBitsAboveWidth)
{
    check_no_high_bits<KmerAcgtMsb>();
    check_no_high_bits<KmerAcgtLsb>();
    check_no_high_bits<KmerActgMsb>();
    check_no_high_bits<KmerActgLsb>();
}

// -------------------------------------------------------------------------------------------------
//     K-mer Vectors
// -------------------------------------------------------------------------------------------------

// See check_kmer_vector() in simd_lanes.hpp.
TEST(CoreCast, KmerVector)
{
    check_kmer_vector<KmerAcgtMsb>(909090);
    check_kmer_vector<KmerAcgtLsb>(909090);
    check_kmer_vector<KmerActgMsb>(909090);
    check_kmer_vector<KmerActgLsb>(909090);
}

// =================================================================================================
//     Wide K-mers
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Compile-Time Guarantees
// -------------------------------------------------------------------------------------------------

// wide_kmer_cast() deduces the capacity from the number of words; kmer_cast() stays the narrow one.
static_assert(std::is_same_v<
    decltype(wide_kmer_cast<Encoding::kACTG, Layout::kLSB>(std::array<std::uint64_t, 3>{}, 65)),
    WideKmer<Encoding::kACTG, Layout::kLSB, 96>
>);
static_assert(std::is_same_v<
    decltype(
        wide_kmer_cast_unchecked<Encoding::kACGT, Layout::kMSB>(std::array<std::uint64_t, 1>{})
    ),
    WideKmer<Encoding::kACGT, Layout::kMSB, 32>
>);
static_assert(std::is_same_v<
    decltype(kmer_cast<Encoding::kACGT, Layout::kMSB>(std::uint64_t{0}, 1)), KmerAcgtMsb
>);

// The predicate behind the debug check in wide_kmer_cast(): words fit a width iff they have no bits
// set above it, neither in the partly used word nor in any wholly unused one.
using Words2 = std::array<std::uint64_t, 2>;
using Words3 = std::array<std::uint64_t, 3>;
static_assert(wide_fits_width_(Words2{0x3, 0}, 1) && !wide_fits_width_(Words2{0x4, 0}, 1));
static_assert(!wide_fits_width_(Words2{0x3, 1}, 1));
static_assert(wide_fits_width_(Words2{~std::uint64_t{0}, 0}, 32));
static_assert(!wide_fits_width_(Words2{~std::uint64_t{0}, 1}, 32));
static_assert(wide_fits_width_(Words2{~std::uint64_t{0}, 0x3}, 33));
static_assert(!wide_fits_width_(Words2{~std::uint64_t{0}, 0x4}, 33));
static_assert(wide_fits_width_(Words2{~std::uint64_t{0}, ~std::uint64_t{0}}, 64));
static_assert(!wide_fits_width_(Words3{~std::uint64_t{0}, ~std::uint64_t{0}, 1}, 64));

// =================================================================================================
//     Spaced K-mers
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Values and Vectors
// -------------------------------------------------------------------------------------------------

// spaced_kmer_cast() and spaced_kmer_cast_unchecked() must hold exactly the word they are cast
// from, under every convention.
template <typename K>
static void check_spaced_value_round_trip()
{
    constexpr Encoding E = K::encoding;
    constexpr Layout L = K::layout;
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 515151 + width)) {
            auto const raw = oracle_encode<E, L>(seq).value();
            auto const spaced = spaced_kmer_cast<E, L>(raw, width);
            EXPECT_EQ(kmer_value(spaced), raw);
            EXPECT_EQ((spaced_kmer_cast_unchecked<E, L>(raw)), spaced);
            EXPECT_EQ(kmer_decode(spaced, width), seq);
        }
    }
}

TEST(CoreCast, SpacedValueRoundTrip)
{
    check_spaced_value_round_trip<SpacedKmerAcgtMsb>();
    check_spaced_value_round_trip<SpacedKmerAcgtLsb>();
    check_spaced_value_round_trip<SpacedKmerActgMsb>();
    check_spaced_value_round_trip<SpacedKmerActgLsb>();
}

// See check_kmer_vector() in simd_lanes.hpp.
TEST(CoreCast, SpacedKmerVector)
{
    check_kmer_vector<SpacedKmerAcgtMsb>(919190);
    check_kmer_vector<SpacedKmerAcgtLsb>(919190);
    check_kmer_vector<SpacedKmerActgMsb>(919190);
    check_kmer_vector<SpacedKmerActgLsb>(919190);
}
