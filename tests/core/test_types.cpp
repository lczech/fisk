#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_wide.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/core/types.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     K-mers
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Compile-Time Guarantees
// -------------------------------------------------------------------------------------------------

// All four instantiations are distinct types, and none of them converts to another or to a raw
// word in either direction. This is the guarantee the whole type exists for.
static_assert(!std::is_same_v<KmerAcgtMsb, KmerAcgtLsb>);
static_assert(!std::is_same_v<KmerAcgtMsb, KmerActgMsb>);
static_assert(!std::is_convertible_v<KmerAcgtMsb, KmerActgMsb>);
static_assert(!std::is_convertible_v<KmerAcgtMsb, KmerAcgtLsb>);
static_assert(!std::is_convertible_v<KmerAcgtMsb, std::uint64_t>);
static_assert(!std::is_convertible_v<std::uint64_t, KmerAcgtMsb>);

// A vector of k-mers is not a k-mer, also not with the single word of SimdScalar, so that the
// operations on k-mers do not take it.
using KmerVectorScalar = KmerVector<Encoding::kACGT, Layout::kMSB, SimdScalar>;
static_assert(!KmerType<KmerVectorScalar>);
static_assert(KmerVectorType<KmerVectorScalar>);
static_assert(ContiguousKmerVectorType<KmerVectorScalar>);
static_assert(!SpacedKmerVectorType<KmerVectorScalar>);
static_assert(!KmerVectorType<KmerAcgtMsb>);
static_assert(!std::is_convertible_v<KmerVectorScalar, KmerAcgtMsb>);

namespace {

template <typename A, typename B>
concept AssignableFrom = requires(A a, B b) { a = b; };

template <typename A>
concept Addable = requires(A a, A b) { a + b; };

template <typename A>
concept Shiftable = requires(A a) { a << 2; };

} // namespace

// Mixing conventions must not compile even through assignment.
static_assert( AssignableFrom<KmerAcgtMsb, KmerAcgtMsb>);
static_assert(!AssignableFrom<KmerAcgtMsb, KmerActgMsb>);
static_assert(!AssignableFrom<KmerAcgtMsb, KmerAcgtLsb>);
static_assert(!AssignableFrom<KmerAcgtMsb, std::uint64_t>);

// Raw integer manipulation is deliberately not available on the type: such operations have no
// k-mer meaning and would not preserve the invariant that only the lowest 2*width bits are set.
// They go through kmer_value() instead, where they are visibly low level.
static_assert(!Addable<KmerAcgtMsb>);
static_assert(!Shiftable<KmerAcgtMsb>);

// The k-mer-like concept accepts k-mers and nothing accidental.
static_assert(KmerType<KmerAcgtMsb>);
static_assert(KmerType<KmerActgLsb>);
static_assert(!KmerType<std::uint64_t>);
static_assert(!KmerType<PackedSequence<Encoding::kACGT, Layout::kMSB>>);

// Any convention-tagged container yields its matching k-mer type through the trait, without the
// container itself having to know that k-mers exist.
static_assert(
    std::is_same_v<kmer_type_of<PackedSequence<Encoding::kACGT, Layout::kMSB>>, KmerAcgtMsb>
);
static_assert(
    std::is_same_v<kmer_type_of<PackedSequence<Encoding::kACTG, Layout::kLSB>>, KmerActgLsb>
);
static_assert(
    std::is_same_v<kmer_type_of<KmerAcgtLsb>, KmerAcgtLsb>
);

// -------------------------------------------------------------------------------------------------
//     Ordering
// -------------------------------------------------------------------------------------------------

// Only ACGT combined with MSB orders k-mers the way the decoded strings order: ACTG because its
// codes are not alphabetical, LSB because the first base sits in the low bits.
TEST(CoreTypes, LexicographicOrderOnlyForAcgtMsb)
{
    std::size_t const width = 8;
    auto seqs = random_seqs(width, 40, 123123);
    std::sort(seqs.begin(), seqs.end());

    bool acgt_msb_ordered = true;
    bool acgt_lsb_ordered = true;
    bool actg_msb_ordered = true;
    for (std::size_t i = 1; i < seqs.size(); ++i) {
        acgt_msb_ordered &= oracle_kmer<KmerAcgtMsb>(seqs[i - 1]) < oracle_kmer<KmerAcgtMsb>(seqs[i]);
        acgt_lsb_ordered &= oracle_kmer<KmerAcgtLsb>(seqs[i - 1]) < oracle_kmer<KmerAcgtLsb>(seqs[i]);
        actg_msb_ordered &= oracle_kmer<KmerActgMsb>(seqs[i - 1]) < oracle_kmer<KmerActgMsb>(seqs[i]);
    }
    EXPECT_TRUE(acgt_msb_ordered);
    EXPECT_FALSE(acgt_lsb_ordered);
    EXPECT_FALSE(actg_msb_ordered);
}

// Equality and ordering are by raw word, and both are usable without naming a comparator.
TEST(CoreTypes, Comparison)
{
    auto const a = KmerAcgtMsb{7};
    auto const b = KmerAcgtMsb{9};
    EXPECT_TRUE(a == a);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= a);
}

// =================================================================================================
//     Wide K-mers
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Helpers
// -------------------------------------------------------------------------------------------------

// Ordering is numeric over all words, highest word first, independent of the conventions. The
// first pair differs in opposite directions in its lowest and highest word, so that comparing from
// the wrong end gives the wrong answer; the second pair differs only in the lowest word.
template <typename K>
static void check_word_order()
{
    static_assert(K::word_count >= 2);
    using Words = std::array<std::uint64_t, K::word_count>;

    Words low_heavy{};
    low_heavy[0] = ~std::uint64_t{0};
    Words high_light{};
    high_light[K::word_count - 1] = 1;
    auto high_light_plus = high_light;
    high_light_plus[0] = 1;

    K const a{low_heavy};
    K const b{high_light};
    K const c{high_light_plus};
    EXPECT_TRUE(a == a);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= a);
    EXPECT_TRUE(b >= b);
    EXPECT_TRUE(b < c);
    EXPECT_FALSE(c < b);
}

// -------------------------------------------------------------------------------------------------
//     Compile-Time Guarantees
// -------------------------------------------------------------------------------------------------

// Capacity sets the number of words, while the width remains a runtime argument of every operation.
static_assert(WideKmerAcgtMsb::capacity == 64 && WideKmerAcgtMsb::word_count == 2);
static_assert(WideKmer<Encoding::kACGT, Layout::kMSB, 32>::word_count == 1);
static_assert(WideKmer<Encoding::kACGT, Layout::kMSB, 128>::word_count == 4);
static_assert(std::is_same_v<
    decltype(kmer_value(WideKmerAcgtMsb{})), std::array<std::uint64_t, 2>
>);

// Wide, contiguous and spaced k-mers satisfy distinct concepts, and rebind keeps the capacity.
static_assert(WideKmerType<WideKmerAcgtMsb>);
static_assert(!WideKmerType<KmerAcgtMsb>);
static_assert(!WideKmerType<std::uint64_t>);
static_assert(!KmerType<WideKmerAcgtMsb>);
static_assert(!ContiguousKmerType<WideKmerAcgtMsb>);
static_assert(!std::is_same_v<WideKmerAcgtMsb, WideKmerActgMsb>);
static_assert(!std::is_same_v<WideKmerAcgtMsb, WideKmerAcgtLsb>);
static_assert(!std::is_convertible_v<WideKmerAcgtMsb, WideKmerActgMsb>);
static_assert(!std::is_convertible_v<WideKmerAcgtMsb, WideKmerAcgtLsb>);
static_assert(std::is_same_v<
    WideKmer<Encoding::kACGT, Layout::kMSB, 96>::rebind<Encoding::kACTG, Layout::kLSB>,
    WideKmer<Encoding::kACTG, Layout::kLSB, 96>
>);

// -------------------------------------------------------------------------------------------------
//     Ordering
// -------------------------------------------------------------------------------------------------

TEST(CoreTypes, WideComparison)
{
    check_word_order<WideKmerAcgtMsb>();
    check_word_order<WideKmerActgLsb>();
    check_word_order<WideKmer<Encoding::kACGT, Layout::kLSB, 96>>();
    check_word_order<WideKmer<Encoding::kACTG, Layout::kMSB, 128>>();
}

// As for Kmer, the order matches lexicographic order of the decoded strings for ACGT/MSB, here
// across word boundaries.
TEST(CoreTypes, WideLexicographicOrderForAcgtMsb)
{
    using K = WideKmer<Encoding::kACGT, Layout::kMSB, 128>;
    for (std::size_t const width : {std::size_t{33}, std::size_t{64}, std::size_t{100}}) {
        auto seqs = random_seqs(width, 40, 160000 + width);
        std::sort(seqs.begin(), seqs.end());
        seqs.erase(std::unique(seqs.begin(), seqs.end()), seqs.end());

        bool ordered = true;
        for (std::size_t i = 1; i < seqs.size(); ++i) {
            ordered &= kmer_encode<K>(seqs[i - 1]) < kmer_encode<K>(seqs[i]);
        }
        EXPECT_TRUE(ordered);
    }
}

// =================================================================================================
//     Spaced K-mers
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Spaced K-mer Type
// -------------------------------------------------------------------------------------------------

static_assert(KmerType<SpacedKmerAcgtMsb>);
static_assert(!ContiguousKmerType<SpacedKmerAcgtMsb>);
static_assert(ContiguousKmerType<KmerAcgtMsb>);
static_assert(SpacedKmerType<SpacedKmerActgLsb>);
static_assert(!SpacedKmerType<KmerAcgtMsb>);
static_assert(!SpacedKmerType<std::uint64_t>);
static_assert(std::is_same_v<SpacedKmerActgLsb, SpacedKmer<Encoding::kACTG, Layout::kLSB>>);

using SpacedKmerVectorScalar = SpacedKmerVector<Encoding::kACGT, Layout::kMSB, SimdScalar>;
static_assert(KmerVectorType<SpacedKmerVectorScalar>);
static_assert(SpacedKmerVectorType<SpacedKmerVectorScalar>);
static_assert(!ContiguousKmerVectorType<SpacedKmerVectorScalar>);
static_assert(!KmerType<SpacedKmerVectorScalar>);

static_assert(SpacedKmerMaskType<SpacedKmerMask<Layout::kMSB, BitExtractMask>>);
static_assert(
    SpacedKmerMaskType<SpacedKmerMask<Layout::kLSB, BitExtractKernelButterfly<SimdScalar>>>
);
static_assert(!SpacedKmerMaskType<BitExtractMask>);
static_assert(!SpacedKmerMaskType<BitExtractKernelButterfly<SimdScalar>>);
static_assert(!std::is_aggregate_v<SpacedKmerMask<Layout::kMSB, BitExtractMask>>);

TEST(CoreTypes, SpacedComparison)
{
    auto const a = SpacedKmerAcgtMsb{7};
    auto const b = SpacedKmerAcgtMsb{9};
    EXPECT_TRUE(a == a);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= a);
}

// =================================================================================================
//     Packed Sequences
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Packed Sequence Classes
// -------------------------------------------------------------------------------------------------

// Construction, resize() and moves of a sequence with validity, which resizes its codes as a
// PackedSequence does: codes that remain keep their values, new ones are zero, and everything
// beyond the length is zero, see check_zero_beyond_length().
template <Layout L>
static void check_construct_resize_move()
{
    using Packed = PackedSequenceWithValidity<Encoding::kACGT, L>;
    auto const fill = [](std::span<std::uint8_t> bytes) {
        std::fill(bytes.begin(), bytes.end(), std::uint8_t{0xFF});
    };

    // Empty without storage, and of the given length with all codes 0 and all invalid.
    Packed const empty;
    EXPECT_EQ(empty.length(), std::size_t{0});
    EXPECT_TRUE(empty.codes().empty());
    EXPECT_TRUE(empty.validity().empty());
    Packed packed(9);
    EXPECT_EQ(packed.length(), std::size_t{9});
    auto const all_zero = [](std::span<std::uint8_t const> bytes) {
        return std::all_of(bytes.begin(), bytes.end(), [](auto b) { return b == 0; });
    };
    EXPECT_TRUE(all_zero(packed.codes()));
    EXPECT_TRUE(all_zero(packed.validity()));
    check_zero_beyond_length(packed);

    // Shrinking clears the bits beyond the new length, and the codes after it, which are now
    // padding: 6 nucleotides use 2 bits of the second byte, at its top for kMSB.
    fill(packed.codes());
    fill(packed.validity());
    packed.resize(6);
    std::uint8_t const second = (L == Layout::kMSB) ? 0xF0 : 0x0F;
    ASSERT_EQ(packed.codes().size(), std::size_t{2});
    EXPECT_EQ(packed.codes()[0], 0xFFu);
    EXPECT_EQ(packed.codes()[1], second);
    ASSERT_EQ(packed.validity().size(), std::size_t{1});
    EXPECT_EQ(packed.validity()[0], 0x3Fu);
    check_zero_beyond_length(packed);

    // Growing keeps those, and adds zero codes and invalid nucleotides.
    packed.resize(13);
    ASSERT_EQ(packed.codes().size(), std::size_t{4});
    EXPECT_EQ(packed.codes()[1], second);
    EXPECT_EQ(packed.codes()[2], 0u);
    EXPECT_EQ(packed.codes()[3], 0u);
    ASSERT_EQ(packed.validity().size(), std::size_t{2});
    EXPECT_EQ(packed.validity()[0], 0x3Fu);
    EXPECT_EQ(packed.validity()[1], 0u);
    check_zero_beyond_length(packed);

    // A move leaves the moved-from sequence empty.
    Packed moved = std::move(packed);
    EXPECT_EQ(moved.length(), std::size_t{13});
    EXPECT_EQ(packed.length(), std::size_t{0});
    EXPECT_TRUE(packed.codes().empty());
    EXPECT_TRUE(packed.validity().empty());
    packed = std::move(moved);
    EXPECT_EQ(packed.length(), std::size_t{13});
    EXPECT_EQ(moved.length(), std::size_t{0});
    EXPECT_TRUE(moved.codes().empty());
    check_zero_beyond_length(packed);

    packed.resize(0);
    EXPECT_EQ(packed.length(), std::size_t{0});
    EXPECT_TRUE(packed.codes().empty());
    EXPECT_TRUE(packed.validity().empty());
}

TEST(CoreTypes, PackedSequenceConstructResizeMoveMsb)
{
    check_construct_resize_move<Layout::kMSB>();
}

TEST(CoreTypes, PackedSequenceConstructResizeMoveLsb)
{
    check_construct_resize_move<Layout::kLSB>();
}

// -------------------------------------------------------------------------------------------------
//     Compile-Time Conventions
// -------------------------------------------------------------------------------------------------

namespace {

// Encoding and Layout are part of PackedSequence's type, so mixing them up between producer and
// consumer is rejected by the compiler. These checks pin that guarantee down.
using AcgtMsb = PackedSequence<Encoding::kACGT, Layout::kMSB>;
using ActgMsb = PackedSequence<Encoding::kACTG, Layout::kMSB>;

} // namespace

// Nor can a sequence of one encoding be passed where another is expected.
static_assert(!std::is_convertible_v<ActgMsb const&, AcgtMsb const&>);

// A sequence with validity is not a plain one, so that its codes cannot be resized without its
// validity bits, but it gives one with sequence().
using AcgtMsbWithValidity = PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>;
static_assert(!std::is_convertible_v<AcgtMsbWithValidity&, AcgtMsb&>);
static_assert(!std::is_convertible_v<AcgtMsbWithValidity const&, AcgtMsb const&>);
static_assert(std::is_same_v<
    decltype(std::declval<AcgtMsbWithValidity const&>().sequence()), AcgtMsb const&
>);

// Both are packed sequences, and only the one with validity has validity bits. A type with
// validity bits but without the rest of a packed sequence is neither.
struct OnlyValidityBits
{
    std::span<std::uint8_t const> validity() const;
};

static_assert( PackedSequenceType<AcgtMsb>);
static_assert( PackedSequenceType<AcgtMsbWithValidity>);
static_assert(!PackedSequenceType<std::vector<std::uint8_t>>);
static_assert(!PackedSequenceType<OnlyValidityBits>);
static_assert(!PackedSequenceWithValidityType<AcgtMsb>);
static_assert( PackedSequenceWithValidityType<AcgtMsbWithValidity>);
static_assert(!PackedSequenceWithValidityType<OnlyValidityBits>);
