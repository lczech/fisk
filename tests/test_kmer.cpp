#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "fisk/core/cast.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/kmer_hash.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "simd_lanes.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Ground truth comes from oracle.hpp, built from nucleotide strings and deliberately independent
// of core/kmer.hpp, so that a mistake in the bit tricks under test cannot cancel itself out
// against the reference. Random inputs come from random_seqs() in corpus.hpp.

// =================================================================================================
//     Compile-Time Guarantees
// =================================================================================================

// All four instantiations are distinct types, and none of them converts to another or to a raw
// word in either direction. This is the guarantee the whole type exists for.
static_assert(!std::is_same_v<KmerAcgtMsb, KmerAcgtLsb>);
static_assert(!std::is_same_v<KmerAcgtMsb, KmerActgMsb>);
static_assert(!std::is_convertible_v<KmerAcgtMsb, KmerActgMsb>);
static_assert(!std::is_convertible_v<KmerAcgtMsb, KmerAcgtLsb>);
static_assert(!std::is_convertible_v<KmerAcgtMsb, std::uint64_t>);
static_assert(!std::is_convertible_v<std::uint64_t, KmerAcgtMsb>);

// A vector of k-mers is not a k-mer, also not with the single word of SimdScalar, so that the
// operations on k-mers do not take it, and kmer_value() gives its vector.
using KmerVectorScalar = KmerVector<Encoding::kACGT, Layout::kMSB, SimdScalar>;
static_assert(!KmerType<KmerVectorScalar>);
static_assert(KmerVectorType<KmerVectorScalar>);
static_assert(!KmerVectorType<KmerAcgtMsb>);
static_assert(!std::is_convertible_v<KmerVectorScalar, KmerAcgtMsb>);
static_assert(std::is_same_v<decltype(kmer_value(KmerVectorScalar{})), std::uint64_t>);

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

// kmer_convert() picks its overload purely from which kind of template argument it is given: an
// Encoding changes the encoding, a Layout the layout, and both change both.
static_assert(
    std::is_same_v<decltype(kmer_convert<Encoding::kACTG>(KmerAcgtMsb{0}, 1)), KmerActgMsb>
);
static_assert(
    std::is_same_v<decltype(kmer_convert<Layout::kLSB>(KmerAcgtMsb{0}, 1)), KmerAcgtLsb>
);
static_assert(
    std::is_same_v<decltype(kmer_convert<Encoding::kACTG, Layout::kLSB>(KmerAcgtMsb{0}, 1)), KmerActgLsb>
);

// The predicate behind the debug check in kmer_cast(): a word fits a width iff it has no bits set
// above it, including at the boundary widths 1 and 32.
static_assert(fits_width_(0x06, 3) && fits_width_(0x3F, 3));
static_assert(!fits_width_(0x40, 3));
static_assert(fits_width_(0x3, 1) && !fits_width_(0x4, 1));
static_assert(fits_width_(std::uint64_t{1} << 61, 31) && !fits_width_(std::uint64_t{1} << 62, 31));
static_assert(fits_width_(~std::uint64_t{0}, 32));

// Every operation is constexpr, so k-mers can be built and manipulated at compile time. Spelled
// out on one worked example, ACG under ACGT/MSB, whose expected words are short enough to read.
namespace {
    constexpr auto acg = kmer_cast<Encoding::kACGT, Layout::kMSB>(0x06, 3);
}
static_assert(kmer_value(acg) == 0x06);
static_assert(kmer_encode("ACG") == acg);
static_assert(base_at(acg, 0, 3) == 0 && base_at(acg, 2, 3) == 2);
static_assert(complement(acg, 3) == KmerAcgtMsb{0x39});          // TGC
static_assert(reverse(acg, 3) == KmerAcgtMsb{0x24});             // GCA
static_assert(reverse_complement(acg, 3) == KmerAcgtMsb{0x1B});  // CGT
static_assert(canonical(acg, 3) == acg);
static_assert(kmer_convert<Layout::kLSB>(acg, 3) == KmerAcgtLsb{0x24});
static_assert(kmer_convert<Encoding::kACTG>(acg, 3) == KmerActgMsb{0x07});

// =================================================================================================
//     Decoding and Base Access
// =================================================================================================

// kmer_decode() must return the string the k-mer was built from, under every convention.
template <typename K>
static void check_decode()
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 8, 90210 + width)) {
            EXPECT_EQ(kmer_decode(oracle_kmer<K>(seq), width), seq);
        }
    }
}

TEST(Kmer, Decode)
{
    check_decode<KmerAcgtMsb>();
    check_decode<KmerAcgtLsb>();
    check_decode<KmerActgMsb>();
    check_decode<KmerActgLsb>();
}

// base_at() must agree with indexing the decoded string, so callers never need the shift.
template <typename K>
static void check_base_at()
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 424242 + width)) {
            auto const kmer = oracle_kmer<K>(seq);
            for (std::size_t i = 0; i < width; ++i) {
                EXPECT_EQ(
                    static_cast<int>(base_at(kmer, i, width)),
                    static_cast<int>(oracle_code<K::encoding>(seq[i]))
                );
            }
        }
    }
}

TEST(Kmer, BaseAt)
{
    check_base_at<KmerAcgtMsb>();
    check_base_at<KmerAcgtLsb>();
    check_base_at<KmerActgMsb>();
    check_base_at<KmerActgLsb>();
}

// kmer_value() must be exactly the word the k-mer was cast from, in both directions.
TEST(Kmer, ValueRoundTrip)
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 131313 + width)) {
            auto const raw = oracle_encode<Encoding::kACGT, Layout::kMSB>(seq).value();
            auto const kmer = kmer_cast<Encoding::kACGT, Layout::kMSB>(raw, width);
            EXPECT_EQ(kmer_value(kmer), raw);
            EXPECT_EQ((kmer_cast<Encoding::kACGT, Layout::kMSB>(kmer_value(kmer), width)), kmer);
        }
    }
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

TEST(Kmer, NoBitsAboveWidth)
{
    check_no_high_bits<KmerAcgtMsb>();
    check_no_high_bits<KmerAcgtLsb>();
    check_no_high_bits<KmerActgMsb>();
    check_no_high_bits<KmerActgLsb>();
}

// =================================================================================================
//     Nucleotide Operations
// =================================================================================================

// complement, reverse and reverse_complement must match doing the same thing to the string, and
// each must be its own inverse. Checked under both encodings, since the complement constant
// differs between them.
template <typename K>
static void check_nucleotide_ops()
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 8, 222222 + width)) {
            auto const kmer = oracle_kmer<K>(seq);

            EXPECT_EQ(kmer_decode(complement(kmer, width), width), oracle_complement_string(seq));
            EXPECT_EQ(kmer_decode(reverse(kmer, width), width), oracle_reverse_string(seq));
            EXPECT_EQ(
                kmer_decode(reverse_complement(kmer, width), width), oracle_reverse_complement_string(seq)
            );

            // Each of the three is an involution: applying it twice restores the original.
            EXPECT_EQ(complement(complement(kmer, width), width), kmer);
            EXPECT_EQ(reverse(reverse(kmer, width), width), kmer);
            EXPECT_EQ(reverse_complement(reverse_complement(kmer, width), width), kmer);

            // And they compose the way their names say they do.
            EXPECT_EQ(complement(reverse(kmer, width), width), reverse_complement(kmer, width));
            EXPECT_EQ(reverse(complement(kmer, width), width), reverse_complement(kmer, width));
        }
    }
}

TEST(Kmer, NucleotideOps)
{
    check_nucleotide_ops<KmerAcgtMsb>();
    check_nucleotide_ops<KmerAcgtLsb>();
    check_nucleotide_ops<KmerActgMsb>();
    check_nucleotide_ops<KmerActgLsb>();
}

// Canonicalization must pick the same representative for a k-mer and its reverse complement,
// which is the property that makes it usable to merge both strands onto one key, and must be
// idempotent so that canonicalizing an already canonical k-mer is harmless.
template <typename K>
static void check_canonical()
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 8, 333333 + width)) {
            auto const kmer = oracle_kmer<K>(seq);
            auto const rev_comp = reverse_complement(kmer, width);
            auto const canon = canonical(kmer, width);

            EXPECT_EQ(canon, canonical(rev_comp, width));
            EXPECT_EQ(canon, canonical(canon, width));
            EXPECT_TRUE(canon == kmer || canon == rev_comp);
            EXPECT_TRUE(kmer_value(canon) <= kmer_value(kmer));
            EXPECT_TRUE(kmer_value(canon) <= kmer_value(rev_comp));
        }
    }
}

TEST(Kmer, Canonical)
{
    check_canonical<KmerAcgtMsb>();
    check_canonical<KmerAcgtLsb>();
    check_canonical<KmerActgMsb>();
    check_canonical<KmerActgLsb>();
}

// Within one layout, reversing the bases yields the same bit pattern as reading the unchanged
// word under the opposite layout. Documented in core/kmer.hpp as the clearest illustration of
// why a bare word cannot be interpreted without knowing its conventions.
TEST(Kmer, ReverseMirrorsLayout)
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 555555 + width)) {
            auto const kmer = oracle_kmer<KmerAcgtMsb>(seq);
            EXPECT_EQ(
                kmer_value(reverse(kmer, width)),
                kmer_value(kmer_convert<Layout::kLSB>(kmer, width))
            );
        }
    }
}

// =================================================================================================
//     Convention Conversion
// =================================================================================================

// Converting must produce exactly the k-mer that the target conventions would have encoded the
// same string as, and the result must still decode to that same string. The two directions
// together pin the conversion down: a pair of cancelling bugs could satisfy either alone.
template <typename From, Encoding E2, Layout L2>
static void check_convert_pair(std::string const& seq, std::size_t width)
{
    using To = Kmer<E2, L2>;
    auto const converted = kmer_convert<E2, L2>(oracle_kmer<From>(seq), width);
    static_assert(std::is_same_v<decltype(converted), To const>);

    EXPECT_EQ(converted, oracle_kmer<To>(seq));
    EXPECT_EQ(kmer_decode(converted, width), seq);
    EXPECT_EQ(kmer_decode(converted, width), kmer_decode(oracle_kmer<From>(seq), width));
}

template <typename From>
static void check_convert_from(std::string const& seq, std::size_t width)
{
    check_convert_pair<From, Encoding::kACGT, Layout::kMSB>(seq, width);
    check_convert_pair<From, Encoding::kACGT, Layout::kLSB>(seq, width);
    check_convert_pair<From, Encoding::kACTG, Layout::kMSB>(seq, width);
    check_convert_pair<From, Encoding::kACTG, Layout::kLSB>(seq, width);
}

TEST(Kmer, ConvertMatchesOracle)
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 666666 + width)) {
            check_convert_from<KmerAcgtMsb>(seq, width);
            check_convert_from<KmerAcgtLsb>(seq, width);
            check_convert_from<KmerActgMsb>(seq, width);
            check_convert_from<KmerActgLsb>(seq, width);
        }
    }
}

// Each single-axis conversion must be an involution, and the combined one must agree with
// composing them in either order, since re-coding is per-group and re-ordering is positional.
TEST(Kmer, ConvertRoundTripAndComposition)
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 777777 + width)) {
            auto const kmer = oracle_kmer<KmerAcgtMsb>(seq);

            EXPECT_EQ(
                kmer_convert<Encoding::kACGT>(kmer_convert<Encoding::kACTG>(kmer, width), width),
                kmer
            );
            EXPECT_EQ(
                kmer_convert<Layout::kMSB>(kmer_convert<Layout::kLSB>(kmer, width), width), kmer
            );
            EXPECT_EQ(
                (kmer_convert<Encoding::kACGT, Layout::kMSB>(
                    kmer_convert<Encoding::kACTG, Layout::kLSB>(kmer, width), width
                )),
                kmer
            );

            auto const combined = kmer_convert<Encoding::kACTG, Layout::kLSB>(kmer, width);
            auto const encoding_first =
                kmer_convert<Layout::kLSB>(kmer_convert<Encoding::kACTG>(kmer, width), width);
            auto const layout_first =
                kmer_convert<Encoding::kACTG>(kmer_convert<Layout::kLSB>(kmer, width), width);
            EXPECT_EQ(combined, encoding_first);
            EXPECT_EQ(combined, layout_first);
        }
    }
}

// Converting to the conventions a k-mer already has must be the identity.
TEST(Kmer, ConvertToSameIsIdentity)
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 888888 + width)) {
            auto const kmer = oracle_kmer<KmerAcgtMsb>(seq);
            EXPECT_EQ(kmer_convert<Encoding::kACGT>(kmer, width), kmer);
            EXPECT_EQ(kmer_convert<Layout::kMSB>(kmer, width), kmer);
            EXPECT_EQ((kmer_convert<Encoding::kACGT, Layout::kMSB>(kmer, width)), kmer);
        }
    }
}

// =================================================================================================
//     Ordering and Hashing
// =================================================================================================

// Only ACGT combined with MSB orders k-mers the way the decoded strings order: ACTG because its
// codes are not alphabetical, LSB because the first base sits in the low bits.
TEST(Kmer, LexicographicOrderOnlyForAcgtMsb)
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
TEST(Kmer, Comparison)
{
    auto const a = KmerAcgtMsb{7};
    auto const b = KmerAcgtMsb{9};
    EXPECT_TRUE(a == a);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= a);
}

// Both hashers must be usable, agree with equality, and the default must actually mix.
TEST(Kmer, Hashing)
{
    std::size_t const width = 12;
    auto const seqs = random_seqs(width, 64, 321321);

    std::unordered_set<KmerAcgtMsb> mixed;
    std::unordered_set<KmerAcgtMsb, KmerHashIdentity> identity;
    for (auto const& seq : seqs) {
        auto const kmer = oracle_kmer<KmerAcgtMsb>(seq);
        mixed.insert(kmer);
        identity.insert(kmer);

        // Equal k-mers hash equally, under both hashers.
        EXPECT_EQ(std::hash<KmerAcgtMsb>{}(kmer), std::hash<KmerAcgtMsb>{}(kmer));
        EXPECT_EQ(KmerHashIdentity{}(kmer), static_cast<std::size_t>(kmer_value(kmer)));
    }
    EXPECT_EQ(mixed.size(), identity.size());

    // The default hasher is the mixing one, not the identity.
    auto const probe = oracle_kmer<KmerAcgtMsb>(seqs.front());
    EXPECT_EQ(std::hash<KmerAcgtMsb>{}(probe), KmerHashMix{}(probe));
    EXPECT_NE(std::hash<KmerAcgtMsb>{}(probe), KmerHashIdentity{}(probe));
}

// =================================================================================================
//     Encoding
// =================================================================================================

// kmer_encode() must be the exact inverse of kmer_decode(): building a k-mer from a string and
// decoding it again must return that same string, under every convention.
template <typename K>
static void check_encode_decode_round_trip()
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 141414 + width)) {
            EXPECT_EQ(kmer_decode(kmer_encode<K>(seq), width), seq);
        }
    }
}

TEST(Kmer, EncodeDecodeRoundTrip)
{
    check_encode_decode_round_trip<KmerAcgtMsb>();
    check_encode_decode_round_trip<KmerAcgtLsb>();
    check_encode_decode_round_trip<KmerActgMsb>();
    check_encode_decode_round_trip<KmerActgLsb>();
}

// kmer_encode() must agree with the oracle, not merely round-trip through kmer_decode(): a pair
// of cancelling bugs in encode and decode could otherwise satisfy the round-trip alone.
template <typename K>
static void check_encode_matches_oracle()
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 151515 + width)) {
            EXPECT_EQ(kmer_encode<K>(seq), oracle_kmer<K>(seq));
        }
    }
}

TEST(Kmer, EncodeMatchesOracle)
{
    check_encode_matches_oracle<KmerAcgtMsb>();
    check_encode_matches_oracle<KmerAcgtLsb>();
    check_encode_matches_oracle<KmerActgMsb>();
    check_encode_matches_oracle<KmerActgLsb>();
}

// The template argument names the result type exactly. Only contiguous k-mers can be built from a
// string alone; a spaced k-mer also needs its mask, see kmer_spaced/kmer_spaced.hpp.
template <typename K>
concept EncodableFromString = requires(std::string_view seq) { kmer_encode<K>(seq); };

static_assert(std::is_same_v<decltype(kmer_encode<KmerActgLsb>("A")), KmerActgLsb>);
static_assert(EncodableFromString<KmerAcgtMsb>);
static_assert(!EncodableFromString<SpacedKmerAcgtMsb>);
static_assert(!EncodableFromString<std::uint64_t>);

// The default template arguments must be ACGT/MSB, matching what for_each_kmer() emits.
TEST(Kmer, EncodeDefaultsMatchForEachKmer)
{
    static_assert(std::is_same_v<decltype(kmer_encode("ACG")), KmerAcgtMsb>);

    for (auto const& seq : random_seqs(11, 8, 161616)) {
        std::vector<KmerAcgtMsb> from_for_each;
        for_each_kmer(seq, seq.size(), [&](KmerAcgtMsb kmer) { from_for_each.push_back(kmer); });
        ASSERT_EQ(from_for_each.size(), std::size_t{1});
        EXPECT_EQ(kmer_encode(seq), from_for_each[0]);
    }
}

// The char-to-code table inside kmer_encode() is a deliberate duplication of the encoders in
// core/char_encoder.hpp (see char_to_code_() in core/kmer.hpp for why). This is what keeps the two
// from silently drifting apart: every code kmer_encode() assigns must be the one the ACGT/ACTG
// lookup tables would assign to the same character, upper and lower case alike.
TEST(Kmer, EncodeAgreesWithSeqEnc)
{
    for (char c = 0; c < 127; ++c) {
        std::string const seq(1, c);
        bool const is_acgt = CharEncoderTable<Encoding::kACGT>{}(c) < 4;
        bool const is_actg = CharEncoderTable<Encoding::kACTG>{}(c) < 4;

        if (is_acgt) {
            EXPECT_EQ(
                kmer_value(kmer_encode(seq)),
                std::uint64_t{CharEncoderTable<Encoding::kACGT>{}(c)}
            );
        } else {
            EXPECT_ANY_THROW((void) kmer_encode(seq));
        }

        if (is_actg) {
            EXPECT_EQ(
                kmer_value(kmer_encode<KmerActgMsb>(seq)),
                std::uint64_t{CharEncoderTable<Encoding::kACTG>{}(c)}
            );
        } else {
            EXPECT_ANY_THROW((void) kmer_encode<KmerActgMsb>(seq));
        }
    }
}

// Boundary and error contract: length must be in [1, 32], and every character must be one of the
// four nucleotides (case-insensitively); anything else throws rather than silently misencoding.
TEST(Kmer, EncodeInvalidInput)
{
    EXPECT_ANY_THROW((void) kmer_encode(""));
    EXPECT_ANY_THROW((void) kmer_encode(std::string(33, 'A')));
    EXPECT_ANY_THROW((void) kmer_encode("ACGN"));
    EXPECT_ANY_THROW((void) kmer_encode("acgx"));
    EXPECT_EQ(kmer_decode(kmer_encode(std::string(32, 'T')), 32), std::string(32, 'T'));

    // Every byte that is not a nucleotide throws, wherever it sits in the sequence, including '\0'
    // and the bytes that are negative as a signed char.
    for (std::size_t const width : {std::size_t{1}, std::size_t{5}, std::size_t{32}}) {
        for (std::size_t const pos : {std::size_t{0}, width / 2, width - 1}) {
            for (char const c : all_invalid_bytes()) {
                std::string seq(width, 'A');
                seq[pos] = c;
                EXPECT_THROW((void) kmer_encode(seq), std::invalid_argument);
                EXPECT_THROW((void) kmer_encode<KmerActgMsb>(seq), std::invalid_argument);
            }
        }
    }

    // Lower case is accepted, and normalized to upper case on the way back out.
    EXPECT_EQ(kmer_encode("acgt"), kmer_encode("ACGT"));
    EXPECT_EQ(kmer_decode(kmer_encode("acgt"), 4), "ACGT");
}

// =================================================================================================
//     SIMD Re-entry
// =================================================================================================

// The valid lanes with bit `j` of `bits` set for lane `j`. Each arch makes its lane mask by
// comparing the lanes of two vectors, so this compares lanes that are equal where the bits are set.
template <typename Simd>
static KmerVectorValidLanes<Simd> valid_lanes_of_bits(std::uint64_t bits)
{
    std::array<std::uint64_t, Simd::lanes> zeros{};
    std::array<std::uint64_t, Simd::lanes> lanes{};
    for (std::size_t j = 0; j < Simd::lanes; ++j) {
        lanes[j] = ((bits >> j) & 1) ? 0 : 1;
    }
    return {Simd::compare_eq_lanes(Simd::from_values(lanes), Simd::from_values(zeros))};
}

// A vector of k-mers holds the lanes it is cast from. for_each_valid_lane() hands out exactly its
// valid k-mers, in lane order and with their positions, for each count of valid lanes and each
// mask of valid lanes, in both callback forms, and store_valid_lanes() stores the same k-mers. Run
// once per vector ISA compiled in, see for_each_lane_vector() in simd_lanes.hpp.
TEST(Kmer, KmerVector)
{
    std::size_t const width = 10;
    std::size_t const pos = 100;
    for_each_lane_vector(width, 909090, [&]<typename Simd>(Simd, auto vec, auto const& seqs) {
        constexpr std::size_t lanes = Simd::lanes;
        constexpr Encoding E = Encoding::kACGT;
        constexpr Layout L = Layout::kMSB;
        auto const kmers = kmer_vector_cast<E, L, Simd>(vec, width);
        auto const kmers_unchecked = kmer_vector_cast_unchecked<E, L, Simd>(vec);
        auto const value = kmer_value(kmers_unchecked);
        EXPECT_EQ(std::memcmp(&kmers.value, &vec, sizeof(vec)), 0);
        EXPECT_EQ(std::memcmp(&value, &vec, sizeof(vec)), 0);

        // The k-mers that each way hands out, for `valid`, which has the lanes of `bits`.
        auto const check = [&](auto const& valid, std::uint64_t bits) {
            std::vector<std::size_t> positions;
            std::vector<KmerAcgtMsb> with_pos;
            std::vector<KmerAcgtMsb> without_pos;
            for_each_valid_lane(pos, kmers, valid, [&](std::size_t p, KmerAcgtMsb kmer) {
                positions.push_back(p);
                with_pos.push_back(kmer);
            });
            for_each_valid_lane(pos, kmers, valid, [&](KmerAcgtMsb kmer) {
                without_pos.push_back(kmer);
            });
            KmerAcgtMsb stored[lanes] = {};
            std::size_t const count = store_valid_lanes(kmers, valid, stored);

            std::size_t i = 0;
            for (std::size_t j = 0; j < lanes; ++j) {
                if (((bits >> j) & 1) == 0) {
                    continue;
                }
                ASSERT_TRUE(i < positions.size());
                EXPECT_EQ(positions[i], pos + j);
                EXPECT_EQ(with_pos[i], oracle_kmer<KmerAcgtMsb>(seqs[j]));
                EXPECT_EQ(without_pos[i], with_pos[i]);
                EXPECT_EQ(stored[i], with_pos[i]);
                ++i;
            }
            EXPECT_EQ(positions.size(), i);
            EXPECT_EQ(without_pos.size(), i);
            EXPECT_EQ(count, i);
        };
        for (std::size_t count = 0; count <= lanes; ++count) {
            check(count, (std::uint64_t{1} << count) - 1);
        }
        for (std::uint64_t bits = 0; bits < (std::uint64_t{1} << lanes); ++bits) {
            check(valid_lanes_of_bits<Simd>(bits), bits);
        }
    });
}
