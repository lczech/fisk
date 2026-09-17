#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "fisk/core/intrinsics.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/random.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_extract/kmer_extract.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Ground truth built from nucleotide strings, deliberately independent of core/kmer.hpp: the
// conventions are re-implemented here as plain per-character rules, so that a mistake in the
// bit tricks under test cannot cancel itself out against the reference. Both convention chains
// end in a static_assert, for the same reason the ones in core/kmer.hpp do: a third Encoding or
// Layout must break the oracle loudly rather than be silently treated as one of the existing two.

template <Encoding E>
static std::uint64_t code_of(char c)
{
    if constexpr (E == Encoding::kACGT) {
        switch (c) {
            case 'A': return 0;
            case 'C': return 1;
            case 'G': return 2;
            case 'T': return 3;
        }
    } else if constexpr (E == Encoding::kACTG) {
        switch (c) {
            case 'A': return 0;
            case 'C': return 1;
            case 'T': return 2;
            case 'G': return 3;
        }
    } else {
        static_assert(dependent_false_v<E>, "Unhandled Encoding in the test oracle's code_of()");
    }
    return 0;
}

// Encode a nucleotide string into a raw word, base by base, honoring both conventions.
template <Encoding E, Layout L>
static std::uint64_t encode_oracle(std::string const& seq)
{
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < seq.size(); ++i) {
        std::size_t shift = 0;
        if constexpr (L == Layout::kMSB) {
            shift = 2 * (seq.size() - 1 - i);
        } else if constexpr (L == Layout::kLSB) {
            shift = 2 * i;
        } else {
            static_assert(dependent_false_v<L>, "Unhandled Layout in the test oracle");
        }
        value |= code_of<E>(seq[i]) << shift;
    }
    return value;
}

// Build the k-mer for a string under the conventions of K, via the oracle rather than via
// any of the library's own encoders.
template <typename K>
static K oracle_kmer(std::string const& seq)
{
    return K{encode_oracle<K::encoding, K::layout>(seq)};
}

static char complement_char(char c)
{
    switch (c) {
        case 'A': return 'T';
        case 'C': return 'G';
        case 'G': return 'C';
        case 'T': return 'A';
    }
    return 'N';
}

static std::string complement_string(std::string s)
{
    for (auto& c : s) {
        c = complement_char(c);
    }
    return s;
}

static std::string reverse_string(std::string s)
{
    std::reverse(s.begin(), s.end());
    return s;
}

static std::string reverse_complement_string(std::string const& s)
{
    return reverse_string(complement_string(s));
}

// Random nucleotide strings of a given length, from a fixed seed so failures reproduce.
static std::vector<std::string> random_seqs(std::size_t width, std::size_t count, std::uint64_t seed)
{
    static char const bases[4] = {'A', 'C', 'G', 'T'};
    Splitmix64 rng(seed);

    std::vector<std::string> out;
    out.reserve(count);
    for (std::size_t n = 0; n < count; ++n) {
        std::string s;
        s.resize(width);
        for (std::size_t i = 0; i < width; ++i) {
            s[i] = bases[rng.get_uint64() % 4];
        }
        out.push_back(s);
    }
    return out;
}

// =================================================================================================
//     Compile-Time Guarantees
// =================================================================================================

// The type must stay a bare word as far as the ABI is concerned: an aggregate with no user-declared
// constructors, so that it is passed in a register exactly like the uint64_t it wraps. Guards
// against someone later adding a constructor or a second member and silently losing that.
static_assert(sizeof(KmerAcgtMsb) == sizeof(std::uint64_t));
static_assert(alignof(KmerAcgtMsb) == alignof(std::uint64_t));
static_assert(std::is_trivially_copyable_v<KmerAcgtMsb>);
static_assert(std::is_standard_layout_v<KmerAcgtMsb>);
static_assert(std::is_aggregate_v<KmerAcgtMsb>);

// All four instantiations are distinct types, and none of them converts to another or to a raw
// word in either direction. This is the guarantee the whole type exists for.
static_assert(!std::is_same_v<KmerAcgtMsb, KmerAcgtLsb>);
static_assert(!std::is_same_v<KmerAcgtMsb, KmerActgMsb>);
static_assert(!std::is_convertible_v<KmerAcgtMsb, KmerActgMsb>);
static_assert(!std::is_convertible_v<KmerAcgtMsb, KmerAcgtLsb>);
static_assert(!std::is_convertible_v<KmerAcgtMsb, std::uint64_t>);
static_assert(!std::is_convertible_v<std::uint64_t, KmerAcgtMsb>);

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

// Every operation is constexpr, so k-mers can be built and manipulated at compile time. Spelled
// out on one worked example, ACG under ACGT/MSB, whose expected words are short enough to read.
namespace {
    constexpr auto acg = kmer_cast<Encoding::kACGT, Layout::kMSB>(0x06, 3);
}
static_assert(kmer_value(acg) == 0x06);
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
                EXPECT_EQ(std::uint64_t{base_at(kmer, i, width)}, code_of<K::encoding>(seq[i]));
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
            auto const raw = encode_oracle<Encoding::kACGT, Layout::kMSB>(seq);
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

            EXPECT_EQ(kmer_decode(complement(kmer, width), width), complement_string(seq));
            EXPECT_EQ(kmer_decode(reverse(kmer, width), width), reverse_string(seq));
            EXPECT_EQ(
                kmer_decode(reverse_complement(kmer, width), width), reverse_complement_string(seq)
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
// decoding it again must return that same string, under every convention and every way of naming
// the template arguments.
TEST(Kmer, EncodeDecodeRoundTrip)
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 141414 + width)) {
            EXPECT_EQ(kmer_decode(kmer_encode(seq), width), seq);
            EXPECT_EQ(kmer_decode((kmer_encode<Encoding::kACTG>(seq)), width), seq);
            EXPECT_EQ(kmer_decode(kmer_encode<Layout::kLSB>(seq), width), seq);
            EXPECT_EQ(
                kmer_decode((kmer_encode<Encoding::kACTG, Layout::kLSB>(seq)), width), seq
            );
            // Same as the line above, but with the two template arguments given in the other
            // order, exercising the reversed-order overload.
            EXPECT_EQ(
                kmer_decode((kmer_encode<Layout::kLSB, Encoding::kACTG>(seq)), width), seq
            );
        }
    }
}

// kmer_encode() must agree with the oracle, not merely round-trip through kmer_decode(): a pair
// of cancelling bugs in encode and decode could otherwise satisfy the round-trip alone.
TEST(Kmer, EncodeMatchesOracle)
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 151515 + width)) {
            EXPECT_EQ(kmer_encode(seq), oracle_kmer<KmerAcgtMsb>(seq));
            EXPECT_EQ((kmer_encode<Encoding::kACTG>(seq)), oracle_kmer<KmerActgMsb>(seq));
            EXPECT_EQ(kmer_encode<Layout::kLSB>(seq), oracle_kmer<KmerAcgtLsb>(seq));
            EXPECT_EQ(
                (kmer_encode<Encoding::kACTG, Layout::kLSB>(seq)), oracle_kmer<KmerActgLsb>(seq)
            );
            // The reversed-order overload must agree with the canonical order, not just compile.
            EXPECT_EQ(
                (kmer_encode<Layout::kLSB, Encoding::kACTG>(seq)),
                (kmer_encode<Encoding::kACTG, Layout::kLSB>(seq))
            );
        }
    }
}

// Both argument orders name the exact same type, not merely equal values.
TEST(Kmer, EncodeArgumentOrderIsInterchangeable)
{
    static_assert(std::is_same_v<
        decltype(kmer_encode<Encoding::kACTG, Layout::kLSB>("A")),
        decltype(kmer_encode<Layout::kLSB, Encoding::kACTG>("A"))>);
}

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
                kmer_value((kmer_encode<Encoding::kACTG>(seq))),
                std::uint64_t{CharEncoderTable<Encoding::kACTG>{}(c)}
            );
        } else {
            EXPECT_ANY_THROW((void) (kmer_encode<Encoding::kACTG>(seq)));
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

    // Lower case is accepted, and normalized to upper case on the way back out.
    EXPECT_EQ(kmer_encode("acgt"), kmer_encode("ACGT"));
    EXPECT_EQ(kmer_decode(kmer_encode("acgt"), 4), "ACGT");
}

// =================================================================================================
//     SIMD Re-entry
// =================================================================================================

// The vector overloads of kmer_cast() must hand back exactly the lanes they were given, in lane
// order and tagged with the conventions asked for, so that consumers of the vector-emitting
// extractors can return to the typed world. Exercised once per ISA below, since each builds its
// vectors with its own set intrinsic and it is precisely the lane ordering that must agree.
template <std::size_t Lanes, typename Vec>
static void check_vector_cast(Vec vec, std::vector<std::string> const& seqs, std::size_t width)
{
    ASSERT_EQ(seqs.size(), Lanes);

    // The full-vector overload writes every lane.
    KmerAcgtMsb out[Lanes] = {};
    EXPECT_EQ((kmer_cast<Encoding::kACGT, Layout::kMSB>(vec, width, out)), Lanes);
    for (std::size_t i = 0; i < Lanes; ++i) {
        EXPECT_EQ(out[i], oracle_kmer<KmerAcgtMsb>(seqs[i]));
        EXPECT_EQ(kmer_decode(out[i], width), seqs[i]);
    }

    // The counted overload writes only the lanes it is told are valid, as the zero-padded final
    // vector of an extraction run requires.
    KmerAcgtMsb partial[Lanes] = {};
    EXPECT_EQ((kmer_cast<Encoding::kACGT, Layout::kMSB>(vec, 1, width, partial)), std::size_t{1});
    EXPECT_EQ(partial[0], oracle_kmer<KmerAcgtMsb>(seqs[0]));
    if constexpr (Lanes > 1) {
        EXPECT_EQ(partial[1], KmerAcgtMsb{0});
    }
}

// Lane values for a test vector, as raw words under ACGT/MSB.
static std::uint64_t lane_word(std::vector<std::string> const& seqs, std::size_t i)
{
    return encode_oracle<Encoding::kACGT, Layout::kMSB>(seqs[i]);
}

#if defined(FISK_HAS_SSE2)

TEST(Kmer, CastVectorSse2)
{
    std::size_t const width = 10;
    auto const seqs = random_seqs(width, 2, 909090);
    auto const vec = _mm_set_epi64x(
        static_cast<long long>(lane_word(seqs, 1)),
        static_cast<long long>(lane_word(seqs, 0))
    );
    check_vector_cast<2>(vec, seqs, width);
}

#endif

#if defined(FISK_HAS_AVX2)

TEST(Kmer, CastVectorAvx2)
{
    std::size_t const width = 10;
    auto const seqs = random_seqs(width, 4, 909091);
    auto const vec = _mm256_set_epi64x(
        static_cast<long long>(lane_word(seqs, 3)),
        static_cast<long long>(lane_word(seqs, 2)),
        static_cast<long long>(lane_word(seqs, 1)),
        static_cast<long long>(lane_word(seqs, 0))
    );
    check_vector_cast<4>(vec, seqs, width);
}

#endif

#if defined(FISK_HAS_AVX512)

TEST(Kmer, CastVectorAvx512)
{
    std::size_t const width = 10;
    auto const seqs = random_seqs(width, 8, 909092);
    auto const vec = _mm512_set_epi64(
        static_cast<long long>(lane_word(seqs, 7)),
        static_cast<long long>(lane_word(seqs, 6)),
        static_cast<long long>(lane_word(seqs, 5)),
        static_cast<long long>(lane_word(seqs, 4)),
        static_cast<long long>(lane_word(seqs, 3)),
        static_cast<long long>(lane_word(seqs, 2)),
        static_cast<long long>(lane_word(seqs, 1)),
        static_cast<long long>(lane_word(seqs, 0))
    );
    check_vector_cast<8>(vec, seqs, width);
}

#endif

#if defined(FISK_HAS_NEON)

TEST(Kmer, CastVectorNeon)
{
    std::size_t const width = 10;
    auto const seqs = random_seqs(width, 2, 909093);
    std::uint64_t const lanes[2] = {lane_word(seqs, 0), lane_word(seqs, 1)};
    auto const vec = vld1q_u64(lanes);
    check_vector_cast<2>(vec, seqs, width);
}

#endif
