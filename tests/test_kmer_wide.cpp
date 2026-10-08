#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_hash.hpp"
#include "fisk/core/kmer_wide.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// Ground truth comes from oracle_wide_kmer() in oracle.hpp, built from nucleotide strings and
// deliberately independent of core/kmer_wide.hpp. Comparing whole wide k-mers against it pins down
// every word, including the zero padding above the width. Random inputs come from random_seqs() in
// corpus.hpp.

// Numeric order of two word arrays, highest word first, written independently of the operator<=>
// of WideKmer.
template <std::size_t N>
static bool words_less(
    std::array<std::uint64_t, N> const& lhs, std::array<std::uint64_t, N> const& rhs
) {
    return std::lexicographical_compare(lhs.rbegin(), lhs.rend(), rhs.rbegin(), rhs.rend());
}

// Every producer and operation, for one sequence under the conventions of K, against the oracle.
template <typename K>
static void check_sequence(std::string const& seq)
{
    std::size_t const width = seq.size();
    auto const expected = oracle_wide_kmer<K>(seq);

    // Producers and decoding.
    auto const kmer = kmer_encode<K>(seq);
    EXPECT_EQ(kmer, expected);
    EXPECT_EQ((kmer_cast<K::encoding, K::layout>(kmer_value(expected), width)), expected);
    EXPECT_EQ((kmer_cast_unchecked<K::encoding, K::layout>(kmer_value(expected))), expected);
    EXPECT_EQ(kmer_decode(kmer, width), seq);
    bool bases_match = true;
    for (std::size_t i = 0; i < width; ++i) {
        bases_match &= base_at(kmer, i, width) == oracle_code<K::encoding>(seq[i]);
    }
    EXPECT_TRUE(bases_match);

    // Nucleotide operations, each of them its own inverse.
    auto const rev_comp_seq = oracle_reverse_complement_string(seq);
    auto const comp = complement(kmer, width);
    auto const rev = reverse(kmer, width);
    auto const rev_comp = reverse_complement(kmer, width);
    EXPECT_EQ(comp, oracle_wide_kmer<K>(oracle_complement_string(seq)));
    EXPECT_EQ(rev, oracle_wide_kmer<K>(oracle_reverse_string(seq)));
    EXPECT_EQ(rev_comp, oracle_wide_kmer<K>(rev_comp_seq));
    EXPECT_EQ(complement(comp, width), kmer);
    EXPECT_EQ(reverse(rev, width), kmer);
    EXPECT_EQ(reverse_complement(rev_comp, width), kmer);

    // Canonical form: the numerically smaller of the two strands, and for ACGT/MSB therefore also
    // the lexicographically smaller one.
    auto const expected_rev_comp = oracle_wide_kmer<K>(rev_comp_seq);
    auto const expected_canon = words_less(kmer_value(expected_rev_comp), kmer_value(expected))
        ? expected_rev_comp
        : expected;
    EXPECT_EQ(canonical(kmer, width), expected_canon);
    EXPECT_EQ(canonical(rev_comp, width), expected_canon);
    if constexpr (K::encoding == Encoding::kACGT && K::layout == Layout::kMSB) {
        EXPECT_EQ(kmer_decode(canonical(kmer, width), width), std::min(seq, rev_comp_seq));
    }

    // Convention conversion keeps the capacity, and converting to the same conventions is identity.
    constexpr Encoding other_encoding = K::encoding == Encoding::kACGT
        ? Encoding::kACTG
        : Encoding::kACGT;
    constexpr Layout other_layout = K::layout == Layout::kLSB
        ? Layout::kMSB
        : Layout::kLSB;
    using Reencoded = typename K::template rebind<other_encoding, K::layout>;
    using Relaid = typename K::template rebind<K::encoding, other_layout>;
    using Converted = typename K::template rebind<other_encoding, other_layout>;
    EXPECT_EQ(kmer_convert<other_encoding>(kmer, width), oracle_wide_kmer<Reencoded>(seq));
    EXPECT_EQ(kmer_convert<other_layout>(kmer, width), oracle_wide_kmer<Relaid>(seq));
    EXPECT_EQ(
        (kmer_convert<other_encoding, other_layout>(kmer, width)), oracle_wide_kmer<Converted>(seq)
    );
    EXPECT_EQ(kmer_convert<K::encoding>(kmer, width), kmer);
    EXPECT_EQ(kmer_convert<K::layout>(kmer, width), kmer);
    EXPECT_EQ((kmer_convert<K::encoding, K::layout>(kmer, width)), kmer);
}

// Up to 32 bases, a wide k-mer holds exactly the word of the Kmer of the same bases in its lowest
// word, under every operation.
template <typename K>
static void check_agrees_with_narrow(std::string const& seq)
{
    using Narrow = Kmer<K::encoding, K::layout>;
    std::size_t const width = seq.size();
    auto const as_words = [](Narrow narrow) {
        std::array<std::uint64_t, K::word_count> words{};
        words[0] = kmer_value(narrow);
        return words;
    };

    auto const wide = kmer_encode<K>(seq);
    auto const narrow = kmer_encode<Narrow>(seq);
    EXPECT_EQ(kmer_value(wide), as_words(narrow));
    EXPECT_EQ(kmer_value(complement(wide, width)), as_words(complement(narrow, width)));
    EXPECT_EQ(kmer_value(reverse(wide, width)), as_words(reverse(narrow, width)));
    EXPECT_EQ(
        kmer_value(reverse_complement(wide, width)), as_words(reverse_complement(narrow, width))
    );
    EXPECT_EQ(kmer_value(canonical(wide, width)), as_words(canonical(narrow, width)));
}

// Every width up to the capacity, a few random sequences each.
template <typename K>
static void check_all_widths(std::uint64_t seed)
{
    for (std::size_t width = 1; width <= K::capacity; ++width) {
        for (auto const& seq : random_seqs(width, 3, seed + width)) {
            check_sequence<K>(seq);
            if (width <= 32) {
                check_agrees_with_narrow<K>(seq);
            }
        }
    }
}

template <std::size_t Capacity>
static void check_all_conventions(std::uint64_t seed)
{
    check_all_widths<WideKmer<Encoding::kACGT, Layout::kMSB, Capacity>>(seed);
    check_all_widths<WideKmer<Encoding::kACGT, Layout::kLSB, Capacity>>(seed + 1000);
    check_all_widths<WideKmer<Encoding::kACTG, Layout::kMSB, Capacity>>(seed + 2000);
    check_all_widths<WideKmer<Encoding::kACTG, Layout::kLSB, Capacity>>(seed + 3000);
}

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

// =================================================================================================
//     Compile-Time Guarantees
// =================================================================================================

// WideKmer is a public aggregate of its word array, just as Kmer is of its single word. Capacity
// sets the number of words, while the width remains a runtime argument of every operation.
static_assert(std::is_aggregate_v<WideKmerAcgtMsb>);
static_assert(std::is_trivially_copyable_v<WideKmerAcgtMsb>);
static_assert(std::is_standard_layout_v<WideKmerAcgtMsb>);
static_assert(sizeof(WideKmerAcgtMsb) == 2 * sizeof(std::uint64_t));
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

// Naming a wide type selects the wide kmer_encode(); the narrow one keeps its default.
static_assert(std::is_same_v<decltype(kmer_encode<WideKmerActgLsb>("A")), WideKmerActgLsb>);
static_assert(std::is_same_v<decltype(kmer_encode("A")), KmerAcgtMsb>);

// kmer_cast() deduces the capacity from the number of words, next to the unchanged narrow casts.
static_assert(std::is_same_v<
    decltype(kmer_cast<Encoding::kACTG, Layout::kLSB>(std::array<std::uint64_t, 3>{}, 65)),
    WideKmer<Encoding::kACTG, Layout::kLSB, 96>
>);
static_assert(std::is_same_v<
    decltype(kmer_cast_unchecked<Encoding::kACGT, Layout::kMSB>(std::array<std::uint64_t, 1>{})),
    WideKmer<Encoding::kACGT, Layout::kMSB, 32>
>);
static_assert(std::is_same_v<
    decltype(kmer_cast<Encoding::kACGT, Layout::kMSB>(std::uint64_t{0}, 1)), KmerAcgtMsb
>);

// The predicate behind the debug check in kmer_cast(): words fit a width iff they have no bits set
// above it, neither in the partly used word nor in any wholly unused one.
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

// Every operation is constexpr. Spelled out on 33 bases, G, then 31 times A, then T, whose two
// endpoints land in separate words, and whose expected words are short enough to read.
namespace {
    constexpr auto lsb33 = kmer_encode<WideKmerAcgtLsb>("GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT");
    constexpr auto msb33 = kmer_encode<WideKmerAcgtMsb>("GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT");
}
static_assert(lsb33.words == std::array<std::uint64_t, 2>{2, 3});
static_assert(msb33.words == std::array<std::uint64_t, 2>{3, 2});
static_assert(kmer_cast<Encoding::kACGT, Layout::kLSB>(lsb33.words, 33) == lsb33);
static_assert(base_at(lsb33, 0, 33) == 2 && base_at(lsb33, 32, 33) == 3);
static_assert(reverse(lsb33, 33).words == std::array<std::uint64_t, 2>{3, 2});  // TAA...AG
static_assert(complement(lsb33, 33).words == std::array<std::uint64_t, 2>{     // CTT...TA
    0xFFFFFFFFFFFFFFFDull, 0
});
static_assert(kmer_convert<Layout::kMSB>(lsb33, 33) == msb33);
static_assert(canonical(msb33, 33) == reverse_complement(msb33, 33));       // ATT...TC < GAA...AT

// =================================================================================================
//     Construction, Conversion, and Nucleotide Operations
// =================================================================================================

// Every width, under every convention, at every capacity up to four words, must agree with the
// character-level oracle, and with the narrow Kmer wherever both apply.
TEST(WideKmer, Capacity32)
{
    check_all_conventions<32>(110000);
}

TEST(WideKmer, Capacity64)
{
    check_all_conventions<64>(120000);
}

TEST(WideKmer, Capacity96)
{
    check_all_conventions<96>(130000);
}

TEST(WideKmer, Capacity128)
{
    check_all_conventions<128>(140000);
}

// =================================================================================================
//     Encoding Validation
// =================================================================================================

// Length must be in [1, capacity], and every character must be one of the four nucleotides
// (case-insensitively), wherever it sits relative to the word boundaries.
TEST(WideKmer, EncodeInvalidInput)
{
    EXPECT_THROW((void) kmer_encode<WideKmerAcgtMsb>(""), std::invalid_argument);
    EXPECT_THROW((void) kmer_encode<WideKmerAcgtMsb>(std::string(65, 'A')), std::invalid_argument);
    EXPECT_EQ(
        kmer_decode(kmer_encode<WideKmerAcgtMsb>(std::string(64, 'T')), 64), std::string(64, 'T')
    );

    for (std::size_t const pos : {std::size_t{0}, std::size_t{31}, std::size_t{32}, std::size_t{63}}) {
        for (char const c : all_invalid_bytes()) {
            std::string seq(64, 'A');
            seq[pos] = c;
            EXPECT_THROW((void) kmer_encode<WideKmerAcgtMsb>(seq), std::invalid_argument);
            EXPECT_THROW((void) kmer_encode<WideKmerActgLsb>(seq), std::invalid_argument);
        }
    }

    // Lower case is accepted, and normalized to upper case on the way back out.
    auto const seq = random_seqs(50, 1, 150000).front();
    std::string lower = seq;
    for (char& c : lower) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    EXPECT_EQ(kmer_encode<WideKmerAcgtMsb>(lower), kmer_encode<WideKmerAcgtMsb>(seq));
    EXPECT_EQ(kmer_decode(kmer_encode<WideKmerAcgtMsb>(lower), 50), seq);
}

// =================================================================================================
//     Ordering and Hashing
// =================================================================================================

TEST(WideKmer, Comparison)
{
    check_word_order<WideKmerAcgtMsb>();
    check_word_order<WideKmerActgLsb>();
    check_word_order<WideKmer<Encoding::kACGT, Layout::kLSB, 96>>();
    check_word_order<WideKmer<Encoding::kACTG, Layout::kMSB, 128>>();
}

// As for Kmer, the order matches lexicographic order of the decoded strings for ACGT/MSB, here
// across word boundaries.
TEST(WideKmer, LexicographicOrderForAcgtMsb)
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

// Both hashers must be usable as container hashers and agree with equality. The mixing hash, which
// is the default, must depend on every word; the identity hash is the lowest word only.
TEST(WideKmer, Hashing)
{
    using K = WideKmer<Encoding::kACGT, Layout::kMSB, 128>;
    auto const seqs = random_seqs(100, 64, 170000);

    std::unordered_set<K> mixed;
    std::unordered_set<K, KmerHashIdentity> identity;
    for (auto const& seq : seqs) {
        auto const kmer = kmer_encode<K>(seq);
        mixed.insert(kmer);
        identity.insert(kmer);
        EXPECT_EQ(mixed.count(kmer), std::size_t{1});
        EXPECT_EQ(identity.count(kmer), std::size_t{1});
    }
    EXPECT_EQ(mixed.size(), std::set<std::string>(seqs.begin(), seqs.end()).size());
    EXPECT_EQ(identity.size(), mixed.size());

    auto const probe = kmer_encode<K>(seqs.front());
    EXPECT_EQ(std::hash<K>{}(probe), KmerHashMix{}(probe));
    EXPECT_EQ(KmerHashIdentity{}(probe), static_cast<std::size_t>(kmer_value(probe)[0]));
    for (std::size_t i = 0; i < K::word_count; ++i) {
        auto words = kmer_value(probe);
        words[i] ^= 1;
        K const flipped{words};
        EXPECT_NE(KmerHashMix{}(flipped), KmerHashMix{}(probe));
        EXPECT_EQ(KmerHashIdentity{}(flipped) == KmerHashIdentity{}(probe), i != 0);
    }

    // A single word hashes as the Kmer of the same value.
    auto const seq32 = random_seqs(32, 1, 180000).front();
    EXPECT_EQ(
        KmerHashMix{}(kmer_encode<WideKmer<Encoding::kACGT, Layout::kMSB, 32>>(seq32)),
        KmerHashMix{}(kmer_encode<KmerAcgtMsb>(seq32))
    );
}
