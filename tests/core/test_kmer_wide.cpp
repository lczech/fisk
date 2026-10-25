#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "fisk/core/cast.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/core/kmer_wide.hpp"
#include "fisk/core/types.hpp"
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
    EXPECT_EQ((wide_kmer_cast<K::encoding, K::layout>(kmer_value(expected), width)), expected);
    EXPECT_EQ((wide_kmer_cast_unchecked<K::encoding, K::layout>(kmer_value(expected))), expected);
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

// =================================================================================================
//     Compile-Time Guarantees
// =================================================================================================

// Naming a wide type selects the wide kmer_encode(); the narrow one keeps its default.
static_assert(std::is_same_v<decltype(kmer_encode<WideKmerActgLsb>("A")), WideKmerActgLsb>);
static_assert(std::is_same_v<decltype(kmer_encode("A")), KmerAcgtMsb>);

namespace {

// Every operation is constexpr. Spelled out on 33 bases, G, then 31 times A, then T, whose two
// endpoints land in separate words, and whose expected words are short enough to read.
    constexpr auto lsb33 = kmer_encode<WideKmerAcgtLsb>("GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT");
    constexpr auto msb33 = kmer_encode<WideKmerAcgtMsb>("GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT");

} // namespace

static_assert(lsb33.words == std::array<std::uint64_t, 2>{2, 3});
static_assert(msb33.words == std::array<std::uint64_t, 2>{3, 2});
static_assert(wide_kmer_cast<Encoding::kACGT, Layout::kLSB>(lsb33.words, 33) == lsb33);
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
TEST(CoreKmerWide, Capacity32)
{
    check_all_conventions<32>(110000);
}

TEST(CoreKmerWide, Capacity64)
{
    check_all_conventions<64>(120000);
}

TEST(CoreKmerWide, Capacity96)
{
    check_all_conventions<96>(130000);
}

TEST(CoreKmerWide, Capacity128)
{
    check_all_conventions<128>(140000);
}

// =================================================================================================
//     Encoding Validation
// =================================================================================================

// Length must be in [1, capacity], and every character must be one of the four nucleotides
// (case-insensitively), wherever it sits relative to the word boundaries.
TEST(CoreKmerWide, EncodeInvalidInput)
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
