#include <cctype>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "fisk/core/cast.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/core/kmer.hpp"
#include "fisk/util/random.hpp"
#include "fisk/core/types.hpp"
#include "fisk/kmer_extract/ascii_rolling_scalar.hpp"
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

namespace {

// Every operation is constexpr, so k-mers can be built and manipulated at compile time. Spelled
// out on one worked example, ACG under ACGT/MSB, whose expected words are short enough to read.
    constexpr auto acg = kmer_cast<Encoding::kACGT, Layout::kMSB>(0x06, 3);

} // namespace

static_assert(kmer_value(acg) == 0x06);
static_assert(kmer_encode("ACG") == acg);
static_assert(base_at(acg, 0, 3) == 0 && base_at(acg, 2, 3) == 2);
static_assert(complement(acg, 3) == KmerAcgtMsb{0x39});          // TGC
static_assert(reverse(acg, 3) == KmerAcgtMsb{0x24});             // GCA
static_assert(reverse_complement(acg, 3) == KmerAcgtMsb{0x1B});  // CGT
static_assert(canonical(acg, 3) == acg);
static_assert(kmer_convert<Layout::kLSB>(acg, 3) == KmerAcgtLsb{0x24});
static_assert(kmer_convert<Encoding::kACTG>(acg, 3) == KmerActgMsb{0x07});

// -------------------------------------------------------------------------------------------------
//     Decoding and Base Access
// -------------------------------------------------------------------------------------------------

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

TEST(CoreKmer, Decode)
{
    check_decode<KmerAcgtMsb>();
    check_decode<KmerAcgtLsb>();
    check_decode<KmerActgMsb>();
    check_decode<KmerActgLsb>();
}

// Valid-only sequences (pure ACGT/acgt), lengths 0..70 plus a batch of longer random lengths, as in
// the extraction tests of kmer_extract/.
static std::vector<std::string> const& decode_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(90210);
        for (std::size_t len = 0; len <= 70; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        for (int i = 0; i < 20; ++i) {
            std::size_t const len = static_cast<std::size_t>(rng.get_uint64() % 300);
            out.push_back(random_sequence(rng, len));
        }
        return out;
    }();
    return seqs;
}

// Round-trips kmer_decode() against the oracle: decoding an oracle-computed k-mer must give back
// the (uppercased) substring it was extracted from.
TEST(CoreKmer, DecodeOracleWindows)
{
    for (auto const& seq : decode_sequences()) {
        for (auto const k : test_ks()) {
            auto const windows = oracle_kmers_all_windows<Encoding::kACGT, Layout::kMSB>(seq, k);
            for (auto const& window : windows) {
                std::string upper = seq.substr(window.pos, k);
                for (auto& c : upper) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                }
                auto const kmer = kmer_cast<Encoding::kACGT, Layout::kMSB>(window.value.value(), k);
                EXPECT_EQ(kmer_decode(kmer, k), upper);
            }
        }
    }
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

TEST(CoreKmer, BaseAt)
{
    check_base_at<KmerAcgtMsb>();
    check_base_at<KmerAcgtLsb>();
    check_base_at<KmerActgMsb>();
    check_base_at<KmerActgLsb>();
}

// -------------------------------------------------------------------------------------------------
//     Nucleotide Operations
// -------------------------------------------------------------------------------------------------

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

TEST(CoreKmer, NucleotideOps)
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

TEST(CoreKmer, Canonical)
{
    check_canonical<KmerAcgtMsb>();
    check_canonical<KmerAcgtLsb>();
    check_canonical<KmerActgMsb>();
    check_canonical<KmerActgLsb>();
}

// Within one layout, reversing the bases yields the same bit pattern as reading the unchanged
// word under the opposite layout. Documented in core/kmer.hpp as the clearest illustration of
// why a bare word cannot be interpreted without knowing its conventions.
TEST(CoreKmer, ReverseMirrorsLayout)
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

// -------------------------------------------------------------------------------------------------
//     Convention Conversion
// -------------------------------------------------------------------------------------------------

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

TEST(CoreKmer, ConvertMatchesOracle)
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
TEST(CoreKmer, ConvertRoundTripAndComposition)
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
TEST(CoreKmer, ConvertToSameIsIdentity)
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

// -------------------------------------------------------------------------------------------------
//     Encoding
// -------------------------------------------------------------------------------------------------

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

TEST(CoreKmer, EncodeDecodeRoundTrip)
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

TEST(CoreKmer, EncodeMatchesOracle)
{
    check_encode_matches_oracle<KmerAcgtMsb>();
    check_encode_matches_oracle<KmerAcgtLsb>();
    check_encode_matches_oracle<KmerActgMsb>();
    check_encode_matches_oracle<KmerActgLsb>();
}

// The template argument names the result type exactly. Only contiguous k-mers can be built from a
// string alone; a spaced k-mer also needs its mask, see core/kmer_spaced.hpp.
template <typename K>
concept EncodableFromString = requires(std::string_view seq) { kmer_encode<K>(seq); };

static_assert(std::is_same_v<decltype(kmer_encode<KmerActgLsb>("A")), KmerActgLsb>);
static_assert(EncodableFromString<KmerAcgtMsb>);
static_assert(!EncodableFromString<SpacedKmerAcgtMsb>);
static_assert(!EncodableFromString<std::uint64_t>);

// The default template arguments must be ACGT/MSB, matching what the extraction emits by default.
TEST(CoreKmer, EncodeDefaultsMatchForEachKmer)
{
    static_assert(std::is_same_v<decltype(kmer_encode("ACG")), KmerAcgtMsb>);

    for (auto const& seq : random_seqs(11, 8, 161616)) {
        std::vector<KmerAcgtMsb> from_for_each;
        for_each_kmer_ascii_checked_rolling_scalar(
            seq, seq.size(), CharEncoderTable<Encoding::kACGT>{},
            [&](KmerAcgtMsb kmer) { from_for_each.push_back(kmer); }
        );
        ASSERT_EQ(from_for_each.size(), std::size_t{1});
        EXPECT_EQ(kmer_encode(seq), from_for_each[0]);
    }
}

// The char-to-code table inside kmer_encode() is a deliberate duplication of the encoders in
// core/char_encoder.hpp (see char_to_code_() in core/kmer.hpp for why). This is what keeps the two
// from silently drifting apart: every code kmer_encode() assigns must be the one the ACGT/ACTG
// lookup tables would assign to the same character, upper and lower case alike.
TEST(CoreKmer, EncodeAgreesWithSeqEnc)
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
TEST(CoreKmer, EncodeInvalidInput)
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
//     Spaced K-mers
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Nucleotide Operations and Conversion
// -------------------------------------------------------------------------------------------------

template <typename K>
concept CanReverse = requires(K kmer) { reverse(kmer, 3); };

template <typename K>
concept CanReverseComplement = requires(K kmer) { reverse_complement(kmer, 3); };

template <typename K>
concept CanCanonical = requires(K kmer) { canonical(kmer, 3); };

static_assert(CanReverse<KmerAcgtMsb>);
static_assert(CanReverseComplement<KmerAcgtMsb>);
static_assert(CanCanonical<KmerAcgtMsb>);

static_assert(!CanReverse<SpacedKmerAcgtMsb> && !CanReverseComplement<SpacedKmerAcgtMsb>);
static_assert(!CanCanonical<SpacedKmerAcgtMsb>);

namespace {

    constexpr auto spaced_acg = spaced_kmer_cast<Encoding::kACGT, Layout::kMSB>(0x06, 3);

} // namespace

static_assert(spaced_kmer_cast_unchecked<Encoding::kACGT, Layout::kMSB>(0x06) == spaced_acg);
static_assert(kmer_cast_unchecked<Encoding::kACGT, Layout::kMSB>(0x06) == KmerAcgtMsb{0x06});
static_assert(kmer_value(spaced_acg) == 0x06);
static_assert(base_at(spaced_acg, 0, 3) == 0 && base_at(spaced_acg, 2, 3) == 2);
static_assert(complement(spaced_acg, 3) == SpacedKmerAcgtMsb{0x39});
static_assert(kmer_convert<Layout::kLSB>(spaced_acg, 3) == SpacedKmerAcgtLsb{0x24});
static_assert(kmer_convert<Encoding::kACTG>(spaced_acg, 3) == SpacedKmerActgMsb{0x07});
static_assert(std::is_same_v<
    decltype(kmer_convert<Encoding::kACTG, Layout::kLSB>(spaced_acg, 3)), SpacedKmerActgLsb
>);

// Bases kept by a spaced mask carry the same meaning as the contiguous k-mer of the same bases, so
// decoding and the convention-only operations must agree with the contiguous results.
TEST(CoreKmer, SpacedAgreesWithContiguousOnBaseOps)
{
    std::size_t const width = 9;
    for (auto const& seq : random_seqs(width, 32, 4242)) {
        auto const contiguous = oracle_kmer<KmerAcgtMsb>(seq);
        auto const spaced = spaced_kmer_cast<Encoding::kACGT, Layout::kMSB>(
            kmer_value(contiguous), width
        );
        EXPECT_EQ(kmer_decode(spaced, width), seq);
        EXPECT_EQ(
            kmer_value(complement(spaced, width)),
            kmer_value(complement(contiguous, width))
        );
        EXPECT_EQ(
            kmer_value(kmer_convert<Layout::kLSB>(spaced, width)),
            kmer_value(kmer_convert<Layout::kLSB>(contiguous, width))
        );
    }
}

TEST(CoreKmer, SpacedConvertRoundTripAndComposition)
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 888888 + width)) {
            auto const kmer = spaced_kmer_cast<Encoding::kACGT, Layout::kMSB>(
                kmer_value(oracle_kmer<KmerAcgtMsb>(seq)), width
            );

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
