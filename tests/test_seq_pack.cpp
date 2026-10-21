#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/core/random.hpp"
#include "fisk/core/types.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Ground truth two-bit codes come from oracle_code() in oracle.hpp, independent of the encoders
// under test.

// Decode base `i` per PackedSequence's documented contract (see Layout in core/types.hpp),
// independent of pack_sequence()'s internals.
template <PackedSequenceType Seq>
static int decode_base(Seq const& s, std::size_t i)
{
    std::uint8_t const byte = s.codes()[i / 4];
    std::size_t const in_byte = i % 4;
    if constexpr (Seq::layout == Layout::kMSB) {
        return (byte >> (6 - 2 * in_byte)) & 0x3;
    } else {
        return (byte >> (2 * in_byte)) & 0x3;
    }
}

// Sequence lengths 0..65 plus a batch of longer random lengths, each with random per-position
// content, so bit- and byte-order correctness are both exercised at every length.
static std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(3003);
        for (std::size_t len = 0; len <= 65; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        for (int i = 0; i < 30; ++i) {
            std::size_t const len = static_cast<std::size_t>(rng.get_uint64() % 200);
            out.push_back(random_sequence(rng, len));
        }
        return out;
    }();
    return seqs;
}

// Sequences with invalid characters from the full byte range. The word encoders assume valid
// input, so the code they give an invalid character is unspecified; but it must not affect any
// other position, nor the padding.
static std::vector<std::string> const& invalid_sequences()
{
    static std::vector<std::string> const seqs = invalid_injected_sequences(40, 200, 0.1, 3004);
    return seqs;
}

// Checks a packed sequence against `seq` and the oracle: length, number of bytes of codes, the
// code of every valid base, and zeros beyond the length.
template <PackedSequenceType Seq>
static void check_packed(std::string const& seq, Seq const& packed)
{
    EXPECT_EQ(packed.length(), seq.size());
    EXPECT_EQ(packed.codes().size(), (seq.size() + 3) / 4);

    for (std::size_t i = 0; i < seq.size(); ++i) {
        std::uint8_t const code = oracle_code<Seq::encoding>(seq[i]);
        if (code < kInvalidNucleotide) {
            EXPECT_EQ(decode_base(packed, i), static_cast<int>(code));
        }
    }
    check_zero_beyond_length(packed);
}

// Checks pack_sequence() -- both overloads, and reuse of one PackedSequence across calls -- for a
// given word encoder, on valid and invalid input.
template <typename Encoder>
static void check_seq_pack()
{
    Encoder encoder{};
    PackedSequence<Encoder::encoding, Encoder::layout> out;

    for (auto const* seqs : {&test_sequences(), &invalid_sequences()}) {
        for (auto const& seq : *seqs) {
            pack_sequence(seq, encoder, out);
            check_packed(seq, out);

            auto const out2 = pack_sequence(seq, encoder);
            check_packed(seq, out2);
        }
    }

    // Reuse must not leak stale bytes from a previous, larger pack.
    pack_sequence(std::string(100, 'T'), encoder, out);
    pack_sequence(std::string("AC"), encoder, out);
    check_packed(std::string("AC"), out);
}

// -----------------------------------------------------------------------------
//     Validity
// -----------------------------------------------------------------------------

static bool is_valid_nucleotide(char c)
{
    return kValidBases.find(c) != std::string_view::npos;
}

// Checks `valid_bits`, called on a pointer to `Bytes` characters, against the oracle, for every
// byte value at every position. The other positions hold random valid characters, random invalid
// ones, which include those that differ from a valid one in a single bit, or random bytes.
template <std::size_t Bytes, typename ValidBits>
static void check_valid_nucleotide_bits(ValidBits valid_bits)
{
    Splitmix64 rng(3005);
    std::string const any_byte = [] {
        std::string out;
        for (int i = 0; i < 256; ++i) {
            out += static_cast<char>(i);
        }
        return out;
    }();

    for (std::string_view background : {kValidBases, std::string_view(all_invalid_bytes()),
                                        std::string_view(any_byte)}) {
        std::string buffer = random_sequence(rng, Bytes, background);
        for (std::size_t pos = 0; pos < Bytes; ++pos) {
            char const kept = buffer[pos];
            for (int value = 0; value < 256; ++value) {
                buffer[pos] = static_cast<char>(value);
                std::uint64_t expected = 0;
                for (std::size_t j = 0; j < Bytes; ++j) {
                    expected |= static_cast<std::uint64_t>(is_valid_nucleotide(buffer[j])) << j;
                }
                EXPECT_EQ(valid_bits(buffer.data()), expected);
            }
            buffer[pos] = kept;
        }
    }
}

// Checks valid_nucleotide_bits(), and its lookup and compare forms where they exist, for an arch.
template <typename Simd>
static void check_valid_nucleotide_bits_arch()
{
    constexpr std::size_t bytes = 8 * Simd::lanes;
    auto const load = [](char const* p) {
        return Simd::load_unaligned(reinterpret_cast<std::uint64_t const*>(p));
    };
    check_valid_nucleotide_bits<bytes>([&](char const* p) {
        return valid_nucleotide_bits<Simd>(load(p));
    });
    check_valid_nucleotide_bits<bytes>([&](char const* p) {
        return valid_nucleotide_bits_compare_<Simd>(load(p));
    });
    if constexpr (SimdArchWithByteShuffle<Simd>) {
        check_valid_nucleotide_bits<bytes>([&](char const* p) {
            return valid_nucleotide_bits_lookup_<Simd>(load(p));
        });
    }
}

// Every length up to 130, which covers the end of a word (8), of a vector of each arch (16, 32,
// 64), and of a double step (64, 128) with both of their neighbors, and some longer ones. Each
// length comes valid only, with invalid bytes of the whole range, and with many hand-picked ones.
static std::vector<std::string> const& validity_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(3006);
        std::vector<std::size_t> lengths;
        for (std::size_t len = 0; len <= 130; ++len) {
            lengths.push_back(len);
        }
        lengths.insert(lengths.end(), {255, 256, 257, 1000});
        for (std::size_t const len : lengths) {
            out.push_back(random_sequence(rng, len));
            std::string seq = random_sequence(rng, len);
            inject_invalid(seq, rng, 0.1, all_invalid_bytes());
            out.push_back(seq);
            seq = random_sequence(rng, len);
            inject_invalid(seq, rng, 0.5, kHandPickedInvalid);
            out.push_back(seq);
        }
        return out;
    }();
    return seqs;
}

// Checks the validity bits of a PackedSequenceWithValidity against `seq`: one bit per character,
// no more bytes than these need, and zeros beyond the end, see check_zero_beyond_length().
template <Encoding E, Layout L>
static void check_validity(std::string const& seq, PackedSequenceWithValidity<E, L> const& packed)
{
    auto const validity = packed.validity();
    ASSERT_EQ(validity.size(), (seq.size() + 7) / 8);
    for (std::size_t i = 0; i < 8 * validity.size(); ++i) {
        bool const bit = (validity[i / 8] >> (i % 8)) & 1;
        EXPECT_EQ(bit, i < seq.size() && is_valid_nucleotide(seq[i]));
    }
    check_zero_beyond_length(packed);
}

// Checks pack_sequence_with_validity() -- both overloads, the overload of pack_sequence() that
// forwards to it, and reuse across calls -- for a given word encoder. Its codes have to be the
// same as those of pack_sequence(), including the unspecified ones of invalid characters.
template <typename Encoder>
static void check_seq_pack_validity()
{
    Encoder encoder{};
    PackedSequence<Encoder::encoding, Encoder::layout> plain;
    PackedSequenceWithValidity<Encoder::encoding, Encoder::layout> out;
    PackedSequenceWithValidity<Encoder::encoding, Encoder::layout> forwarded;

    for (auto const& seq : validity_sequences()) {
        pack_sequence(seq, encoder, plain);

        pack_sequence_with_validity(seq, encoder, out);
        check_packed(seq, out);
        check_validity(seq, out);
        EXPECT_EQ(out.length(), plain.length());
        EXPECT_TRUE(std::ranges::equal(out.codes(), plain.codes()));

        auto const out2 = pack_sequence_with_validity(seq, encoder);
        check_packed(seq, out2);
        check_validity(seq, out2);

        pack_sequence(seq, encoder, forwarded);
        check_validity(seq, forwarded);
    }

    // Reuse must not leak stale bits from a previous, larger pack.
    pack_sequence_with_validity(std::string(100, 'T'), encoder, out);
    pack_sequence_with_validity(std::string("ANC"), encoder, out);
    check_packed(std::string("ANC"), out);
    check_validity(std::string("ANC"), out);
}

// Runs check_seq_pack_validity() for all four combinations of Encoding and Layout.
template <template <Encoding, Layout, typename> class Encoder, typename Simd>
static void check_seq_pack_validity_all()
{
    check_seq_pack_validity<Encoder<Encoding::kACGT, Layout::kLSB, Simd>>();
    check_seq_pack_validity<Encoder<Encoding::kACGT, Layout::kMSB, Simd>>();
    check_seq_pack_validity<Encoder<Encoding::kACTG, Layout::kLSB, Simd>>();
    check_seq_pack_validity<Encoder<Encoding::kACTG, Layout::kMSB, Simd>>();
}

// =================================================================================================
//     Packed Sequence Classes
// =================================================================================================

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
    EXPECT_TRUE(std::ranges::all_of(packed.codes(), [](auto b) { return b == 0; }));
    EXPECT_TRUE(std::ranges::all_of(packed.validity(), [](auto b) { return b == 0; }));
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

TEST(PackedSequence, ConstructResizeMoveMsb)
{
    check_construct_resize_move<Layout::kMSB>();
}

TEST(PackedSequence, ConstructResizeMoveLsb)
{
    check_construct_resize_move<Layout::kLSB>();
}

// =================================================================================================
//     Compile-Time Conventions
// =================================================================================================

// Encoding and Layout are part of PackedSequence's type, so mixing them up between producer and
// consumer is rejected by the compiler. These checks pin that guarantee down.

namespace {

using AcgtMsb = PackedSequence<Encoding::kACGT, Layout::kMSB>;
using AcgtLsb = PackedSequence<Encoding::kACGT, Layout::kLSB>;
using ActgMsb = PackedSequence<Encoding::kACTG, Layout::kMSB>;

template <typename Encoder, Encoding E, Layout L>
constexpr bool tagged_as = Encoder::encoding == E && Encoder::layout == L;

template <typename Encoder, typename Out>
concept PackableInto = requires(std::string_view seq, Encoder encoder, Out& out) {
    pack_sequence(seq, encoder, out);
};

using UntaggedWordEncoder = decltype([](std::uint64_t word) -> std::uint64_t { return word; });

} // namespace

// Every word encoder satisfies the concept that pack_sequence() requires; the per-character
// encoders do not, and neither does a bare callable that states no conventions.
static_assert( WordEncoder<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>>);
static_assert( WordEncoder<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB>>);
#if defined(FISK_HAS_BMI2)
static_assert( WordEncoder<WordEncoderPext<Encoding::kACGT, Layout::kLSB>>);
#endif
static_assert( WordEncoder<WordEncoderDirect<Encoding::kACGT, Layout::kLSB>>);
static_assert( WordEncoder<WordEncoderFold<Encoding::kACGT, Layout::kLSB>>);
#if defined(FISK_HAS_SSE2)
static_assert( WordEncoder<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB, SimdSSE2>>);
static_assert( WordEncoder<WordEncoderDirect<Encoding::kACTG, Layout::kMSB, SimdSSE2>>);
static_assert( WordEncoder<WordEncoderFold<Encoding::kACTG, Layout::kMSB, SimdSSE2>>);
#endif
static_assert(!WordEncoder<CharEncoderTable<Encoding::kACGT>>);
static_assert(!WordEncoder<UntaggedWordEncoder>);
static_assert(!PackableInto<UntaggedWordEncoder, AcgtMsb>);

// Scalar word encoders carry `encoding` as a pure tag that packing never consults, so the
// byte-level tests below could not catch a wrong one; check that each instantiation forwards its
// template arguments into its tags instead.
static_assert(
    tagged_as<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB>, Encoding::kACTG, Layout::kLSB>
);
static_assert(
    tagged_as<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB>, Encoding::kACTG, Layout::kMSB>
);
static_assert(
    tagged_as<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>, Encoding::kACGT, Layout::kLSB>
);
static_assert(
    tagged_as<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>, Encoding::kACGT, Layout::kMSB>
);
#if defined(FISK_HAS_BMI2)
static_assert(tagged_as<WordEncoderPext<Encoding::kACTG, Layout::kLSB>, Encoding::kACTG, Layout::kLSB>);
static_assert(tagged_as<WordEncoderPext<Encoding::kACTG, Layout::kMSB>, Encoding::kACTG, Layout::kMSB>);
static_assert(tagged_as<WordEncoderPext<Encoding::kACGT, Layout::kLSB>, Encoding::kACGT, Layout::kLSB>);
static_assert(tagged_as<WordEncoderPext<Encoding::kACGT, Layout::kMSB>, Encoding::kACGT, Layout::kMSB>);
#endif // FISK_HAS_BMI2

// The number of lanes is what selects the vector loop of pack_sequence().
static_assert(WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>::lanes == 1);
static_assert(WordEncoderDirect<Encoding::kACGT, Layout::kMSB>::lanes == 1);
static_assert(WordEncoderFold<Encoding::kACGT, Layout::kMSB>::lanes == 1);
#if defined(FISK_HAS_BMI2)
static_assert(WordEncoderPext<Encoding::kACGT, Layout::kMSB>::lanes == 1);
#if defined(FISK_HAS_SSE2)
static_assert(WordEncoderPext<Encoding::kACGT, Layout::kMSB, SimdSSE2>::lanes == 2);
#endif
#if defined(FISK_HAS_AVX2)
static_assert(WordEncoderPext<Encoding::kACGT, Layout::kMSB, SimdAVX2>::lanes == 4);
#endif
#if defined(FISK_HAS_AVX512)
static_assert(WordEncoderPext<Encoding::kACGT, Layout::kMSB, SimdAVX512>::lanes == 8);
#endif
#endif // FISK_HAS_BMI2
#if defined(FISK_HAS_SSE2)
static_assert(WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdSSE2>::lanes == 2);
static_assert(WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdSSE2>::lanes == 2);
static_assert(WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdSSE2>::lanes == 2);
#if defined(FISK_HAS_SSSE3)
static_assert(WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdSSE2>::lanes == 4);
#endif
#endif
#if defined(FISK_HAS_AVX2)
static_assert(WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdAVX2>::lanes == 4);
static_assert(WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdAVX2>::lanes == 4);
static_assert(WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdAVX2>::lanes == 4);
static_assert(WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdAVX2>::lanes == 8);
static_assert( WordEncoder<WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdAVX2>>);
#endif
#if defined(FISK_HAS_AVX512)
static_assert(WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdAVX512>::lanes == 8);
static_assert(WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdAVX512>::lanes == 8);
static_assert(WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdAVX512>::lanes == 8);
static_assert(WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdAVX512>::lanes == 16);
#endif
#if defined(FISK_HAS_NEON)
static_assert(WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdNEON>::lanes == 2);
static_assert(WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdNEON>::lanes == 2);
static_assert(WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdNEON>::lanes == 2);
static_assert(WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdNEON>::lanes == 4);
#endif

// The packed sequence type follows the encoder's tags.
static_assert(std::is_same_v<
    decltype(pack_sequence("", WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{})),
    AcgtMsb
>);

// Packing into an existing sequence of a different encoding or layout does not compile.
static_assert( PackableInto<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>, AcgtMsb>);
static_assert(!PackableInto<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>, ActgMsb>);
static_assert(!PackableInto<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>, AcgtLsb>);

// Nor can a sequence of one encoding be passed where another is expected.
static_assert(!std::is_convertible_v<ActgMsb const&, AcgtMsb const&>);

// A sequence with validity is not a plain one, so that its codes cannot be resized without its
// validity bits, but it gives one with sequence(). Packing it with either function fills its
// validity, see check_seq_pack_validity().
using AcgtMsbWithValidity = PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>;
static_assert(!std::is_convertible_v<AcgtMsbWithValidity&, AcgtMsb&>);
static_assert(!std::is_convertible_v<AcgtMsbWithValidity const&, AcgtMsb const&>);
static_assert(std::is_same_v<
    decltype(std::declval<AcgtMsbWithValidity const&>().sequence()), AcgtMsb const&
>);
static_assert(std::is_same_v<
    decltype(pack_sequence_with_validity("", WordEncoderDirect<Encoding::kACGT, Layout::kMSB>{})),
    AcgtMsbWithValidity
>);
static_assert(
    PackableInto<WordEncoderDirect<Encoding::kACGT, Layout::kMSB>, AcgtMsbWithValidity>
);
static_assert(
    !PackableInto<WordEncoderDirect<Encoding::kACGT, Layout::kLSB>, AcgtMsbWithValidity>
);

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

// =================================================================================================
//     Scalar Word Encoders
// =================================================================================================

// -----------------------------------------------------------------------------
//     Butterfly Table
// -----------------------------------------------------------------------------

TEST(SeqPack, ScalarButterflyActgLsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB>>();
}

TEST(SeqPack, ScalarButterflyActgMsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB>>();
}

TEST(SeqPack, ScalarButterflyAcgtLsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>>();
}

TEST(SeqPack, ScalarButterflyAcgtMsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>>();
}

// -----------------------------------------------------------------------------
//     Fold
// -----------------------------------------------------------------------------

TEST(SeqPack, ScalarFoldActgLsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACTG, Layout::kLSB>>();
}

TEST(SeqPack, ScalarFoldActgMsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACTG, Layout::kMSB>>();
}

TEST(SeqPack, ScalarFoldAcgtLsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACGT, Layout::kLSB>>();
}

TEST(SeqPack, ScalarFoldAcgtMsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACGT, Layout::kMSB>>();
}

// -----------------------------------------------------------------------------
//     Direct
// -----------------------------------------------------------------------------

TEST(SeqPack, ScalarDirectActgLsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACTG, Layout::kLSB>>();
}

TEST(SeqPack, ScalarDirectActgMsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACTG, Layout::kMSB>>();
}

TEST(SeqPack, ScalarDirectAcgtLsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACGT, Layout::kLSB>>();
}

TEST(SeqPack, ScalarDirectAcgtMsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACGT, Layout::kMSB>>();
}

// -----------------------------------------------------------------------------
//     PEXT
// -----------------------------------------------------------------------------

#if defined(FISK_HAS_BMI2)

TEST(SeqPack, ScalarPextActgLsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACTG, Layout::kLSB>>();
}

TEST(SeqPack, ScalarPextActgMsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACTG, Layout::kMSB>>();
}

TEST(SeqPack, ScalarPextAcgtLsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACGT, Layout::kLSB>>();
}

TEST(SeqPack, ScalarPextAcgtMsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACGT, Layout::kMSB>>();
}

#endif // FISK_HAS_BMI2

// =================================================================================================
//     SIMD Word Encoders
// =================================================================================================

#if defined(FISK_HAS_SSE2)

TEST(SeqPackSimd, Sse2ActgLsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2ActgMsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2AcgtLsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2AcgtMsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2FoldActgLsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACTG, Layout::kLSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2FoldActgMsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACTG, Layout::kMSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2FoldAcgtLsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACGT, Layout::kLSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2FoldAcgtMsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2DirectActgLsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACTG, Layout::kLSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2DirectActgMsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACTG, Layout::kMSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2DirectAcgtLsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACGT, Layout::kLSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2DirectAcgtMsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdSSE2>>();
}
#if defined(FISK_HAS_SSSE3)
TEST(SeqPackSimd, Sse2DirectDoubleActgLsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACTG, Layout::kLSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2DirectDoubleActgMsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACTG, Layout::kMSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2DirectDoubleAcgtLsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACGT, Layout::kLSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2DirectDoubleAcgtMsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdSSE2>>();
}
#endif // FISK_HAS_SSSE3
#if defined(FISK_HAS_BMI2)
TEST(SeqPackSimd, Sse2PextActgLsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACTG, Layout::kLSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2PextActgMsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACTG, Layout::kMSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2PextAcgtLsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACGT, Layout::kLSB, SimdSSE2>>();
}
TEST(SeqPackSimd, Sse2PextAcgtMsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACGT, Layout::kMSB, SimdSSE2>>();
}
#endif // FISK_HAS_BMI2

#endif // FISK_HAS_SSE2

#if defined(FISK_HAS_AVX2)

TEST(SeqPackSimd, Avx2ActgLsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2ActgMsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2AcgtLsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2AcgtMsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2FoldActgLsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACTG, Layout::kLSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2FoldActgMsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACTG, Layout::kMSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2FoldAcgtLsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACGT, Layout::kLSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2FoldAcgtMsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2DirectActgLsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACTG, Layout::kLSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2DirectActgMsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACTG, Layout::kMSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2DirectAcgtLsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACGT, Layout::kLSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2DirectAcgtMsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2DirectDoubleActgLsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACTG, Layout::kLSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2DirectDoubleActgMsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACTG, Layout::kMSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2DirectDoubleAcgtLsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACGT, Layout::kLSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2DirectDoubleAcgtMsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdAVX2>>();
}
#if defined(FISK_HAS_BMI2)
TEST(SeqPackSimd, Avx2PextActgLsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACTG, Layout::kLSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2PextActgMsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACTG, Layout::kMSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2PextAcgtLsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACGT, Layout::kLSB, SimdAVX2>>();
}
TEST(SeqPackSimd, Avx2PextAcgtMsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACGT, Layout::kMSB, SimdAVX2>>();
}
#endif // FISK_HAS_BMI2

#endif // FISK_HAS_AVX2

#if defined(FISK_HAS_AVX512)

TEST(SeqPackSimd, Avx512ActgLsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512ActgMsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512AcgtLsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512AcgtMsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512FoldActgLsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACTG, Layout::kLSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512FoldActgMsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACTG, Layout::kMSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512FoldAcgtLsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACGT, Layout::kLSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512FoldAcgtMsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512DirectActgLsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACTG, Layout::kLSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512DirectActgMsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACTG, Layout::kMSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512DirectAcgtLsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACGT, Layout::kLSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512DirectAcgtMsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512DirectDoubleActgLsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACTG, Layout::kLSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512DirectDoubleActgMsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACTG, Layout::kMSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512DirectDoubleAcgtLsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACGT, Layout::kLSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512DirectDoubleAcgtMsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdAVX512>>();
}
#if defined(FISK_HAS_BMI2)
TEST(SeqPackSimd, Avx512PextActgLsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACTG, Layout::kLSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512PextActgMsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACTG, Layout::kMSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512PextAcgtLsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACGT, Layout::kLSB, SimdAVX512>>();
}
TEST(SeqPackSimd, Avx512PextAcgtMsb)
{
    check_seq_pack<WordEncoderPext<Encoding::kACGT, Layout::kMSB, SimdAVX512>>();
}
#endif // FISK_HAS_BMI2

#endif // FISK_HAS_AVX512

#if defined(FISK_HAS_NEON)

TEST(SeqPackSimd, NeonActgLsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonActgMsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonAcgtLsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonAcgtMsb)
{
    check_seq_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonFoldActgLsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACTG, Layout::kLSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonFoldActgMsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACTG, Layout::kMSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonFoldAcgtLsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACGT, Layout::kLSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonFoldAcgtMsb)
{
    check_seq_pack<WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonDirectActgLsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACTG, Layout::kLSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonDirectActgMsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACTG, Layout::kMSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonDirectAcgtLsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACGT, Layout::kLSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonDirectAcgtMsb)
{
    check_seq_pack<WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonDirectDoubleActgLsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACTG, Layout::kLSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonDirectDoubleActgMsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACTG, Layout::kMSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonDirectDoubleAcgtLsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACGT, Layout::kLSB, SimdNEON>>();
}
TEST(SeqPackSimd, NeonDirectDoubleAcgtMsb)
{
    check_seq_pack<WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdNEON>>();
}

#endif // FISK_HAS_NEON

// =================================================================================================
//     Validity
// =================================================================================================

// -----------------------------------------------------------------------------
//     Nucleotide Bits
// -----------------------------------------------------------------------------

TEST(SeqPackValidity, NucleotideBitsScalar)
{
    check_valid_nucleotide_bits_arch<SimdScalar>();
}

#if defined(FISK_HAS_SSE2)
TEST(SeqPackValidity, NucleotideBitsSse2)
{
    check_valid_nucleotide_bits_arch<SimdSSE2>();
}
#endif

#if defined(FISK_HAS_AVX2)
TEST(SeqPackValidity, NucleotideBitsAvx2)
{
    check_valid_nucleotide_bits_arch<SimdAVX2>();
}
#endif

#if defined(FISK_HAS_AVX512)
TEST(SeqPackValidity, NucleotideBitsAvx512)
{
    check_valid_nucleotide_bits_arch<SimdAVX512>();
}
#endif

#if defined(FISK_HAS_NEON)
TEST(SeqPackValidity, NucleotideBitsNeon)
{
    check_valid_nucleotide_bits_arch<SimdNEON>();
}
#endif

// -----------------------------------------------------------------------------
//     Pack Sequence With Validity
// -----------------------------------------------------------------------------

TEST(SeqPackValidity, Scalar)
{
    check_seq_pack_validity_all<WordEncoderButterfly, SimdScalar>();
    check_seq_pack_validity_all<WordEncoderFold, SimdScalar>();
    check_seq_pack_validity_all<WordEncoderDirect, SimdScalar>();
    #if defined(FISK_HAS_BMI2)
    check_seq_pack_validity_all<WordEncoderPext, SimdScalar>();
    #endif
}

#if defined(FISK_HAS_SSE2)
TEST(SeqPackValidity, Sse2)
{
    check_seq_pack_validity_all<WordEncoderButterfly, SimdSSE2>();
    check_seq_pack_validity_all<WordEncoderFold, SimdSSE2>();
    check_seq_pack_validity_all<WordEncoderDirect, SimdSSE2>();
    #if defined(FISK_HAS_SSSE3)
    check_seq_pack_validity_all<WordEncoderDirectDouble, SimdSSE2>();
    #endif
    #if defined(FISK_HAS_BMI2)
    check_seq_pack_validity_all<WordEncoderPext, SimdSSE2>();
    #endif
}
#endif

#if defined(FISK_HAS_AVX2)
TEST(SeqPackValidity, Avx2)
{
    check_seq_pack_validity_all<WordEncoderButterfly, SimdAVX2>();
    check_seq_pack_validity_all<WordEncoderFold, SimdAVX2>();
    check_seq_pack_validity_all<WordEncoderDirect, SimdAVX2>();
    check_seq_pack_validity_all<WordEncoderDirectDouble, SimdAVX2>();
    #if defined(FISK_HAS_BMI2)
    check_seq_pack_validity_all<WordEncoderPext, SimdAVX2>();
    #endif
}
#endif

#if defined(FISK_HAS_AVX512)
TEST(SeqPackValidity, Avx512)
{
    check_seq_pack_validity_all<WordEncoderButterfly, SimdAVX512>();
    check_seq_pack_validity_all<WordEncoderFold, SimdAVX512>();
    check_seq_pack_validity_all<WordEncoderDirect, SimdAVX512>();
    check_seq_pack_validity_all<WordEncoderDirectDouble, SimdAVX512>();
    #if defined(FISK_HAS_BMI2)
    check_seq_pack_validity_all<WordEncoderPext, SimdAVX512>();
    #endif
}
#endif

#if defined(FISK_HAS_NEON)
TEST(SeqPackValidity, Neon)
{
    check_seq_pack_validity_all<WordEncoderButterfly, SimdNEON>();
    check_seq_pack_validity_all<WordEncoderFold, SimdNEON>();
    check_seq_pack_validity_all<WordEncoderDirect, SimdNEON>();
    check_seq_pack_validity_all<WordEncoderDirectDouble, SimdNEON>();
}
#endif
