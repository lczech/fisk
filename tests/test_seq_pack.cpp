#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/core/random.hpp"
#include "fisk/core/types.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
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
template <Encoding E, Layout L>
static int decode_base(PackedSequence<E, L> const& s, std::size_t i)
{
    std::uint8_t const byte = s.data[i / 4];
    std::size_t const in_byte = i % 4;
    if constexpr (L == Layout::kMSB) {
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

// Checks a PackedSequence against `seq` and the oracle: length, data size, the code of every
// valid base, and zero padding.
template <Encoding E, Layout L>
static void check_packed(std::string const& seq, PackedSequence<E, L> const& packed)
{
    EXPECT_EQ(packed.length, seq.size());

    std::size_t const expected_content_bytes = (seq.size() + 3) / 4;
    EXPECT_EQ(packed.data.size(), expected_content_bytes);

    for (std::size_t i = 0; i < seq.size(); ++i) {
        std::uint8_t const code = oracle_code<E>(seq[i]);
        if (code < kInvalidNucleotide) {
            EXPECT_EQ(decode_base(packed, i), static_cast<int>(code));
        }
    }
    check_packed_padding_zero(packed);
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
#if defined(FISK_HAS_SSE2)
static_assert( WordEncoder<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB, SimdSSE2>>);
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
#if defined(FISK_HAS_BMI2)
static_assert(WordEncoderPext<Encoding::kACGT, Layout::kMSB>::lanes == 1);
#endif
#if defined(FISK_HAS_SSE2)
static_assert(WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdSSE2>::lanes == 2);
#endif
#if defined(FISK_HAS_AVX2)
static_assert(WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdAVX2>::lanes == 4);
#endif
#if defined(FISK_HAS_AVX512)
static_assert(WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdAVX512>::lanes == 8);
#endif
#if defined(FISK_HAS_NEON)
static_assert(WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdNEON>::lanes == 2);
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

#endif // FISK_HAS_NEON
