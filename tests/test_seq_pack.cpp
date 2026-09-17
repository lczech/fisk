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
#include "fisk/seq_pack/simd.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Ground truth two-bit codes, independent of the encoders under test.

static int expected_acgt(char c)
{
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return -1;
    }
}

static int expected_actg(char c)
{
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'T': case 't': return 2;
        case 'G': case 'g': return 3;
        default:            return -1;
    }
}

// Decode base `i` per PackedSequence's documented contract (see Layout in core/types.hpp),
// independent of pack_sequence()/pack_sequence_simd()'s internals.
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
        char const bases[] = "ACGTacgt";
        Splitmix64 rng(3003);

        auto random_seq = [&](std::size_t len) {
            std::string s;
            s.reserve(len);
            for (std::size_t k = 0; k < len; ++k) {
                s += bases[rng.get_uint64() % 8];
            }
            return s;
        };

        for (std::size_t len = 0; len <= 65; ++len) {
            out.push_back(random_seq(len));
        }
        for (int i = 0; i < 30; ++i) {
            std::size_t const len = static_cast<std::size_t>(rng.get_uint64() % 200);
            out.push_back(random_seq(len));
        }
        return out;
    }();
    return seqs;
}

// Checks a PackedSequence against `seq` and its oracle: length, data size, and every
// base's code.
template <Encoding E, Layout L, typename OracleFn>
static void check_packed(
    std::string const& seq, PackedSequence<E, L> const& packed, OracleFn&& oracle
) {
    EXPECT_EQ(packed.length, seq.size());

    std::size_t const expected_content_bytes = (seq.size() + 3) / 4;
    EXPECT_EQ(packed.data.size(), expected_content_bytes);

    for (std::size_t i = 0; i < seq.size(); ++i) {
        EXPECT_EQ(decode_base(packed, i), oracle(seq[i]));
    }
}

// Checks pack_sequence() -- both overloads, and reuse of one PackedSequence across calls -- for a
// given scalar word encoder and its matching oracle.
template <typename Encoder, typename OracleFn>
static void check_scalar_pack(OracleFn&& oracle)
{
    Encoder encoder{};
    PackedSequence<Encoder::encoding, Encoder::layout> out;

    for (auto const& seq : test_sequences()) {
        pack_sequence(seq, encoder, out);
        check_packed(seq, out, oracle);

        auto const out2 = pack_sequence(seq, encoder);
        check_packed(seq, out2, oracle);
    }

    // Reuse must not leak stale bytes from a previous, larger pack.
    pack_sequence(std::string(100, 'T'), encoder, out);
    pack_sequence(std::string("AC"), encoder, out);
    check_packed(std::string("AC"), out, oracle);
}

#if defined(FISK_HAS_SSE2)   || \
    defined(FISK_HAS_AVX2)   || \
    defined(FISK_HAS_AVX512) || \
    defined(FISK_HAS_NEON)

// Same as check_scalar_pack(), but for pack_sequence_simd() and a SIMD word encoder.
template <typename Encoder, typename OracleFn>
static void check_simd_pack(OracleFn&& oracle)
{
    Encoder encoder{};
    PackedSequence<Encoder::encoding, Encoder::layout> out;

    for (auto const& seq : test_sequences()) {
        pack_sequence_simd(seq, encoder, out);
        check_packed(seq, out, oracle);

        auto const out2 = pack_sequence_simd(seq, encoder);
        check_packed(seq, out2, oracle);
    }

    pack_sequence_simd(std::string(100, 'T'), encoder, out);
    pack_sequence_simd(std::string("AC"), encoder, out);
    check_packed(std::string("AC"), out, oracle);
}

#endif

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
static_assert( WordEncoder<WordEncoderButterflySSE2<Encoding::kACTG, Layout::kMSB>>);
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
//     Scalar pack_sequence()
// =================================================================================================

// -----------------------------------------------------------------------------
//     Butterfly Table
// -----------------------------------------------------------------------------

TEST(SeqPack, ScalarButterflyActgLsb)
{
    check_scalar_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB>>(expected_actg);
}

TEST(SeqPack, ScalarButterflyActgMsb)
{
    check_scalar_pack<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB>>(expected_actg);
}

TEST(SeqPack, ScalarButterflyAcgtLsb)
{
    check_scalar_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>>(expected_acgt);
}

TEST(SeqPack, ScalarButterflyAcgtMsb)
{
    check_scalar_pack<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>>(expected_acgt);
}

// -----------------------------------------------------------------------------
//     PEXT
// -----------------------------------------------------------------------------

#if defined(FISK_HAS_BMI2)

TEST(SeqPack, ScalarPextActgLsb)
{
    check_scalar_pack<WordEncoderPext<Encoding::kACTG, Layout::kLSB>>(expected_actg);
}

TEST(SeqPack, ScalarPextActgMsb)
{
    check_scalar_pack<WordEncoderPext<Encoding::kACTG, Layout::kMSB>>(expected_actg);
}

TEST(SeqPack, ScalarPextAcgtLsb)
{
    check_scalar_pack<WordEncoderPext<Encoding::kACGT, Layout::kLSB>>(expected_acgt);
}

TEST(SeqPack, ScalarPextAcgtMsb)
{
    check_scalar_pack<WordEncoderPext<Encoding::kACGT, Layout::kMSB>>(expected_acgt);
}

#endif // FISK_HAS_BMI2

// =================================================================================================
//     SIMD pack_sequence_simd()
// =================================================================================================

#if defined(FISK_HAS_SSE2)

TEST(SeqPackSimd, Sse2ActgLsb)
{
    check_simd_pack<WordEncoderButterflySSE2<Encoding::kACTG, Layout::kLSB>>(expected_actg);
}
TEST(SeqPackSimd, Sse2ActgMsb)
{
    check_simd_pack<WordEncoderButterflySSE2<Encoding::kACTG, Layout::kMSB>>(expected_actg);
}
TEST(SeqPackSimd, Sse2AcgtLsb)
{
    check_simd_pack<WordEncoderButterflySSE2<Encoding::kACGT, Layout::kLSB>>(expected_acgt);
}
TEST(SeqPackSimd, Sse2AcgtMsb)
{
    check_simd_pack<WordEncoderButterflySSE2<Encoding::kACGT, Layout::kMSB>>(expected_acgt);
}

#endif // FISK_HAS_SSE2

#if defined(FISK_HAS_AVX2)

TEST(SeqPackSimd, Avx2ActgLsb)
{
    check_simd_pack<WordEncoderButterflyAVX2<Encoding::kACTG, Layout::kLSB>>(expected_actg);
}
TEST(SeqPackSimd, Avx2ActgMsb)
{
    check_simd_pack<WordEncoderButterflyAVX2<Encoding::kACTG, Layout::kMSB>>(expected_actg);
}
TEST(SeqPackSimd, Avx2AcgtLsb)
{
    check_simd_pack<WordEncoderButterflyAVX2<Encoding::kACGT, Layout::kLSB>>(expected_acgt);
}
TEST(SeqPackSimd, Avx2AcgtMsb)
{
    check_simd_pack<WordEncoderButterflyAVX2<Encoding::kACGT, Layout::kMSB>>(expected_acgt);
}

#endif // FISK_HAS_AVX2

#if defined(FISK_HAS_AVX512)

TEST(SeqPackSimd, Avx512ActgLsb)
{
    check_simd_pack<WordEncoderButterflyAVX512<Encoding::kACTG, Layout::kLSB>>(expected_actg);
}
TEST(SeqPackSimd, Avx512ActgMsb)
{
    check_simd_pack<WordEncoderButterflyAVX512<Encoding::kACTG, Layout::kMSB>>(expected_actg);
}
TEST(SeqPackSimd, Avx512AcgtLsb)
{
    check_simd_pack<WordEncoderButterflyAVX512<Encoding::kACGT, Layout::kLSB>>(expected_acgt);
}
TEST(SeqPackSimd, Avx512AcgtMsb)
{
    check_simd_pack<WordEncoderButterflyAVX512<Encoding::kACGT, Layout::kMSB>>(expected_acgt);
}

#endif // FISK_HAS_AVX512

#if defined(FISK_HAS_NEON)

TEST(SeqPackSimd, NeonActgLsb)
{
    check_simd_pack<WordEncoderButterflyNEON<Encoding::kACTG, Layout::kLSB>>(expected_actg);
}
TEST(SeqPackSimd, NeonActgMsb)
{
    check_simd_pack<WordEncoderButterflyNEON<Encoding::kACTG, Layout::kMSB>>(expected_actg);
}
TEST(SeqPackSimd, NeonAcgtLsb)
{
    check_simd_pack<WordEncoderButterflyNEON<Encoding::kACGT, Layout::kLSB>>(expected_acgt);
}
TEST(SeqPackSimd, NeonAcgtMsb)
{
    check_simd_pack<WordEncoderButterflyNEON<Encoding::kACGT, Layout::kMSB>>(expected_acgt);
}

#endif // FISK_HAS_NEON
