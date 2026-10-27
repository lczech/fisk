#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/util/random.hpp"
#include "fisk/simd/simd.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "sequence_pack/common.hpp"
#include "corpus.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

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

// Checks pack_sequence_unchecked() -- both overloads, and reuse of one PackedSequence across calls
// -- for a given word encoder, on valid and invalid input.
template <typename Encoder>
static void check_word_encoder()
{
    Encoder encoder{};
    PackedSequence<Encoder::encoding, Encoder::layout> out;

    for (auto const* seqs : {&test_sequences(), &invalid_sequences()}) {
        for (auto const& seq : *seqs) {
            pack_sequence_unchecked(seq, encoder, out);
            check_packed(seq, out);

            auto const out2 = pack_sequence_unchecked(seq, encoder);
            check_packed(seq, out2);
        }
    }

    // Reuse must not leak stale bytes from a previous, larger pack.
    pack_sequence_unchecked(std::string(100, 'T'), encoder, out);
    pack_sequence_unchecked(std::string("AC"), encoder, out);
    check_packed(std::string("AC"), out);
}

// -------------------------------------------------------------------------------------------------
//     Validity
// -------------------------------------------------------------------------------------------------

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
        return Simd::load_unaligned(p);
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

// =================================================================================================
//     Compile-Time Conventions
// =================================================================================================

namespace {

template <typename Encoder, Encoding E, Layout L>
constexpr bool tagged_as = Encoder::encoding == E && Encoder::layout == L;

using UntaggedWordEncoder = decltype([](std::uint64_t word) -> std::uint64_t { return word; });

} // namespace

// Every word encoder satisfies the concept that pack_sequence_unchecked() requires; the
// per-character encoders do not, and neither does a bare callable that states no conventions.
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

#endif

// The number of lanes is what selects the vector loop of pack_sequence_unchecked().
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

#endif

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

// =================================================================================================
//     Scalar Word Encoders
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Butterfly
// -------------------------------------------------------------------------------------------------

TEST(SequencePackWordEncoder, ScalarButterflyActgLsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB>>();
}

TEST(SequencePackWordEncoder, ScalarButterflyActgMsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB>>();
}

TEST(SequencePackWordEncoder, ScalarButterflyAcgtLsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>>();
}

TEST(SequencePackWordEncoder, ScalarButterflyAcgtMsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>>();
}

// -------------------------------------------------------------------------------------------------
//     Fold
// -------------------------------------------------------------------------------------------------

TEST(SequencePackWordEncoder, ScalarFoldActgLsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACTG, Layout::kLSB>>();
}

TEST(SequencePackWordEncoder, ScalarFoldActgMsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACTG, Layout::kMSB>>();
}

TEST(SequencePackWordEncoder, ScalarFoldAcgtLsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACGT, Layout::kLSB>>();
}

TEST(SequencePackWordEncoder, ScalarFoldAcgtMsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACGT, Layout::kMSB>>();
}

// -------------------------------------------------------------------------------------------------
//     Direct
// -------------------------------------------------------------------------------------------------

TEST(SequencePackWordEncoder, ScalarDirectActgLsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACTG, Layout::kLSB>>();
}

TEST(SequencePackWordEncoder, ScalarDirectActgMsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACTG, Layout::kMSB>>();
}

TEST(SequencePackWordEncoder, ScalarDirectAcgtLsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACGT, Layout::kLSB>>();
}

TEST(SequencePackWordEncoder, ScalarDirectAcgtMsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACGT, Layout::kMSB>>();
}

// -------------------------------------------------------------------------------------------------
//     PEXT
// -------------------------------------------------------------------------------------------------

#if defined(FISK_HAS_BMI2)

TEST(SequencePackWordEncoder, ScalarPextActgLsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACTG, Layout::kLSB>>();
}

TEST(SequencePackWordEncoder, ScalarPextActgMsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACTG, Layout::kMSB>>();
}

TEST(SequencePackWordEncoder, ScalarPextAcgtLsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACGT, Layout::kLSB>>();
}

TEST(SequencePackWordEncoder, ScalarPextAcgtMsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACGT, Layout::kMSB>>();
}

#endif

// =================================================================================================
//     SIMD Word Encoders
// =================================================================================================

#if defined(FISK_HAS_SSE2)

TEST(SequencePackWordEncoder, Sse2ActgLsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2ActgMsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2AcgtLsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2AcgtMsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2FoldActgLsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACTG, Layout::kLSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2FoldActgMsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACTG, Layout::kMSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2FoldAcgtLsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACGT, Layout::kLSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2FoldAcgtMsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2DirectActgLsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACTG, Layout::kLSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2DirectActgMsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACTG, Layout::kMSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2DirectAcgtLsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACGT, Layout::kLSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2DirectAcgtMsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdSSE2>>();
}

#if defined(FISK_HAS_SSSE3)

TEST(SequencePackWordEncoder, Sse2DirectDoubleActgLsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACTG, Layout::kLSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2DirectDoubleActgMsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACTG, Layout::kMSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2DirectDoubleAcgtLsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACGT, Layout::kLSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2DirectDoubleAcgtMsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdSSE2>>();
}

#endif

#if defined(FISK_HAS_BMI2)

TEST(SequencePackWordEncoder, Sse2PextActgLsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACTG, Layout::kLSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2PextActgMsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACTG, Layout::kMSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2PextAcgtLsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACGT, Layout::kLSB, SimdSSE2>>();
}

TEST(SequencePackWordEncoder, Sse2PextAcgtMsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACGT, Layout::kMSB, SimdSSE2>>();
}

#endif

#endif

#if defined(FISK_HAS_AVX2)

TEST(SequencePackWordEncoder, Avx2ActgLsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2ActgMsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2AcgtLsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2AcgtMsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2FoldActgLsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACTG, Layout::kLSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2FoldActgMsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACTG, Layout::kMSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2FoldAcgtLsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACGT, Layout::kLSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2FoldAcgtMsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2DirectActgLsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACTG, Layout::kLSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2DirectActgMsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACTG, Layout::kMSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2DirectAcgtLsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACGT, Layout::kLSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2DirectAcgtMsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2DirectDoubleActgLsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACTG, Layout::kLSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2DirectDoubleActgMsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACTG, Layout::kMSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2DirectDoubleAcgtLsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACGT, Layout::kLSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2DirectDoubleAcgtMsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdAVX2>>();
}

#if defined(FISK_HAS_BMI2)

TEST(SequencePackWordEncoder, Avx2PextActgLsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACTG, Layout::kLSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2PextActgMsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACTG, Layout::kMSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2PextAcgtLsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACGT, Layout::kLSB, SimdAVX2>>();
}

TEST(SequencePackWordEncoder, Avx2PextAcgtMsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACGT, Layout::kMSB, SimdAVX2>>();
}

#endif

#endif

#if defined(FISK_HAS_AVX512)

TEST(SequencePackWordEncoder, Avx512ActgLsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512ActgMsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512AcgtLsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512AcgtMsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512FoldActgLsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACTG, Layout::kLSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512FoldActgMsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACTG, Layout::kMSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512FoldAcgtLsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACGT, Layout::kLSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512FoldAcgtMsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512DirectActgLsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACTG, Layout::kLSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512DirectActgMsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACTG, Layout::kMSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512DirectAcgtLsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACGT, Layout::kLSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512DirectAcgtMsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512DirectDoubleActgLsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACTG, Layout::kLSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512DirectDoubleActgMsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACTG, Layout::kMSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512DirectDoubleAcgtLsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACGT, Layout::kLSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512DirectDoubleAcgtMsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdAVX512>>();
}

#if defined(FISK_HAS_BMI2)

TEST(SequencePackWordEncoder, Avx512PextActgLsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACTG, Layout::kLSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512PextActgMsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACTG, Layout::kMSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512PextAcgtLsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACGT, Layout::kLSB, SimdAVX512>>();
}

TEST(SequencePackWordEncoder, Avx512PextAcgtMsb)
{
    check_word_encoder<WordEncoderPext<Encoding::kACGT, Layout::kMSB, SimdAVX512>>();
}

#endif

#endif

#if defined(FISK_HAS_NEON)

TEST(SequencePackWordEncoder, NeonActgLsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACTG, Layout::kLSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonActgMsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACTG, Layout::kMSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonAcgtLsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACGT, Layout::kLSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonAcgtMsb)
{
    check_word_encoder<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonFoldActgLsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACTG, Layout::kLSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonFoldActgMsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACTG, Layout::kMSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonFoldAcgtLsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACGT, Layout::kLSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonFoldAcgtMsb)
{
    check_word_encoder<WordEncoderFold<Encoding::kACGT, Layout::kMSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonDirectActgLsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACTG, Layout::kLSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonDirectActgMsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACTG, Layout::kMSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonDirectAcgtLsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACGT, Layout::kLSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonDirectAcgtMsb)
{
    check_word_encoder<WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonDirectDoubleActgLsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACTG, Layout::kLSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonDirectDoubleActgMsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACTG, Layout::kMSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonDirectDoubleAcgtLsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACGT, Layout::kLSB, SimdNEON>>();
}

TEST(SequencePackWordEncoder, NeonDirectDoubleAcgtMsb)
{
    check_word_encoder<WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdNEON>>();
}

#endif

// =================================================================================================
//     Validity
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Nucleotide Bits
// -------------------------------------------------------------------------------------------------

TEST(SequencePackWordEncoder, NucleotideBitsScalar)
{
    check_valid_nucleotide_bits_arch<SimdScalar>();
}

#if defined(FISK_HAS_SSE2)

TEST(SequencePackWordEncoder, NucleotideBitsSse2)
{
    check_valid_nucleotide_bits_arch<SimdSSE2>();
}

#endif

#if defined(FISK_HAS_AVX2)

TEST(SequencePackWordEncoder, NucleotideBitsAvx2)
{
    check_valid_nucleotide_bits_arch<SimdAVX2>();
}

#endif

#if defined(FISK_HAS_AVX512)

TEST(SequencePackWordEncoder, NucleotideBitsAvx512)
{
    check_valid_nucleotide_bits_arch<SimdAVX512>();
}

#endif

#if defined(FISK_HAS_NEON)

TEST(SequencePackWordEncoder, NucleotideBitsNeon)
{
    check_valid_nucleotide_bits_arch<SimdNEON>();
}

#endif
