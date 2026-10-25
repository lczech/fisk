#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "sequence_pack/common.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// -------------------------------------------------------------------------------------------------
//     Validity
// -------------------------------------------------------------------------------------------------

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

// Checks pack_sequence_checked() -- both overloads, the overload of pack_sequence_unchecked() that
// forwards to it, and reuse across calls -- for a given word encoder. Its codes have to be the
// same as those of pack_sequence_unchecked(), including the unspecified ones of invalid characters.
template <typename Encoder>
static void check_pack_sequence_checked()
{
    Encoder encoder{};
    PackedSequence<Encoder::encoding, Encoder::layout> plain;
    PackedSequenceWithValidity<Encoder::encoding, Encoder::layout> out;

    for (auto const& seq : validity_sequences()) {
        pack_sequence_unchecked(seq, encoder, plain);

        pack_sequence_checked(seq, encoder, out);
        check_packed(seq, out);
        check_validity(seq, out);
        EXPECT_EQ(out.length(), plain.length());
        auto const out_codes = out.codes();
        auto const plain_codes = plain.codes();
        EXPECT_TRUE(std::equal(
            out_codes.begin(), out_codes.end(), plain_codes.begin(), plain_codes.end()
        ));

        auto const out2 = pack_sequence_checked(seq, encoder);
        check_packed(seq, out2);
        check_validity(seq, out2);
    }

    // Reuse must not leak stale bits from a previous, larger pack.
    pack_sequence_checked(std::string(100, 'T'), encoder, out);
    pack_sequence_checked(std::string("ANC"), encoder, out);
    check_packed(std::string("ANC"), out);
    check_validity(std::string("ANC"), out);
}

// Runs check_pack_sequence_checked() for all four combinations of Encoding and Layout.
template <template <Encoding, Layout, typename> class Encoder, typename Simd>
static void check_pack_sequence_checked_all()
{
    check_pack_sequence_checked<Encoder<Encoding::kACGT, Layout::kLSB, Simd>>();
    check_pack_sequence_checked<Encoder<Encoding::kACGT, Layout::kMSB, Simd>>();
    check_pack_sequence_checked<Encoder<Encoding::kACTG, Layout::kLSB, Simd>>();
    check_pack_sequence_checked<Encoder<Encoding::kACTG, Layout::kMSB, Simd>>();
}

// =================================================================================================
//     Compile-Time Conventions
// =================================================================================================

namespace {

// Encoding and Layout are part of PackedSequence's type, so mixing them up between producer and
// consumer is rejected by the compiler. These checks pin that guarantee down.
using AcgtMsb = PackedSequence<Encoding::kACGT, Layout::kMSB>;
using AcgtLsb = PackedSequence<Encoding::kACGT, Layout::kLSB>;
using ActgMsb = PackedSequence<Encoding::kACTG, Layout::kMSB>;

template <typename Encoder, typename Out>
concept UncheckedPackableInto = requires(std::string_view seq, Encoder encoder, Out& out) {
    pack_sequence_unchecked(seq, encoder, out);
};

template <typename Encoder, typename Out>
concept CheckedPackableInto = requires(std::string_view seq, Encoder encoder, Out& out) {
    pack_sequence_checked(seq, encoder, out);
};

using UntaggedWordEncoder = decltype([](std::uint64_t word) -> std::uint64_t { return word; });

} // namespace

static_assert(!UncheckedPackableInto<UntaggedWordEncoder, AcgtMsb>);

// The packed sequence type follows the encoder's tags.
static_assert(std::is_same_v<
    decltype(pack_sequence_unchecked("", WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{})),
    AcgtMsb
>);

// Packing into an existing sequence of a different encoding or layout does not compile.
static_assert( UncheckedPackableInto<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>, AcgtMsb>);
static_assert(!UncheckedPackableInto<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>, ActgMsb>);
static_assert(!UncheckedPackableInto<WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>, AcgtLsb>);

// Only pack_sequence_checked() packs a sequence with validity, so that its validity is always
// filled (see check_pack_sequence_checked()), and it packs nothing else.
using AcgtMsbWithValidity = PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>;

static_assert(std::is_same_v<
    decltype(pack_sequence_checked("", WordEncoderDirect<Encoding::kACGT, Layout::kMSB>{})),
    AcgtMsbWithValidity
>);
static_assert(
    CheckedPackableInto<WordEncoderDirect<Encoding::kACGT, Layout::kMSB>, AcgtMsbWithValidity>
);
static_assert(
    !CheckedPackableInto<WordEncoderDirect<Encoding::kACGT, Layout::kLSB>, AcgtMsbWithValidity>
);
static_assert(
    !UncheckedPackableInto<WordEncoderDirect<Encoding::kACGT, Layout::kMSB>, AcgtMsbWithValidity>
);
static_assert(!CheckedPackableInto<WordEncoderDirect<Encoding::kACGT, Layout::kMSB>, AcgtMsb>);

// =================================================================================================
//     Pack Sequence Checked
// =================================================================================================

TEST(SequencePackPackSequence, CheckedScalar)
{
    check_pack_sequence_checked_all<WordEncoderButterfly, SimdScalar>();
    check_pack_sequence_checked_all<WordEncoderFold, SimdScalar>();
    check_pack_sequence_checked_all<WordEncoderDirect, SimdScalar>();
    #if defined(FISK_HAS_BMI2)
    check_pack_sequence_checked_all<WordEncoderPext, SimdScalar>();
    #endif
}

#if defined(FISK_HAS_SSE2)

TEST(SequencePackPackSequence, CheckedSse2)
{
    check_pack_sequence_checked_all<WordEncoderButterfly, SimdSSE2>();
    check_pack_sequence_checked_all<WordEncoderFold, SimdSSE2>();
    check_pack_sequence_checked_all<WordEncoderDirect, SimdSSE2>();
    #if defined(FISK_HAS_SSSE3)
    check_pack_sequence_checked_all<WordEncoderDirectDouble, SimdSSE2>();
    #endif
    #if defined(FISK_HAS_BMI2)
    check_pack_sequence_checked_all<WordEncoderPext, SimdSSE2>();
    #endif
}

#endif

#if defined(FISK_HAS_AVX2)

TEST(SequencePackPackSequence, CheckedAvx2)
{
    check_pack_sequence_checked_all<WordEncoderButterfly, SimdAVX2>();
    check_pack_sequence_checked_all<WordEncoderFold, SimdAVX2>();
    check_pack_sequence_checked_all<WordEncoderDirect, SimdAVX2>();
    check_pack_sequence_checked_all<WordEncoderDirectDouble, SimdAVX2>();
    #if defined(FISK_HAS_BMI2)
    check_pack_sequence_checked_all<WordEncoderPext, SimdAVX2>();
    #endif
}

#endif

#if defined(FISK_HAS_AVX512)

TEST(SequencePackPackSequence, CheckedAvx512)
{
    check_pack_sequence_checked_all<WordEncoderButterfly, SimdAVX512>();
    check_pack_sequence_checked_all<WordEncoderFold, SimdAVX512>();
    check_pack_sequence_checked_all<WordEncoderDirect, SimdAVX512>();
    check_pack_sequence_checked_all<WordEncoderDirectDouble, SimdAVX512>();
    #if defined(FISK_HAS_BMI2)
    check_pack_sequence_checked_all<WordEncoderPext, SimdAVX512>();
    #endif
}

#endif

#if defined(FISK_HAS_NEON)

TEST(SequencePackPackSequence, CheckedNeon)
{
    check_pack_sequence_checked_all<WordEncoderButterfly, SimdNEON>();
    check_pack_sequence_checked_all<WordEncoderFold, SimdNEON>();
    check_pack_sequence_checked_all<WordEncoderDirect, SimdNEON>();
    check_pack_sequence_checked_all<WordEncoderDirectDouble, SimdNEON>();
}

#endif
