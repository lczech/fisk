#include "fisk/core/char_encoder.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// Check one encoder against the oracle for every byte value.
template <template <Encoding> class Encoder, Encoding E>
static void check_encoder_under()
{
    for (int i = 0; i < 256; ++i) {
        auto const c = static_cast<char>(i);
        EXPECT_EQ(static_cast<int>(Encoder<E>{}(c)), static_cast<int>(oracle_code<E>(c)));
    }
}

// Check one encoder template under both encodings.
template <template <Encoding> class Encoder>
static void check_encoder()
{
    check_encoder_under<Encoder, Encoding::kACGT>();
    check_encoder_under<Encoder, Encoding::kACTG>();
}

// =================================================================================================
//     Compile-Time Guarantees
// =================================================================================================

// Every encoder states the Encoding it was instantiated with, which is what lets the k-mer
// extraction loops tag what they build from its codes.
static_assert(CharEncoderIfs<Encoding::kACTG>::encoding == Encoding::kACTG);
static_assert(CharEncoderSwitch<Encoding::kACGT>::encoding == Encoding::kACGT);
static_assert(CharEncoderAscii<Encoding::kACTG>::encoding == Encoding::kACTG);
static_assert(CharEncoderAscii<Encoding::kACGT, InputValidity::kAssumeValid>::encoding == Encoding::kACGT);
static_assert(CharEncoderTable<Encoding::kACTG>::encoding == Encoding::kACTG);

// Encoders are usable at compile time, and the two encodings genuinely differ where they should:
// agreeing on A and C, and swapping the codes of G and T.
static_assert(
    CharEncoderTable<Encoding::kACGT>{}('G') == 2 && CharEncoderTable<Encoding::kACTG>{}('G') == 3
);
static_assert(
    CharEncoderTable<Encoding::kACGT>{}('T') == 3 && CharEncoderTable<Encoding::kACTG>{}('T') == 2
);
static_assert(
    CharEncoderAscii<Encoding::kACGT>{}('c') == CharEncoderAscii<Encoding::kACTG>{}('c')
);
static_assert(
    CharEncoderSwitch<Encoding::kACTG>{}('N') == kInvalidNucleotide
);

// =================================================================================================
//     Encoders
// =================================================================================================

// Each technique must produce exactly the oracle's code for every byte value, under both
// encodings, including kInvalidNucleotide for everything that is not ACGT in either case.

TEST(CharEncoder, Ifs)
{
    check_encoder<CharEncoderIfs>();
}

TEST(CharEncoder, Switch)
{
    check_encoder<CharEncoderSwitch>();
}

TEST(CharEncoder, Table)
{
    check_encoder<CharEncoderTable>();
}

// CharEncoderAscii takes a second, defaulted InputValidity parameter (for kAssumeValid, see
// core/char_encoder.hpp), so it does not have the single-parameter shape check_encoder()'s
// template-template-parameter expects. This alias fixes V at its default (kValidate) to restore
// that shape exactly, rather than relying on compilers to accept a defaulted trailing parameter.
template <Encoding E>
using CharEncoderAsciiValidating = CharEncoderAscii<E, InputValidity::kValidate>;

TEST(CharEncoder, Ascii)
{
    check_encoder<CharEncoderAsciiValidating>();
}

// The kAssumeValid variant must agree with the (default, kValidate) checked variant on the
// characters it actually supports. For any other byte its code is unspecified, but it must still
// be a valid-looking 0-3 code, as documented in core/char_encoder.hpp.
TEST(CharEncoder, AsciiAssumeValid)
{
    for (int i = 0; i < 256; ++i) {
        auto const c = static_cast<char>(i);
        EXPECT_TRUE((CharEncoderAscii<Encoding::kACGT, InputValidity::kAssumeValid>{}(c)) < 4);
        EXPECT_TRUE((CharEncoderAscii<Encoding::kACTG, InputValidity::kAssumeValid>{}(c)) < 4);
    }

    char const valid[] = {'A', 'C', 'G', 'T', 'a', 'c', 'g', 't'};
    for (char c : valid) {
        EXPECT_EQ(
            static_cast<int>((CharEncoderAscii<Encoding::kACGT, InputValidity::kAssumeValid>{}(c))),
            static_cast<int>(CharEncoderAscii<Encoding::kACGT>{}(c))
        );
        EXPECT_EQ(
            static_cast<int>((CharEncoderAscii<Encoding::kACTG, InputValidity::kAssumeValid>{}(c))),
            static_cast<int>(CharEncoderAscii<Encoding::kACTG>{}(c))
        );
    }
}

// All techniques must agree with each other, not just with the oracle: they exist to be
// benchmarked against one another, which is only meaningful if they compute the same thing.
template <Encoding E>
static void check_techniques_agree_under()
{
    for (int i = 0; i < 256; ++i) {
        auto const c = static_cast<char>(i);
        auto const table = CharEncoderTable<E>{}(c);
        EXPECT_EQ(CharEncoderIfs<E>{}(c), table);
        EXPECT_EQ(CharEncoderSwitch<E>{}(c), table);
        EXPECT_EQ(CharEncoderAscii<E>{}(c), table);
    }
}

TEST(CharEncoder, TechniquesAgree)
{
    check_techniques_agree_under<Encoding::kACGT>();
    check_techniques_agree_under<Encoding::kACTG>();
}
