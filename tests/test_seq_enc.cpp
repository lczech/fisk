#include "fisk/core/seq_enc.hpp"
#include "testing.hpp"

// Ground truth for the ACGT and ACTG two-bit encodings, independent of any of the
// implementations under test below. Every char_to_nt_*_acgt()/char_to_nt_*_actg() variant is
// checked against these over all 256 byte values.

static int expected_acgt(unsigned char c)
{
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return 4;
    }
}

static int expected_actg(unsigned char c)
{
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'T': case 't': return 2;
        case 'G': case 'g': return 3;
        default:            return 4;
    }
}

// -----------------------------------------------------------------------------
//     ACGT
// -----------------------------------------------------------------------------

TEST(SeqEnc, CharToNtIfsAcgt)
{
    for (int i = 0; i < 256; ++i) {
        auto const c = static_cast<unsigned char>(i);
        EXPECT_EQ(static_cast<int>(char_to_nt_ifs_acgt(static_cast<char>(c))), expected_acgt(c));
    }
}

TEST(SeqEnc, CharToNtSwitchAcgt)
{
    for (int i = 0; i < 256; ++i) {
        auto const c = static_cast<unsigned char>(i);
        EXPECT_EQ(static_cast<int>(char_to_nt_switch_acgt(static_cast<char>(c))), expected_acgt(c));
    }
}

TEST(SeqEnc, CharToNtAsciiAcgt)
{
    for (int i = 0; i < 256; ++i) {
        auto const c = static_cast<unsigned char>(i);
        EXPECT_EQ(static_cast<int>(char_to_nt_ascii_acgt(static_cast<char>(c))), expected_acgt(c));
    }
}

TEST(SeqEnc, CharToNtTableAcgt)
{
    for (int i = 0; i < 256; ++i) {
        auto const c = static_cast<unsigned char>(i);
        EXPECT_EQ(static_cast<int>(char_to_nt_table_acgt(static_cast<char>(c))), expected_acgt(c));
    }
}

TEST(SeqEnc, NucleotideEncoderAcgtEncode)
{
    for (int i = 0; i < 256; ++i) {
        auto const c = static_cast<unsigned char>(i);
        EXPECT_EQ(
            static_cast<int>(NucleotideEncoderAcgt::encode(static_cast<char>(c))), expected_acgt(c)
        );
    }
}

TEST(SeqEnc, CharToNtAsciiUncheckedAcgt)
{
    // Unspecified for non-ACGT input, so only check it agrees with the checked variant on the
    // chars it actually supports.
    char const valid[] = {'A', 'C', 'G', 'T', 'a', 'c', 'g', 't'};
    for (char c : valid) {
        EXPECT_EQ(
            static_cast<int>(char_to_nt_ascii_unchecked_acgt(c)),
            static_cast<int>(char_to_nt_ascii_acgt(c))
        );
    }
}

// -----------------------------------------------------------------------------
//     ACTG
// -----------------------------------------------------------------------------

TEST(SeqEnc, CharToNtAsciiActg)
{
    for (int i = 0; i < 256; ++i) {
        auto const c = static_cast<unsigned char>(i);
        EXPECT_EQ(static_cast<int>(char_to_nt_ascii_actg(static_cast<char>(c))), expected_actg(c));
    }
}

TEST(SeqEnc, CharToNtTableActg)
{
    for (int i = 0; i < 256; ++i) {
        auto const c = static_cast<unsigned char>(i);
        EXPECT_EQ(static_cast<int>(char_to_nt_table_actg(static_cast<char>(c))), expected_actg(c));
    }
}

TEST(SeqEnc, NucleotideEncoderActgEncode)
{
    for (int i = 0; i < 256; ++i) {
        auto const c = static_cast<unsigned char>(i);
        EXPECT_EQ(
            static_cast<int>(NucleotideEncoderActg::encode(static_cast<char>(c))), expected_actg(c)
        );
    }
}

TEST(SeqEnc, CharToNtAsciiUncheckedActg)
{
    char const valid[] = {'A', 'C', 'G', 'T', 'a', 'c', 'g', 't'};
    for (char c : valid) {
        EXPECT_EQ(
            static_cast<int>(char_to_nt_ascii_unchecked_actg(c)),
            static_cast<int>(char_to_nt_ascii_actg(c))
        );
    }
}
