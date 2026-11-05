#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/extractor/spaced_kmer_extractor.hpp"
#include "fisk/extractor/spaced_kmer_masks.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "extractor/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kChecked = InputCheck::kChecked;

using TestMasks = SpacedKmerMasks<kMsb, kMaxTestMasks>;

// =================================================================================================
//     Forms
// =================================================================================================

// The masks in each form equal those of the functions of mask.hpp, piece by piece, for fewer masks
// than the set takes, of a span.
static void check_forms(std::size_t span)
{
    auto const raw = make_spaced_kmer_masks<kMsb>(random_masks(kMaxTestMasks - 1, span, 7));
    TestMasks const masks(raw);
    ASSERT_EQ(masks.size(), raw.size());

    auto const check_common = [&](auto const& forms) {
        ASSERT_EQ(forms.size(), raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) {
            EXPECT_EQ(forms[i].span_k, raw[i].span_k);
            EXPECT_EQ(forms[i].validity_pieces, raw[i].validity_pieces);
            EXPECT_EQ(forms[i].mask_shifts, raw[i].mask_shifts);
        }
    };

    auto const plain = masks.get<BitExtractMask>();
    check_common(plain);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        for (std::size_t j = 0; j < raw[i].max_pieces; ++j) {
            EXPECT_EQ(plain[i].mask_pieces[j].mask, raw[i].mask_pieces[j].mask);
        }
    }

    auto const butterfly = masks.get<BitExtractButterflyMask>();
    auto const expected_butterfly = make_spaced_kmer_butterfly_masks(raw);
    check_common(butterfly);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        for (std::size_t j = 0; j < raw[i].max_pieces; ++j) {
            auto const& piece = butterfly[i].mask_pieces[j];
            auto const& expected = expected_butterfly[i].mask_pieces[j];
            EXPECT_EQ(piece.mask, expected.mask);
            EXPECT_EQ(piece.sieves, expected.sieves);
        }
    }

    #if defined(FISK_HAS_BMI2)
    auto const pext = masks.get<BitExtractKernelPext>();
    check_common(pext);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        for (std::size_t j = 0; j < raw[i].max_pieces; ++j) {
            EXPECT_EQ(pext[i].mask_pieces[j].mask.mask, raw[i].mask_pieces[j].mask);
        }
    }
    #endif

    // The butterfly kernels of each SIMD arch of the candidates.
    [&]<typename... Simd>(TypeList<Simd...>) {
        (
            [&] {
                auto const kernels = masks.get<BitExtractKernelButterfly<Simd>>();
                check_common(kernels);
                for (std::size_t i = 0; i < raw.size(); ++i) {
                    for (std::size_t j = 0; j < raw[i].max_pieces; ++j) {
                        auto const& piece = kernels[i].mask_pieces[j].mask;
                        auto const& expected = expected_butterfly[i].mask_pieces[j];
                        EXPECT_EQ(piece.mask, expected.mask);
                        EXPECT_EQ(piece.sieves, expected.sieves);
                    }
                }
            }(),
            ...
        );
    }(CandidateSimdArchs{});
}

// One piece, and three, the most of the default MaxSpan.
TEST(SpacedKmerMasks, Forms)
{
    check_forms(13);
    check_forms(64);
}

// =================================================================================================
//     Extractor
// =================================================================================================

// An extractor of a prepared set extracts the same as one of the strings of the masks.
TEST(SpacedKmerMasks, Extractor)
{
    using Extractor = SpacedKmerExtractor<
        InputFormat::kAscii, kChecked, kAcgt, kMsb, kMaxTestMasks
    >;
    auto const strings = random_masks(kMaxTestMasks, 11, 3);
    Extractor const from_masks(TestMasks{strings});
    Extractor const from_strings(strings);
    for (auto const& seq : extractor_test_sequences<kChecked>()) {
        SpacedKmerRecorder expected;
        from_strings.for_each(seq, expected);
        SpacedKmerRecorder result;
        from_masks.for_each(seq, result);
        check_spaced_events(result.events, expected.events);
    }
}

// =================================================================================================
//     Errors
// =================================================================================================

TEST(SpacedKmerMasks, Errors)
{
    using Masks = std::vector<std::string>;
    EXPECT_THROW(TestMasks(Masks{}), std::invalid_argument);
    EXPECT_THROW(TestMasks(Masks{"1101", "11011"}), std::invalid_argument);
    EXPECT_THROW(TestMasks(Masks{"0110"}), std::invalid_argument);
    EXPECT_THROW(TestMasks(random_masks(kMaxTestMasks + 1, 11, 3)), std::invalid_argument);

    // Too many masks: the message says that the maximum is a compile-time one, and how to raise it.
    std::string message;
    try {
        TestMasks masks(random_masks(kMaxTestMasks + 1, 11, 3));
    } catch (std::invalid_argument const& error) {
        message = error.what();
    }
    EXPECT_TRUE(message.find("compile-time maximum is 3") != std::string::npos);
    EXPECT_TRUE(message.find("MaxMasks") != std::string::npos);
}

// Masks above 32: up to MaxSpan, with at most 32 kept positions, and all of one span.
TEST(SpacedKmerMasks, WideErrors)
{
    using Masks = std::vector<std::string>;
    auto const mask = [](std::size_t span) {
        return std::string("1").append(span - 2, '0').append("1");
    };
    EXPECT_EQ(TestMasks(Masks{mask(kDefaultMaxSpan)}).size(), 1u);
    EXPECT_THROW(TestMasks(Masks{mask(kDefaultMaxSpan + 1)}), std::invalid_argument);
    EXPECT_THROW(TestMasks(Masks{std::string(40, '1')}), std::invalid_argument);
    EXPECT_THROW(TestMasks(Masks{"11", mask(45)}), std::invalid_argument);
    EXPECT_THROW(TestMasks(Masks{mask(45), mask(46)}), std::invalid_argument);

    // A span above MaxSpan: the message says that the maximum is a compile-time one, and how to
    // raise it.
    std::string message;
    try {
        TestMasks masks(Masks{mask(kDefaultMaxSpan + 1)});
    } catch (std::invalid_argument const& error) {
        message = error.what();
    }
    EXPECT_TRUE(message.find("span " + std::to_string(kDefaultMaxSpan + 1)) != std::string::npos);
    EXPECT_TRUE(message.find("MaxSpan") != std::string::npos);
    EXPECT_TRUE(message.find("FISK_DEFAULT_MAX_SPAN") != std::string::npos);

    // Other maximum spans than the default.
    using NarrowMasks = SpacedKmerMasks<kMsb, 1, 32>;
    EXPECT_EQ(NarrowMasks(Masks{mask(32)}).size(), 1u);
    EXPECT_THROW(NarrowMasks(Masks{mask(33)}), std::invalid_argument);
    using WideMasks = SpacedKmerMasks<kMsb, 1, 128>;
    WideMasks const wide(Masks{mask(128)});
    EXPECT_EQ(wide.get<BitExtractMask>()[0].span_k, 128u);
    EXPECT_EQ(WideMasks::mask_type::max_pieces, 5u);
    EXPECT_THROW(WideMasks(Masks{mask(129)}), std::invalid_argument);
}
