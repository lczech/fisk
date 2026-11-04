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

// The masks in each form equal those of the functions of mask.hpp, for fewer masks than the set
// takes.
TEST(SpacedKmerMasks, Forms)
{
    auto const raw = make_spaced_kmer_masks<kMsb>(random_masks(kMaxTestMasks - 1, 13, 7));
    TestMasks const masks(raw);
    ASSERT_EQ(masks.size(), raw.size());

    auto const check_common = [&](auto const& forms) {
        ASSERT_EQ(forms.size(), raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) {
            EXPECT_EQ(forms[i].span_k, raw[i].span_k);
            EXPECT_EQ(forms[i].validity_mask, raw[i].validity_mask);
        }
    };

    auto const plain = masks.get<BitExtractMask>();
    check_common(plain);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        EXPECT_EQ(plain[i].mask, raw[i].mask);
    }

    auto const butterfly = masks.get<BitExtractButterflyMask>();
    auto const expected_butterfly = make_spaced_kmer_butterfly_masks(raw);
    check_common(butterfly);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        EXPECT_EQ(butterfly[i].mask, expected_butterfly[i].mask);
        EXPECT_EQ(butterfly[i].sieves, expected_butterfly[i].sieves);
    }

    #if defined(FISK_HAS_BMI2)
    auto const pext = masks.get<BitExtractKernelPext>();
    check_common(pext);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        EXPECT_EQ(pext[i].mask.mask, raw[i].mask);
    }
    #endif

    // The butterfly kernels of each SIMD arch of the candidates.
    [&]<typename... Simd>(TypeList<Simd...>) {
        (
            [&] {
                auto const kernels = masks.get<BitExtractKernelButterfly<Simd>>();
                check_common(kernels);
                for (std::size_t i = 0; i < raw.size(); ++i) {
                    EXPECT_EQ(kernels[i].mask.mask, expected_butterfly[i].mask);
                    EXPECT_EQ(kernels[i].mask.sieves, expected_butterfly[i].sieves);
                }
            }(),
            ...
        );
    }(CandidateSimdArchs{});
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
