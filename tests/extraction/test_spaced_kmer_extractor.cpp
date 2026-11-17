#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extraction/spaced_kmer_extractor.hpp"
#include "fisk/util/cpu_runtime.hpp"
#include "extraction/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kChecked = InputCheck::kChecked;

// =================================================================================================
//     Largest Mask Sets
// =================================================================================================

// The most masks of the default MaxMasks, with every algorithm, with one callback, which keeps the
// compile time of all mask counts down to one callback.
TEST(SpacedKmerExtractor, DefaultMaxMasks)
{
    using Extractor = SpacedKmerExtractor<InputFormat::kAscii, kChecked, kAcgt, kMsb>;
    auto const masks = random_masks(kDefaultMaxMasks, 32, 2);
    auto const seqs = extraction_test_sequences<kChecked>();
    Extractor extractor(masks);
    for (auto const& name : extractor.algorithm_names()) {
        extractor.set_algorithm(name);
        for (auto const& seq : seqs) {
            SpacedKmerRecorder result;
            extractor.for_each(seq, result);
            check_spaced_events(
                by_mask(result.events), by_mask(oracle_spaced_events<kAcgt, kMsb>(seq, masks))
            );
        }
    }
}

// =================================================================================================
//     Maximum Span
// =================================================================================================

// Other maximum spans than the default: 32, which takes no wider masks, and 128, with every
// algorithm, for masks of four and five pieces, the most of that span.
TEST(SpacedKmerExtractor, MaxSpan)
{
    using Masks = std::vector<std::string>;
    using NarrowExtractor = SpacedKmerExtractor<InputFormat::kAscii, kChecked, kAcgt, kMsb, 1, 32>;
    EXPECT_EQ(NarrowExtractor(Masks{std::string(32, '1')}).mask_forms().size(), 1u);
    EXPECT_THROW(NarrowExtractor(random_masks(1, 33, 5)), std::invalid_argument);

    using WideExtractor = SpacedKmerExtractor<InputFormat::kAscii, kChecked, kAcgt, kMsb, 1, 128>;
    auto const seqs = extraction_test_sequences<kChecked>();
    for (std::size_t span : {85u, 128u}) {
        auto const masks = random_masks(1, span, span);
        WideExtractor extractor(masks);
        for (auto const& name : extractor.algorithm_names()) {
            extractor.set_algorithm(name);
            for (auto const& seq : seqs) {
                SpacedKmerRecorder result;
                extractor.for_each(seq, result);
                check_spaced_events(result.events, oracle_spaced_events<kAcgt, kMsb>(seq, masks));
            }
        }
    }
    EXPECT_THROW(WideExtractor(random_masks(1, 129, 5)), std::invalid_argument);
}

// =================================================================================================
//     Estimate and Tuning
// =================================================================================================

using AsciiCheckedSpacedExtractor = SpacedKmerExtractor<
    InputFormat::kAscii, kChecked, kAcgt, kMsb, kMaxTestMasks
>;

// The estimate, see expected_spaced_estimate().
TEST(SpacedKmerExtractor, Estimate)
{
    AsciiCheckedSpacedExtractor ascii({"1101101"});
    ascii.estimate();
    EXPECT_EQ(
        ascii.algorithm_name(),
        expected_spaced_estimate("for_each_spaced_kmer_ascii_checked_chunked")
    );

    // Packed input, with a single mask, which only builds the candidates, not their loops.
    SpacedKmerExtractor<InputFormat::kPacked, InputCheck::kUnchecked, kAcgt, kMsb, 1> packed(
        {"1101101"}
    );
    packed.estimate();
    EXPECT_EQ(
        packed.algorithm_name(),
        expected_spaced_estimate("for_each_spaced_kmer_packed_unchecked_bytewise")
    );
}

// =================================================================================================
//     Masks
// =================================================================================================

TEST(SpacedKmerExtractor, Masks)
{
    auto const seqs = extraction_test_sequences<kChecked>();

    // Made from mask objects, and with as many masks as the extractor takes.
    auto const strings = random_masks(kMaxTestMasks, 13, 5);
    auto const masks = make_spaced_kmer_masks<kMsb>(strings);
    AsciiCheckedSpacedExtractor extractor(masks);
    ASSERT_EQ(extractor.mask_forms().size(), kMaxTestMasks);
    for (std::size_t i = 0; i < kMaxTestMasks; ++i) {
        EXPECT_EQ(
            extractor.mask_forms().get<BitExtractMask>()[i].mask_pieces[0].mask,
            masks[i].mask_pieces[0].mask
        );
    }
    for (auto const& seq : seqs) {
        SpacedKmerRecorder result;
        extractor.for_each(seq, result);
        check_spaced_events(
            by_mask(result.events), by_mask(oracle_spaced_events<kAcgt, kMsb>(seq, strings))
        );
    }
}

// =================================================================================================
//     Errors
// =================================================================================================

TEST(SpacedKmerExtractor, Errors)
{
    using Masks = std::vector<std::string>;
    EXPECT_THROW(AsciiCheckedSpacedExtractor(Masks{}), std::invalid_argument);
    EXPECT_THROW(AsciiCheckedSpacedExtractor(Masks{"1101", "11011"}), std::invalid_argument);
    EXPECT_THROW(AsciiCheckedSpacedExtractor(Masks{"0110"}), std::invalid_argument);
    EXPECT_THROW(
        AsciiCheckedSpacedExtractor(random_masks(kMaxTestMasks + 1, 11, 3)),
        std::invalid_argument
    );

    // Too many masks: the message says that the maximum is a compile-time one, and how to raise it.
    std::string message;
    try {
        AsciiCheckedSpacedExtractor extractor(random_masks(kMaxTestMasks + 1, 11, 3));
    } catch (std::invalid_argument const& error) {
        message = error.what();
    }
    EXPECT_TRUE(message.find("compile-time maximum is 3") != std::string::npos);
    EXPECT_TRUE(message.find("MaxMasks") != std::string::npos);

    // A span above the default MaxSpan.
    EXPECT_THROW(
        AsciiCheckedSpacedExtractor(random_masks(1, kDefaultMaxSpan + 1, 3)),
        std::invalid_argument
    );

    AsciiCheckedSpacedExtractor extractor({"1101101"});
    EXPECT_THROW(extractor.set_algorithm("unknown"), std::invalid_argument);
}
