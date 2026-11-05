#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extractor/spaced_kmer_extractor.hpp"
#include "fisk/util/cpu_runtime.hpp"
#include "extractor/common.hpp"
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
    auto const seqs = extractor_test_sequences<kChecked>();
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
    auto const seqs = extractor_test_sequences<kChecked>();
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

// The estimate: PEXT where likely fast, else the butterfly in the SIMD loop of AVX2, SSE2, or Neon,
// the first one compiled, else in the scalar loop; `loop` is the name of the function of the loop
// for the input.
static std::string expected_spaced_estimate(std::string const& loop)
{
    std::string expected = loop + "_scalar_butterfly";
    #if defined(FISK_HAS_AVX2)
    if (avx2_enabled()) {
        expected = loop + "_simd_butterfly_avx2";
    }
    #elif defined(FISK_HAS_SSE2)
    if (sse2_enabled()) {
        expected = loop + "_simd_butterfly_sse2";
    }
    #elif defined(FISK_HAS_NEON)
    if (neon_enabled()) {
        expected = loop + "_simd_butterfly_neon";
    }
    #endif
    if (pext_likely_fast()) {
        expected = loop + "_scalar_pext";
    }
    return expected;
}

TEST(SpacedKmerExtractor, Estimate)
{
    // The same for callbacks of single spaced k-mers and for those that also take vectors.
    auto const vector_callback = callback_overload{
        [](std::size_t, std::size_t, SpacedKmerAcgtMsb) {},
        [](std::size_t, std::size_t, auto, auto) {}
    };
    AsciiCheckedSpacedExtractor ascii({"1101101"});
    auto const ascii_estimate = expected_spaced_estimate(
        "for_each_spaced_kmer_ascii_checked_chunked"
    );
    EXPECT_EQ(ascii.algorithm_name(), ascii_estimate);
    EXPECT_EQ(ascii.algorithm_name(vector_callback), ascii_estimate);

    // Packed input, with a single mask, which only builds the candidates, not their loops.
    SpacedKmerExtractor<InputFormat::kPacked, InputCheck::kUnchecked, kAcgt, kMsb, 1> packed(
        {"1101101"}
    );
    auto const packed_estimate = expected_spaced_estimate(
        "for_each_spaced_kmer_packed_unchecked_bytewise"
    );
    EXPECT_EQ(packed.algorithm_name(), packed_estimate);
    EXPECT_EQ(packed.algorithm_name(vector_callback), packed_estimate);
}

TEST(SpacedKmerExtractor, Tune)
{
    std::vector<std::string> const masks = {"110110111", "101101011"};
    auto const seqs = extractor_test_sequences<kChecked>();
    auto const contains = [](std::vector<std::string> const& names, std::string const& name) {
        return std::find(names.begin(), names.end(), name) != names.end();
    };

    TuneOptions options;
    options.total_length = 20000;
    options.sequence_length = 150;

    AsciiCheckedSpacedExtractor extractor(masks);
    extractor.tune(options);
    EXPECT_TRUE(contains(extractor.algorithm_names(), extractor.algorithm_name()));
    extractor.tune(seqs, options);
    EXPECT_TRUE(contains(extractor.algorithm_names(), extractor.algorithm_name()));

    SpacedKmerRecorder recorder;
    extractor.tune(seqs.back(), recorder, options);
    EXPECT_FALSE(recorder.events.empty());

    // The tuned extractor, and a copy, extract all spaced k-mers.
    auto const copy = extractor;
    EXPECT_EQ(copy.algorithm_name(), extractor.algorithm_name());
    for (auto const& seq : seqs) {
        auto const expected = by_mask(oracle_spaced_events<kAcgt, kMsb>(seq, masks));
        SpacedKmerRecorder result;
        copy.for_each(seq, result);
        check_spaced_events(by_mask(result.events), expected);
    }
}

// =================================================================================================
//     Errors
// =================================================================================================

TEST(SpacedKmerExtractor, TuneEdgeCases)
{
    std::vector<std::string> const masks = {"110110111", "101101011"};
    AsciiCheckedSpacedExtractor extractor(masks);
    auto const names = extractor.algorithm_names();
    auto const valid = [&] {
        return std::find(names.begin(), names.end(), extractor.algorithm_name()) != names.end();
    };

    // No random data, and samples without spaced k-mers: each still picks an algorithm.
    TuneOptions options;
    options.total_length = 0;
    extractor.tune(options);
    EXPECT_TRUE(valid());
    options = TuneOptions{};
    extractor.tune(std::vector<std::string>{}, options);
    EXPECT_TRUE(valid());
    extractor.tune(std::vector<std::string>{"", "ACGT", "ACGTACGT"}, options);
    EXPECT_TRUE(valid());
}

// =================================================================================================
//     Masks
// =================================================================================================

TEST(SpacedKmerExtractor, Masks)
{
    auto const seqs = extractor_test_sequences<kChecked>();

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
