#include <algorithm>
#include <chrono>
#include <cstddef>
#include <initializer_list>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extraction/kmer_extractor.hpp"
#include "fisk/util/cpu_runtime.hpp"
#include "extraction/kmer_common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Algorithms
// =================================================================================================

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kActg = Encoding::kACTG;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kLsb = Layout::kLSB;
constexpr auto kChecked = InputCheck::kChecked;
constexpr auto kUnchecked = InputCheck::kUnchecked;

TEST(KmerExtractor, AsciiChecked)
{
    check_kmer_extractor_algorithms<InputFormat::kAscii, kChecked, kAcgt, kMsb>();
    check_kmer_extractor_algorithms<InputFormat::kAscii, kChecked, kActg, kLsb>();
}

TEST(KmerExtractor, AsciiUnchecked)
{
    check_kmer_extractor_algorithms<InputFormat::kAscii, kUnchecked, kAcgt, kMsb>();
}

TEST(KmerExtractor, PackedChecked)
{
    check_kmer_extractor_algorithms<InputFormat::kPacked, kChecked, kAcgt, kMsb>();
}

TEST(KmerExtractor, PackedUnchecked)
{
    check_kmer_extractor_algorithms<InputFormat::kPacked, kUnchecked, kAcgt, kLsb>();
}

// =================================================================================================
//     Estimate and Tuning
// =================================================================================================

using AsciiCheckedExtractor = KmerExtractor<
    InputFormat::kAscii, InputCheck::kChecked, Encoding::kACGT, Layout::kMSB
>;

TEST(KmerExtractor, Estimate)
{
    // The estimate of for_each_kmer(), see expected_kmer_estimate().
    AsciiCheckedExtractor extractor(21);
    extractor.estimate();
    std::string const scalar = "for_each_kmer_ascii_checked_chunked_scalar";
    std::string const expected = expected_kmer_estimate("for_each_kmer_ascii_checked_chunked");
    EXPECT_EQ(extractor.algorithm_name(), expected);

    KmerExtractor<InputFormat::kPacked, kUnchecked, kAcgt, kMsb> packed(21);
    packed.estimate();
    EXPECT_EQ(
        packed.algorithm_name(), expected_kmer_estimate("for_each_kmer_packed_unchecked_bytewise")
    );

    // A set algorithm stays until the estimate again.
    extractor.set_algorithm(scalar);
    EXPECT_EQ(extractor.algorithm_name(), scalar);
    extractor.estimate();
    EXPECT_EQ(extractor.algorithm_name(), expected);
}

// =================================================================================================
//     Callbacks
// =================================================================================================

// A callback of single k-mers without their positions, with each algorithm, and while tuning, on
// the pieces of the long sequence.
TEST(KmerExtractor, CallbackWithoutPositions)
{
    AsciiCheckedExtractor extractor(21);
    auto const seqs = extraction_test_sequences<InputCheck::kChecked>();
    auto const check = [&] {
        for (auto const& seq : seqs) {
            std::vector<std::uint64_t> values;
            extractor.for_each(seq, [&](KmerAcgtMsb kmer) { values.push_back(kmer_value(kmer)); });
            check_emitted_kmers(
                oracle_kmers_skipping_invalid<kAcgt, kMsb>(seq, 21), 21, values, nullptr
            );
        }
    };
    for (auto const& name : extractor.algorithm_names()) {
        extractor.set_algorithm(name);
        check();
    }
    TuneOptions options;
    options.time_per_algorithm = std::chrono::nanoseconds(0);
    options.piece_length = kExtractorPieceUnit;
    extractor.tune(options);
    check();
    EXPECT_FALSE(extractor.tuning_pending());
}

// =================================================================================================
//     Threads
// =================================================================================================

// One extractor, used from several threads at once.
TEST(KmerExtractor, Threads)
{
    AsciiCheckedExtractor const extractor(21);
    auto const seqs = extraction_test_sequences<InputCheck::kChecked>();
    std::vector<std::vector<ExpectedKmer>> expected;
    for (auto const& seq : seqs) {
        expected.push_back(oracle_kmers_skipping_invalid<Encoding::kACGT, Layout::kMSB>(seq, 21));
    }

    std::size_t const thread_count = 4;
    std::vector<std::vector<KmerRecorder>> results(thread_count);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < thread_count; ++t) {
        threads.emplace_back([&, t] {
            for (std::size_t round = 0; round < 20; ++round) {
                results[t].assign(seqs.size(), KmerRecorder{});
                for (std::size_t i = 0; i < seqs.size(); ++i) {
                    extractor.for_each(seqs[i], results[t][i]);
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (std::size_t t = 0; t < thread_count; ++t) {
        for (std::size_t i = 0; i < seqs.size(); ++i) {
            check_emitted_kmers(
                expected[i], 21, results[t][i].values, &results[t][i].positions
            );
        }
    }
}

// =================================================================================================
//     Errors
// =================================================================================================

TEST(KmerExtractor, Errors)
{
    EXPECT_THROW(AsciiCheckedExtractor(0), std::invalid_argument);
    EXPECT_THROW(AsciiCheckedExtractor(33), std::invalid_argument);

    AsciiCheckedExtractor extractor(21);
    EXPECT_THROW(extractor.set_algorithm("unknown"), std::invalid_argument);
    EXPECT_THROW(extractor.set_algorithm(""), std::invalid_argument);
}
