#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extractor/kmer_extractor.hpp"
#include "fisk/util/cpu_runtime.hpp"
#include "extractor/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// The k-mers that an extractor is expected to hand out.
template <typename Extractor>
static std::vector<ExpectedKmer> extractor_test_expected(std::string const& seq, std::size_t k)
{
    constexpr auto E = Extractor::encoding;
    constexpr auto L = Extractor::layout;
    if constexpr (Extractor::input_check == InputCheck::kChecked) {
        return oracle_kmers_skipping_invalid<E, L>(seq, k);
    } else {
        return oracle_kmers_all_windows<E, L>(seq, k);
    }
}

// A callback of single k-mers that records them.
struct KmerRecorder
{
    std::vector<std::size_t> positions;
    std::vector<std::uint64_t> values;

    template <typename K>
    void operator()(std::size_t pos, K const& kmer)
    {
        positions.push_back(pos);
        values.push_back(kmer_value(kmer));
    }
};

// Every algorithm of the extractor, run with a callback of single k-mers, and with one that also
// takes vectors.
template <InputFormat I, InputCheck C, Encoding E, Layout L>
static void check_kmer_extractor_algorithms()
{
    using Extractor = KmerExtractor<I, C, E, L>;
    auto const seqs = extractor_test_sequences<C>();

    for (std::size_t k : std::initializer_list<std::size_t>{1, 7, 15, 31, 32}) {
        Extractor extractor(k);
        auto const names = extractor.algorithm_names();
        EXPECT_FALSE(names.empty());

        for (auto const& name : names) {
            extractor.set_algorithm(name);
            EXPECT_EQ(extractor.algorithm_name(), name);

            for (auto const& seq : seqs) {
                auto const expected = extractor_test_expected<Extractor>(seq, k);
                auto const input = extractor_test_input<Extractor>(seq);

                KmerRecorder single;
                extractor.for_each(input, single);
                check_emitted_kmers(expected, k, single.values, &single.positions);

                // The SIMD algorithms hand their vectors to a callback that takes them, apart from
                // rolling_simd, which only encodes and checks the characters with its arch.
                KmerRecorder unpacked;
                std::size_t vectors = 0;
                auto const both = callback_overload{
                    [&](std::size_t pos, Kmer<E, L> kmer) { unpacked(pos, kmer); },
                    [&](std::size_t pos, auto kmers, auto valid) {
                        ++vectors;
                        for_each_valid_lane(pos, kmers, valid, unpacked);
                    }
                };
                extractor.for_each(input, both);
                check_emitted_kmers(expected, k, unpacked.values, &unpacked.positions);
                bool const simd = name.find("simd") != std::string::npos
                    && !name.starts_with("rolling_simd");
                EXPECT_TRUE(simd ? (vectors > 0 || expected.empty()) : vectors == 0);
            }
        }
    }
}

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

// The SIMD arch that the estimate prefers for callbacks that take vectors, if any.
static std::string preferred_simd_arch_name()
{
    #if defined(FISK_HAS_AVX2)
    if (avx2_enabled()) {
        return "avx2";
    }
    #endif
    #if defined(FISK_HAS_NEON)
    if (neon_enabled()) {
        return "neon";
    }
    #endif
    return "";
}

TEST(KmerExtractor, Estimate)
{
    // Single k-mers: the scalar loop; vectors: the SIMD loop of AVX2 or Neon.
    AsciiCheckedExtractor extractor(21);
    auto const vector_callback = callback_overload{
        [](std::size_t, KmerAcgtMsb) {},
        [](std::size_t, auto, auto) {}
    };
    auto const arch = preferred_simd_arch_name();
    EXPECT_EQ(extractor.algorithm_name(), "chunked_scalar");
    EXPECT_EQ(
        extractor.algorithm_name(vector_callback),
        arch.empty() ? "chunked_scalar" : "chunked_simd_" + arch
    );

    KmerExtractor<InputFormat::kPacked, kUnchecked, kAcgt, kMsb> packed(21);
    EXPECT_EQ(packed.algorithm_name(), "bytewise_scalar");

    // A set algorithm is used for all callbacks.
    extractor.set_algorithm("chunked_scalar");
    EXPECT_EQ(extractor.algorithm_name(vector_callback), "chunked_scalar");

    // An estimate brings the choice by callback back.
    TuneOptions options;
    options.level = TuneLevel::kEstimate;
    extractor.tune(options);
    EXPECT_EQ(
        extractor.algorithm_name(vector_callback),
        arch.empty() ? "chunked_scalar" : "chunked_simd_" + arch
    );
}

TEST(KmerExtractor, Tune)
{
    auto const seqs = extractor_test_sequences<InputCheck::kChecked>();
    auto const contains = [](std::vector<std::string> const& names, std::string const& name) {
        return std::find(names.begin(), names.end(), name) != names.end();
    };

    TuneOptions options;
    options.total_length = 20000;
    options.sequence_length = 150;

    // Random data.
    AsciiCheckedExtractor extractor(15);
    extractor.tune(options);
    EXPECT_TRUE(contains(extractor.algorithm_names(), extractor.algorithm_name()));

    // A sample of reads, and a single sequence.
    extractor.tune(seqs, options);
    EXPECT_TRUE(contains(extractor.algorithm_names(), extractor.algorithm_name()));
    extractor.tune(seqs.back(), options);
    EXPECT_TRUE(contains(extractor.algorithm_names(), extractor.algorithm_name()));

    // With the callback, which then sees the k-mers.
    KmerRecorder recorder;
    extractor.tune(recorder, options);
    EXPECT_FALSE(recorder.values.empty());
    std::size_t calls = 0;
    extractor.tune(seqs, [&](KmerAcgtMsb) { ++calls; }, options);
    EXPECT_TRUE(calls > 0);

    // The tuned extractor, and a copy, extract all k-mers.
    auto const copy = extractor;
    EXPECT_EQ(copy.algorithm_name(), extractor.algorithm_name());
    for (auto const& seq : seqs) {
        auto const expected = oracle_kmers_skipping_invalid<Encoding::kACGT, Layout::kMSB>(seq, 15);
        KmerRecorder result;
        copy.for_each(seq, result);
        check_emitted_kmers(expected, 15, result.values, &result.positions);
    }
}

TEST(KmerExtractor, TunePacked)
{
    using Extractor = KmerExtractor<
        InputFormat::kPacked, InputCheck::kChecked, Encoding::kACGT, Layout::kMSB
    >;
    std::vector<PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>> reads;
    for (auto const& seq : extractor_test_sequences<InputCheck::kChecked>()) {
        reads.push_back(extractor_test_input<Extractor>(seq));
    }

    TuneOptions options;
    options.total_length = 20000;
    Extractor extractor(31);
    extractor.tune(options);
    extractor.tune(reads, options);
    extractor.tune(reads.back(), options);
    auto const names = extractor.algorithm_names();
    EXPECT_TRUE(std::find(names.begin(), names.end(), extractor.algorithm_name()) != names.end());
}

TEST(KmerExtractor, TuneEdgeCases)
{
    auto const names = AsciiCheckedExtractor(15).algorithm_names();
    auto const valid = [&](AsciiCheckedExtractor const& extractor) {
        return std::find(names.begin(), names.end(), extractor.algorithm_name()) != names.end();
    };
    AsciiCheckedExtractor extractor(15);

    // No random data, reads longer than the total, and no repeats: each still picks an algorithm.
    TuneOptions options;
    options.total_length = 0;
    extractor.tune(options);
    EXPECT_TRUE(valid(extractor));
    options.total_length = 1000;
    options.sequence_length = 5000;
    options.repeats = 0;
    extractor.tune(options);
    EXPECT_TRUE(valid(extractor));

    // Samples without k-mers: empty, or all sequences shorter than k.
    options = TuneOptions{};
    extractor.tune(std::vector<std::string>{}, options);
    EXPECT_TRUE(valid(extractor));
    extractor.tune(std::vector<std::string>{"", "ACGT", "ACGTACGTACGTAC"}, options);
    EXPECT_TRUE(valid(extractor));

    auto const seq = extractor_test_sequences<InputCheck::kChecked>().back();
    KmerRecorder result;
    extractor.for_each(seq, result);
    check_emitted_kmers(
        oracle_kmers_skipping_invalid<Encoding::kACGT, Layout::kMSB>(seq, 15), 15, result.values,
        &result.positions
    );
}

// =================================================================================================
//     Threads
// =================================================================================================

// One extractor, used from several threads at once.
TEST(KmerExtractor, Threads)
{
    AsciiCheckedExtractor const extractor(21);
    auto const seqs = extractor_test_sequences<InputCheck::kChecked>();
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
