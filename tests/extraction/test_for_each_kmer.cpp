#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extraction/for_each.hpp"
#include "extraction/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kActg = Encoding::kACTG;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kLsb = Layout::kLSB;
constexpr auto kAscii = InputFormat::kAscii;
constexpr auto kPacked = InputFormat::kPacked;
constexpr auto kChecked = InputCheck::kChecked;
constexpr auto kUnchecked = InputCheck::kUnchecked;

// =================================================================================================
//     K-mers
// =================================================================================================

// for_each_kmer() extracts all k-mers, with a callback of single k-mers, and with one that also
// takes vectors, which gets vectors from a SIMD algorithm, and runs the algorithm of the estimate,
// see expected_kmer_estimate(), the same for both callbacks.
template <InputFormat I, InputCheck C, Encoding E, Layout L>
static void check_for_each_kmer()
{
    auto const seqs = extraction_test_sequences<C>();
    auto const estimate = expected_kmer_estimate(
        std::string("for_each_kmer_") + (I == kAscii ? "ascii_" : "packed_") +
        (C == kChecked ? "checked_" : "unchecked_") + (I == kAscii ? "chunked" : "bytewise")
    );

    for (std::size_t k : std::initializer_list<std::size_t>{1, 15, 31, 32}) {
        for (auto const& seq : seqs) {
            auto const expected = extraction_test_expected<C, E, L>(seq, k);
            auto const input = extraction_test_input<I, C, E, L>(seq);

            KmerRecorder single;
            for_each_kmer<C, E, L>(input, k, single);
            check_emitted_kmers(expected, k, single.values, &single.positions);
            auto const single_name = for_each_kmer_algorithm_name<C, E, L>(input, k);
            EXPECT_EQ(single_name, estimate);

            KmerRecorder unpacked;
            std::size_t vectors = 0;
            auto const both = callback_overload{
                [&](std::size_t pos, Kmer<E, L> kmer) { unpacked(pos, kmer); },
                [&](std::size_t pos, auto kmers, auto valid) {
                    ++vectors;
                    for_each_valid_lane(pos, kmers, valid, unpacked);
                }
            };
            for_each_kmer<C, E, L>(input, k, both);
            check_emitted_kmers(expected, k, unpacked.values, &unpacked.positions);
            bool const simd = estimate.find("simd") != std::string::npos;
            EXPECT_TRUE(simd ? (vectors > 0 || expected.empty()) : vectors == 0);

            // Packed input gives the template arguments by its type.
            if constexpr (I == kPacked) {
                KmerRecorder deduced;
                for_each_kmer(input, k, deduced);
                EXPECT_EQ(deduced.values, single.values);
                EXPECT_EQ(for_each_kmer_algorithm_name(input, k), estimate);
            }
        }
    }
}

TEST(ForEachKmer, AsciiChecked)
{
    check_for_each_kmer<kAscii, kChecked, kAcgt, kMsb>();
    check_for_each_kmer<kAscii, kChecked, kActg, kLsb>();
}

TEST(ForEachKmer, AsciiUnchecked)
{
    check_for_each_kmer<kAscii, kUnchecked, kAcgt, kMsb>();
}

TEST(ForEachKmer, PackedChecked)
{
    check_for_each_kmer<kPacked, kChecked, kAcgt, kMsb>();
}

TEST(ForEachKmer, PackedUnchecked)
{
    check_for_each_kmer<kPacked, kUnchecked, kAcgt, kLsb>();
}

// ASCII input as std::string, std::string_view, and C string.
TEST(ForEachKmer, AsciiInputTypes)
{
    std::string const seq = "ACGTNACGTACGTTGCA";
    auto const run = [](auto const& input) {
        KmerRecorder recorder;
        for_each_kmer<kChecked, kAcgt, kMsb>(input, 4, recorder);
        return recorder.values;
    };
    auto const expected = run(seq);
    EXPECT_FALSE(expected.empty());
    EXPECT_EQ(run(std::string_view(seq)), expected);
    EXPECT_EQ(run(seq.c_str()), expected);
}

TEST(ForEachKmer, Errors)
{
    KmerRecorder recorder;
    auto const run = [&](std::size_t k) {
        for_each_kmer<kChecked, kAcgt, kMsb>("ACGT", k, recorder);
    };
    EXPECT_THROW(run(0), std::invalid_argument);
    EXPECT_THROW(run(33), std::invalid_argument);
}

// =================================================================================================
//     Threads
// =================================================================================================

// for_each_kmer() on several threads at once, each with the buffer of its thread for the chunks of
// the ASCII input.
TEST(ForEachKmer, Threads)
{
    auto const seqs = extraction_test_sequences<kChecked>();
    std::vector<std::vector<ExpectedKmer>> expected;
    for (auto const& seq : seqs) {
        expected.push_back(oracle_kmers_skipping_invalid<kAcgt, kMsb>(seq, 21));
    }

    std::size_t const thread_count = 4;
    std::vector<std::vector<KmerRecorder>> results(thread_count);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < thread_count; ++t) {
        threads.emplace_back([&, t] {
            for (std::size_t round = 0; round < 20; ++round) {
                results[t].assign(seqs.size(), KmerRecorder{});
                for (std::size_t i = 0; i < seqs.size(); ++i) {
                    for_each_kmer<kChecked, kAcgt, kMsb>(seqs[i], 21, results[t][i]);
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
