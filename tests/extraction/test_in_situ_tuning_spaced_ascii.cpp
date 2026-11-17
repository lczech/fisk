#include <cstddef>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "extraction/in_situ_spaced_common.hpp"

using namespace fisk;

// The in-situ tuning of the spaced k-mer extractor for ASCII input, see in_situ_spaced_common.hpp,
// and threads. The rest of the tuning is shared with the k-mer extractor, see
// test_in_situ_tuning_kmer.cpp.

TEST(InSituTuning, SpacedAsciiChecked)
{
    check_in_situ_spaced<InputFormat::kAscii, InputCheck::kChecked, Layout::kLSB>();
}

TEST(InSituTuning, SpacedAsciiUnchecked)
{
    check_in_situ_spaced<InputFormat::kAscii, InputCheck::kUnchecked, Layout::kMSB>();
}

// Threads that call for_each() of one extractor while it tunes all get the spaced k-mers of their
// sequences.
TEST(InSituTuning, SpacedThreads)
{
    using Extractor = SpacedKmerExtractor<
        InputFormat::kAscii, InputCheck::kChecked, Encoding::kACGT, Layout::kLSB, kMaxTestMasks
    >;
    Splitmix64 rng(11);
    auto const masks = random_masks(2, 21, 21);
    std::vector<std::string> seqs;
    for (std::size_t i = 0; i < 50; ++i) {
        auto seq = random_sequence(rng, 100 + rng.get_uint64() % 2000, "ACGT");
        inject_invalid(seq, rng, 0.01, "N");
        seqs.push_back(std::move(seq));
    }
    for (std::size_t repetition = 0; repetition < 10; ++repetition) {
        Extractor ex(masks);
        ex.tune(in_situ_test_options(3));
        std::vector<int> ok(4, 1);
        std::vector<std::thread> threads;
        for (std::size_t t = 0; t < ok.size(); ++t) {
            threads.emplace_back([&, t] {
                for (auto const& seq : seqs) {
                    SpacedKmerRecorder recorder;
                    ex.for_each(extraction_test_input<Extractor>(seq), recorder);
                    auto const expected = by_mask(
                        oracle_spaced_events<Encoding::kACGT, Layout::kLSB>(seq, masks)
                    );
                    if (!in_situ_same_events(by_mask(recorder.events), expected)) {
                        ok[t] = 0;
                    }
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        for (int o : ok) {
            EXPECT_EQ(o, 1);
        }
        EXPECT_FALSE(ex.tuning_pending());
    }
}
