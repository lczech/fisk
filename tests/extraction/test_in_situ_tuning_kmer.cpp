#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extraction/kmer_extractor.hpp"
#include "extraction/common.hpp"
#include "extraction/in_situ_common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// The in-situ tuning of the k-mer extractor, see ExtractorPlan::tune() and
// in_situ_common.hpp: the k-mers of the calls of for_each() during and after the tuning, the end of
// the tuning, its first turn, its turns with a time, copies of extractors while tuning, and
// threads.

namespace {

// Whether a recorder holds the expected k-mers of a sequence.
bool same_kmers(std::vector<ExpectedKmer> const& expected, KmerRecorder const& got)
{
    if (got.positions.size() != expected.size() || got.values.size() != expected.size()) {
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (got.positions[i] != expected[i].pos || got.values[i] != *expected[i].value) {
            return false;
        }
    }
    return true;
}

// A callback that also takes the vectors of the SIMD algorithms, into a recorder, and counts them.
template <Encoding E, Layout L>
auto vector_recorder(KmerRecorder& recorder, std::size_t& vectors)
{
    return callback_overload{
        [&recorder](std::size_t pos, Kmer<E, L> kmer) { recorder(pos, kmer); },
        [&recorder, &vectors](std::size_t pos, auto kmers, auto valid) {
            ++vectors;
            for_each_valid_lane(pos, kmers, valid, recorder);
        }
    };
}

// The k-mers of all calls of for_each() while and after tuning, for inputs whose tuning ends at
// each piece of their long sequence, see in_situ_test_input(), with a callback of single k-mers
// and with one that also takes vectors.
template <InputFormat I, InputCheck C, Layout L>
void check_in_situ_kmers()
{
    constexpr auto E = Encoding::kACGT;
    using Extractor = KmerExtractor<I, C, E, L>;
    Splitmix64 rng(4242);

    for (std::size_t k : std::initializer_list<std::size_t>{1, 5, 31, 32}) {
        auto const lengths = in_situ_test_lengths(k - 1, rng);
        for (std::size_t rounds = 1; rounds <= 3; ++rounds) {
            for (std::size_t prefix = 0; prefix <= 5; ++prefix) {
                for (std::size_t length : lengths) {
                    auto const seqs = in_situ_test_input<C>(prefix, length, rng);
                    Extractor single(k);
                    Extractor both(k);
                    single.tune(in_situ_test_options(rounds));
                    both.tune(in_situ_test_options(rounds));
                    for (std::size_t s = 0; s < seqs.size(); ++s) {
                        auto const expected = extraction_test_expected<Extractor>(seqs[s], k);
                        auto const input = extraction_test_input<Extractor>(seqs[s]);

                        KmerRecorder got;
                        single.for_each(input, got);
                        KmerRecorder unpacked;
                        std::size_t vectors = 0;
                        both.for_each(input, vector_recorder<E, L>(unpacked, vectors));
                        if (!same_kmers(expected, got) || !same_kmers(expected, unpacked)) {
                            std::cerr << "    in-situ mismatch: k " << k << ", rounds " << rounds
                                      << ", prefix " << prefix << ", length " << length
                                      << ", sequence " << s << "\n";
                            EXPECT_TRUE(false);
                        }
                    }
                }
            }
        }
    }
}

// A sequence with exactly one piece of k-mers, so that a turn takes one call of for_each().
template <typename Extractor>
auto one_piece(fisk::Splitmix64& rng, std::size_t k)
{
    return extraction_test_input<Extractor>(random_sequence(rng, kInSituTestPiece + k - 1, "ACGT"));
}

using LifecycleExtractor =
    KmerExtractor<InputFormat::kAscii, InputCheck::kChecked, Encoding::kACGT, Layout::kMSB>;

} // namespace

TEST(InSituTuning, KmerAsciiChecked)
{
    check_in_situ_kmers<InputFormat::kAscii, InputCheck::kChecked, Layout::kMSB>();
}

TEST(InSituTuning, KmerAsciiUnchecked)
{
    check_in_situ_kmers<InputFormat::kAscii, InputCheck::kUnchecked, Layout::kLSB>();
}

TEST(InSituTuning, KmerPackedChecked)
{
    check_in_situ_kmers<InputFormat::kPacked, InputCheck::kChecked, Layout::kLSB>();
}

TEST(InSituTuning, KmerPackedUnchecked)
{
    check_in_situ_kmers<InputFormat::kPacked, InputCheck::kUnchecked, Layout::kMSB>();
}

// The tuning is pending from tune() until each algorithm had its turns, and every other choice of
// the algorithm ends it. While it is pending, the algorithm is the one from before.
TEST(InSituTuning, Lifecycle)
{
    Splitmix64 rng(7);
    LifecycleExtractor ex(31);
    std::size_t const count = ex.algorithm_names().size();
    std::string const estimated = ex.algorithm_name();

    // A new extractor tunes, until estimate() or another choice.
    EXPECT_TRUE(ex.tuning_pending());
    ex.estimate();
    EXPECT_FALSE(ex.tuning_pending());
    EXPECT_EQ(ex.algorithm_name(), estimated);

    for (std::size_t rounds = 1; rounds <= 3; ++rounds) {
        ex.estimate();
        ex.tune(in_situ_test_options(rounds));
        for (std::size_t turn = 0; turn < count * rounds; ++turn) {
            EXPECT_TRUE(ex.tuning_pending());
            EXPECT_EQ(ex.algorithm_name(), estimated);
            KmerRecorder recorder;
            ex.for_each(one_piece<LifecycleExtractor>(rng, 31), recorder);
            EXPECT_EQ(recorder.positions.size(), kInSituTestPiece);
        }
        EXPECT_FALSE(ex.tuning_pending());
        auto const names = ex.algorithm_names();
        EXPECT_TRUE(std::find(names.begin(), names.end(), ex.algorithm_name()) != names.end());
    }

    // Sequences without k-mers do not count.
    ex.tune(in_situ_test_options(1));
    for (std::size_t i = 0; i < 10; ++i) {
        KmerRecorder recorder;
        ex.for_each(std::string(30, 'A'), recorder);
        ex.for_each(std::string(), recorder);
    }
    EXPECT_TRUE(ex.tuning_pending());

    // The other choices end the tuning.
    std::string const first = ex.algorithm_names().front();
    ex.set_algorithm(first);
    EXPECT_FALSE(ex.tuning_pending());
    EXPECT_EQ(ex.algorithm_name(), first);

    ex.tune(in_situ_test_options(1));
    ex.estimate();
    EXPECT_FALSE(ex.tuning_pending());
    EXPECT_EQ(ex.algorithm_name(), estimated);
}

// The first turn is the estimate's: a callback that takes vectors gets vectors from the first
// call, where the estimate is a SIMD algorithm, although the first candidate is scalar.
TEST(InSituTuning, FirstTurnIsTheEstimate)
{
    Splitmix64 rng(8);
    LifecycleExtractor ex(31);
    KmerRecorder recorder;
    std::size_t vectors = 0;
    auto callback = vector_recorder<Encoding::kACGT, Layout::kMSB>(recorder, vectors);
    bool const simd = ex.algorithm_name().find("simd") != std::string::npos;
    ex.tune(in_situ_test_options(3));
    ex.for_each(one_piece<LifecycleExtractor>(rng, 31), callback);
    EXPECT_EQ(vectors > 0, simd);
    EXPECT_TRUE(ex.tuning_pending());
}

// With a time, a turn ends at the first reading of the clock after its time ran out, after every
// piece_length k-mers, across the calls of for_each(): a turn on reads of 100 k-mers takes several
// calls, and a long sequence has several turns, each of one piece.
TEST(InSituTuning, TimedTurns)
{
    Splitmix64 rng(11);
    TuneOptions options;
    options.time_per_algorithm = std::chrono::nanoseconds(3);
    options.rounds = 3;
    options.piece_length = kInSituTestPiece;
    for (std::size_t length : std::initializer_list<std::size_t>{130, 100000}) {
        LifecycleExtractor ex(31);
        std::size_t const turns = ex.algorithm_names().size() * options.rounds;
        ex.tune(options);
        std::size_t calls = 0;
        while (ex.tuning_pending() && calls < 100000) {
            auto seq = random_sequence(rng, length, "ACGT");
            inject_invalid(seq, rng, 0.01, "N");
            KmerRecorder recorder;
            ex.for_each(seq, recorder);
            EXPECT_TRUE(same_kmers(extraction_test_expected<LifecycleExtractor>(seq, 31), recorder));
            ++calls;
        }
        EXPECT_FALSE(ex.tuning_pending());

        // Each turn has at least one piece of k-mers, and less than two: the last call of a turn
        // has at most one piece.
        std::size_t const starts = length - 30;
        std::size_t const least = turns * kInSituTestPiece;
        std::size_t const most = turns * 2 * kInSituTestPiece;
        EXPECT_TRUE(calls >= (least + starts - 1) / starts);
        EXPECT_TRUE(calls <= (most + starts - 1) / starts);
    }
}

// Options beyond their usual range, see TuneOptions: 0 rounds count as 1, and more than
// kTuneMaxRounds as kTuneMaxRounds; a piece_length of 0 counts as 256, and more than
// kTuneMaxPieceLength as kTuneMaxPieceLength, also with a time, so that the clock is read; a
// negative time ends each turn after a piece, as 0 does. With such turns, the calls until the
// tuning ends tell the options apart, and the k-mers stay those of the oracle.
TEST(InSituTuning, EdgeOptions)
{
    Splitmix64 rng(12);
    LifecycleExtractor ex(31);
    std::size_t const count = ex.algorithm_names().size();
    auto const calls_until_done = [&](TuneOptions const& options, std::size_t length) {
        ex.tune(options);
        std::size_t calls = 0;
        while (ex.tuning_pending() && calls < 100 * count) {
            auto seq = random_sequence(rng, length, "ACGT");
            inject_invalid(seq, rng, 0.01, "N");
            KmerRecorder recorder;
            ex.for_each(seq, recorder);
            EXPECT_TRUE(same_kmers(extraction_test_expected<LifecycleExtractor>(seq, 31), recorder));
            ++calls;
        }
        EXPECT_FALSE(ex.tuning_pending());
        return calls;
    };
    constexpr std::size_t max = std::numeric_limits<std::size_t>::max();

    // One turn per call of one piece: as many calls as algorithms times rounds.
    EXPECT_EQ(calls_until_done(in_situ_test_options(0), kInSituTestPiece + 30), count);
    TuneOptions options = in_situ_test_options(1);
    options.time_per_algorithm = std::chrono::nanoseconds(-1);
    EXPECT_EQ(calls_until_done(options, kInSituTestPiece + 30), count);

    // A piece_length of 0: four turns per call of 1000 k-mers.
    options = in_situ_test_options(1);
    options.piece_length = 0;
    EXPECT_EQ(calls_until_done(options, 1030), (count + 3) / 4);

    // The largest piece_lengths, longer than the calls: one turn per call.
    for (std::size_t length : {max, max - 100, kTuneMaxPieceLength + 1}) {
        options.piece_length = length;
        EXPECT_EQ(calls_until_done(options, 1030), count);
    }

    // The largest piece_length with a time: the clock is read after kTuneMaxPieceLength k-mers, so
    // that each turn takes as many calls of 1000 k-mers, with a callback that only counts.
    {
        options.piece_length = max;
        options.time_per_algorithm = std::chrono::nanoseconds(1);
        ex.tune(options);
        std::string const seq = random_sequence(rng, 1030, "ACGT");
        std::size_t const per_turn = (kTuneMaxPieceLength + 999) / 1000;
        std::size_t calls = 0;
        while (ex.tuning_pending() && calls <= count * per_turn) {
            ex.for_each(seq, [](std::size_t, KmerAcgtMsb) {});
            ++calls;
        }
        EXPECT_FALSE(ex.tuning_pending());
        EXPECT_EQ(calls, count * per_turn);
    }

    // The largest rounds, as kTuneMaxRounds: one turn per call of one k-mer, with a callback that
    // only counts, as these are many calls.
    options = in_situ_test_options(max);
    ex.tune(options);
    std::string const seq = random_sequence(rng, 31, "ACGT");
    std::size_t kmers = 0;
    std::size_t calls = 0;
    while (ex.tuning_pending() && calls <= count * kTuneMaxRounds) {
        ex.for_each(seq, [&](std::size_t, KmerAcgtMsb) { ++kmers; });
        ++calls;
    }
    EXPECT_FALSE(ex.tuning_pending());
    EXPECT_EQ(calls, count * kTuneMaxRounds);
    EXPECT_EQ(kmers, calls);
}

// A copy of an extractor whose tuning is armed or running tunes on its own; a copy of a tuned one
// keeps its algorithm.
TEST(InSituTuning, Copies)
{
    Splitmix64 rng(9);
    LifecycleExtractor a(31);
    std::size_t const count = a.algorithm_names().size();
    auto const run = [&](LifecycleExtractor const& ex, std::size_t calls) {
        for (std::size_t i = 0; i < calls; ++i) {
            auto const seq = random_sequence(rng, kInSituTestPiece + 30, "ACGT");
            KmerRecorder recorder;
            ex.for_each(seq, recorder);
            EXPECT_TRUE(same_kmers(extraction_test_expected<LifecycleExtractor>(seq, 31), recorder));
        }
    };

    // Armed.
    a.tune(in_situ_test_options(1));
    LifecycleExtractor b = a;
    EXPECT_TRUE(b.tuning_pending());
    run(b, count);
    EXPECT_FALSE(b.tuning_pending());
    EXPECT_TRUE(a.tuning_pending());

    // Running: the copy starts its own turns, and the original goes on with its own.
    if (count > 1) {
        run(a, 1);
        EXPECT_TRUE(a.tuning_pending());
        LifecycleExtractor c(31);
        c = a;
        EXPECT_TRUE(c.tuning_pending());
        run(c, count);
        EXPECT_FALSE(c.tuning_pending());
        EXPECT_TRUE(a.tuning_pending());
        run(a, count - 1);
    } else {
        run(a, count);
    }
    EXPECT_FALSE(a.tuning_pending());

    // Tuned.
    LifecycleExtractor d = a;
    EXPECT_FALSE(d.tuning_pending());
    EXPECT_EQ(d.algorithm_name(), a.algorithm_name());
}

// Threads that call for_each() of one extractor while it tunes all get the k-mers of their
// sequences: one of them tunes, the others run the estimate.
TEST(InSituTuning, Threads)
{
    Splitmix64 rng(10);
    std::vector<std::string> seqs;
    for (std::size_t i = 0; i < 50; ++i) {
        auto seq = random_sequence(rng, 100 + rng.get_uint64() % 2000, "ACGT");
        inject_invalid(seq, rng, 0.01, "N");
        seqs.push_back(std::move(seq));
    }
    for (std::size_t repetition = 0; repetition < 10; ++repetition) {
        LifecycleExtractor ex(31);
        ex.tune(in_situ_test_options(3));
        std::vector<int> ok(4, 1);
        std::vector<std::thread> threads;
        for (std::size_t t = 0; t < ok.size(); ++t) {
            threads.emplace_back([&, t] {
                for (auto const& seq : seqs) {
                    KmerRecorder recorder;
                    ex.for_each(seq, recorder);
                    auto const expected = extraction_test_expected<LifecycleExtractor>(seq, 31);
                    if (!same_kmers(expected, recorder)) {
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

// -------------------------------------------------------------------------------------------------
//     Threads that Stop or Copy
// -------------------------------------------------------------------------------------------------

namespace {

// One call of for_each() on a sequence of one piece, which is one turn while tuning; whether its
// k-mers are those of the oracle.
bool call_one_piece(LifecycleExtractor const& ex, Splitmix64& rng)
{
    auto const seq = random_sequence(rng, kInSituTestPiece + 30, "ACGT");
    KmerRecorder recorder;
    ex.for_each(seq, recorder);
    return same_kmers(extraction_test_expected<LifecycleExtractor>(seq, 31), recorder);
}

// The calls of one piece until the tuning of `ex` ends, at most `limit`.
std::size_t calls_until_tuned(LifecycleExtractor const& ex, Splitmix64& rng, std::size_t limit)
{
    std::size_t calls = 0;
    while (ex.tuning_pending() && calls < limit) {
        EXPECT_TRUE(call_one_piece(ex, rng));
        ++calls;
    }
    return calls;
}

} // namespace

// An owner thread that stops calling for_each() leaves its tuning pending: the other threads run
// the algorithm from before, with the k-mers of the oracle, and tune() arms the tuning again, for
// the next thread that calls for_each().
TEST(InSituTuning, OwnerStops)
{
    Splitmix64 rng(16);
    LifecycleExtractor ex(31);
    std::size_t const count = ex.algorithm_names().size();
    ex.tune(in_situ_test_options(1));
    int ok = 0;
    std::thread([&] {
        Splitmix64 thread_rng(17);
        ok = call_one_piece(ex, thread_rng) ? 1 : 0;
    }).join();
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(calls_until_tuned(ex, rng, 3 * count), 3 * count);
    EXPECT_TRUE(ex.tuning_pending());
    ex.tune(in_situ_test_options(1));
    EXPECT_EQ(calls_until_tuned(ex, rng, 10 * count), count);
}

// Copies made while another thread tunes the extractor are pending, and tune on their own, on the
// thread that calls them, here with fewer rounds than the extractor, which stays pending.
TEST(InSituTuning, CopiesWhileTuning)
{
    Splitmix64 rng(18);
    LifecycleExtractor ex(31);
    std::size_t const count = ex.algorithm_names().size();
    ex.tune(in_situ_test_options(kTuneMaxRounds));
    std::atomic<bool> stop{false};
    int ok = 1;
    std::thread tuner([&] {
        Splitmix64 thread_rng(19);
        while (!stop.load()) {
            if (!call_one_piece(ex, thread_rng)) {
                ok = 0;
            }
        }
    });
    for (std::size_t i = 0; i < 20; ++i) {
        LifecycleExtractor copy = ex;
        EXPECT_TRUE(copy.tuning_pending());
        copy.tune(in_situ_test_options(1));
        EXPECT_EQ(calls_until_tuned(copy, rng, 10 * count), count);
    }
    stop.store(true);
    tuner.join();
    EXPECT_EQ(ok, 1);
    EXPECT_TRUE(ex.tuning_pending());
}

// -------------------------------------------------------------------------------------------------
//     Callbacks while Tuning
// -------------------------------------------------------------------------------------------------

// A callback that throws in the second turn: the exception leaves for_each(), that turn does not
// count, and the next calls give the k-mers of the oracle and take the remaining turns.
TEST(InSituTuning, ThrowingCallback)
{
    Splitmix64 rng(20);
    LifecycleExtractor ex(31);
    std::size_t const count = ex.algorithm_names().size();
    ex.tune(in_situ_test_options(1));
    EXPECT_TRUE(call_one_piece(ex, rng));
    bool thrown = false;
    try {
        std::size_t kmers = 0;
        ex.for_each(
            random_sequence(rng, kInSituTestPiece + 30, "ACGT"),
            [&](std::size_t, KmerAcgtMsb) {
                if (++kmers == 100) {
                    throw std::runtime_error("callback");
                }
            }
        );
    } catch (std::runtime_error const&) {
        thrown = true;
    }
    EXPECT_TRUE(thrown);
    EXPECT_EQ(ex.tuning_pending(), count > 1);
    EXPECT_EQ(calls_until_tuned(ex, rng, 10 * count), count - 1);
}

// Two callback types in turn within one tuning, one of single k-mers and one that takes vectors:
// the k-mers stay those of the oracle, and the tuning ends after its turns, each timed with the
// callback that ran it.
TEST(InSituTuning, CallbackTypesInTurn)
{
    Splitmix64 rng(21);
    LifecycleExtractor ex(31);
    std::size_t const count = ex.algorithm_names().size();
    ex.tune(in_situ_test_options(2));
    std::size_t calls = 0;
    while (ex.tuning_pending() && calls < 10 * count) {
        auto const seq = random_sequence(rng, kInSituTestPiece + 30, "ACGT");
        KmerRecorder recorder;
        if (calls % 2 == 0) {
            ex.for_each(seq, recorder);
        } else {
            std::size_t vectors = 0;
            ex.for_each(seq, vector_recorder<Encoding::kACGT, Layout::kMSB>(recorder, vectors));
        }
        EXPECT_TRUE(same_kmers(extraction_test_expected<LifecycleExtractor>(seq, 31), recorder));
        ++calls;
    }
    EXPECT_EQ(calls, 2 * count);
}

// While a tuning again is pending, the algorithm is the one from before: the one set, here the
// first and the last candidate, which another thread runs while this one tunes.
TEST(InSituTuning, TuningAgainKeepsTheAlgorithm)
{
    Splitmix64 rng(22);
    LifecycleExtractor ex(31);
    std::size_t const count = ex.algorithm_names().size();
    for (auto const& name : {ex.algorithm_names().front(), ex.algorithm_names().back()}) {
        ex.set_algorithm(name);
        ex.tune(in_situ_test_options(1));
        EXPECT_TRUE(call_one_piece(ex, rng));
        EXPECT_EQ(ex.algorithm_name(), name);
        if (count > 1) {
            EXPECT_TRUE(ex.tuning_pending());
        }
        std::size_t vectors = 0;
        int ok = 0;
        std::thread([&] {
            Splitmix64 thread_rng(23);
            auto const seq = random_sequence(thread_rng, kInSituTestPiece + 30, "ACGT");
            KmerRecorder recorder;
            ex.for_each(seq, vector_recorder<Encoding::kACGT, Layout::kMSB>(recorder, vectors));
            ok = same_kmers(extraction_test_expected<LifecycleExtractor>(seq, 31), recorder) ? 1 : 0;
        }).join();
        EXPECT_EQ(ok, 1);
        EXPECT_EQ(vectors > 0, name.find("simd") != std::string::npos);
        calls_until_tuned(ex, rng, 10 * count);
    }
}

// The tuning keeps the fastest algorithm: with a callback that is slow for vectors, the scalar
// one, which hands out single k-mers; with one that is slow for single k-mers, a SIMD one, which
// hands out vectors. The slow form waits for 20 us per call, far longer than the extraction.
TEST(InSituTuning, KeepsTheFastest)
{
    Splitmix64 rng(24);
    auto const wait = [] {
        auto const until = std::chrono::steady_clock::now() + std::chrono::microseconds(20);
        while (std::chrono::steady_clock::now() < until) {
        }
    };
    for (bool slow_vectors : {true, false}) {
        LifecycleExtractor ex(31);
        std::string const scalar = ex.algorithm_names().front();
        if (ex.algorithm_names().size() < 2) {
            continue;
        }
        auto const callback = callback_overload{
            [&](std::size_t, KmerAcgtMsb) {
                if (!slow_vectors) {
                    wait();
                }
            },
            [&](std::size_t, auto, auto) {
                if (slow_vectors) {
                    wait();
                }
            }
        };
        ex.tune(in_situ_test_options(1));
        std::size_t calls = 0;
        while (ex.tuning_pending() && calls < 100) {
            ex.for_each(random_sequence(rng, kInSituTestPiece + 30, "ACGT"), callback);
            ++calls;
        }
        EXPECT_FALSE(ex.tuning_pending());
        EXPECT_EQ(ex.algorithm_name() == scalar, slow_vectors);
    }
}
