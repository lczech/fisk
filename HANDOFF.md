# In-situ tuning: handoff

This branch adds `KmerExtractor` and `SpacedKmerExtractor` back on top of master, which has only the
free functions `for_each_kmer()` and `for_each_spaced_kmer()` with the estimate. The extractors tune
in situ: their first calls of `for_each()` time the candidate algorithms in turns, on the caller's
sequences and with the caller's callback, and keep the fastest. The branch is a backup, kept to
revive the feature; it is not meant to be merged as it is.

## Why it is not on master

- The estimate of the free functions picks the fastest algorithm, or one close to it, for
  realistic callbacks on the CPUs that we measured, see the benchmarks of master.
- Tuning that measures something else than the real run is misleading, and making it measure the
  real run took a design (the tuning inside `for_each()`) that is complex, with costs of its own,
  see the open issues.
- Tuning in user code (a sample, a loop until `tuning_pending()` is false) is a burden that most
  users would not take on.

## Design

- `ExtractorPlan` (`extraction/extractor_plan.hpp`) holds the candidates, the index of the chosen
  one (an atomic, `CopyableAtomic`, so that a plan stays copyable), and the state of the tuning.
- `for_each()` has one loop with one jump to the candidates, for the tuning and for the
  extraction, so that both run the same machine code. Without tuning, the loop runs once.
- While tuning, a sequence is cut into pieces of `piece_length` k-mers, and the candidates run in
  turns on them. Packed sequences are cut without a copy (`ExtractorPackedPiece`), at multiples
  of 256 nucleotides, see `make_extractor_packed_piece()`. The positions of a piece are shifted by
  its offset, see `ExtractorOffsetCallback`.
- A turn is timed from its first call to its last, across the calls of `for_each()`, and the clock
  is read once per `piece_length` k-mers. Reading it for every short sequence kept the CPU from
  overlapping the work of consecutive sequences, which slowed some candidates more than others.
- The turns start with the estimate's candidate, then go round the other available ones, `rounds`
  times; the fastest turn of each candidate counts.
- Threads: the first thread that calls `for_each()` after `tune()` claims the tuning (phases
  off, armed, claiming, running, see `ExtractorTuningPhase`), and tunes in its own calls; the
  other threads run the plan as it was. A copy of a plan that tunes is armed, and tunes on its own.
- The candidates are those of the estimate (`extraction/kmer_estimate.hpp`,
  `spaced_kmer_estimate.hpp`) and a few more: AVX512 next to AVX2 (`ExtractorSimdArchs`), as the
  estimate stops at AVX2 but AVX512 is faster on some CPUs and for some callbacks, and PEXT as a
  kernel of the packed SIMD loop. The estimate
  prefers PEXT only where it is likely fast, but the tuning measures it wherever it is available.
- `random_acgt()` (`util/random.hpp`) makes a sample for tuning before the real input, see the
  README.

## Open issues

1. **With GCC, the plan makes some loops slower than when they are called directly.** Clang is not
   affected, and neither are the free functions, which run the same loops.
   - Found on an Intel Core 7 240H with GCC 15, with the example on the human genome: spaced
     k-mers (scalar PEXT) took 2.35 s instead of 1.23 s, and on a Xeon Platinum 8568Y the GCC
     tuning picked the butterfly instead of PEXT. A bisect with the example on chr1 found the commit that moved the
     tuning into `for_each()` (spaced 100 -> 187 ms): GCC then loaded and stored the fields of the
     callback per k-mer. Any wrapper object around the callback in that function,
     `ExtractorOffsetCallback` here, also only in the tuning path, had this effect there, and
     passing the piece offset as a thread-local position base instead of the wrapper fixed it.
   - A harness of its own (appendix) does not confirm the wrapper as the cause. Times in ms on
     32 Mbp of random ACGT, the best of 5, with a fixed scalar algorithm:

     | | k-mers ASCII | k-mers packed | spaced PEXT ASCII | spaced PEXT packed |
     |---|---|---|---|---|
     | direct loop | 30.7 | 14.0 | 16.4 | 14.8 |
     | plan of this branch | 30.9 | 32.0 | 31.3 | 31.0 |
     | plan without the wrapper | 31.0 | 32.0 | 31.4 | 31.0 |
     | plan of master before the in-situ tuning | 33.6 | 29.3 | 31.5 | 30.1 |

     (Xeon Platinum 8568Y, GCC 15.2). With GCC 14 on a Ryzen 7 PRO 4750U, the plan is 36%
     slower than the direct loop for k-mers, ASCII and packed, also without the wrapper and also
     with master's extractor; spaced with the butterfly, which does more work per k-mer, shows no
     difference.
   - Master met the same effect without the plan: an AVX512 candidate compiled into the same
     function as the scalar PEXT loop made GCC keep the state of a counting callback in memory
     there, stored and reloaded per byte (spaced k-mers 1.8x slower in the example on the Xeon),
     which is why the estimate stops at AVX2, see `CandidateSimdArchs`. The extractors compile
     AVX512 candidates next to the PEXT loops, so the same likely applies to them.
   - So GCC's code for the loops in the plan depends on the code around them, which the example
     and the harness differ in; a fix of the wrapper alone, such as an offset parameter through
     the candidates and loops instead of `ExtractorOffsetCallback`, would fix the case of the
     example, but likely not the harness. This is the same kind of problem as that of the tuning
     before it ran in situ: the measured code is not the code that runs. Next step: compare the
     machine code of the plan and of the direct loop in the harness (perf annotate, or
     `-fopt-info`), for spaced PEXT and packed k-mers.
2. **Input that misleads the tuning**: runs of `N`, waits between the calls, and two extractors
   that tune at the same time on the same thread. The README recipe of tuning on a sample works
   around it, at the cost of user code.
3. **Feedback on the API, not done**: `kTuneMaxRounds` and `kTuneMaxPieceLength` are global
   constants and clamp the options; they could be internal constants that make `tune()` throw
   when exceeded.

## Reviving it

- Rebase onto master; the branch only adds files and a few lines to the shared extraction code.
- Tests: `tests/extraction/` (`test_kmer_extractor.cpp`, `test_spaced_kmer_extractor*.cpp`,
  `test_in_situ_tuning_*.cpp`, `test_packed_pieces_*.cpp`, `test_extractor_names.cpp`).
- The benchmarks and the example use the free functions. For an extractor benchmark, see the
  `extractor_scaling` suite in the history of master, before it became `for_each_scaling`.

## Appendix: the harness of open issue 1

Build each variant as its own program, against `include/` of the tree to measure, and run it
pinned to one core, e.g. with GCC: `g++ -std=c++20 -O3 -march=native -DNDEBUG
-falign-functions=64 -falign-loops=64 -DVARIANT_PLAN -DSPACED_PEXT -Iinclude wrap_bench.cpp`;
`-DVARIANT_DIRECT` for the direct loops, `-DINPUT_PACKED` for packed input, and `-DPLAIN_STRING`
for trees without `AlignedString`. It prints the variant, the input, the ms of k-mers and of
spaced k-mers, and their counts and checksums, which all variants share.

```cpp
// GCC wrapper check: the extractor (ExtractorPlan::for_each) with a fixed scalar algorithm, against
// the same loop called directly. Build with -DVARIANT_PLAN or -DVARIANT_DIRECT, and -DINPUT_PACKED
// or not. Prints ms per run of k-mers (k = 31) and spaced k-mers (span 31, weight 22), the best of
// 5, on 32 Mbp of random ACGT with N at 0.001.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include "fisk/fisk.hpp"
using namespace fisk;

struct MyCounter
{
    std::uint64_t count = 0;
    std::uint64_t checksum = 0;
    template <typename Kmer> void operator()(std::size_t, Kmer kmer) { ++count; checksum += kmer_value(kmer); }
    template <typename Kmer> void operator()(std::size_t, std::size_t, Kmer kmer) { ++count; checksum += kmer_value(kmer); }
};

constexpr auto C = InputCheck::kChecked;
constexpr auto E = Encoding::kACGT;
constexpr auto L = Layout::kLSB;
#if defined(INPUT_PACKED)
constexpr auto I = InputFormat::kPacked;
#else
constexpr auto I = InputFormat::kAscii;
#endif

// The bit extraction of the spaced k-mers: PEXT with -DSPACED_PEXT, else the butterfly.
#if defined(SPACED_PEXT)
using SpacedForm = BitExtractMask;
#define SPACED_SUFFIX "_pext"
#else
using SpacedForm = BitExtractButterflyMask;
#define SPACED_SUFFIX "_butterfly"
#endif

#if defined(INPUT_PACKED)
using KmerDirect = KmerCandidatePackedBytewiseScalar<C, E, L>;
using SpacedDirect = SpacedKmerCandidatePackedBytewiseScalar<C, E, L, 1, SpacedForm>;
#else
using KmerDirect = KmerCandidateAsciiChunkedScalar<C, E, L>;
using SpacedDirect = SpacedKmerCandidateAsciiChunkedScalar<C, E, L, 1, SpacedForm>;
#endif

template <typename F>
double best_ms(F&& f, MyCounter& out)
{
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
        MyCounter c;
        auto const t0 = std::chrono::steady_clock::now();
        f(c);
        std::chrono::duration<double, std::milli> const d = std::chrono::steady_clock::now() - t0;
        best = std::min(best, d.count());
        out = c;
    }
    return best;
}

int main()
{
    std::mt19937_64 rng(42);
#if defined(PLAIN_STRING)
    std::string seq(std::size_t{32} << 20, 'A');
#else
    AlignedString seq(std::size_t{32} << 20, 'A');
#endif
    for (auto& ch : seq) { auto r = rng(); ch = (r >> 11) % 1000 == 0 ? 'N' : "ACGT"[r & 3]; }
    std::vector<std::string> const masks{"1111011101110010111001011011111"};
    SpacedKmerMaskForms<L, 1> const forms(masks);
#if defined(INPUT_PACKED)
    auto const input = pack_sequence_checked(std::string_view(seq), WordEncoderButterfly<E, L>{});
    char const* kname = "for_each_kmer_packed_checked_bytewise_scalar";
    char const* sname = "for_each_spaced_kmer_packed_checked_bytewise_scalar" SPACED_SUFFIX;
#else
    std::string_view const input(seq);
    char const* kname = "for_each_kmer_ascii_checked_chunked_scalar";
    char const* sname = "for_each_spaced_kmer_ascii_checked_chunked_scalar" SPACED_SUFFIX;
#endif
    MyCounter kc, sc;
#if defined(VARIANT_PLAN)
    KmerExtractor<I, C, E, L> kx(31);
    kx.set_algorithm(kname);
    SpacedKmerExtractor<I, C, E, L, 1> sx(forms);
    sx.set_algorithm(sname);
    double const kms = best_ms([&](MyCounter& c) { kx.for_each(input, c); }, kc);
    double const sms = best_ms([&](MyCounter& c) { sx.for_each(input, c); }, sc);
    char const* variant = "plan";
#else
    double const kms = best_ms([&](MyCounter& c) {
        KmerDirect{}.run(31, input, c);
    }, kc);
    double const sms = best_ms([&](MyCounter& c) {
        SpacedDirect{}.run(forms, input, c);
    }, sc);
    char const* variant = "direct";
#endif
    std::cout << variant << (I == InputFormat::kPacked ? ",packed," : ",ascii,") << kms << "," << sms
              << "," << kc.count << "," << kc.checksum << "," << sc.count << "," << sc.checksum << "\n";
}
```
