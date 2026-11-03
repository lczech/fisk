# Wide k-mer exploration

## Task

- Source: user prompt, 2026-10-07.
- Goal: compare practical designs for `k > 32`, including wide k-mer extraction from sequence data and extraction of spaced k-mers with at most 32 selected positions. Measure direct sequence-to-spaced extraction where useful, while preserving existing narrow-path performance.
- Mode: explore, with experimental correctness tests and microbenchmarks; no production integration unless separately approved.
- Writable locations: this worktree only (`/home/lucas/Repos/fisk/.worktrees/wide-kmers`) plus its Git metadata.
- Allowance: original passes used the then-current 5-hour window. Round 3 may use 15 percentage points of the weekly allowance, starting at 16% and stopping at 31%.
- Round 3 writable locations: this worktree and a checkout on gaia. Further rounds: `gaia-agent:~/fisk` only; never log in to gaia as `lucas`.

## Status

Archived (2026-11-03) on branch `feature/wide-kmers`; see [HANDOFF.md](HANDOFF.md) for the
current state and for how to revive the feature on `master` (re-implement in clean commits, never
merge this branch). The original status below is kept for the record.

Partial: the approved first production step, `WideKmer` and its value operations, is implemented in
the task worktree but is not committed or compiled. Work stopped when `trai check` reported 37% of
the weekly allowance, above the agreed 31% ceiling. The master checkout has no implementation edits.

## Improve-pass candidates

Ranked by expected value/risk, with the committed CSVs as baselines:

1. Replace two-bit-per-base wide validity with one bit per base (`spaced_ascii`: 3.5--7.8 ns/window for current best variants).
2. Make packed four-window butterfly extraction explicitly SIMD so performance survives non-vectorizing callbacks (`window_butterfly2`: 2.45--3.44 ns vectorized, 5.55--5.61 ns without vectorization).
3. Add an assume-valid ASCII path, matching the repository's existing narrow split (`wide_ascii array<2>`: 1.19--1.46 ns/window with validation).
4. Try grouped/unrolled ASCII rolling only after the lower-risk changes above.

### Round 3 candidates

Ranked after collecting a matched Intel Core 7 240H baseline:

1. Specialize for fast Intel hardware PEXT, including an assume-valid ASCII path.
2. Batch/unroll direct packed PEXT to increase instruction-level parallelism.
3. Compare scalar-built packed windows with AVX2 variable-lane shifts.
4. Increase AVX2 batch width where it hides loop and extraction latency on both hosts.
5. Exercise realistic scalar/vector consumer shapes before recommending an API.

Intel reverses the Zen 2 extractor result: direct packed PEXT is ~0.79 ns/window and two-limb
window PEXT is 0.94--1.03 ns, versus 1.46--1.50 ns for the best vector-sink AVX2 butterfly path.
ASCII PEXT with compact validity is 1.52--1.54 ns, versus 1.97--2.01 ns for AVX2 butterfly.

### Improve results

| Change | GCC 14.2 | Clang 17 | Decision |
|---|---:|---:|---|
| One-bit validity, butterfly | 7.59--7.61 -> 6.39--6.41 ns | 6.48--6.56 -> 6.07--6.14 ns | Keep |
| One-bit validity, block | 5.76--17.12 -> 5.27--16.42 ns | 3.48--18.57 -> 3.08--17.28 ns | Keep |
| Explicit packed AVX2 butterfly | 3.33 -> 2.81 ns | 2.44--2.46 -> 2.00 ns | Keep |
| Explicit packed AVX2, vectorization disabled | 5.50--5.54 -> 2.81--2.83 ns | 5.52--5.56 -> 1.67--1.68 ns | Keep |
| Assume-valid wide rolling | ~1.45 -> ~1.45 ns | ~1.30 -> ~1.25 ns | Keep as separate API |
| Assume-valid rolling butterfly | 6.38--6.45 -> 6.21--6.29 ns | 6.36--6.40 -> 6.16--6.18 ns | Keep |
| Four-at-once rolling + AVX2 butterfly | 6.15--6.18 -> 2.74--2.77 ns | 6.15--6.18 -> 2.12--2.13 ns | Keep |
| Same, vectorization disabled | 5.80--5.82 -> 2.75--2.77 ns | 6.02--6.05 -> 2.26--2.28 ns | Keep |
| AVX2 immediate shifts, rolling | 2.74--2.77 -> 2.46--2.49 ns | ~2.12 -> ~2.12 ns | Keep |
| AVX2 immediate shifts, packed | 2.81--2.83 -> 2.02--2.05 ns | ~1.99 -> ~1.99 ns | Keep |
| Register-native sink, rolling AVX2 | 2.36--2.37 -> 2.21--2.25 ns | ~2.12 -> 1.94 ns | Keep as vector-output API |
| Register-native sink, packed AVX2 | 2.01--2.03 -> 1.66--1.68 ns | 1.98--2.00 -> 1.66--1.67 ns | Keep as vector-output API |

### Round 3 results

| Change | AMD Zen 2 | Intel Core 7 240H | Decision |
|---|---:|---:|---|
| Assume-valid two-limb PEXT | Still 15--70 ns | 1.50--1.54 -> 1.39--1.46 ns | Keep for Intel-selected path |
| Group four packed PEXT offsets | Safe loop 17.7--70.6 -> 17.2--70.1 ns | 0.60--0.94 -> 0.53--0.73 ns | Keep |
| Four-window rolling ASCII PEXT | Mixed; still 16--69 ns | 1.37--1.45 -> 0.98--1.13 ns | Keep for Intel-selected path |

The Intel path is also 29--31% faster than the vector-sink AVX2 butterfly. BMI2 feature detection
alone is insufficient: the same instruction is one to two orders of magnitude slower on Zen 2.
The grouped packed path improves Intel by 3--33% relative to the safe loop, depending on compiler
and span, and fixes a shift-by-64 undefined operation in the original three-word prototype.
Materializing four dependent ASCII windows together improves Intel PEXT by 19--32%. It remains
unsuitable for Zen 2, where PEXT dominates the cost and compiler results are mixed.

Rejected attempts:

- Assume-valid direct position gathering: 4--18% faster under Clang, but 5--13% slower under GCC.
- Assume-valid block extraction: improved many masks, but regressed several medium/dense Clang cases by up to ~8%.
- Scalar four-at-once wide rolling: ~4% faster under GCC, but ~1.4% slower under Clang.
- Per-output zero-mask PEXT branch: helped Clang, but regressed Intel GCC at span 64; superseded by
  explicitly grouping the four packed offsets.
- Scalar construction of four packed windows before AVX2 extraction: 25--40% slower on both hosts
  with both compilers; vector variable-lane shifts are clearly preferable.
- Two AVX2 batches per packed-loop iteration: neutral for AMD GCC but 1--2% slower with Clang and
  generally slower on Intel; the four-window loop is already throughput-saturated.
- Four-window validated ASCII PEXT: ~9% faster with Intel Clang, but 8--12% slower with Intel GCC
  and generally worse with AMD GCC; retain scalar compact-validity rolling.

The register-native measurements are a lower bound for consumers that accept four k-mers at once;
scalar callbacks still require a lane handoff. With auto-vectorization disabled, GCC retained the
same 6% rolling and 18% packed gains. Clang retained a 14% rolling gain, while its scalar packed
reduction already optimized to the same ~1.67 ns result.

## What was done

- Mapped all narrow type, callback, ASCII, packed, spaced, and bit-extraction dependencies.
- Built correctness-checked fixed-array, `__uint128_t`, vector, wide-to-spaced, fused ASCII, and fused packed prototypes.
- Benchmarked GCC and Clang with and without auto-vectorization on the named host.
- Checkpoint commits: `532662e` (prototype and primary results), `889cf4e` (vectorization sensitivity), `8f913ab` (compact validity), `647a1cf` (packed AVX2 batches), `18b9b68` (assume-valid rolling), `bf9b90c` (rolling AVX2 batches), `99e0280` (immediate shifts), and `464b206` (register-native consumption).
- Round 3 commits: `0b22889` (Intel baseline), `395028e` (assume-valid Intel PEXT), `1a812de` (grouped packed PEXT), `b4c8f2f` (rejected scalar windows), `e553c26` (rejected wider AVX2 batches), `5dc09cb` (four-window rolling PEXT), and `ae15781` (rejected validated batching).
- After the user's design decisions, began the isolated public-type implementation in this worktree:
  added `core/wide_kmer.hpp` with fixed-array storage, cast/value access, base/string conversion,
  nucleotide operations, convention conversion, comparison and hashing; declared hash overloads in
  `kmer.hpp` and included the new header from `fisk.hpp`. No sequence or spaced extraction was added.

## Repository findings

- Existing `Kmer`/`SpacedKmer`, callback dispatch, and all public operations assume one `uint64_t` and widths in `[1,32]`.
- ASCII extraction is a one-word rolling recurrence. Spaced extraction rolls the complete span plus a validity word, then applies a 64-bit bit-extract implementation.
- Packed extraction already uses internal `hi:lo` accumulators for boundary handling, but emits one-word k-mers only.
- Existing narrow code can remain untouched by introducing a separate wide header/API. Generalizing the current concepts and callbacks would be broad and risks narrow-path code-generation changes.
- Test host: AMD Ryzen 7 PRO 4750U (BMI2/AVX2), GCC 14.2, Clang 17.0.

## Options evaluated

### 1. Independent fixed-capacity wide type (favored)

Sketch: `WideKmer<MaxK, E, L>` containing `std::array<uint64_t, (2 * MaxK + 63) / 64>` in least-significant-limb-first order; width remains a runtime argument, as for `Kmer`. For the practical first target, `MaxK = 64` gives exactly two words.

- Keeps every existing narrow header/type/hot loop unchanged.
- Generalizes mechanically: rolling shift, complement, reverse, conversion, and per-limb bit extraction are `O(word_count)` loops whose bounds are compile-time constants.
- A matching preprocessed wide mask stores one existing bit-extract table per limb. Results concatenate low limb first; with mask weight <=32 the result remains `SpacedKmer`/`uint64_t`.
- Cost: template/API surface and code size grow with distinct `MaxK` instantiations. Recommend supporting/test-driving 64 first while making storage algorithms generic.

### 2. Dedicated `k <= 64` pair / `__uint128_t`

- Simplest implementation, but `__uint128_t` is not portable to MSVC and was 8--16% slower than the explicit two-word array here.
- A named `hi`/`lo` pair is effectively option 1 specialized to two limbs and forecloses cheap generalization. Strictly dominated by the two-element array unless minimizing initial source size is the only concern.

### 3. Dynamic `std::vector<uint64_t>` arbitrary width

- Supports arbitrary runtime width without template instantiations.
- GCC measured ~4.8x slower than `array<2>` for rolling extraction; Clang scalar-replaced the fixed-size allocation and tied the array, showing fragile compiler dependence. It also adds allocation/ownership questions to every emitted value.
- Not recommended for the hot value type. It could remain an out-of-band representation for truly unbounded data.

### Direct sequence-to-spaced extraction

- ASCII position gathering is best for very sparse masks (weight 4: 3.5--4.6 ns/window) but scales with weight (weight 32: 24--32 ns/window).
- Rolling a two-word span and applying the existing butterfly extractor is stable across weight (6.5--7.8 ns/window). A block-table extractor wins for some sparse or highly clustered masks, reinforcing the existing selector approach.
- From `PackedSequence`, loading a two-word window and applying the existing extraction strategies is the clear general mechanism. Butterfly was 2.45--3.44 ns/window with batching and 5.55--5.61 ns with vectorization disabled; block/position methods can win for sparse or clustered masks. Direct three-limb PEXT was 23--76 ns/window on this Zen 2 host.
- Recommendation: offer both wide-value extraction and a fused spaced iterator. Both should select among position gathering, block tables, butterfly, and hardware PEXT for the complete multi-limb operation. Packed input should use grouped aligned window loads rather than rolling through a public wide value.

## Compatible wide semantics

- Store limbs least-significant first: this makes rolling carry propagation and low-to-high bit extraction simple. Do **not** default `operator<=>` on the array; compare limbs high-to-low to preserve the current raw-numeric ordering.
- Keep runtime `width`, matching `Kmer`. Clear unused high bits at every producer boundary so equality, ordering, and hashing do not depend on garbage capacity bits.
- Implement complement/recode independently per limb, then mask the last limb. Reverse/relayout reverses both bit-pairs within limbs and limb order, followed by a capacity-padding shift when `2 * width` is not a multiple of 64.
- Hash all limbs through a mixing combiner. A lowest-limb-only identity hash would be legal but collision-prone and should not be the default.
- Use a separate wide callback helper and concepts initially. Generalizing `KmerType`, `kmer_value()`, callbacks, and `std::hash` now would touch nearly every narrow operation for little immediate benefit.
- Suggested first API shape: `WideKmer<MaxK, E, L>`, `for_each_wide_kmer<MaxK>()`, `WideSpacedMask<MaxK, Extractor>`, and `for_each_wide_spaced_kmer<MaxK>()`; the latter still emits existing `SpacedKmer<E,L>` while mask weight is <=32.

## Benchmark highlights

Best of 7, 1 MiB valid sequence, pinned core, nanoseconds per produced window/value. Full CSVs are in `experiments/wide_kmers_{gcc,clang}.csv`.

| Case | GCC 14.2 | Clang 17 |
|---|---:|---:|
| Wide ASCII, `array<2>` | 1.44--1.46 | 1.19--1.27 |
| Wide ASCII, `__uint128_t` | 1.68--1.70 | 1.30--1.35 |
| Wide ASCII, `vector<2>` | 6.95--6.96 | 1.20--1.24 |
| Wide-to-spaced, butterfly | 5.43--5.56 | 1.50--1.54 |
| ASCII fused, butterfly | 7.72--7.83 | 6.48--6.55 |
| Packed fused, butterfly | 3.33--3.44 | 2.45--2.47 |
| Existing-style scalar vs generic `array<1>` | 0.855 vs 0.852 | 1.145 vs 1.133 |

With compiler auto-vectorization disabled, wide-array timings were unchanged; isolated butterfly converged to 4.9--5.2 ns and packed butterfly to 5.5--5.6 ns. This confirms the storage conclusion while showing that extractor choice must use the repository's realistic sink modes and selector machinery. The narrow microbenchmark shows no array-loop penalty, but is not enough evidence to refactor existing public types/callbacks. Separate implementation remains the low-risk first step.

## Experimental artifact

- `experiments/wide_kmers.cpp`: standalone correctness checks and benchmarks. It validates spans 33/48/64, weights 4/16/32, invalid-character semantics, ASCII and packed input, both compilers, and all candidate extraction routes.
- The experiment uses existing `CharEncoderTable`, butterfly, and block-table implementations instead of duplicating their algorithms.

## Verification

- Experimental program: GCC 14.2 and Clang 17, `-O3 -march=native -Wall -Wextra -Werror`; internal correctness checks pass.
- Final vector-output candidate: GCC 14.2 and Clang 17, with and without auto-vectorization; all internal correctness checks pass.
- Round 3 experiment: AMD GCC 14.2/Clang 17 and Intel GCC 15.2/Clang 21.1; strict `-Wall -Wextra -Werror` builds and internal correctness checks pass.
- Repository suite: GCC Release, 1/1 CTest passed (5.75 s).
- Repository suite: Clang Release, 1/1 CTest passed (9.38 s).
- Final repository suite rerun: GCC and Clang Release, 1/1 CTest passed for each (9.85 s and 9.82 s while run concurrently).
- Implementation pass: `git diff --check` passed. No build or tests were run; implementation remains uncompiled.

## Uncertainties

- Results are from one AMD Zen 2 machine. Intel PEXT, newer AMD, and ARM behavior may change extractor selection, though not the storage recommendation.
- The sum sink can batch independent outputs. The no-vectorization runs bound that effect, but the production benchmark should still use all existing Sum/Barrier/Write modes.
- The generic one-limb loop matched the narrow scalar prototype, but the experiment does not establish that refactoring existing public narrow types is regression-free.

## Next steps

1. Resume within allowance; review the uncommitted wide type in this worktree and compile/test it.
2. Fix any API or correctness issues, then checkpoint the type before beginning extraction work.
3. Add ASCII rolling and grouped packed-window iterators; emit existing `SpacedKmer` while weight is <=32.
4. Benchmark retained PEXT and AVX2 paths under repository Sum, Barrier, and Write sink shapes.
5. Benchmark on newer AMD and ARM before fixing selector thresholds; consider narrow/wide unification only after assembly and full benchmark equivalence.

## Noticed but out of scope

1. Hardware PEXT is exceptionally poor on this Zen 2 host; any wide implementation that assumes PEXT from BMI2 availability alone will regress badly.
2. Dynamic-vector performance changes sharply with compiler optimization decisions, even when its logical size is always two.

## Research and sub-agents

- No literature phase: this was a repository-specific representation and code-generation question.
- No sub-agents used, as agreed in pre-flight.
- Gaia's named directory was not a Git repository, so a linked worktree could not be created there;
  `.worktrees/wide-kmers-intel` was used as an isolated benchmark workspace populated from the
  local task branch. No paths outside the approved Gaia directory were written.

## Log

- 2026-10-07 13:00 CEST · starting repository and design investigation · codex: OK · 5h 6% (resets in 2h 55m) · 7d 1% (resets Wed 10:55)
- 2026-10-07 13:01 CEST · repository mapping complete; starting isolated prototypes · codex: OK · 5h 6% (resets in 2h 53m) · 7d 1% (resets Wed 10:55)
- 2026-10-07 13:08 CEST · first GCC/Clang comparison complete; refining encoder and isolated extraction measurements · codex: OK · 5h 12% (resets in 2h 47m) · 7d 2% (resets Wed 10:55)
- 2026-10-07 13:13 CEST · prototype and benchmark checkpoint committed (`532662e`); starting full repository verification · codex: OK · 5h 21% (resets in 2h 42m) · 7d 3% (resets Wed 10:55)
- 2026-10-07 13:18 CEST · sensitivity checkpoint committed (`889cf4e`); GCC/Clang repository tests passed; wrapping up · codex: OK · 5h 28% (resets in 2h 37m) · 7d 4% (resets Wed 10:55)
- 2026-10-07 13:19 CEST · complete · codex: OK · 5h 28% (resets in 2h 36m) · 7d 4% (resets Wed 10:55)
- 2026-10-07 13:32 CEST · follow-up improve pass started; baseline candidates ranked · codex: OK · 5h 32% (resets in 2h 23m) · 7d 5% (resets Wed 10:55)
- 2026-10-07 13:35 CEST · one-bit validity optimization committed (`8f913ab`); fresh allowance query unavailable in sandbox, recent cache used · codex: OK · 5h 32% (resets in 2h 19m) · 7d 5% (resets Wed 10:55)
- 2026-10-07 13:39 CEST · explicit AVX2 packed extraction committed (`647a1cf`) · codex: OK · 5h 32% (resets in 2h 16m) · 7d 5% (resets Wed 10:55)
- 2026-10-07 13:44 CEST · assume-valid path committed (`18b9b68`); compiler-specific alternatives rejected · codex: OK · 5h 32% (resets in 2h 11m) · 7d 5% (resets Wed 10:55)
- 2026-10-07 13:50 CEST · four-at-once AVX2 rolling extraction committed (`bf9b90c`); scalar unrolling rejected · codex: OK · 5h 32% (resets in 2h 5m) · 7d 5% (resets Wed 10:55)
- 2026-10-07 13:54 CEST · AVX2 immediate-shift optimization committed (`99e0280`) · codex: OK · 5h 32% (resets in 2h 1m) · 7d 5% (resets Wed 10:55)
- 2026-10-07 16:02 CEST · register-native output optimization committed (`464b206`); resumed after interruption · codex: OK · 5h 3% (resets in 4h 58m) · 7d 16% (resets Wed 10:55)
- 2026-10-07 16:24 CEST · round 3 started; matched AMD/Intel baselines next · codex: OK · 5h 0% (resets in 4h 59m) · 7d 16% (resets Wed 10:55)
- 2026-10-07 21:24 CEST · Intel baseline complete; first round-3 timestamp above was recorded incorrectly · codex: OK · 5h 5% (resets in 4h 55m) · 7d 17% (resets Wed 10:55)
- 2026-10-07 21:27 CEST · assume-valid Intel PEXT path kept; checkpointing · codex: OK · 5h 13% (resets in 4h 52m) · 7d 18% (resets Wed 10:55)
- 2026-10-08 16:39 CEST · wide value type drafted in task worktree; stopping at allowance ceiling · codex: OK · 5h 37% (resets in 1h 12m) · 7d 37% (resets Wed 10:55)
- 2026-10-07 21:36 CEST · grouped packed PEXT kept; unsafe baseline removed · codex: OK · 5h 30% (resets in 4h 43m) · 7d 21% (resets Wed 10:55)
- 2026-10-07 21:39 CEST · scalar-built AVX2 windows rejected on AMD and Intel · codex: OK · 5h 42% (resets in 4h 40m) · 7d 23% (resets Wed 10:55)
- 2026-10-07 21:43 CEST · two-batch AVX2 unrolling rejected on AMD and Intel · codex: OK · 5h 49% (resets in 4h 36m) · 7d 24% (resets Wed 10:55)
- 2026-10-07 21:46 CEST · four-window rolling Intel PEXT kept · codex: OK · 5h 64% (resets in 4h 33m) · 7d 26% (resets Wed 10:55)
- 2026-10-07 21:51 CEST · validated four-window PEXT rejected; no new work after WARN · codex: WARN · 5h 78% (resets in 4h 28m) · 7d 28% (resets Wed 10:55)
- 2026-10-07 21:52 CEST · final GCC/Clang local tests and Intel builds passed; stopped at allowance guard · codex: WARN · 5h 89% (resets in 4h 27m) · 7d 30% (resets Wed 10:55)
