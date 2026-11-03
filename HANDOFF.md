# Wide k-mer handoff

## Purpose

Branch `feature/wide-kmers` archives the wide k-mer work (k > 32) that is not on `master`:

- The `WideKmer` value type with its operations, casts, hashing and tests (`core/types.hpp`,
  `core/kmer_wide.hpp`, `core/cast.hpp`, `core/kmer_hash.hpp`, `tests/core/`). `master` removed
  it, as nothing extracts wide k-mers from sequences yet. On this branch it is the state of
  `master` just before the removal, i.e. complete, tested and in the current project style.
- The extraction experiments in `experiments/`, as a log of 15 commits with benchmark CSVs.
- [SUMMARY.md](SUMMARY.md), the canonical account of the experiments, benchmark results, rejected
  alternatives, and original next steps, and this handoff.

## Reviving the feature

This branch is reference material, never to be merged or cherry-picked: its experiment commits
must not enter the history of `master`. To bring wide k-mers back:

1. Start a fresh worktree from the then current `master`.
2. Re-implement the feature there in clean commits, using this branch as reference: its
   `WideKmer` files (copying them with `git checkout feature/wide-kmers -- <file>` as a starting
   point is fine, as that copies no history), the prototype in `experiments/` for the algorithms,
   and SUMMARY.md and this handoff for the measured design choices.
3. Re-measure on the current hardware and compilers; the CSVs here are from October 2026.

Spaced k-mers with spans above 32 are no longer part of this: branch `task/wide-spaced-kmers`
(topic doc 10) extracts them from packed sequences with the mask cut into pieces, up to a maximum
span set at compile time, and emits `SpacedKmer`. What remains here is the contiguous extraction
into `WideKmer`, for which that branch's multi-word packed windows may be a template. The
wide-to-spaced items below are superseded by it.

## Commits

The branch was rebased onto `master` (2026-11-03) at the state just before the removal of
`WideKmer`. Its commits only add files, so it can be rebased again without conflicts. The
experiment commits are an experiment log, not a proposed production history: most add benchmark
CSVs and intermediate candidates.

`experiments/wide_kmers.cpp` was written against an older `master` (`373f990`) and no longer
compiles against the current headers, as the bit extract tables have been renamed since (e.g.
`BitExtractButterflyTable` is now `BitExtractButterflyMask`). Read it for the algorithms, or
build it from its own commits. `experiments/wide_kmers_write.cpp` still compiles.

Useful review commands:

```sh
git log --reverse --oneline master..feature/wide-kmers
git show 532662e -- experiments/wide_kmers.cpp
git show 464b206 -- experiments/wide_kmers.cpp
git show 1a812de -- experiments/wide_kmers.cpp
git show 5dc09cb -- experiments/wide_kmers.cpp
```

The source of the final accumulated prototype is
`experiments/wide_kmers.cpp`; how to build/run it is in `experiments/README.md`.

## What the checkpoint establishes

The important conclusion is storage and dispatch, not a ready API:

- A fixed `std::array<uint64_t, N>` wide representation is the recommended hot value type.
  `__uint128_t` was slower on the AMD host and is not portable to MSVC; dynamic vectors had
  fragile compiler-dependent performance.
- BMI2 availability alone must not select PEXT. It is extremely slow on the Zen 2 test host but
  exceptionally fast on the Intel Core 7 240H host. Keep the existing measured-selector model.
- Intel benefits from grouped packed PEXT and four-window ASCII PEXT; Zen 2 instead favors the
  AVX2 butterfly route. Exact measurements and compiler/machine matrix are in `SUMMARY.md`.
- The four-at-once AVX2/register-native figures are lower bounds for a consumer that can receive a
  vector batch. Scalar callback materialization costs must be measured separately.

One benchmark guardrail discovered during review: a sum-only consumer can let compilers optimize
contiguous extraction into something that does not materially produce k-mers. Before accepting a
new result, benchmark a materializing/write-vector sink as well as any sum or register sink. The
repository's established Sum/Barrier/Write modes are the right eventual production benchmark
shape.

## Later design decisions to preserve

These decisions postdate parts of `SUMMARY.md` and supersede its older API sketch:

- The intended public type is `WideKmer<Encoding E, Layout L, std::size_t Capacity = 64>`.
  Capacity is compile-time, measured in bases, may be non-multiple-of-32, and runtime `width` is
  supplied to operations rather than stored in the object.
- The storage invariant is slot-based: word 0 contains slots 0--31, word 1 contains slots 32--63;
  used slots are `[0, width)` and every higher slot is zero. `kLSB` maps input base `i` to slot
  `i`; `kMSB` maps it to slot `width - 1 - i`. The layout documentation should include a 40-base
  example.
- Keep WideKmer separate from the narrow `KmerType` and narrow hot path initially. Reuse public
  function names through constrained overloads only where this cannot alter narrow codegen.
- Wide hashing belongs in the separate `core/kmer_hash.hpp` layer, not in `kmer.hpp`. The narrow
  header should not need knowledge of WideKmer.
- Match project style: no `namespace detail`; internal helpers use trailing underscores; Doxygen
  comments use the project's three-line form; tests reuse `tests/oracle.hpp` helpers rather than
  duplicating reverse-complement code.
- Wide spaced extraction may still emit existing `SpacedKmer` while mask weight is at most 32.

The `WideKmer` on this branch implements these decisions, except that its capacity is a positive
multiple of 32 bases; where it differs from the sketch above, the code and its tests are
authoritative. An earlier uncommitted draft of the type (`core/wide_kmer.hpp`) was discarded.

## Suggested next session sequence

1. Start from a fresh worktree of the current `master`, see "Reviving the feature". Read the
   accumulated prototype only for algorithms/measurements.
2. Bring back the `WideKmer` value type and its tests first, before adding extraction.
3. Add wide contiguous extraction, then wide-to-spaced extraction, keeping output materialization
   explicit in benchmark cases.
4. Reuse the selector for PEXT versus butterfly/block paths; do not encode microarchitecture
   guesses beyond safe instruction availability.
5. Update benchmark CSVs/plots once at the end, rather than committing intermediate runs.

## Future implementation and exploration ideas

Prioritize a correct, materializing baseline before another broad optimization pass:

1. Add WideKmer and SpacedKmer write-vector sink modes to the benchmark harness. A sum-only
   reducer can collapse contiguous extraction into computation that does not materially produce
   k-mers, so it is insufficient evidence for throughput claims.
2. Implement a wide-to-spaced reference path. Preprocess one two-bit selection mask per input
   word, extract each word with the selected existing extractor, then concatenate the partial
   results into the existing `SpacedKmer` while the total mask weight is at most 32. This provides
   the correctness baseline for subsequent selector candidates.
3. Implement wide contiguous extraction independently for each input representation: two-word
   ASCII rolling with compact one-bit validity, and packed grouped-window loads. A fused spaced
   path may consume the latter directly rather than materializing a public WideKmer per window.
4. Integrate wide PEXT, butterfly, block, and position-gather behind measured selection. Do not
   infer PEXT speed from BMI2: PEXT wins on the measured Intel CPU and loses catastrophically on
   the measured Zen 2 CPU.
5. Only after those baselines, evaluate the retained optimizations: four-window AVX2 butterfly
   (especially for vector-batch consumers), grouped packed PEXT, and four-window rolling ASCII
   PEXT on an Intel-selected path.

Do not reopen the following without a new reason or changed hardware/compiler evidence:

- Dynamic `std::vector<uint64_t>` values or `__uint128_t` storage.
- Narrow/wide API unification before narrow-path assembly and benchmarks prove equivalence.
- Scalar-built AVX2 packed windows, wider/two-batch AVX2 loops, or validated four-window PEXT
  batching; all were rejected in the recorded experiments.

## Suggested skills

- `mattpocock-skills:grill-me` before locking a new public API or changing representation
  semantics.
- `mattpocock-skills:diagnose` if a compiler, sink shape, or machine produces a surprising
  performance regression.
- `work-on-task` only for a separate, autonomous experimental pass; do not use it for interactive
  main-branch review work.
