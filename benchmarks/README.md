# Benchmarks

The benchmark program `fisk_benchmarks`, which produced the results in the manuscript.

## Reproducing the results

To build and run the benchmarks for all compilers and modes, call, in the main directory of the repository,

```
scripts/run_benchmarks.sh
```

This needs CMake (3.20 or later) and a C++20 compiler; on macOS, the Xcode command line tools, and CMake, e.g., from Homebrew. For stable timings, run it on mains power, with other programs closed.

The script builds and runs the benchmarks with GCC and Clang on Linux, as far as they are installed, or with Apple Clang on macOS, each in the three modes of handling the results, see [Measurements](#measurements). Each run is pinned to one CPU core on Linux. On the human genome, which it downloads once into `example/data/`, it also runs the genome sweep (see [Suites](#suites)) in each mode, and the [example](../example/README.md) with each compiler; this needs curl, and about 5 GB of disk space and 4 GB of memory. `--skip-genome` leaves both out. `scripts/run_benchmarks.sh --help` lists all options.

The results go to `results/<CPU>/<Compiler>_<version>/`, such as `results/Intel_Core_7_240H/GCC_15/`:

 - `<mode>/`: one CSV per suite, `sys_info.txt` with the system, the build, and the benchmark settings, and `log.txt` with the output; `genome_sweep.csv`, with its own `genome_sweep_sys_info.txt` and `genome_sweep_log.txt`.
 - `human_genome.csv`: the output of the example on the human genome.
 - `DuoHash.csv`: the results of the DuoHash benchmark, see [External comparisons](#external-comparisons), which is run separately.

Existing results of the same CPU, compiler, and mode are moved to `results/_archive/<date>/` first. The repository has these results for all CPUs that we tested, as CSV and text files. The plots are not in git; to make them, see [plot/README.md](../plot/README.md).

## Running the program directly

`make` in the main directory of the repository builds the benchmarks and the tests, as `bin/fisk_benchmarks` and `bin/fisk_tests`. The benchmarks are built in the mode of the CMake option `FISK_BENCHMARK_SINK_MODE` (`FISK_BENCHMARK_SINK_SUM` by default). A single run, pinned to one CPU on Linux:

```
taskset -c 2 ./bin/fisk_benchmarks
```

`./bin/fisk_benchmarks --help` lists its options, such as the input (`--input-fasta`, `--input-length`), the value of k (`-k`), the masks (`--masks-file`), and the timing (`--sample-ms`, `--repeats`). Without `--masks-file`, the program reads its masks from `masks/` in the repository, next to `bin/`.

## Measurements

Each suite times variants of an algorithm on the same input: by default, 256 Ki random bases from a fixed seed. The exceptions are the `bit_extract_*` suites, which make random words and masks, `for_each_scaling`, which makes its own input of each length, and `genome_sweep`, which reads its FASTA file. Each measurement takes whole passes over the input, as many as fill the sample time (10 ms by default), after a short warm-up; each variant is measured several times (5 by default), in turn with the other variants of its case. The CSVs give the time per operation, such as per extracted k-mer: the minimum over the measurements as `ns_per_op`, as well as their median and mean, and the passes per measurement (`rounds`).

The mode (`sink.hpp`) sets what each variant does with its results:

 - `sum` adds them up, which the compiler can combine with the extraction, e.g., vectorize.
 - `barrier` passes each one through an optimization barrier, so that they cannot be combined.
 - `write` stores them in a small buffer, as a consumer that keeps its results does.

The `sequence_pack_*` and `kmer_clark` suites do not use the mode, and their `sink` column is always `sum`.

## Suites

| Suite | Measures |
|---|---|
| `kmer_extract_<input>_<check>` | The extraction of k-mers. |
| `spaced_kmer_extract_<input>_<check>_single`, `…_multi` | The extraction of spaced k-mers, for one mask, or several. |
| `spaced_kmer_extract_<input>_<check>_spans` | The extraction of spaced k-mers, for one mask per span from 1 to 128. |
| `sequence_pack_<check>` | The packing of ASCII input, with each word encoder. |
| `char_encoder` | The per-character encoding, from ASCII to 2-bit codes. |
| `bit_extract_weights` | The bit extraction, for masks of each weight. |
| `bit_extract_blocks` | The bit extraction, for masks of each number of blocks of consecutive 1s. |
| `kmer_clark` | The spaced k-mer code of CLARK-S, to examine its bottlenecks. |
| `for_each_scaling` | `for_each_kmer()` and `for_each_spaced_kmer()` over the length of the input. |
| `genome_sweep` | The algorithms of practical use, on the human genome. |

`<input>` is `ascii` or `packed`, and `<check>` is `checked` or `unchecked`. The extraction suites are named after the functions they test (see [Algorithms](../README.md#algorithms)), and their variants complete that name: for instance, the variant `rolling_scalar_ifs` of the suite `kmer_extract_ascii_checked` is `for_each_kmer_ascii_checked_rolling_scalar()` with the `CharEncoderIfs` encoder. The `sequence_pack_<check>` suites test `pack_sequence_<check>()`, with variants named after their word encoder. The Encoding and Layout are in the `case` column.

**`for_each_scaling`** runs k-mers (k = 31) and spaced k-mers (mask W22L31), checked and unchecked, from 64 bases up to 4 Gi bases (`--scaling-max-length`), to show where the caches and the memory set their speed, and what a call costs on short input. Its input is in memory as a whole, 4 GiB by default. Each runs with a callback of one k-mer at a time, or also of the vectors of the SIMD algorithms (`callback` in the `case` column).

**`genome_sweep`** runs with `--genome-sweep --input-fasta <fasta>`, and then only this suite. It times k-mers (k = 31) and spaced k-mers (mask W22L31), checked, from ASCII and from packed input, as well as the packing: each algorithm as the variant of its suite, under the same name, and `for_each_kmer` and `for_each_spaced_kmer` alongside, with their choice in the `case` column. Each time is the best of 3 single passes over the genome. The algorithms with PEXT only run where it is likely fast, see `pext_likely_fast()`.

Each suite has a directory here, with its `bench.cpp` and one `var_*.cpp` per variant, each in its own translation unit, so that the code of one cannot change how the compiler builds another. The shared driver code is in the files at the top level of this directory.

## External comparisons

The benchmarks compare against the code of other projects, in [external/](external/), each with its source and license in its header:

| Code | Suites |
|---|---|
| [zp7](https://github.com/zwegner/zp7) and [InstLatX64_Demo](https://github.com/InstLatx64/InstLatX64_Demo), software PEXT | `bit_extract_weights`, `bit_extract_blocks` |
| [CLARK](https://github.com/rouni001/CLARK), spaced k-mers of CLARK-S (GPL v3) | `kmer_clark` |
| [cute-nucleotides](https://github.com/Daniel-Liu-c0deb0t/cute-nucleotides) and [fastdna](https://github.com/eitanf/fastdna), packing of ASCII input | `sequence_pack_unchecked` |

The spaced k-mer methods of [DuoHash](https://github.com/CominLab/DuoHash) are timed by a benchmark program of their own, in [our fork of DuoHash](https://github.com/lczech/DuoHash), whose results are the `DuoHash.csv` files in `results/`.
