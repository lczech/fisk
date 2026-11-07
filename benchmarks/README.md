# Benchmarks

The benchmark program `fisk_benchmarks`, which produced the results in the manuscript. To build and run it for all compilers and modes, see [Reproducing the benchmarks](../README.md#reproducing-the-benchmarks).

## Running it directly

`make` in the repository root builds it as `bin/fisk_benchmarks`, in the mode of the CMake option `FISK_BENCHMARK_SINK_MODE` (`FISK_BENCHMARK_SINK_SUM` by default). A single run, pinned to one CPU on Linux:

```
taskset -c 2 ./bin/fisk_benchmarks
```

`./bin/fisk_benchmarks --help` lists its options, such as the input (`--input-fasta`, `--input-length`), the value of k (`-k`), and the timing (`--sample-ms`, `--repeats`).

## Measurements

Each suite times variants of an algorithm on the same input, by default 256 Ki random bases from a fixed seed. Each measurement takes whole passes over the input, as many as fill the sample time (10 ms by default), after a short warm-up; each variant is measured several times (5 by default), in turn with the other variants of its case. The CSVs give the time per operation, such as per extracted k-mer: the minimum over the measurements as `ns_per_op`, as well as their median and mean, and the passes per measurement (`rounds`). `sys_info.txt` records the system, the build, and these settings.

The mode (`sink.hpp`) sets what each variant does with its results:

 - `sum` adds them up, which the compiler can combine with the extraction, e.g. vectorize.
 - `barrier` passes each one through an optimization barrier, so that they cannot be combined.
 - `write` stores them in a small buffer, as a consumer that keeps its results does.

## Suites

The suites of the k-mer extraction and the sequence packing are named after the functions they test (see [K-mer extraction functions](../README.md#k-mer-extraction-functions)), and their variants complete that name: for instance, the variant `rolling_scalar_ifs` of the suite `kmer_extract_ascii_checked` is `for_each_kmer_ascii_checked_rolling_scalar()` with the `CharEncoderIfs` encoder. The Encoding and Layout are in the `case` column of the tables.

 - `kmer_extract_<input>_<check>`: k-mers
 - `spaced_kmer_extract_<input>_<check>_single` and `…_multi`: spaced k-mers, one mask or several
 - `spaced_kmer_extract_<input>_<check>_spans`: spaced k-mers, one mask per span from 1 to 128
 - `sequence_pack_<check>`: sequence packing

with `<input>` being `ascii` or `packed`, and `<check>` being `checked` or `unchecked`.

Furthermore:

 - `bit_extract_weights`: Performance of bit extracting, for different weights of the mask.
 - `bit_extract_blocks`: Performance of bit extracting, for masks with different numbers of blocks of consecutive 1s.
 - `char_encoder`: Per-character encoding, from ASCII to 2-bit codes.
 - `kmer_clark`: Small test to examine bottlenecks in the CLARK-S implementation.
 - `extractor_scaling`: The tuned `KmerExtractor` (k = 31) and `SpacedKmerExtractor` (mask W22L31), checked and unchecked, over the length of the input, from 64 bases to 4 Gi bases (`--scaling-max-length`), to show where the caches and the memory set their speed. Each is tuned for its callback, of one k-mer at a time, or also of the vectors of the SIMD algorithms (`callback` in the `case` column), and named after the algorithm that `tune()` chose. Its input is in memory as a whole, 4 GiB by default.

Each suite has a directory here, with its `bench.cpp` and one `var_*.cpp` per variant, each variant in its own translation unit, so that the code of one cannot change how the compiler builds another.
