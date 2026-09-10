<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/fisk-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="logo/fisk-light.png">
  <img alt="fisk logo" src="logo/fisk-light.png">
</picture>

# FISK: Fast Iteration of Spaced K-mers

This is a code exploration project to test different implementations to extract spaced k-mers from a genomic sequence. We find that the algorithms implemented here are up to an order of magnitude faster than existing approaches to extract spaced k-mers.

For details, see our preprint:

> **Fast Iteration of Spaced k-mers** <br />
> Lucas Czech  <br />
> arXiv, 2026, doi:[10.48550/arXiv.2603.25417](https://doi.org/10.48550/arXiv.2603.25417)

For questions or bugs, please [open an issue](https://github.com/lczech/fisk/issues).


## Build and execute

Simply call

```
make
```

to build the benchmark program, and

```
./bin/fisk_benchmarks
```

to run it.

<!--
With thread pinning:
```
taskset -c 2 ./bin/fisk_benchmarks
```
-->

This will produce performance measurements for all implemented benchmarks on the given hardware, measuring time per operation, such as per extracted k-mer:

 - `bit_extract_weights.csv`: Performance of bit extracting, for different weights of the mask.
 - `bit_extract_blocks.csv`: Performance of bit extracting, for masks with different numbers of blocks of consecutive 1s.
 - `seq_enc.csv`: Sequence encoding, from ASCII to 2-bit codes.
 - `kmer_extract.csv`: Regular k-mer extraction, traversing along an input sequence.
 - `kmer_spaced_single.csv`: Spaced k-mer extraction, using a single mask.
 - `kmer_spaced_multi.csv`: Spaced k-mer extraction, for multiple masks at once.
 - `kmer_clark.csv`: Small test to examine bottlenecks in the CLARK-S implementation.

We provide these results for all hardware architectures tested in `results`. We also added
the results of our [DuoHash benchmark](https://github.com/lczech/DuoHash) there, called `DuoHash.csv`.

To create the plots from the manuscript, run `./plot/plot_all_cpus.sh`. This requires some standard Python packages to be installed; a conda env file with these packages is provided in `conda-env.yaml`.


## Implementation

The repository separates the library (bit extraction and k-mer functionality, for general reuse) from the benchmark program used to produce the results in the manuscript:

 - `include/fisk/`: The library. Header-only, organized by theme; `#include` the specific headers you need (e.g., `fisk/kmer_spaced/kmer_spaced_selector.hpp`), or `fisk/fisk.hpp` for everything at once.
 - `benchmarks/`: The benchmark program (`main.cpp` and driver code) used to produce the results in the manuscript. Not needed to use the library itself.
 - `results/`: Recorded benchmark results for all hardware architectures we tested.

Overview of the library headers (in `include/fisk/`), and their most important functions and algorithms:

 - `core/`: Shared building blocks. `seq_enc.hpp` for nucleotide-to-2-bit encoding; `intrinsics.hpp`/`cpu_runtime.hpp` for compile-time and runtime CPU feature detection; `random.hpp` for a fast PRNG used internally by the adaptive/selector algorithms.
 - `bit_extract/bit_extract.hpp`: Main bit extraction functions. This is probably the most relevant part, containing the core algorithms.
 - `bit_extract/bit_extract_simd.hpp`: SIMD implementations of the bit extract algorithms.
 - `bit_extract/bit_extract_selector.hpp`: Helper that runs a quick benchmark to find the most performant bit extraction algorithm for a given mask.
 - `bit_extract/bit_extract_adaptive.hpp`, `bit_extract_instlatx64.hpp`, `bit_extract_zp7.hpp`: Alternative implementations of the selector and of bit extraction algorithms. Not recommended, but kept here for reference.
 - `kmer_extract/kmer_extract.hpp`: Basic extraction loop of k-mers from a sequence.
 - `kmer_extract/kmer_extract_simd.hpp`: SIMD variant of the rolling extraction, probably overkill for most use cases.
 - `kmer_spaced/kmer_spaced.hpp`: Extraction loop for spaced k-mers from a sequence, templated with the bit extract function. Also contains the naive implementation, and some helper functions, e.g., to prepare the mask from a string of 1s and 0s.
 - `kmer_spaced/kmer_spaced_simd.hpp`: SIMD variant of the spaced k-mer extraction, taking one of the `bit_extract_simd.hpp` implementations as template parameter.
 - `kmer_spaced/kmer_spaced_selector.hpp`: Helper that runs a quick benchmark to find the most performant spaced k-mer extraction algorithm for a given mask. Similar to `bit_extract_selector.hpp`.

The `benchmarks/` directory contains the driver code for the benchmark program: `main.cpp`, `arg_parser.hpp`, `microbench.hpp`, `utils.hpp`, `seq_data.hpp`, `kmer_clark.hpp`, and one `bench_*.hpp` per benchmark shown in the manuscript. These are not part of the library, but might be useful to see how each algorithm is intended to be called and used.

Functions that process k-mers are mostly templated here, in order to allow us to benchmark different implemenations of, e.g., the nucleotide to two-bit encoding and the bit extraction. Thus, to use these function in your code, you might want to replace those template parameters with hard-coded versions for simplicity - no need to use any of the sub-par alterative implementations if you can just use the fastest one.

To use this in our own code, we recommend to use PEXT for bit extraction where possible, and a SIMD-accelerated Butterfly network otherwise. In the simplest case, use `bit_extract_selector()` as a dynamic check to test this, and a switch as shown in `switch_bit_extract_mode()` and `run_bit_extract_mode()` to run. This avoids overhead for function pointers in the hot loop per extracted value.

Similarly, for spaced k-mer extraction, use `spaced_kmer_selector()` to decide which extraction algorithm to use dynamically on a given hardware, and again a switch outside of the hot loop to dispatch between the algorithms.

Note that the SIMD implementations need to be guarded by hardware checks to ensure not calling invalid intrinsics. We mostly solve this here via preprocessor checks (e.g., for AVX). These checks likely need to be adapted as needed for your build system. They can be adapted to perform a dynamic check, allowing to cross-compile for different hardware architectures.


## Sources

Existing approaches for spaced k-mers:

 - [DuoHash](https://github.com/CominLab/DuoHash), and [our fork](https://github.com/lczech/DuoHash) containing a simple benchmark program for their methods
 - [FISH](https://bitbucket.org/samu661/fish/src/master/)
 - [MISSH](https://github.com/CominLab/MISSH)
 - [CLARK-S](https://github.com/rouni001/CLARK)
 - [MaskJelly](https://github.com/hhaentze/MaskJelly)


Information on PEXT:

 - [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=pext&ig_expand=5088)
 - [Wikipedia: BMI2](https://en.wikipedia.org/wiki/X86_Bit_manipulation_instruction_set#Parallel_bit_deposit_and_extract)
 - [Félix Cloutier's x86 reference](https://www.felixcloutier.com/x86/pext)
 - Sirrida pages on Bit permutations for [BMI2](https://programming.sirrida.de/bit_perm.html#bmi2) and [Compress and expand](https://programming.sirrida.de/bit_perm.html#c_e)

<!--
Unused implementations
https://github.com/InstLatx64/InstLatX64_Demo/blob/master/PEXT_PDEP_Emu.cpp
https://github.com/zwegner/zp7/blob/master/zp7.c
-->
