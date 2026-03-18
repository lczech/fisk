# FISK: Fast Iteration of Spaced K-mers

This is a code exploration project to test different implementations to extract spaced k-mers from a genomic sequence. We find that the algorithms implemented here are up to an order of magnitude faster than existing approaches to extract spaced k-mers.


## Build and execute

Simply call

```
make
```

to build the program, and

```
./bin/fisk
```

<!--
With thread pinning:
```
taskset -c 2 ./bin/fisk
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

We provide these results for all hardware architectures tested in `benchmarks`. We also added
the results of our [DuoHash benchmark](https://github.com/lczech/DuoHash) there, called `DuoHash.csv`.

To create the plots from the manuscript, run `./plot/plot_all_cpus.sh`. This requires some standard Python packages to be installed; a conda env file with these packages is provided in `conda-env.yaml`.


## Implementation

The code separates the bit extraction functionality from the k-mer functionality. The bit extraction functions can thus be used in a general context as well.

Functions that process k-mers are mostly templated here, in order to allow us to benchmark different implemenations of, e.g., the nucleotide to two-bit encoding and the bit extraction. Thus, to use these function in your code, you might want to replace those template parameters with hard-coded versions for simplicity.

To use this in our own code, we recommend to use PEXT for bit extraction where possible, and a SIMD-accelerated Butterfly network otherwise. In the simplest case, use `bit_extract_selector()` as a dynamic check to test this, and a switch as shown in `switch_bit_extract_mode()` and `run_bit_extract_mode()` to run. This avoids overhead for function pointers in the hot loop per extracted value.

Similarly, for spaced k-mer extraction, use `spaced_kmer_selector()` to decide which extraction algorithm to use dynamically on a given hardware, and again a switch outside of the hot loop to dispatch between the algorithms.


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
