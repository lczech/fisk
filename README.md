<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/fisk-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="logo/fisk-light.png">
  <img alt="fisk logo" src="logo/fisk-light.png">
</picture>

# FISK: Fast Iteration of Spaced K-mers

FISK is a header-only C++20 library for fast extraction of (spaced) k-mers from genomic sequences. It grew out of a research project comparing different implementations for this task; we find that the algorithms implemented here are about an order of magnitude faster than existing approaches for a single spaced k-mer mask, and up to 20 times faster for several masks. The repository also contains the benchmark suite and results used to produce that comparison.

For details, see our preprint:

> **Fast Iteration of Spaced k-mers** <br />
> Lucas Czech  <br />
> arXiv, 2026, doi:[10.48550/arXiv.2603.25417](https://doi.org/10.48550/arXiv.2603.25417)

For questions or bugs, please [open an issue](https://github.com/lczech/fisk/issues).


## Quick start

Add fisk to your CMake project, either via `FetchContent`:

```cmake
include(FetchContent)
FetchContent_Declare(
    fisk
    GIT_REPOSITORY https://github.com/lczech/fisk.git
    GIT_TAG master
)
FetchContent_MakeAvailable(fisk)

target_link_libraries(your_target PRIVATE fisk::fisk)
```

or as a subdirectory, e.g., a git submodule, via `add_subdirectory(path/to/fisk)` and the same `target_link_libraries()`. Then `#include "fisk/fisk.hpp"` for the whole library, or just the headers you need, see the [library reference](include/fisk/README.md).

`for_each_kmer()` calls back with each k-mer of a sequence (and its position), with the algorithm that is likely the fastest by what your build and CPU support, here for a given sequence `seq` and `k=31`:

```cpp
#include "fisk/extraction/for_each.hpp"

using namespace fisk;

for_each_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kMSB>(
    seq, 31, [&](std::size_t pos, KmerAcgtMsb kmer) {
        // ...
    }
);
```

`for_each_spaced_kmer()` does the same for a set of spaced k-mer masks, prepared once in all forms of its algorithms as `SpacedKmerMaskForms`, and calls back as `callback(pos, mask_idx, spaced_kmer)`:

```cpp
SpacedKmerMaskForms<Layout::kMSB> const mask_forms({
    "1101101101101101101101101101101",
    "1011011011011011011011011011011"
});
for_each_spaced_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kMSB>(
    seq, mask_forms, [&](std::size_t pos, std::size_t mask_idx, SpacedKmerAcgtMsb kmer) {
        // ...
    }
);
```

The input is a `std::string_view` (or anything that converts to one, such as a `std::string`), or a [packed sequence](#k-mer-extraction), whose type then gives the template arguments. `InputCheck::kChecked` skips the k-mers with characters other than ACGT, such as N; `kUnchecked` is faster for input without them.

By default, fisk runs on any CPU of its architecture, using SSE2 on x86-64 and Neon on Apple Silicon. For the faster algorithms of AVX2 and PEXT, build for recent x86-64 CPUs, see [Performance and build options](#performance-and-build-options). The [example](example/README.md) is a complete program that uses both functions.


## K-mer extraction

**Encoding and Layout.** `Encoding::kACGT` and `kACTG` set the 2-bit codes of the nucleotides, in that order from 0 to 3. `kACGT` is alphabetical, and its complementary nucleotides are bitwise complements, which makes reverse complements cheap; `kACTG` is cheaper to encode. `Layout::kMSB` puts the first nucleotide of a k-mer in its highest bits, so that with `kACGT` the k-mers sort as their strings do; `kLSB` puts it in the lowest bits, which is cheaper to produce with PEXT. Both are template parameters of the k-mer types, such as `KmerAcgtMsb` for `Kmer<Encoding::kACGT, Layout::kMSB>`.

**Packed sequences.** `PackedSequence<E, L>` stores a sequence at 2 bits per nucleotide, a quarter of its size as ASCII, and `PackedSequenceWithValidity<E, L>` also stores one bit per nucleotide that tells whether it was one of ACGT. `pack_sequence_unchecked<E, L>(seq)` and `pack_sequence_checked<E, L>(seq)` create them from ASCII, up to 2 times faster than existing methods, and both functions take them as input, the latter for the checked extraction. From ASCII, both functions pack the input chunk by chunk anyway, so packing first pays off where a sequence is extracted several times, such as for several values of k or sets of masks, or is kept in memory.

**Choosing the algorithm.** Both functions run the algorithm that is likely the fastest by what the build and the CPU support: the SIMD loop of the widest instruction set of the build of AVX2, SSE2, and Neon, and for spaced k-mers the scalar loop with PEXT where the CPU has a fast one. The AVX-512 algorithms are not among the choices, as with callbacks of single k-mers they are in general not faster than those of AVX2. `for_each_kmer_algorithm_name()` and `for_each_spaced_kmer_algorithm_name()` return the chosen algorithm name. To run another algorithm, call its function directly, see [Algorithms](#algorithms).

**Vectors.** The callback can also take the vectors of the SIMD algorithms directly, for consumers that process the k-mers with SIMD themselves, which saves unpacking them:

```cpp
for_each_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kMSB>(seq, 31, callback_overload{
    [&](std::size_t pos, KmerAcgtMsb kmer) {
        // a single k-mer
    },
    [&](std::size_t pos, auto kmers, auto valid_lanes) {
        // a KmerVector of any SIMD arch
    }
});
```

The algorithms that are not SIMD hand the callback single k-mers. This includes the PEXT choice for spaced k-mers, so that a callback that takes vectors can there be faster with a SIMD algorithm called directly.

**Order.** The k-mers are handed out in order of their positions. For spaced k-mers with several masks, the scalar loops hand out the spaced k-mers of all masks at one position before the next position, and the SIMD loops those of one mask for a whole vector of positions before the next mask.


## Algorithms

The extraction functions that `for_each_kmer()` and `for_each_spaced_kmer()` choose from can also be called directly. They are named by what they extract (`kmer` or `spaced_kmer`), from which input (`ascii` or `packed`), with or without checking the input (`checked` or `unchecked`), and with which `algorithm` and `technique`:

```
for_each_[kmer|spaced_kmer]_[ascii|packed]_[checked|unchecked]_[algorithm]_[technique]()
```

For example, `for_each_kmer_packed_checked_bytewise_scalar()` extracts the valid k-mers of a `PackedSequenceWithValidity`, one at a time. The algorithms are:

 - `rolling` shifts each nucleotide into the k-mer of the previous position. It is the standard approach, but slower due to the data dependency of consecutive steps, takes spaced k-mer masks of spans up to 32 only, and `for_each_kmer()` and `for_each_spaced_kmer()` do not choose it.
 - `bytewise` reads a packed sequence one byte at a time, with one word load for the four k-mers that start in that byte; it is the fastest scalar algorithm.
 - `chunked` packs ASCII input first as chunks into a buffer, and extracts from there with the `bytewise` algorithm.
 - `reextract` builds each k-mer anew from its characters, for comparison only.

The technique is `scalar`, which calls back with single k-mers, or `simd`, which runs on a SIMD arch (SSE2, AVX2, AVX-512, or Neon) and calls back with vectors of k-mers. These combinations exist, as `[algorithm]_[technique]`:

| | `for_each_kmer_…` | `for_each_spaced_kmer_…` |
|---|---|---|
| `ascii_checked` | `rolling_scalar`, `reextract_scalar`, `rolling_simd`, `chunked_scalar`, `chunked_simd` | `rolling_scalar`, `rolling_simd`, `chunked_scalar`, `chunked_simd` |
| `ascii_unchecked` | `chunked_scalar`, `chunked_simd` | `chunked_scalar`, `chunked_simd` |
| `packed_checked` | `bytewise_scalar`, `rolling_scalar`, `bytewise_simd` | `bytewise_scalar`, `bytewise_simd` |
| `packed_unchecked` | `bytewise_scalar`, `rolling_scalar`, `bytewise_simd` | `bytewise_scalar`, `bytewise_simd` |

A spaced k-mer mask is a string such as `"1101101"`, of the positions of a window of `span` nucleotides that the spaced k-mer keeps. Its weight, the number of kept positions, is at most 32. The `bytewise` and `chunked` functions take spans up to a maximum that is set at compile time, 64 by default, and can be raised, see the [library reference](include/fisk/README.md#spaced-k-mer-masks).

**Bit extraction.** Extracting a spaced k-mer gathers the 2-bit codes of the kept positions from the window of the sequence, and packs them densely: a bit extraction. PEXT, an instruction of BMI2 on x86-64, does this in one step, but is slow on AMD CPUs before Zen 3, which `pext_likely_fast()` checks for. Without a fast PEXT, `for_each_spaced_kmer()` uses the butterfly network, which gathers the bits in a fixed number of shift and mask steps, and which also runs on all lanes of a SIMD vector at once. fisk has more algorithms, such as the block shift, with one step per block of consecutive kept positions. The spaced k-mer functions take their masks in the form of a bit extraction algorithm, see the [library reference](include/fisk/README.md#spaced-k-mer-masks), which also shows how to call them.


## Performance and build options

These are optional: fisk works without them, at some cost in speed.

**Instruction sets.** By default, `fisk::fisk` adds no instruction set flags, and a program built with it runs on any CPU of its architecture. The faster algorithms need instruction sets beyond that, which the options below turn on, set before fisk is added, or on the command line, e.g., `-DFISK_ENABLE_AVX2=ON`. Compiler flags such as `-march=x86-64-v3` (which includes AVX2 and BMI2) turn them on as well. Each option adds its compiler flag to all targets that link against `fisk::fisk`, so that the compiler may use these instructions anywhere in their code, and the program then crashes on CPUs without them. Hence, choose them for the oldest CPU that the binary needs to run on: AVX2 and BMI2 are available on most x86-64 CPUs since about 2015, while AVX-512 is not; `-march=native` only fits programs that run where they are built, as the [example](example/CMakeLists.txt) does. Of the instruction sets that are compiled in, fisk checks at runtime which ones the CPU has.

```cmake
set(FISK_ENABLE_AVX2 ON)
set(FISK_ENABLE_BMI2 ON)
FetchContent_MakeAvailable(fisk)  # or add_subdirectory(path/to/fisk)
```

| CMake option | Default | Effect |
|---|---|---|
| `FISK_ENABLE_AVX2`, `FISK_ENABLE_AVX512` | `OFF` | The SIMD algorithms of AVX2 and AVX-512 (F and BW). |
| `FISK_ENABLE_BMI2` | `OFF` | PEXT, for the bit extraction of spaced k-mers. |
| `FISK_ENABLE_SSSE3` | `OFF` | Faster steps of the SSE2 algorithms, such as the packing of ASCII input. |
| `FISK_ENABLE_SSE2`, `FISK_ENABLE_NEON` | `OFF` | Already on with every x86-64 and arm64 compiler, so not needed there. |
| `FISK_DEFAULT_MAX_MASKS` | 4 | Default largest number of spaced k-mer masks in a `std::vector` that the extraction takes; each count up to it compiles the extraction once. |
| `FISK_DEFAULT_MAX_SPAN` | 64 | Default largest span of a spaced k-mer mask. |
| `FISK_DEFAULT_MAX_COMPILED_PIECES` | 2 | Default largest number of pieces of a mask with a span above 32 for which the extraction has its own copy of its loop, i.e., spans up to 57; longer spans run in one loop over the pieces. |
| `FISK_BUILD_BENCHMARKS`, `FISK_BUILD_TESTS` | `ON` when fisk is built on its own, else `OFF` | The [benchmarks](benchmarks/README.md) and the [tests](tests/README.md). |

The ARM target is Apple Silicon; other ARM platforms are untested.

**Compile time.** Each callback type compiles every algorithm that the functions can choose, with aggressive inlining to maximize performance. Thus, the above `max` constants are set to small values, so that only users who need more get longer compile times. For spaced k-mer functions, the call is compiled for each number of masks up to the `MaxMasks` of `SpacedKmerMaskForms`, and each number of pieces up to its `MaxSpan`. Setting smaller values, such as the number of masks you use, or a span of 32 for masks that are not wider, shorten the compile time: `SpacedKmerMaskForms<Layout::kMSB, 2, 32>`.

**Inlining.** The `bytewise` and `chunked` loops, which both functions choose from, call the callback at every step of their unrolled blocks. A small callback should be inlined there, which `FISK_ALWAYS_INLINE_LAMBDA`, placed after the parameter list of a lambda, ensures with GCC, which otherwise tends to keep it as a call. A large callback, inlined into each step, can multiply the code and its compile time; `FISK_NOINLINE_LAMBDA` keeps it as one function, at the cost of a function call per k-mer. Which one is faster depends on the callback, so it is worth measuring with your own code. The extraction loops themselves are forced inline via `FISK_ALWAYS_INLINE_FOR_EACH`; defining it as empty before including fisk shortens the compile time of code with many instantiations of them. All three are in `util/intrinsics.hpp`.

**Aligned input.** The SIMD algorithms read the ASCII input in vectors of up to 64 bytes, which they can read faster when the input starts at a cache line (64 bytes), mostly with AVX-512. `fisk::AlignedString` and `fisk::AlignedVector<T>` are a `std::basic_string` and a `std::vector` whose elements start there, except for strings short enough to be stored inside the object, see `util/aligned_allocator.hpp`; the extraction functions take them as any other string. The buffers of fisk itself, such as those of `PackedSequence`, are always aligned.

**Code alignment.** `-falign-functions=64 -falign-loops=64` (GCC, and Clang 14 or later) start each function and loop at a 64-byte boundary, so that the speed of a hot loop does not depend on where the code around it puts it. The benchmarks and the example build with them. Unfortunately, this does not make them more performant per-se, but it avoids fluctuations in performance due to code alignment, giving stable results.

**Threads.** Each call runs on one thread; fisk does not split a sequence across threads. To use several threads, call the functions for different sequences on each, such as per chromosome or per batch of reads; they can run on several threads at once. A single thread is usually fast enough: on the human genome, the k-mers take about a second, and the spaced k-mers of one mask a few seconds, less than reading the FASTA file.


## Example, benchmarks, and tests

 - [`example/`](example/README.md): A program that extracts the k-mers and spaced k-mers of a FASTA file, and a script that runs it on the human genome.
 - [`benchmarks/`](benchmarks/README.md): The benchmark suite of the manuscript, and how to reproduce its results.
 - [`results/`](results/README.md): The results of the benchmarks, for each CPU and compiler that we tested.
 - [`tests/`](tests/README.md): The test suite, and how to build and run it.
 - [`include/fisk/`](include/fisk/README.md): The library reference: its headers, the spaced k-mer masks, and how to call an algorithm directly.


## Sources

Existing approaches for spaced k-mers:

 - [DuoHash](https://github.com/CominLab/DuoHash)
 - [FISH](https://bitbucket.org/samu661/fish/src/master/)
 - [MISSH](https://github.com/CominLab/MISSH)
 - [CLARK-S](https://github.com/rouni001/CLARK)
 - [MaskJelly](https://github.com/hhaentze/MaskJelly)

Existing approaches for packing sequences:

 - [cute-nucleotides](https://github.com/Daniel-Liu-c0deb0t/cute-nucleotides)
 - [fastdna](https://github.com/eitanf/fastdna)

Information on PEXT, and software implementations of it:

 - [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=pext&ig_expand=5088)
 - [Wikipedia: BMI2](https://en.wikipedia.org/wiki/X86_Bit_manipulation_instruction_set#Parallel_bit_deposit_and_extract)
 - [Félix Cloutier's x86 reference](https://www.felixcloutier.com/x86/pext)
 - Sirrida pages on Bit permutations for [BMI2](https://programming.sirrida.de/bit_perm.html#bmi2) and [Compress and expand](https://programming.sirrida.de/bit_perm.html#c_e)
 - [zp7](https://github.com/zwegner/zp7) and [InstLatX64_Demo](https://github.com/InstLatx64/InstLatX64_Demo)

The code of other projects that the benchmarks compare against is listed in [benchmarks/README.md](benchmarks/README.md#external-comparisons).


## So Long, and Thanks for All the Fish

*Fisk* simply means fish in Danish (as well as Norwegian and Swedish). It seems fitting: I currently live in Copenhagen, and I wanted the `k` of k-mers to make an appearance in the name.

This library, its benchmarks and tests, and its documentation, were developed with the assistance of [Claude Code](https://claude.com/claude-code), which greatly helped in quickly exploring performance optimizations.
