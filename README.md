<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo/fisk-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="logo/fisk-light.png">
  <img alt="fisk logo" src="logo/fisk-light.png">
</picture>

# FISK: Fast Iteration of Spaced K-mers

FISK is a header-only C++20 library for fast extraction of (spaced) k-mers from genomic sequences. It grew out of a research project comparing different implementations for this task; we find that the algorithms implemented here are up to an order of magnitude faster than existing approaches. The repository also contains the benchmark suite and results used to produce that comparison.

For details, see our preprint:

> **Fast Iteration of Spaced k-mers** <br />
> Lucas Czech  <br />
> arXiv, 2026, doi:[10.48550/arXiv.2603.25417](https://doi.org/10.48550/arXiv.2603.25417)

For questions or bugs, please [open an issue](https://github.com/lczech/fisk/issues).


## Using the library

FISK is a header-only library. Add it to your own CMake project either via `FetchContent`:

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

or as a subdirectory, e.g., as a git submodule:

```cmake
add_subdirectory(path/to/fisk)
target_link_libraries(your_target PRIVATE fisk::fisk)
```

Both approaches add `include/` to your target include path. `#include "fisk/fisk.hpp"` for the whole library, or include just the headers you need (e.g., `fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp`). See [Implementation](#implementation) below for an overview of what is where.

By default, `fisk::fisk` adds no instruction set flags, and a program built with it runs on any CPU of its architecture, using SSE2 on x86-64 and Neon on arm64. The faster code paths need instruction sets beyond that, which are turned on via options, set before fisk is added:

```cmake
set(FISK_ENABLE_AVX2 ON)
set(FISK_ENABLE_BMI2 ON)
FetchContent_MakeAvailable(fisk)  # or add_subdirectory(path/to/fisk)
```

or on the command line, e.g., `-DFISK_ENABLE_AVX2=ON`. The options are `FISK_ENABLE_BMI2` (for PEXT), `FISK_ENABLE_SSE2`, `FISK_ENABLE_SSSE3`, `FISK_ENABLE_SSE41`, `FISK_ENABLE_AVX2`, `FISK_ENABLE_AVX512`, and `FISK_ENABLE_NEON`; each takes effect if the compiler supports it. Compiler flags such as `-march=x86-64-v3` (which includes AVX2 and BMI2) turn on the respective code paths as well.

Each option adds its compiler flag to all targets that link against `fisk::fisk`, so that the compiler may use these instructions anywhere in their code, and the program then crashes on CPUs without them. Hence, choose the options for the oldest CPU that the program needs to run on. For instance, AVX2 and BMI2 are available on most x86-64 CPUs since about 2015, while AVX-512 is not; `-march=native` only fits programs that run where they are built. Of the instruction sets that are compiled in, fisk checks at runtime which ones the CPU has, and picks the algorithm accordingly, see [Tuned extraction](#tuned-extraction).

`FISK_DEFAULT_MAX_MASKS` (default 4) is the default largest number of spaced k-mer masks in a `std::vector` that can be handed on to the extraction, see [K-mer extraction functions](#k-mer-extraction-functions); each count up to it compiles the extraction once. Code that needs more masks passes a larger bound as template parameter, or raises this default. Likewise, `FISK_DEFAULT_MAX_COMPILED_PIECES` (default 2) is the default largest number of pieces of a spaced k-mer mask with a span above 32 for which the extraction has a copy of its loop.

When fisk is built standalone (i.e., not pulled in via `add_subdirectory()`/`FetchContent()` from another project), the benchmark suite and test suite below are built by default; set `-DFISK_BUILD_BENCHMARKS=OFF` and/or `-DFISK_BUILD_TESTS=OFF` to skip them.


## Quick start

`for_each_kmer()` calls back with each k-mer of a sequence, with the algorithm that is likely the fastest by what your build and CPU support:

```cpp
#include "fisk/extractor/for_each.hpp"

using namespace fisk;

for_each_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kMSB>(
    seq, 31, [&](std::size_t pos, KmerAcgtMsb kmer) {
        // ...
    }
);
```

`for_each_spaced_kmer()` does the same for a set of spaced k-mer masks, prepared once as `SpacedKmerMasks`, and calls back as `callback(pos, mask_idx, spaced_kmer)`:

```cpp
SpacedKmerMasks<Layout::kMSB> const masks({
    "1101101101101101101101101101101",
    "1011011011011011011011011011011"
});
for_each_spaced_kmer<InputCheck::kChecked, Encoding::kACGT, Layout::kMSB>(
    seq, masks, [&](std::size_t pos, std::size_t mask_idx, SpacedKmerAcgtMsb kmer) {
        // ...
    }
);
```

The input is a `std::string_view` (or anything that converts to one, such as a `std::string`), or a packed sequence, whose type then gives the template arguments. The algorithm is the estimate of the extractors below, and `for_each_kmer_algorithm_name()` and `for_each_spaced_kmer_algorithm_name()` name it. To measure the fastest algorithm on your hardware and data instead, use the extractors; to call one algorithm directly, see the [k-mer extraction functions](#k-mer-extraction-functions).


## Tuned extraction

The extractors choose the fastest of the [k-mer extraction functions](#k-mer-extraction-functions) on the hardware, and run it with your callback:

```cpp
#include "fisk/extractor/kmer_extractor.hpp"

using namespace fisk;

using Extractor = KmerExtractor<
    InputFormat::kAscii,
    InputCheck::kChecked,
    Encoding::kACGT,
    Layout::kMSB
>;
Extractor extractor(31);
extractor.tune();
for (auto const& read : reads) {
    extractor.for_each(read, [&](std::size_t pos, KmerAcgtMsb kmer) {
        // ...
    });
}
```

`SpacedKmerExtractor` does the same for a set of spaced k-mer masks of the same span, and calls back as `callback(pos, mask_idx, spaced_kmer)`:

```cpp
#include "fisk/extractor/spaced_kmer_extractor.hpp"

using namespace fisk;

using Extractor = SpacedKmerExtractor<
    InputFormat::kPacked,
    InputCheck::kUnchecked,
    Encoding::kACGT,
    Layout::kMSB,
    2  // MaxMasks
>;
Extractor extractor({
    "1101101101101101101101101101101",
    "1011011011011011011011011011011"
});
```

Its last template parameter, `MaxMasks`, is the largest number of masks it takes, `FISK_DEFAULT_MAX_MASKS` (4) by default. Each number of masks up to it compiles the extraction once, so a small `MaxMasks` shortens the compile time a lot, see below. More masks throw `std::invalid_argument` when the extractor is constructed; if the users of your program choose the masks, set `MaxMasks` to the most that it accepts.

The other template parameters set, in this order, the input format (`InputFormat::kAscii` for a `std::string_view`, `InputFormat::kPacked` for a `PackedSequence`, or a `PackedSequenceWithValidity` when checked), whether the input is checked for invalid characters (`InputCheck`, see the check in [K-mer extraction functions](#k-mer-extraction-functions)), and the `Encoding` and `Layout` of the k-mers.

**Choosing the algorithm.** A new extractor uses an estimate. For k-mers, it is the scalar `chunked` or `bytewise` loop, or, for a callback that takes vectors (see below), the SIMD loop of AVX2 or Neon. For spaced k-mers, it is the scalar loop with PEXT, unless the CPU is known for a slow PEXT (see `pext_likely_fast()`), else the SIMD loop with the butterfly kernel of AVX2, SSE2, or Neon. `tune()` times the algorithms that are the fastest on some hardware on 1 Mi random bases, and keeps the fastest; with the defaults, this takes some tens of milliseconds. `TuneOptions` sets the amount of data, the length of its sequences (e.g., of short reads), and its rate of invalid characters. `tune(sample)` times on your data instead, one sequence or a range of them, and `tune(callback)` and `tune(sample, callback)` time with your callback, which is then called during tuning as well. `algorithm_name()` names the choice, after the function that it runs, e.g. `"for_each_spaced_kmer_ascii_checked_chunked_scalar_pext"`, and `set_algorithm(name)` sets it, for instance to reuse a measurement in a later run.

**Vectors.** The callback can also take the vectors of the SIMD algorithms, for consumers that process the k-mers with SIMD themselves. This saves unpacking the lanes, and is usually faster when the consumer works on vectors anyway. The SIMD algorithms then hand it the vectors, and else their k-mers one by one. For example, with `callback_overload`:

```cpp
extractor.for_each(seq, callback_overload{
    [&](std::size_t pos, KmerAcgtMsb kmer) {
        // a single k-mer
    },
    [&](std::size_t pos, auto kmers, auto valid_lanes) {
        // a KmerVector of any SIMD arch
    }
});
```

or with a struct:

```cpp
struct Consumer
{
    void operator()(std::size_t pos, KmerAcgtMsb kmer);

    template <typename Simd>
    void operator()(
        std::size_t pos,
        KmerVector<Encoding::kACGT, Layout::kMSB, Simd> kmers,
        SimdVectorValidLanes<Simd> valid_lanes
    );
};
```

The vector forms are those of the extraction functions: with a count of valid lanes when unchecked, and with `SimdVectorValidLanes` when checked; for spaced k-mers with `mask_idx` after `pos`. Such a callback should also be passed to `tune()`, so that the vectors are part of the measurement.

**Threads.** `for_each()` is `const`, and one extractor can be used by several threads at once, but not while `tune()` or `set_algorithm()` run. ASCII input is packed chunk by chunk into a buffer per thread.

**Order and compile time.** The k-mers are handed out in order of their positions; spaced k-mers in order of their positions per mask, while the order between masks differs between the algorithms: the scalar algorithms hand out the spaced k-mers of all masks at one position before going on to the next position, while the SIMD algorithms hand out those of one mask for a whole vector of positions, then those of the next mask for the same positions. Each callback type compiles all algorithms of the extractor, and for spaced k-mers each of them for each number of masks up to `MaxMasks`.


## Reproducing the benchmarks

To build and run the benchmark suite that produced the results in the manuscript, simply call

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

This will produce performance measurements for all implemented benchmarks on the given hardware, measuring time per operation, such as per extracted k-mer, with one CSV file per benchmark suite. The suites of the k-mer extraction and the sequence packing are named after the functions they test (see [K-mer extraction functions](#k-mer-extraction-functions) below), and their variants complete that name: for instance, the variant `rolling_scalar_ifs` of the suite `kmer_extract_ascii_checked` is `for_each_kmer_ascii_checked_rolling_scalar()` with the `CharEncoderIfs` encoder. The Encoding and Layout are in the `case` column of the tables.

 - `kmer_extract_<input>_<check>`: k-mers
 - `spaced_kmer_extract_<input>_<check>_single` and `…_multi`: spaced k-mers, one mask or several
 - `spaced_kmer_extract_packed_<check>_spans`: spaced k-mers, one mask per span from 1 to 128
 - `sequence_pack_<check>`: sequence packing

with `<input>` being `ascii` or `packed`, and `<check>` being `checked` or `unchecked`.

Furthermore:

 - `bit_extract_weights`: Performance of bit extracting, for different weights of the mask.
 - `bit_extract_blocks`: Performance of bit extracting, for masks with different numbers of blocks of consecutive 1s.
 - `char_encoder`: Per-character encoding, from ASCII to 2-bit codes.
 - `kmer_clark`: Small test to examine bottlenecks in the CLARK-S implementation.

We provide these results for all hardware architectures tested in `results`. We also added
the results of our [DuoHash benchmark](https://github.com/lczech/DuoHash) there, called `DuoHash.csv`.

To create the plots from the manuscript, run `./plot/plot_all_cpus.sh`. This requires some standard Python packages to be installed; a conda env file with these packages is provided in `plot/conda-env.yaml`.


## Tests

The test suite can be built and run via CMake/CTest:

```
cmake -B build && cmake --build build -j
ctest --test-dir build
```

or directly via the resulting binary: `./bin/fisk_tests`.


## Implementation

The repository separates the library (bit extraction and k-mer functionality, for general reuse) from the benchmark program used to produce the results in the manuscript:

 - `include/fisk/`: The library. Header-only, organized by theme; `#include` the specific headers you need (e.g., `fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp`), or `fisk/fisk.hpp` for everything at once.
 - `benchmarks/`: The benchmark program (`main.cpp` and driver code) used to produce the results in the manuscript. Not needed to use the library itself.
 - `tests/`: Unit test suite (built as `fisk_tests`, see [Tests](#tests) above). Not needed to use the library itself.
 - `results/`: Recorded benchmark results for all hardware architectures we tested.

Overview of the library headers (in `include/fisk/`), and their most important functions and algorithms:

 - `core/`: Shared building blocks. `char_encoder.hpp` for per-character nucleotide-to-2-bit encoding (the `CharEncoder` family, templated on the `Encoding` they produce); `types.hpp` for the shared `Encoding`/`Layout` conventions and the types that carry them: `PackedSequence`, `PackedSequenceWithValidity`, which also records which nucleotides were valid, both with zero padding after their data so that the extraction loops can read whole words up to the end, the k-mer types `Kmer` and `SpacedKmer` (emitted by spaced k-mer extraction), `SpacedKmerMask` for masks tagged with the `Layout` and span they extract, up to a maximum span (default 64) as a template parameter, with spans above 32 cut into pieces, the SIMD vectors `KmerVector` and `SpacedKmerVector` of the two, and `SimdVectorValidLanes` for which of their lanes are valid, as bits; `cast.hpp` for the casts between raw words and the k-mer types, such as `kmer_cast()`; `kmer.hpp` for the encoding, decoding, reverse-complement, canonicalization and conversion operations of `Kmer` and `SpacedKmer`; `kmer_spaced.hpp` for spaced k-mer masks as strings and bits (`spaced_kmer_weight()`, `is_valid_spaced_kmer_mask()`, etc.), and the encoding and decoding of `SpacedKmer` with its mask (`spaced_kmer_encode()`, `spaced_kmer_decode()`); `kmer_hash.hpp` for hashing both, including `std::hash` support.
 - `util/`: General utilities that are not specific to k-mers. `platform.hpp` for the host assumptions fisk asserts, such as little-endianness; `intrinsics.hpp`/`cpu_runtime.hpp` for compile-time and runtime CPU feature detection; `random.hpp` for a fast PRNG used internally by the selector.
 - `simd/`: The SIMD archs. `simd.hpp` for their concepts, the stateless vector operations that the SIMD kernels are built on, with one header per arch (`scalar.hpp`, `sse2.hpp`, `avx2.hpp`, `avx512.hpp`, `neon.hpp`).
 - `bit_extract/bit_extract.hpp`: Main bit extraction functions. This is probably the most relevant part, containing the core algorithms.
 - `bit_extract/simd.hpp`: SIMD kernels of the bit extract algorithms, templated on the SIMD arch (e.g., `BitExtractKernelButterfly<SimdAVX2>`).
 - `bit_extract/selector.hpp`: Helper that runs a quick benchmark to find the most performant bit extraction algorithm for a given mask.
 - `sequence_pack/`: Packing of ASCII sequences into a compact 2-bit-per-base `PackedSequence`, 8 characters at a time via a word encoder. `pack_sequence_unchecked()` takes the input as valid, while `pack_sequence_checked()` also records one validity bit per nucleotide, in a `PackedSequenceWithValidity`. The word encoders are in `word_encoder.hpp`: the `WordEncoder` family, using the same PEXT/butterfly building blocks as bit extraction, the simple `WordEncoderFold`, or the fastest `WordEncoderDirect` and `WordEncoderDirectDouble`. All are templated on the SIMD arch, to pack several words at once (e.g., `WordEncoderDirect<Encoding::kACGT, Layout::kMSB, SimdAVX2>`).
 - `extractor/`: `KmerExtractor` and `SpacedKmerExtractor`, which choose the fastest extraction function, see [Tuned extraction](#tuned-extraction).
 - `kmer_extract/` and `spaced_kmer_extract/`: The extraction of k-mers and of spaced k-mers, see below. `spaced_kmer_extract/mask.hpp` has the masks: they are created from a string of 1s and 0s via `make_spaced_kmer_mask()` and the related functions for the bit extraction tables (`make_spaced_kmer_butterfly_mask()`, etc.) and SIMD kernels (`make_spaced_kmer_kernel()`), for spaced k-mers in either `Layout` (e.g., `make_spaced_kmer_mask<Layout::kLSB>()`).

### K-mer extraction functions

The extraction functions are named by what they extract, from which input, with or without checking the input, and with which algorithm and technique:

```
for_each_<kmer|spaced_kmer>_<ascii|packed>_<checked|unchecked>_<algorithm>_<technique>()
```

For example, `for_each_kmer_packed_checked_bytewise_scalar()` extracts the valid k-mers of a `PackedSequenceWithValidity`, one at a time. The parts are:

 - What: `kmer` for contiguous k-mers (`Kmer`), `spaced_kmer` for spaced k-mers (`SpacedKmer`), which take a mask, or a set of masks of the same span, with a bit extraction function (scalar) or kernel (SIMD), see `bit_extract/`.
 - Input: `ascii` for a sequence of characters (`std::string_view`), `packed` for a `PackedSequence` or `PackedSequenceWithValidity`.
 - Check: `checked` emits only the (spaced) k-mers whose nucleotides are valid: from ASCII, the characters `ACGTacgt`; from a `PackedSequenceWithValidity`, by its validity bits. A spaced k-mer is valid if the positions that its mask keeps are valid; invalid characters, such as `N`, at the positions that the mask skips do not matter. `unchecked` takes the input as valid and is faster: an invalid character is encoded as one of the four nucleotides, changing every k-mer that overlaps it. The unchecked functions on packed input also take a `PackedSequenceWithValidity`, and ignore its validity bits.
 - Algorithm: `rolling` shifts each nucleotide into the k-mer of the previous position; `reextract` builds each k-mer anew from its characters, for comparison only; `bytewise` reads the packed sequence one byte at a time, with one word load for the four k-mers that start in that byte, and is the fastest scalar algorithm; `chunked` packs the ASCII input chunk by chunk into a buffer, and extracts from there with the `bytewise` algorithm; the buffer is given by the caller, or else is a thread-local one, and the chunk size is a template parameter (`kDefaultAsciiChunkSize` by default).
 - Technique: `scalar` calls back with single k-mers; `simd` runs on a SIMD arch, given as template parameter (e.g., `SimdAVX2`) or by the SIMD kernel of the masks, and calls back with vectors of k-mers (`KmerVector`, `SpacedKmerVector`) and their count, or their valid lanes (`SimdVectorValidLanes`) when checked. Only the contiguous `rolling_simd` calls back with single k-mers, as it uses the arch just to encode and check the characters.

The vectors are meant for consumers that process the k-mers with SIMD themselves. For single k-mers, the scalar functions are faster than taking them out of the vectors with `for_each_valid_lane()`. The Encoding and Layout of the k-mers are those of the packed sequence, of the encoder or masks, or template parameters, and are part of their types.

These combinations exist, as `<algorithm>_<technique>`:

| | `for_each_kmer_…` | `for_each_spaced_kmer_…` |
|---|---|---|
| `ascii_checked` | `rolling_scalar`, `reextract_scalar`, `rolling_simd`, `chunked_scalar`, `chunked_simd` | `rolling_scalar`, `rolling_simd`, `chunked_scalar`, `chunked_simd` |
| `ascii_unchecked` | `chunked_scalar`, `chunked_simd` | `chunked_scalar`, `chunked_simd` |
| `packed_checked` | `bytewise_scalar`, `rolling_scalar`, `bytewise_simd` | `bytewise_scalar`, `bytewise_simd` |
| `packed_unchecked` | `bytewise_scalar`, `rolling_scalar`, `bytewise_simd` | `bytewise_scalar`, `bytewise_simd` |

Each `<input>_<algorithm>_<technique>` has its own header in `kmer_extract/` or `spaced_kmer_extract/`, with its checked and unchecked functions, e.g. `kmer_extract/packed_bytewise_scalar.hpp`. The internals that several of them share are in an `internal.hpp` in each directory, and the chunking that all `chunked` ones share in `kmer_extract/ascii_chunked.hpp`.

The spaced k-mer functions take a set of masks as a `std::span` of compile-time size, so that their loops over the masks are unrolled; an array is handed on as `std::span{array}`. `dispatch_static_span()` (`util/static_span.hpp`) hands on a `std::vector` of masks, as made by `make_spaced_kmer_masks()`, as such a span of its size, which can be up to `kDefaultMaxMasks` (4 by default, see the CMake option `FISK_DEFAULT_MAX_MASKS` above):

```cpp
auto const masks = fisk::make_spaced_kmer_masks({"1101", "1011"});
fisk::dispatch_static_span<fisk::kDefaultMaxMasks>(masks, [&](auto mask_span) {
    fisk::for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
        packed, mask_span, fisk::bit_extract_pext,
        [&](std::size_t pos, std::size_t mask_idx, auto const& spaced_kmer) {
            // ...
        }
    );
});
```

The masks have a maximum span as their last template parameter, 64 by default, e.g. `make_spaced_kmer_mask<fisk::Layout::kMSB, 128>(mask)`; their weight is at most 32 for any span. Of the spaced k-mer functions, `for_each_spaced_kmer_packed_<check>_bytewise_scalar()` with a single mask takes spans up to that maximum, and the others spans up to 32. Above 32, a span is cut into pieces: leading pieces of 28 positions and a tail of up to 29. The functions that take longer spans have a copy of their loop for each number of pieces up to their first template parameter `MaxCompiledPieces`, `FISK_DEFAULT_MAX_COMPILED_PIECES` (2) by default, i.e., spans up to 57, and run longer spans in one loop over the pieces.

The `bytewise` and `chunked` functions call the callback at every step of their unrolled loops. A small callback should be inlined there, which `FISK_ALWAYS_INLINE_LAMBDA` (`util/intrinsics.hpp`, placed after the parameter list) ensures with GCC, which otherwise tends to keep it as a call. A large callback, on the other hand, inlined into each step, can multiply the code and its compile time many times over, in particular with clang before version 18. `FISK_NOINLINE_LAMBDA` keeps it as one function, at the cost of a call per callback. Which one is faster depends on the callback, so it is worth measuring with your own code.

The `benchmarks/` directory contains the driver code for the benchmark program: `main.cpp`, `arg_parser.hpp`, `microbench.hpp`, `sink.hpp`, `utils.hpp`, `seq_data.hpp`, and one directory per suite, with its `bench.cpp` and one `var_*.cpp` per variant. These might be useful to see how each algorithm is intended to be called and used.

Functions that process k-mers are mostly templated here, in order to allow us to benchmark different implemenations of, e.g., the nucleotide to two-bit encoding and the bit extraction. Thus, to use these function in your code, you might want to replace those template parameters with hard-coded versions for simplicity - no need to use any of the sub-par alterative implementations if you can just use the fastest one.

To use this in our own code, we recommend to use PEXT for bit extraction where possible, and a SIMD-accelerated Butterfly network otherwise. For the extraction of k-mers, the extractors of [Tuned extraction](#tuned-extraction) make this choice. In the simplest case, use `bit_extract_selector()` as a dynamic check to test this, and a switch as shown in `switch_bit_extract_mode()` and `run_bit_extract_mode()` to run. This avoids overhead for function pointers in the hot loop per extracted value.

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

## So Long, and Thanks for All the Fish

*Fisk* simply means fish in Danish (as well as Norwegian and Swedish). It seems fitting: I currently live in Copenhagen, and I wanted the `k` of k-mers to make an appearance in the name.

This library, its benchmarks and tests, and its documentation, were developed with the assistance of [Claude Code](https://claude.com/claude-code) and [Codex](https://openai.com/codex/).
