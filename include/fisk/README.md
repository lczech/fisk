# Library reference

The spaced k-mer masks, how to call an extraction algorithm directly, and an overview of the functions and types of fisk. For `for_each_kmer()` and `for_each_spaced_kmer()`, which are enough for most uses, see the [main README](../../README.md).

`fisk/fisk.hpp` includes all headers; the [overview](#functions-and-types) names the header of each function, and the headers document them in full.

## Spaced k-mer masks

A spaced k-mer mask says which positions of a window of `span` nucleotides a spaced k-mer keeps, and is given as a string such as `"1101101"`. For `for_each_spaced_kmer()`, `SpacedKmerMaskForms` prepares a set of masks from their strings, in all forms that it uses, as it chooses its algorithm at runtime:

```cpp
fisk::SpacedKmerMaskForms<fisk::Layout::kMSB> const mask_forms({"1101101", "1011011"});
```

The extraction functions that are called directly take their masks in the form of their bit extraction algorithm instead, which these functions create:

| Form | For | One mask | A set of masks |
|---|---|---|---|
| `BitExtractMask` | PEXT and the other scalar algorithms | `make_spaced_kmer_mask()` | `make_spaced_kmer_masks()` |
| `BitExtractButterflyMask` | The scalar butterfly | `make_spaced_kmer_butterfly_mask()` | `make_spaced_kmer_butterfly_masks()` |
| `BitExtractBlockShiftMask` | The scalar block shift | `make_spaced_kmer_block_shift_mask()` | `make_spaced_kmer_block_shift_masks()` |
| A kernel, such as `BitExtractKernelButterfly<SimdAVX2>` | The SIMD functions | `make_spaced_kmer_kernel<Kernel>()` | `make_spaced_kmer_kernels<Kernel>()` |

The functions for one mask take its string, or a mask in the form `BitExtractMask`. Of those for a set of masks, `make_spaced_kmer_masks()` takes the strings, and the others convert the result of it, a `std::vector` of masks in the form `BitExtractMask`. The ones that take strings take the `Layout` of the spaced k-mers as template parameter, `Layout::kMSB` by default, e.g., `make_spaced_kmer_mask<Layout::kLSB>("1101101")`; the others keep the Layout of their input.

**Forms.** Bit extraction gathers the bits of the kept positions. It has two axes: the algorithm (PEXT, butterfly, block shift, and others), and whether it runs as a plain scalar function on one word, or on a SIMD arch, on all lanes of a vector at once. "Mask" has three meanings here, which the documentation names in full where they could be confused:

 - Spaced k-mer mask: `SpacedKmerMask<L, BitExtractForm, MaxSpan>`, the positions that a spaced k-mer keeps, with its span and Layout, and each of its pieces (see below) in one form.
 - Bit extraction mask: the data that one bit extraction algorithm needs for a mask, prepared once, such as `BitExtractButterflyMask` with the tables of the butterfly.
 - `BitExtractMask`: the plain bit extraction mask, the bits of the kept positions themselves, which several algorithms share.

A bit extraction kernel, such as `BitExtractKernelButterfly<SimdAVX2>`, is one algorithm for one SIMD arch, and holds its bit extraction mask in vectors. The form of a spaced k-mer mask is either of the two: a bit extraction mask, for the scalar functions, or a kernel, for the SIMD functions. These forms exist:

| Algorithm | Scalar functions | Kernels |
|---|---|---|
| PEXT | `BitExtractMask` | `BitExtractKernelPext`, on the scalar arch only, as PEXT has no SIMD form |
| Butterfly | `BitExtractButterflyMask` | `BitExtractKernelButterfly<Simd>`, on all SIMD archs |
| Block shift | `BitExtractBlockShiftMask` | `BitExtractKernelBlockShift<Simd>`, on all SIMD archs |
| Bit loop, split32, byte lookup | `BitExtractMask` | none |

**Long spans.** The largest span of a mask is set at compile time, by the template parameter `MaxSpan` of the masks and of `SpacedKmerMaskForms`. Its default, 64, is the CMake option `FISK_DEFAULT_MAX_SPAN`, and a single mask type can raise it, e.g., `make_spaced_kmer_mask<Layout::kMSB, 128>(mask)` or `SpacedKmerMaskForms<Layout::kMSB, 4, 128>`. The weight of a mask, its number of `1` positions, is at most 32 for any span. Spans above 32 are extracted in pieces. The loop is compiled once for each number of pieces up to `MaxCompiledPieces`, a template parameter of the extraction functions, `FISK_DEFAULT_MAX_COMPILED_PIECES` (2) by default, i.e., spans up to 57; longer spans run in one slower loop over the pieces. Only the `rolling` functions are limited to spans up to 32.

## Calling an algorithm directly

`for_each_kmer()` and `for_each_spaced_kmer()` choose the algorithm themselves. Calling one directly fits where another one is faster for the callback, such as AVX-512 for a callback that takes vectors, or where the algorithm has to be fixed, as in the benchmarks. Their suites, in [benchmarks/](../../benchmarks/), call each function, and show how it is meant to be used.

The spaced k-mer functions take their masks in the form of their algorithm, and a set of them as a `std::span` of compile-time size, so that their loops over the masks are unrolled; an array is handed on as `std::span{array}`, a single mask as it is. `dispatch_static_span()` hands on a `std::vector` of masks as such a span of its size, up to `kDefaultMaxMasks` (`FISK_DEFAULT_MAX_MASKS`, 4 by default):

```cpp
// PEXT needs BMI2; without it, use make_spaced_kmer_butterfly_masks(masks) and bit_extract_butterfly.
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

The SIMD functions take kernels instead, and call back with vectors, here of AVX-512 for a single mask:

```cpp
auto const kernel = fisk::make_spaced_kmer_kernel<fisk::BitExtractKernelButterfly<fisk::SimdAVX512>>(
    "1101101101101101101101101101101"
);
fisk::for_each_spaced_kmer_packed_unchecked_bytewise_simd(
    packed, kernel, [&](std::size_t pos, auto const& spaced_kmers, std::size_t valid_count) {
        // ...
    }
);
```

The k-mers have the Encoding and Layout of the packed sequence, or of the encoder or masks, or of the template parameters, and these are part of their types. The checked SIMD functions call back with the valid lanes of each vector (`SimdVectorValidLanes`) instead of a count. For single k-mers, the scalar functions are faster than taking them out of the vectors with `for_each_valid_lane()`. The `chunked` functions pack the ASCII input chunk by chunk into a buffer given by the caller, or else a thread-local one; the chunk size is a template parameter, `kDefaultAsciiChunkSize` by default.

## Functions and types

### Extraction

 - `for_each_kmer()` and `for_each_spaced_kmer()` (`extraction/for_each.hpp`) call back with each k-mer or spaced k-mer of an ASCII or packed sequence, with the algorithm that is likely the fastest, see the [main README](../../README.md#quick-start).
 - `for_each_kmer_algorithm_name()` and `for_each_spaced_kmer_algorithm_name()` (same header) return the name of the algorithm that they run.
 - `SpacedKmerMaskForms` (`extraction/spaced_kmer_mask_forms.hpp`) holds a set of masks in all forms that `for_each_spaced_kmer()` uses, see [above](#spaced-k-mer-masks).
 - The extraction functions of `kmer_extract/` and `spaced_kmer_extract/` run one algorithm each, see the [main README](../../README.md#algorithms) and [above](#calling-an-algorithm-directly); each header holds the checked and unchecked functions of one `<input>_<algorithm>_<technique>`, such as `kmer_extract/packed_bytewise_scalar.hpp`.

### K-mer types

In `core/types.hpp`:

 - `Encoding`, `Layout`, and `InputCheck` select the 2-bit codes of the nucleotides, their order in a word, and whether the input is checked, see the [main README](../../README.md#k-mer-extraction).
 - `Kmer<E, L>` holds the codes of a k-mer of up to 32 nucleotides in a 64-bit word, `value`. It does not store its length k, which the functions that need it take as an argument. `KmerAcgtMsb`, `KmerAcgtLsb`, `KmerActgMsb`, and `KmerActgLsb` name its four conventions.
 - `SpacedKmer<E, L>` holds the codes of the kept nucleotides of a spaced k-mer, densely packed, and `SpacedKmerAcgtMsb` and the others name its conventions.
 - `KmerVector<E, L, Simd>` and `SpacedKmerVector<E, L, Simd>` hold a SIMD vector of them, one per 64-bit lane, as the SIMD algorithms hand them out. `SimdVectorValidLanes<Simd>` marks which of their lanes are valid, for the checked functions.

### K-mer operations

In `core/kmer.hpp`:

 - `kmer_encode()` builds a k-mer from a string of nucleotides, and `kmer_decode()` turns it back into one.
 - `base_at()` returns the code of the nucleotide at a position of a k-mer.
 - `complement()` complements each nucleotide of a k-mer, `reverse()` reverses their order, and `reverse_complement()` does both.
 - `canonical()` returns the smaller of a k-mer and its reverse complement.
 - `kmer_convert()` converts a k-mer to another Encoding, Layout, or both.

### Spaced k-mers

In `core/kmer_spaced.hpp`; for creating the masks, see [above](#spaced-k-mer-masks):

 - `spaced_kmer_encode()` builds the spaced k-mer that a mask keeps from a window of nucleotides, and `spaced_kmer_decode()` turns it back into a string over the span of the mask, with `-` at the skipped positions.
 - `spaced_kmer_weight()` returns the number of kept positions of a mask, `spaced_kmer_mask_positions()` lists them, and `spaced_kmer_mask_string()` turns a mask back into its string of `0` and `1`.
 - `is_valid_spaced_kmer_mask()` checks that a mask is well-formed.

### Packed sequences

 - `PackedSequence<E, L>` (`core/types.hpp`) stores a sequence at 2 bits per nucleotide. `PackedSequenceWithValidity<E, L>` also stores one bit per nucleotide that tells whether it was one of `ACGTacgt`. Their `codes()` and `validity()` give access to these bytes.
 - `pack_sequence_unchecked()` and `pack_sequence_checked()` (`sequence_pack/pack_sequence.hpp`) pack an ASCII sequence into them, either into a new one, e.g., `pack_sequence_checked<Encoding::kACGT, Layout::kMSB>(seq)`, or into an existing one, whose storage they reuse. The unchecked one leaves the codes of characters other than ACGT unspecified.
 - The word encoders of `sequence_pack/word_encoder.hpp`, such as `WordEncoderDirectDouble<E, L, SimdAVX2>`, encode the characters for the packing, 8 at a time, or a vector of them. `DefaultWordEncoder<E, L>` names the fastest one of the build, which the packing uses unless it is given another one.

### Callbacks

In `core/kmer_callback.hpp`:

 - `callback_overload{...}` combines several lambdas into one callback, such as one for single k-mers and one for vectors, see the [main README](../../README.md#k-mer-extraction).
 - `for_each_valid_lane()` calls a function with each valid k-mer of a vector, and `store_valid_lanes()` stores them in an array and returns their number.
 - `KmerCallback`, `SpacedKmerCallback`, and the other concepts list the forms of callbacks that the functions take. A callback of single k-mers can leave out the position, and one of single spaced k-mers of a single mask can leave out the mask index, or both.

### Casts and hashes

 - `kmer_cast<E, L>()` (`core/cast.hpp`) turns a raw 2-bit word into a k-mer, and checks in debug builds that it fits the length k; `kmer_value()` returns the word of a k-mer. `spaced_kmer_cast()`, `kmer_vector_cast()`, and `spaced_kmer_vector_cast()` do the same for spaced k-mers and for vectors, and the `_unchecked` versions skip the check.
 - `std::hash` (`core/kmer_hash.hpp`) of `Kmer` and `SpacedKmer` mixes their word with `KmerHashMix`, the splitmix64 finalizer, so that they can be keys of the standard unordered containers. `KmerHashIdentity` uses the word as it is instead, for k-mers that are already well distributed.

### Build and CPU

In `util/cpu_runtime.hpp`:

 - `compiled_avx2()` tells whether AVX2 is compiled in, `cpu_avx2()` whether the CPU has it, and `avx2_enabled()` whether both hold; the same functions exist for BMI2, SSE2, SSSE3, SSE4.1, AVX-512, and Neon.
 - `pext_likely_fast()` tells whether PEXT is enabled and likely fast on this CPU, by a list of CPUs where it is slow (e.g., AMD CPUs before Zen 3).
 - `info_print_system()` prints the platform, CPU, caches, compiler, and instruction sets, and `info_cpu_model()`, `info_cache_sizes()`, and the other `info_*()` functions return them one by one.

`util/intrinsics.hpp` defines `FISK_HAS_AVX2` and the other macros of which instruction sets are compiled in, and the inlining macros, see the [main README](../../README.md#performance-and-build-options). `util/platform.hpp` asserts the host assumptions of fisk, such as little-endianness.

### Utilities

 - `AlignedString` and `AlignedVector<T>` (`util/aligned_allocator.hpp`) start their elements at a cache line, see the [main README](../../README.md#performance-and-build-options).
 - `dispatch_static_span<Max>()` (`util/static_span.hpp`) calls a function with a span of compile-time size over the elements of a vector, of up to `Max` elements, see [above](#calling-an-algorithm-directly).
 - `Splitmix64` (`util/random.hpp`) generates fast, deterministic random numbers.

### Building blocks

 - `CharEncoderTable`, `CharEncoderAscii`, `CharEncoderIfs`, and `CharEncoderSwitch` (`core/char_encoder.hpp`) encode a single character, with a lookup table, a bit trick on its ASCII code, a series of `if`s, or a `switch`; the `ascii_checked_rolling_scalar` and `ascii_checked_reextract_scalar` functions take one of them.
 - `bit_extract_pext`, `bit_extract_butterfly`, `bit_extract_block_shift`, and the other functions of `bit_extract/bit_extract.hpp` gather the bits of a mask from a 64-bit word, each with its form of mask, see the [table above](#spaced-k-mer-masks); `bit_extract/simd.hpp` holds the kernels, which do the same for all lanes of a vector.
 - `SimdScalar`, `SimdSSE2`, `SimdAVX2`, `SimdAVX512`, and `SimdNEON` (`simd/`, one header each) wrap the vector operations of their instruction set, on which the SIMD algorithms are built; `simd/simd.hpp` defines their concepts.
