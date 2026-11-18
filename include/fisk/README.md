# Library reference

The headers of fisk, the spaced k-mer masks, and how to call an extraction algorithm directly. For `for_each_kmer()` and `for_each_spaced_kmer()`, which are enough for most uses, see the [main README](../../README.md).

## Headers

`fisk/fisk.hpp` includes all headers. Each directory covers one theme; the headers document their functions and types.

| Directory | Contents |
|---|---|
| `core/` | The types and their operations. `types.hpp`: the conventions `Encoding`, `Layout`, and `InputCheck`; `PackedSequence` and `PackedSequenceWithValidity`, which also records which nucleotides were valid; the k-mer types `Kmer` and `SpacedKmer`; the spaced k-mer masks `SpacedKmerMask`; the SIMD vectors `KmerVector` and `SpacedKmerVector`, and `SimdVectorValidLanes` for which of their lanes are valid. `kmer.hpp`: `kmer_encode()`, `kmer_decode()`, `reverse_complement()`, `canonical()`, and more. `kmer_spaced.hpp`: masks as strings and bits, and `spaced_kmer_encode()`/`spaced_kmer_decode()`. `kmer_hash.hpp`: hashes, also as `std::hash`. `cast.hpp`: the casts between raw words and the k-mer types, such as `kmer_cast()`. `kmer_callback.hpp`: the callback concepts, `callback_overload`, and `for_each_valid_lane()`. `char_encoder.hpp`: the per-character encoders. |
| `extraction/` | `for_each.hpp` with `for_each_kmer()` and `for_each_spaced_kmer()`; `spaced_kmer_mask_forms.hpp` with `SpacedKmerMaskForms`; the `*estimate.hpp` headers with the algorithms that both choose from. |
| `kmer_extract/`, `spaced_kmer_extract/` | The extraction functions, see the [main README](../../README.md#algorithms): one header per `<input>_<algorithm>_<technique>`, such as `kmer_extract/packed_bytewise_scalar.hpp`, with its checked and unchecked functions. `spaced_kmer_extract/mask.hpp` makes the spaced k-mer masks, see below. |
| `sequence_pack/` | `pack_sequence_unchecked()` and `pack_sequence_checked()`, which pack ASCII into a `PackedSequence` or `PackedSequenceWithValidity`, 8 characters at a time via a word encoder of `word_encoder.hpp`, such as `WordEncoderDirectDouble<Encoding::kACGT, Layout::kMSB, SimdAVX2>`. |
| `bit_extract/` | The bit extraction algorithms that gather the kept positions of a spaced k-mer: `bit_extract.hpp` with the scalar functions and their masks, `simd.hpp` with the kernels of the SIMD archs, see below. |
| `simd/` | The SIMD archs `SimdScalar`, `SimdSSE2`, `SimdAVX2`, `SimdAVX512`, and `SimdNEON`, one header each, with the vector operations that the SIMD algorithms are built on; `simd.hpp` has their concepts. |
| `util/` | `intrinsics.hpp`: which instruction sets are compiled in, and the inlining macros; `cpu_runtime.hpp`: which ones the CPU has, `pext_likely_fast()`, and `info_print_system()`; `aligned_allocator.hpp`: `AlignedString` and `AlignedVector`; `static_span.hpp`: `dispatch_static_span()`; `random.hpp`: a fast PRNG, whose mixing the hashes use; `platform.hpp`: the host assumptions that fisk asserts, such as little-endianness. |

## Spaced k-mer masks

A spaced k-mer mask says which positions of a window of `span` nucleotides a spaced k-mer keeps, and is given as a string such as `"1101101"`. Bit extraction then gathers the bits of the kept positions. It has two axes: the algorithm (PEXT, butterfly, block shift, and others), and whether it runs as a plain scalar function on one word, or on a SIMD arch, on all lanes of a vector at once. "Mask" has three meanings here, which the documentation names in full where they could be confused:

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

`make_spaced_kmer_mask()` creates a spaced k-mer mask in the form `BitExtractMask`, and `make_spaced_kmer_butterfly_mask()`, `make_spaced_kmer_block_shift_mask()`, and `make_spaced_kmer_kernel()` convert it into the other forms; their plural versions do the same for a `std::vector` of masks. All take the `Layout` of the spaced k-mers as template parameter, e.g., `make_spaced_kmer_mask<Layout::kLSB>()`. `SpacedKmerMaskForms` holds a set of masks in all forms that `for_each_spaced_kmer()` uses, as it chooses its algorithm at runtime.

**Long spans.** The masks have a maximum span as their last template parameter, `FISK_DEFAULT_MAX_SPAN` (64) by default, e.g., `make_spaced_kmer_mask<Layout::kMSB, 128>(mask)`; their weight is at most 32 for any span. Of the spaced k-mer functions, the `packed_<check>_bytewise_*` and `ascii_<check>_chunked_*` ones take spans up to that maximum, and the others spans up to 32. Above 32, a span is cut into pieces: leading pieces of 28 positions and a tail of up to 29. The functions that take longer spans have a copy of their loop for each number of pieces up to their template parameter `MaxCompiledPieces`, `FISK_DEFAULT_MAX_COMPILED_PIECES` (2) by default, i.e., spans up to 57, and run longer spans in one loop over the pieces.

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
