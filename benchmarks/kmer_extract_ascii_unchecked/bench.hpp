#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/sequence_pack/pack_sequence.hpp"
#include "sink.hpp"

using PackedAcgtMsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using PackedAcgtLsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>;
using PackedActgMsb = fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>;
using PackedActgLsb = fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>;

/**
 * @brief Benchmark the extraction of all k-mers of a sequence from ASCII input without checks.
 *
 * For input known to be valid: for_each_kmer_ascii_unchecked_chunked_scalar(), which
 * packs the sequence in chunks and extracts from those, against packing the whole sequence first
 * and then extracting via for_each_kmer_packed_unchecked_bytewise_scalar(). Both run for both
 * Encodings and both Layouts, and pack with the word encoder of chunked.hpp. The Encoding
 * only affects the packing step, not the extraction from the packed sequence. For ACGT, also
 * for_each_kmer_ascii_unchecked_chunked_simd() with the SIMD archs of the build: the `_scalars`
 * variants take the k-mers out of the vectors with for_each_valid_lane(), the others sum the
 * vectors whole. Both sums match, as the lanes that pad the last vector are zero.
 */
void bench_kmer_extract_ascii_unchecked(
    std::vector<std::string> const& sequences,
    std::size_t k_min,
    std::size_t k_max,
    std::ostream& csv_os
);

// Test all valid k-mer sizes.
void bench_kmer_extract_ascii_unchecked(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one
// constructs its own local Sink (see sink.hpp) via make_sink(sink_buffer), so that Sum's
// accumulation can still auto-vectorize the way plain `hash += kmer_value(kmer)` did before -- see
// the matching comment in bit_extract_weights/bench.hpp for why a Sink built elsewhere and passed
// in by reference would defeat that.
// The scratch-buffer variants take caller-owned PackedSequence storage,
// so it is reused across calls rather than reallocated each time.
std::uint64_t run_var_chunked_scalar_threadlocal_acgt_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_threadlocal_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_threadlocal_actg_msb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_threadlocal_actg_lsb(
    std::string const& seq,
    std::size_t k,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_scratch_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_scratch_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_scratch_actg_msb(
    std::string const& seq,
    std::size_t k,
    PackedActgMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_scalar_scratch_actg_lsb(
    std::string const& seq,
    std::size_t k,
    PackedActgLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_scalar_scalars_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_chunked_simd_scalar_scalars_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_chunked_simd_avx2_scalars_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_AVX2
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_chunked_simd_avx512_scalars_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_AVX512
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_chunked_simd_neon_scalars_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_NEON
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_chunked_simd_avx2_scalars_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_AVX2
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_chunked_simd_avx512_scalars_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_AVX512
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_chunked_simd_neon_scalars_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_NEON
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_chunked_simd_sse2_scalars_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_SSE2
std::uint64_t run_var_chunked_simd_scalar_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_chunked_simd_sse2_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_SSE2
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_chunked_simd_avx2_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_AVX2
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_chunked_simd_avx512_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_AVX512
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_chunked_simd_neon_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_NEON
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_chunked_simd_sse2_scalars_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_SSE2
std::uint64_t run_var_chunked_simd_scalar_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_chunked_simd_sse2_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_SSE2
#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_chunked_simd_avx2_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_AVX2
#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_chunked_simd_avx512_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_AVX512
#if defined(FISK_HAS_NEON)
std::uint64_t run_var_chunked_simd_neon_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
#endif // FISK_HAS_NEON
std::uint64_t run_var_pack_then_bytewise_scalar_acgt_msb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pack_then_bytewise_scalar_acgt_lsb(
    std::string const& seq,
    std::size_t k,
    PackedAcgtLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pack_then_bytewise_scalar_actg_msb(
    std::string const& seq,
    std::size_t k,
    PackedActgMsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pack_then_bytewise_scalar_actg_lsb(
    std::string const& seq,
    std::size_t k,
    PackedActgLsb& scratch,
    std::vector<std::uint64_t>& sink_buffer
);
