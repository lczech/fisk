#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/packed.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "sink.hpp"

// Every sequence benchmarked below is packed with its validity, with an ACGT word encoder; only
// the layout varies.
using CheckedMsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using CheckedLsb = fisk::PackedSequenceWithValidity<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

/**
 * @brief Benchmark checked spaced k-mer extraction from a PackedSequenceWithValidity with
 * single masks, the scalar variants of bench_kmer_spaced_packed_single(), each extracting only the
 * valid spaced k-mers.
 *
 * Packing with validity is done once per sequence, outside the timed region. Which nucleotides are
 * invalid, and whether they come in runs, depends on the input.
 */
void bench_kmer_spaced_packed_checked_single(
    std::vector<std::string> const& sequences,
    std::vector<std::string> const& masks,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one takes,
// alongside the packed sequence, the piece of per-mask setup it needs (which carries the span
// k) -- prepared once per mask in bench.cpp and threaded in here, so that setup cost never
// enters the timed region. Guarded to match the ISA availability of the kernel each one benchmarks
// (see fisk/core/intrinsics.hpp). They are overloaded for both Layouts, defined in var_msb_*.cpp
// and var_lsb_*.cpp, respectively.
std::uint64_t run_var_bitloop(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bitloop(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_byte_table(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_byte_table(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_pext(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pext(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_block_table(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled2(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled2(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled4(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled4(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled8(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled8(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_butterfly_table(
    CheckedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractButterflyTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_butterfly_table(
    CheckedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractButterflyTable> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);

