#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/kmer_spaced/packed.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "sink.hpp"

using PackedMsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using PackedLsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

/**
 * @brief Benchmark spaced k-mer extract from packed sequences with multiple masks at once.
 */
void bench_kmer_spaced_packed_multi(
    std::vector<std::string> const& sequences,
    std::vector<std::vector<std::string>> const& multi_masks,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one takes,
// alongside the packed sequence, the piece of per-mask-set setup it needs (which carries the span
// k) -- prepared once per mask set in bench.cpp and threaded in here, so that setup cost never
// enters the timed region. Guarded to match the ISA availability of the kernel each one benchmarks
// (see fisk/core/intrinsics.hpp). They are overloaded for both Layouts, defined in var_msb_*.cpp
// and var_lsb_*.cpp, respectively.
std::uint64_t run_var_bitloop(
    PackedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bitloop(
    PackedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_byte_table(
    PackedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_byte_table(
    PackedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_pext(
    PackedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_pext(
    PackedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_block_table(
    PackedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table(
    PackedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled2(
    PackedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled2(
    PackedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled4(
    PackedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled4(
    PackedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled8(
    PackedMsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled8(
    PackedLsb const& seq,
    std::vector<fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractBlockTable>> const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_butterfly_table(
    PackedMsb const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractButterflyTable>
    > const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_butterfly_table(
    PackedLsb const& seq,
    std::vector<
        fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractButterflyTable>
    > const& masks,
    std::vector<std::uint64_t>& sink_buffer
);
