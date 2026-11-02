#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/types.hpp"
#include "sink.hpp"
#include "spaced_masks.hpp"

using PackedMsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>;
using PackedLsb = fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>;

/**
 * @brief Benchmark spaced k-mer extraction from packed sequences, with a single mask for each span
 * from 1 to kSpanSweepMaxSpan, see spaced_mask_for_span(), each span its own case.
 */
void bench_spaced_kmer_extract_packed_unchecked_spans(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit (var_*.cpp), and overloaded for
// both Layouts, defined in var_*_msb.cpp and var_*_lsb.cpp, respectively.
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_bytewise_scalar_pext(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kMSB, fisk::BitExtractMask, kSpanSweepMaxSpan> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_pext(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<fisk::Layout::kLSB, fisk::BitExtractMask, kSpanSweepMaxSpan> const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
#endif
std::uint64_t run_var_bytewise_scalar_butterfly(
    PackedMsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kMSB, fisk::BitExtractButterflyMask, kSpanSweepMaxSpan
    > const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_bytewise_scalar_butterfly(
    PackedLsb const& seq,
    fisk::SpacedKmerMask<
        fisk::Layout::kLSB, fisk::BitExtractButterflyMask, kSpanSweepMaxSpan
    > const& mask,
    std::vector<std::uint64_t>& sink_buffer
);
