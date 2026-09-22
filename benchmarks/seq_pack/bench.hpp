#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "fisk/seq_pack/seq_pack.hpp"

/**
 * @brief Benchmark pack_sequence() across all four encoding x layout combinations.
 *
 * Each combination (encoding=acgt/actg, layout=lsb/msb) runs as its own comparison group: within
 * one group, PEXT and butterfly-table candidates are supposed to agree bit-for-bit (same encoding,
 * same layout), which is exactly what the sink cross-validation checks.
 */
void bench_seq_pack(std::vector<std::string> const& sequences, std::ostream& csv_os);

// Kernels compared above, each defined in its own translation unit (var_*.cpp). Each one packs
// into `out`, owned and reused by the calling suite (see bench.cpp) rather than by the kernel
// itself, so that summing it into a sink can happen once, in Microbench::finalize(), outside the
// timed region. Guarded to match the ISA availability of the word encoder each one benchmarks
// (see fisk/core/intrinsics.hpp).
std::uint64_t run_var_acgt_lsb_butterfly(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_butterfly(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_butterfly(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_butterfly(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);

#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_acgt_lsb_pext(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_pext(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_pext(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_pext(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_SSE2)
std::uint64_t run_var_acgt_lsb_butterfly_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_butterfly_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_butterfly_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_butterfly_sse2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_AVX2)
std::uint64_t run_var_acgt_lsb_butterfly_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_butterfly_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_butterfly_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_butterfly_avx2(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_AVX512)
std::uint64_t run_var_acgt_lsb_butterfly_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_butterfly_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_butterfly_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_butterfly_avx512(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif

#if defined(FISK_HAS_NEON)
std::uint64_t run_var_acgt_lsb_butterfly_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_acgt_msb_butterfly_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACGT, fisk::Layout::kMSB>& out
);
std::uint64_t run_var_actg_lsb_butterfly_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kLSB>& out
);
std::uint64_t run_var_actg_msb_butterfly_neon(
    std::string const& seq, fisk::PackedSequence<fisk::Encoding::kACTG, fisk::Layout::kMSB>& out
);
#endif
