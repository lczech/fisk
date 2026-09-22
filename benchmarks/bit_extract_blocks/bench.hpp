#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"

/**
 * @brief Test data for one measurement, as parallel arrays.
 *
 * Kept as arrays-of-fields rather than an array of per-entry structs, so that each implementation
 * streams only the fields it actually uses: the mask-only ones touch 16 bytes per entry, while the
 * preprocessed ones pull in their (much larger) tables. With one struct per entry, every technique
 * would pay for all four fields.
 *
 * All four arrays have the same length, indexed together.
 */
struct BitExtractBlocksBatch
{
    std::vector<std::uint64_t> values;
    std::vector<fisk::BitExtractMask> masks;

    // For the preprocessed implementations, we also pre-compute their tables
    std::vector<fisk::BitExtractBlockTable> block_tables;
    std::vector<fisk::BitExtractButterflyTable> butterfly_tables;
};

/**
 * @brief Benchmark bit extract implementations for masks with different numbers of runs of
 * consecutive 1s.
 *
 * The runs of consecutive 1s are in the range 1 to 32. The latter is the maximum we can get
 * in a 64 bit word, by alternating 0s and 1s.
 */
void bench_bit_extract_blocks(std::ostream& csv_os);

// Implementations compared above, each defined in its own translation unit (var_*.cpp). Each one
// runs @p rounds passes over the whole batch, so that the cost of the one call across the
// translation unit boundary is amortized over all of them, instead of being paid per extraction.
// The batch is small enough to stay in L1 for every implementation, including the preprocessed ones
// with their per-mask tables, so that what is compared is the extraction itself and not how far
// down the cache hierarchy each implementation's helper data happens to land.
//
// Repeating the same passes needs clobber_memory() between them: the inputs do not change across
// rounds, so without it the compiler is free to keep the first round's loads and results and hand
// them back for the rest. Guarded to match the ISA availability of the implementation each one
// benchmarks (see fisk/core/intrinsics.hpp).
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_pext(BitExtractBlocksBatch const& batch, std::size_t rounds);
#endif

std::uint64_t run_var_bitloop(BitExtractBlocksBatch const& batch, std::size_t rounds);
std::uint64_t run_var_split32(BitExtractBlocksBatch const& batch, std::size_t rounds);
std::uint64_t run_var_byte_table(BitExtractBlocksBatch const& batch, std::size_t rounds);

std::uint64_t run_var_block_table(BitExtractBlocksBatch const& batch, std::size_t rounds);
std::uint64_t run_var_block_table_unrolled1(BitExtractBlocksBatch const& batch, std::size_t rounds);
std::uint64_t run_var_block_table_unrolled2(BitExtractBlocksBatch const& batch, std::size_t rounds);
std::uint64_t run_var_block_table_unrolled4(BitExtractBlocksBatch const& batch, std::size_t rounds);
std::uint64_t run_var_block_table_unrolled8(BitExtractBlocksBatch const& batch, std::size_t rounds);

std::uint64_t run_var_butterfly_table(BitExtractBlocksBatch const& batch, std::size_t rounds);

#if defined(PLATFORM_X86_64) && defined(FISK_HAS_CLMUL)
std::uint64_t run_var_instlatx(BitExtractBlocksBatch const& batch, std::size_t rounds);
#endif

std::uint64_t run_var_zp7(BitExtractBlocksBatch const& batch, std::size_t rounds);
