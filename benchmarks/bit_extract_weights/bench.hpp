#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "sink.hpp"

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
struct BitExtractWeightsBatch
{
    std::vector<std::uint64_t> values;
    std::vector<fisk::BitExtractMask> masks;

    // For the preprocessed implementations, we also pre-compute their tables
    std::vector<fisk::BitExtractBlockTable> block_tables;
    std::vector<fisk::BitExtractButterflyTable> butterfly_tables;
};

/**
 * @brief Benchmark different bit extract implementations using randomly generated values and masks,
 * for a range of mask weights (number of bits set) from 0 to 64.
 */
void bench_bit_extract_weights(std::ostream& csv_os);

// Implementations compared above, each defined in its own translation unit (var_*.cpp). Each one
// runs @p rounds passes over the whole batch, so that the cost of the one call across the
// translation unit boundary is amortized over all of them, instead of being paid per extraction.
// The batch is small enough to stay in L1 for every implementation, including the preprocessed ones
// with their per-mask tables, so that what is compared is the extraction itself and not how far
// down the cache hierarchy each implementation's helper data happens to land.
//
// Repeating the same passes needs clobber_memory() between them: the inputs do not change across
// rounds, so without it the compiler is free to keep the first round's loads and results and hand
// them back for the rest.
//
// Each one constructs its own local Sink (see sink.hpp) via make_sink(sink_buffer),
// rather than receiving one built elsewhere: a Sink held only through a local variable is what
// lets Sum's accumulation still auto-vectorize the way plain `hash += ...` did before -- passing
// an already-built Sink in by reference measurably defeats that (confirmed by disassembly), since
// the compiler can no longer prove the accumulator doesn't alias anything else the loop touches.
// sink_buffer itself is still an ordinary reference parameter -- only live for the Write strategy,
// harmlessly unused by Sum and Barrier's make_sink(). Guarded to match the ISA availability of the
// implementation each one benchmarks (see fisk/core/intrinsics.hpp).
#if defined(FISK_HAS_BMI2)
std::uint64_t run_var_pext(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

std::uint64_t run_var_bitloop(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_split32(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_byte_table(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_block_table(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled1(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled2(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled4(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);
std::uint64_t run_var_block_table_unrolled8(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);

std::uint64_t run_var_butterfly_table(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);

#if defined(PLATFORM_X86_64) && defined(FISK_HAS_CLMUL)
std::uint64_t run_var_instlatx(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);
#endif

std::uint64_t run_var_zp7(
    BitExtractWeightsBatch const& batch,
    std::size_t rounds,
    std::vector<std::uint64_t>& sink_buffer
);
