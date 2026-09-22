#include <algorithm>
#include <bit> // std::popcount (C++20)
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "bit_extract_blocks/bench.hpp"
#include "fisk/bit_extract/selector.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

// Number of entries per batch, and how often each implementation passes over that batch per timed
// call. Their product is the number of extractions a measurement is averaged over; the split
// between them only decides how much data is in flight, which we keep small enough for the
// per-mask tables of all implementations to stay in L1.
static constexpr std::size_t kBatchSize = 64;
static constexpr std::size_t kInnerRounds = 1024;

// Sample size for the selector survey below, per number of runs.
static constexpr std::size_t kSelectorSamples = 16;

// bit_extract_selector() runs its own internal self-tuning benchmark on every call. We reduce
// its sample size here (from the default of 2^14) since the result is only used for informational
// mode counts, not for any of the benched implementations.
static constexpr std::size_t kTuneNumVals = 4096;

namespace {

/**
 * @brief Generate a random bitmask of width W (<= 64) with exactly `runs` runs of consecutive 1s.
 *
 * Runs are maximal contiguous segments of 1 bits. For example:
 *   0011100111000  has 2 runs of 1s.
 *
 * Distribution notes:
 * - This produces a reasonably random mask subject to the constraint, but is not
 *   a perfectly uniform distribution over all masks with `runs` runs.
 * - It randomizes: (a) total number of 1s, (b) run lengths, (c) zero gaps.
 *
 * Boundary handling:
 * - runs == 0 -> returns 0
 * - runs must satisfy 0 <= runs <= ceil(W/2). For W=64, max is 32.
 * - Works for any W in [1, 64]; default W=64.
 */
template <std::size_t W = 64, class URBG>
std::uint64_t random_mask_with_runs(std::size_t runs, URBG& rng)
{
    static_assert(W >= 1 && W <= 64, "W must be in [1, 64]");

    const std::size_t max_runs = (W + 1) / 2; // ceil(W/2)
    if (runs > max_runs) {
        throw std::invalid_argument("runs too large for given width");
    }
    if (runs == 0) {
        return 0ULL;
    }

    // Internal gaps between runs must be >= 1 when runs >= 2
    const std::size_t min_internal_zeros = (runs >= 2) ? (runs - 1) : 0;

    // Choose total ones K: must be >= runs (each run at least 1),
    // and must leave room for the required internal zeros.
    const std::size_t min_ones = runs;
    const std::size_t max_ones = W - min_internal_zeros;
    if (min_ones > max_ones) {
        // Should not happen if runs <= ceil(W/2), but keep it airtight.
        throw std::logic_error("No feasible mask for given runs/width");
    }

    std::uniform_int_distribution<std::size_t> dist_ones(min_ones, max_ones);
    const std::size_t K = dist_ones(rng);

    // Helper: random composition of N into k positive integers.
    auto random_positive_composition = [&](std::size_t N, std::size_t k) {
        // If k == 1, only [N]
        std::vector<std::size_t> parts(k, 1);
        N -= k; // remaining to distribute as nonnegative increments
        if (k == 0) return parts;

        // Distribute N as nonnegative into k buckets via stars-and-bars with random cut points.
        // Choose (k-1) cut points in [0..N+k-1]? We can do simpler:
        // Generate k-1 integers in [0..N], sort them, take differences.
        std::vector<std::size_t> cuts;
        cuts.reserve(k > 0 ? k - 1 : 0);
        std::uniform_int_distribution<std::size_t> d(0, N);
        for (std::size_t i = 0; i + 1 < k; ++i) cuts.push_back(d(rng));
        std::sort(cuts.begin(), cuts.end());

        std::size_t prev = 0;
        for (std::size_t i = 0; i + 1 < k; ++i) {
            const std::size_t val = cuts[i] - prev;
            parts[i] += val;
            prev = cuts[i];
        }
        parts[k - 1] += (N - prev);
        return parts;
    };

    // 1-run lengths: composition of K into `runs` positive parts
    std::vector<std::size_t> one_runs = random_positive_composition(K, runs);

    // Now handle zeros. Total zeros:
    const std::size_t Z = W - K;

    // We have (runs+1) zero gaps: [prefix, between..., suffix]
    // Internal gaps (runs-1 of them) must be >= 1 if runs >= 2.
    const std::size_t gaps = runs + 1;
    std::vector<std::size_t> zero_gaps(gaps, 0);

    std::size_t remaining_zeros = Z;

    if (runs >= 2) {
        // Assign mandatory 1 zero to each internal gap
        for (std::size_t i = 1; i <= runs - 1; ++i) {
            zero_gaps[i] = 1;
        }
        remaining_zeros -= (runs - 1);
    }

    // Distribute remaining_zeros freely across all gaps (including internal ones)
    // as a nonnegative composition into `gaps` parts.
    {
        // Nonnegative composition by random cut points in [0..remaining_zeros]
        std::vector<std::size_t> cuts;
        cuts.reserve(gaps > 1 ? gaps - 1 : 0);
        std::uniform_int_distribution<std::size_t> d(0, remaining_zeros);
        for (std::size_t i = 0; i + 1 < gaps; ++i) cuts.push_back(d(rng));
        std::sort(cuts.begin(), cuts.end());

        std::size_t prev = 0;
        for (std::size_t i = 0; i + 1 < gaps; ++i) {
            zero_gaps[i] += cuts[i] - prev;
            prev = cuts[i];
        }
        zero_gaps[gaps - 1] += remaining_zeros - prev;
    }

    // Pack into a uint64_t, MSB..LSB or LSB..MSB. Choose one and be consistent.
    // Here: bit 0 is the least-significant bit, we fill from LSB upward.
    std::uint64_t mask = 0;
    std::size_t pos = 0;

    auto put_zeros = [&](std::size_t n) {
        pos += n;
        if (pos > W) throw std::logic_error("packing overflow (zeros)");
    };
    auto put_ones = [&](std::size_t n) {
        if (n == 0) return;
        if (pos + n > W) throw std::logic_error("packing overflow (ones)");
        // Set n bits starting at pos
        // Special case n == 64 to avoid shift UB (but W<=64 and pos+n<=W).
        if (n == 64) {
            mask = std::numeric_limits<std::uint64_t>::max();
        } else {
            const std::uint64_t ones = (n == 64) ? ~0ULL : ((1ULL << n) - 1ULL);
            mask |= (ones << pos);
        }
        pos += n;
    };

    // prefix zeros
    put_zeros(zero_gaps[0]);

    // alternating 1-run and zero gap
    for (std::size_t i = 0; i < runs; ++i) {
        put_ones(one_runs[i]);
        put_zeros(zero_gaps[i + 1]);
    }

    // pos should be exactly W
    if (pos != W) {
        throw std::logic_error("packing did not fill width exactly");
    }

    // If W < 64, ensure upper bits are 0
    if constexpr (W < 64) {
        const std::uint64_t high_mask = (1ULL << W) - 1ULL;
        mask &= high_mask;
    }

    return mask;
}

/**
 * @brief Count the number of runs of consecutive 1s in a mask.
 *
 * This is mostly used interally to check that we got the right amount.
 */
std::size_t count_runs(std::uint64_t x)
{
    // Bits that start a run of 1s (as seen from the direction of the LSB):
    // - bit is 1
    // - previous bit is 0
    const std::uint64_t run_starts = x & ~(x << 1);
    return static_cast<std::size_t>( std::popcount( run_starts ));
}

/**
 * @brief Print a mask as a bit string.
 */
[[maybe_unused]] void print_bits(std::uint64_t x, std::ostream& os)
{
    for (int i = 63; i >= 0; --i) {
        os << ((x >> i) & 1ULL);
        if (i % 8 == 0 && i != 0) {
            os << ' ';
        }
    }
}

} // namespace

/**
 * @brief Generate @p n random masks with @p runs runs of consecutive 1s each.
 *
 * The function also creates the bit extract helper masks for our software implemetations.
 * Doing all of this here is not good software design, but good enough for our simple benchmark.
 */
static std::vector<BitExtractBlocksBatch> make_input_blocks(
    std::size_t n, std::size_t runs, std::uint64_t seed
) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint64_t> dist_u64;

    BitExtractBlocksBatch batch;
    batch.values.reserve(n);
    batch.masks.reserve(n);
    batch.block_tables.reserve(n);
    batch.butterfly_tables.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t value = dist_u64(rng);
        std::uint64_t mask  = random_mask_with_runs<64>(runs, rng);

        // sanity check that we got what we expected
        if( runs != count_runs(mask) ) {
            throw std::runtime_error( "issue in random_mask_with_runs()" );
        }

        // debug printing
        // print_bits(mask, std::cout);
        // std::cout << " runs=" << runs << "\n";

        // add to the list of inputs
        batch.values.push_back(value);
        batch.masks.push_back(BitExtractMask(mask));
        batch.block_tables.push_back(bit_extract_block_table_preprocess(mask));
        batch.butterfly_tables.push_back(bit_extract_butterfly_table_preprocess(mask));
    }

    // One batch is one input to the benchmark: every implementation is handed the whole thing in
    // a single call, and loops over it internally.
    return std::vector<BitExtractBlocksBatch>{ std::move(batch) };
}

/**
 * @brief Collect which selector mode is chosen how often, across the same range of run counts
 * that the benchmark below uses.
 *
 * This is not really important, but we are curious to see this. It is deliberately kept out of the
 * benchmark input generation: bit_extract_selector() runs a self-tuning benchmark per call, whose
 * cost would otherwise scale with the batch size and the number of repeats, and dominate the
 * runtime of the suite without contributing anything to its measurements.
 */
static void survey_selector_modes(std::vector<std::size_t>& selector_counts)
{
    for (std::size_t runs = 0; runs <= 32; ++runs) {
        auto seed = static_cast<std::uint64_t>(0x5E1EC7ULL) ^ static_cast<std::uint64_t>(runs);
        std::mt19937_64 rng(seed);
        for (std::size_t i = 0; i < kSelectorSamples; ++i) {
            auto const mask = random_mask_with_runs<64>(runs, rng);
            ++selector_counts[static_cast<std::size_t>(bit_extract_selector(mask, kTuneNumVals))];
        }
    }
}

void bench_bit_extract_blocks(std::ostream& csv_os)
{
    std::size_t const n = kBatchSize;
    std::size_t const rounds = kInnerRounds;
    std::size_t const repeats = 16;

    // User output
    std::string const suite_title = "bit_extract_blocks";
    std::cout << "\n=== bit extract blocks ===\n";
    std::cout << "n=" << n << ", rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    auto selector_counts = std::vector<std::size_t>( 6, 0 );
    survey_selector_modes(selector_counts);

    // Run a benchmark for each weight of the mask.
    // Most of our bit extract software implementations have a runtime depending on that,
    // so we want to test the effects of different masks on the implementations.
    for( std::size_t runs_cnt = 0; runs_cnt <= 32; ++runs_cnt ) {
        std::string case_label = "runs=" + std::to_string(runs_cnt);
        if( stdout_is_terminal() ) {
            std::cout << "\rmask runs "
                << std::setw(2) << runs_cnt << " / 32"
                << std::flush;
        }

        // Helper to generate fresh input for each repetition
        auto make_inputs_rep = [runs_cnt]()
        {
            auto seed = static_cast<std::uint64_t>(0xC0FFEEULL) ^ static_cast<std::uint64_t>(runs_cnt);
            return make_input_blocks( n, runs_cnt, seed );
        };

        // One call per timed round, with each implementation looping over the batch internally,
        // so the units per run are all extractions that one call performs.
        Microbench<BitExtractBlocksBatch> suite(suite_title);
        suite
            .rounds(1)
            .repeats(repeats)
            .units_fn([&](BitExtractBlocksBatch const& batch) {
                return static_cast<double>(batch.values.size()) * static_cast<double>(rounds);
            });

        auto results = suite.run(
            make_inputs_rep,
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_pext(b, rounds);
                }
            ),
            #endif
            bench(
                "bitloop",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_bitloop(b, rounds);
                }
            ),
            bench(
                "split32",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_split32(b, rounds);
                }
            ),
            bench(
                "byte_table",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_byte_table(b, rounds);
                }
            ),
            bench(
                "block_table",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_block_table(b, rounds);
                }
            ),
            bench(
                "block_table_unrolled1",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_block_table_unrolled1(b, rounds);
                }
            ),
            bench(
                "block_table_unrolled2",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_block_table_unrolled2(b, rounds);
                }
            ),
            bench(
                "block_table_unrolled4",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_block_table_unrolled4(b, rounds);
                }
            ),
            bench(
                "block_table_unrolled8",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_block_table_unrolled8(b, rounds);
                }
            ),
            bench(
                "butterfly_table",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_butterfly_table(b, rounds);
                }
            ),
            #if defined(PLATFORM_X86_64) && defined(FISK_HAS_CLMUL)
            bench(
                "instlatx",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_instlatx(b, rounds);
                }
            ),
            #endif
            bench(
                "zp7",
                [&](BitExtractBlocksBatch const& b){
                    return run_var_zp7(b, rounds);
                }
            )
        );

        write_csv_rows(csv_os, suite_title, case_label, results);
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }

    std::cout << "selector bit extract counts:\n";
    for( std::size_t i = 0; i < selector_counts.size(); ++i ) {
        std::cout << "  " << selector_counts[i] << " <== " << bit_extract_mode_name(static_cast<BitExtractMode>(i)) << "\n";
    }
    std::cout << "\n";
}
