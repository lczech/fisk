#include <bit> // std::popcount (C++20)
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "bit_extract_weights/bench.hpp"
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

// Sample size for the selector survey below, per mask weight.
static constexpr std::size_t kSelectorSamples = 16;

// bit_extract_selector() runs its own internal self-tuning benchmark on every call. We reduce
// its sample size here (from the default of 2^14) since the result is only used for informational
// mode counts, not for any of the benched implementations.
static constexpr std::size_t kTuneNumVals = 4096;

/**
 * @brief Generate a random bit extract mask with a given amount of set bits.
 */
static std::uint64_t random_mask_with_popcount(std::mt19937_64& rng, int popcnt)
{
    if (popcnt <= 0) return 0ull;
    if (popcnt >= 64) return ~0ull;

    std::uint64_t mask = 0;
    std::uniform_int_distribution<int> dist(0, 63);
    while (static_cast<int>(std::popcount(mask)) < popcnt) {
        mask |= (1ull << dist(rng));
    }
    return mask;
}

/**
 * @brief Generate input for the benchmark here, with @p n entries of @p popcnt weight.
 */
static std::vector<BitExtractWeightsBatch> make_inputs(
    std::size_t n, int popcnt, std::uint64_t seed
) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint64_t> dist_u64;

    BitExtractWeightsBatch batch;
    batch.values.reserve(n);
    batch.masks.reserve(n);
    batch.block_tables.reserve(n);
    batch.butterfly_tables.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t value = dist_u64(rng);
        std::uint64_t mask  = random_mask_with_popcount(rng, popcnt);

        batch.values.push_back(value);
        batch.masks.push_back(BitExtractMask(mask));
        batch.block_tables.push_back(bit_extract_block_table_preprocess(mask));
        batch.butterfly_tables.push_back(bit_extract_butterfly_table_preprocess(mask));
    }

    // One batch is one input to the benchmark: every implementation is handed the whole thing in
    // a single call, and loops over it internally.
    return std::vector<BitExtractWeightsBatch>{ std::move(batch) };
}

/**
 * @brief Collect which selector mode is chosen how often, across the same range of mask weights
 * that the benchmark below uses.
 *
 * This is not really important, but we are curious to see this. It is deliberately kept out of the
 * benchmark input generation: bit_extract_selector() runs a self-tuning benchmark per call, whose
 * cost would otherwise scale with the batch size and the number of repeats, and dominate the
 * runtime of the suite without contributing anything to its measurements.
 */
static void survey_selector_modes(std::vector<std::size_t>& selector_counts)
{
    for (int w = 0; w <= 64; ++w) {
        auto seed = static_cast<std::uint64_t>(0x5E1EC7ULL) ^ static_cast<std::uint64_t>(w);
        std::mt19937_64 rng(seed);
        for (std::size_t i = 0; i < kSelectorSamples; ++i) {
            auto const mask = random_mask_with_popcount(rng, w);
            ++selector_counts[static_cast<std::size_t>(bit_extract_selector(mask, kTuneNumVals))];
        }
    }
}

void bench_bit_extract_weights(std::ostream& csv_os)
{
    std::size_t const n = kBatchSize;
    std::size_t const rounds = kInnerRounds;
    std::size_t const repeats = 32;

    // User output
    std::string const suite_title = "bit_extract_weights";
    std::cout << "\n=== bit extract weights ===\n";
    std::cout << "n=" << n << ", rounds=" << rounds << ", repeats=" << repeats << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

    auto selector_counts = std::vector<std::size_t>( 6, 0 );
    survey_selector_modes(selector_counts);

    // Run a benchmark for each weight of the mask.
    // Most of our bit extract software implementations have a runtime depending on that,
    // so we want to test the effects of different masks on the implementations.
    for (int w = 0; w <= 64; ++w) {
        std::string case_label = "popcount=" + std::to_string(w);
        if( stdout_is_terminal() ) {
            std::cout << "\rmask popcount "
                << std::setw(2) << w << " / 64"
                << std::flush;
        }

        // Helper to generate fresh input for each repetition
        auto make_inputs_rep = [w]()
        {
            auto seed = static_cast<std::uint64_t>(0xC0FFEEULL) ^ static_cast<std::uint64_t>(w);
            return make_inputs( n, w, seed );
        };

        // One call per timed round, with each implementation looping over the batch internally,
        // so the units per run are all extractions that one call performs.
        Microbench<BitExtractWeightsBatch> suite(suite_title);
        suite
            .rounds(1)
            .repeats(repeats)
            .units_fn([&](BitExtractWeightsBatch const& batch) {
                return static_cast<double>(batch.values.size()) * static_cast<double>(rounds);
            });

        auto results = suite.run(
            make_inputs_rep,
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_pext(b, rounds);
                }
            ),
            #endif
            bench(
                "bitloop",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_bitloop(b, rounds);
                }
            ),
            bench(
                "split32",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_split32(b, rounds);
                }
            ),
            bench(
                "byte_table",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_byte_table(b, rounds);
                }
            ),
            bench(
                "block_table",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_block_table(b, rounds);
                }
            ),
            bench(
                "block_table_unrolled1",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_block_table_unrolled1(b, rounds);
                }
            ),
            bench(
                "block_table_unrolled2",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_block_table_unrolled2(b, rounds);
                }
            ),
            bench(
                "block_table_unrolled4",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_block_table_unrolled4(b, rounds);
                }
            ),
            bench(
                "block_table_unrolled8",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_block_table_unrolled8(b, rounds);
                }
            ),
            bench(
                "butterfly_table",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_butterfly_table(b, rounds);
                }
            ),
            #if defined(PLATFORM_X86_64) && defined(FISK_HAS_CLMUL)
            bench(
                "instlatx",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_instlatx(b, rounds);
                }
            ),
            #endif
            bench(
                "zp7",
                [&](BitExtractWeightsBatch const& b){
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
