#include <bit> // std::popcount (C++20)
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "bit_extract_weights/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

// Number of entries per batch, and how often each implementation passes over that batch per call.
// Their product is the number of extractions per call, which Microbench repeats as often as its
// sample time needs; the split between them only decides how much data is in flight, which we
// keep small enough for the per-mask tables of all implementations to stay in L1.
static constexpr std::size_t kBatchSize = 64;
static constexpr std::size_t kInnerRounds = 1024;

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
    batch.block_shift_masks.reserve(n);
    batch.butterfly_masks.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t value = dist_u64(rng);
        std::uint64_t mask  = random_mask_with_popcount(rng, popcnt);

        batch.values.push_back(value);
        batch.masks.push_back(BitExtractMask(mask));
        batch.block_shift_masks.push_back(make_bit_extract_block_shift_mask(mask));
        batch.butterfly_masks.push_back(make_bit_extract_butterfly_mask(mask));
    }

    // One batch is one input to the benchmark: every implementation is handed the whole thing in
    // a single call, and loops over it internally.
    return std::vector<BitExtractWeightsBatch>{ std::move(batch) };
}

void bench_bit_extract_weights(std::ostream& csv_os)
{
    std::size_t const n = kBatchSize;
    std::size_t const inner_rounds = kInnerRounds;

    // User output
    std::string const suite_title = "bit_extract_weights";
    std::cout << "\n=== bit extract weights ===\n";
    std::cout << "n=" << n << ", inner rounds=" << inner_rounds << "\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);

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

        // The buffer of the Write sink, see sink.hpp; Sum and Barrier ignore it.
        auto sink_buffer = make_sink_buffer();

        // One call per timed round, with each implementation looping over the batch internally,
        // so the units per run are all extractions that one call performs.
        Microbench<BitExtractWeightsBatch> suite(suite_title);
        suite
            .units_fn([&](BitExtractWeightsBatch const& batch) {
                return static_cast<double>(batch.values.size()) * static_cast<double>(inner_rounds);
            });

        auto results = suite.run(
            make_inputs_rep,
            #if defined(FISK_HAS_BMI2)
            bench(
                "pext",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_pext(b, inner_rounds, sink_buffer);
                }
            ),
            #endif
            bench(
                "bit_loop",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_bit_loop(b, inner_rounds, sink_buffer);
                }
            ),
            bench(
                "split32",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_split32(b, inner_rounds, sink_buffer);
                }
            ),
            bench(
                "byte_lookup",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_byte_lookup(b, inner_rounds, sink_buffer);
                }
            ),
            bench(
                "block_shift",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_block_shift(b, inner_rounds, sink_buffer);
                }
            ),
            bench(
                "block_shift_unrolled1",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_block_shift_unrolled1(b, inner_rounds, sink_buffer);
                }
            ),
            bench(
                "block_shift_unrolled2",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_block_shift_unrolled2(b, inner_rounds, sink_buffer);
                }
            ),
            bench(
                "block_shift_unrolled4",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_block_shift_unrolled4(b, inner_rounds, sink_buffer);
                }
            ),
            bench(
                "block_shift_unrolled8",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_block_shift_unrolled8(b, inner_rounds, sink_buffer);
                }
            ),
            bench(
                "butterfly",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_butterfly(b, inner_rounds, sink_buffer);
                }
            ),
            #if defined(PLATFORM_X86_64) && defined(FISK_HAS_CLMUL)
            bench(
                "instlatx",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_instlatx(b, inner_rounds, sink_buffer);
                }
            ),
            #endif
            bench(
                "zp7",
                [&](BitExtractWeightsBatch const& b){
                    return run_var_zp7(b, inner_rounds, sink_buffer);
                }
            )
        );

        write_csv_rows(csv_os, suite_title, case_label, results, kSinkName);
    }
    if( stdout_is_terminal() ) {
        std::cout << "\n";
    }
}
