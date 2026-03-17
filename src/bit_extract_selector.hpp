#pragma once

#include <cstdint>
#include <chrono>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bit_extract.hpp"
#include "bit_extract_simd.hpp"
#include "utils.hpp"
#include "sys_info.hpp"

// =================================================================================================
//     Bit Extract Algorithms Enum
// =================================================================================================

/**
 * @brief Modes for bit extraction.
 *
 * For simplicity, we currently only include the implementations that were most performant in our
 * benchmarks (on AMD, Intel, and Apple hardware, i.e., with and without hardware PEXT).
 * If needed, this can be trivially expanded to test for additional implementations.
 */
enum class BitExtractMode : int
{
    /** @brief Use hardware PEXT */
    kPext,

    /** @brief Use the block table algorithm */
    kBlockTable,

    /** @brief Use the block table unrolled by 2 algorithm */
    kBlockTableUnrolled2,

    /** @brief Use the block table unrolled by 4 algorithm */
    kBlockTableUnrolled4,

    /** @brief Use the block table unrolled by 8 algorithm */
    kBlockTableUnrolled8,

    /** @brief Use the butterfly table algorithm */
    kButterflyTable,
};

/**
 * @brief Get a bit extract mode as a printable string, e.g., for user output.
 */
std::string bit_extract_mode_name( BitExtractMode mode )
{
    switch(mode) {
        case BitExtractMode::kPext:                return "PEXT";
        case BitExtractMode::kBlockTable:          return "BlockTable";
        case BitExtractMode::kBlockTableUnrolled2: return "BlockTableUnrolled2";
        case BitExtractMode::kBlockTableUnrolled4: return "BlockTableUnrolled4";
        case BitExtractMode::kBlockTableUnrolled8: return "BlockTableUnrolled8";
        case BitExtractMode::kButterflyTable:      return "ButterflyTable";
        default: {
            throw std::invalid_argument(
                "Invalid BitExtractMode in bit_extract_mode_name(): " +
                std::to_string(static_cast<int>(mode))
            );
        }
    }
}

// =================================================================================================
//     Bit Extract Algorithm Selector
// =================================================================================================

/**
 * @brief Helper to adaptively select the fastest bit extract implementation for a given fixed mask.
 *
 * A benchmark is run that tests hardware PEXT and several flavors of software bit extract
 * implementations for a given @p mask, and returns the most performant one for the given mask.
 * The more values @p num_vals, the more accurate the results will be, at the expense of runtime
 * for the benchmarking process. In our test, the default @p num_vals yields ~2ms for the function.
 *
 * This is intended for usage for a single fixed mask (or few masks, in separate calls of this
 * function). When the mask is not fixed, this strategy is not beneficial, as precomputing lookup
 * tables specific for the mask does not yield performance. In that case, the butterfly table
 * algorithm can be adapted to work without precomputing the mask, and a similar benchmarking
 * process can be used to select the best algorithm, but that is not implemented here.
 *
 * The result of this function can be used to select the desired bit extraction algorithm.
 * We recommend the implementation of the inner loop where this is used to be templated by the bit
 * extractor function, so that the dispatch to the algorithm can be done at compile time without
 * runtime overhead, e.g., via a `switch` statement.
 *
 * @param mask The bitmask to use for extraction.
 * @param num_vals The number of random values to test on, randomly generated (default: 2^14).
 * @return The most performant bit extraction mode.
 */
BitExtractMode bit_extract_selector(
    BitExtractMask const mask,
    std::size_t const num_vals = ( 1 << 14 )
) {
    using clock = std::chrono::steady_clock;

    // Prepare some random data that is long enough for a meaningful test.
    std::vector<std::uint64_t> data;
    data.reserve(num_vals);
    Splitmix64 rng{};
    for( size_t i = 0; i < num_vals; ++i ) {
        data.push_back( rng.get_uint64() );
    }

    // Bit extract mask and block table for the block algorithm
    auto const block_table     = bit_extract_block_table_preprocess(mask);
    auto const butterfly_table = bit_extract_butterfly_table_preprocess(mask);

    // --------------------------------------------------------
    //     Benchmark helper
    // --------------------------------------------------------

    struct CandidateResult
    {
        BitExtractMode mode;
        std::chrono::nanoseconds time;
        std::uint64_t result;
    };

    std::vector<CandidateResult> results;
    auto benchmark_candidate_ = [&](BitExtractMode mode, auto&& func)
    {
        // Warmup. We use a volatile to ensure the compiler cant optimize out the loop below,
        // which would create invalid benchmarks.
        volatile std::uint64_t sink_out = 0;
        for (auto const x : data) {
            sink_out += func(x);
        }

        // Timed runs. Take best-of-N to reduce scheduler noise.
        constexpr std::size_t repeats = 3;
        auto best = std::chrono::nanoseconds::max();
        for (std::size_t r = 0; r < repeats; ++r) {
            std::uint64_t sink = 0;

            auto const start = clock::now();
            for (auto const x : data) {
                sink += func(x);
            }
            auto const stop = clock::now();
            sink_out = sink;

            auto const dt = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            if (dt < best) {
                best = dt;
            }
        }

        // Check that previous result sink is the same as what we got now.
        if( results.size() > 0 && results.back().result != sink_out ) {
            throw std::runtime_error("bit_extract_selector(): candidate result mismatch");
        }
        results.push_back({mode, best, sink_out});
    };

    // --------------------------------------------------------
    //     Benchmark candidates
    // --------------------------------------------------------

    #if defined(FISK_HAS_BMI2)
    if (bmi2_enabled()) {
        benchmark_candidate_(
            BitExtractMode::kPext,
            [&](std::uint64_t x) noexcept {
                return bit_extract_pext(x, mask);
            }
        );
    }
    #endif

    benchmark_candidate_(
        BitExtractMode::kBlockTable,
        [&](std::uint64_t x) noexcept {
            return bit_extract_block_table_unrolled<1>(x, block_table);
        }
    );

    benchmark_candidate_(
        BitExtractMode::kBlockTableUnrolled2,
        [&](std::uint64_t x) noexcept {
            return bit_extract_block_table_unrolled<2>(x, block_table);
        }
    );

    benchmark_candidate_(
        BitExtractMode::kBlockTableUnrolled4,
        [&](std::uint64_t x) noexcept {
            return bit_extract_block_table_unrolled<4>(x, block_table);
        }
    );

    benchmark_candidate_(
        BitExtractMode::kBlockTableUnrolled8,
        [&](std::uint64_t x) noexcept {
            return bit_extract_block_table_unrolled<8>(x, block_table);
        }
    );

    benchmark_candidate_(
        BitExtractMode::kButterflyTable,
        [&](std::uint64_t x) noexcept {
            return bit_extract_butterfly_table(x, butterfly_table);
        }
    );

    // --------------------------------------------------------
    //     Return the fastest candidate
    // --------------------------------------------------------

    if (results.empty()) {
        throw std::runtime_error("bit_extract_selector(): no candidate implementations available");
    }

    auto const best = std::min_element(
        results.begin(),
        results.end(),
        [](auto const& a, auto const& b) {
            return a.time < b.time;
        }
    );

    return best->mode;
}

/**
 * @brief Select the best bit extraction mode for a given mask and number of values.
 *
 * Same as the overload, but takes the mask as an 64-bit unsigned int.
 */
BitExtractMode bit_extract_selector(
    std::uint64_t const mask,
    std::size_t const num_vals = ( 1 << 14 )
) {
    return bit_extract_selector( BitExtractMask(mask), num_vals );
}

// =================================================================================================
//     Exemplary usage
// =================================================================================================

void switch_bit_extract_mode( std::uint64_t const mask )
{
    auto const mode = bit_extract_selector(mask);
    switch (mode) {
        #if defined(FISK_HAS_BMI2)
        case BitExtractMode::kPext:
            // hot loop with bit_extract_pext(x, mask)
            break;
        #endif
        case BitExtractMode::kBlockTable:
        {
            auto const bt = bit_extract_block_table_preprocess(mask);
            (void) bt;
            // hot loop with bit_extract_block_table_unrolled<1>(x, bt)
            break;
        }
        case BitExtractMode::kBlockTableUnrolled2:
        {
            auto const bt = bit_extract_block_table_preprocess(mask);
            (void) bt;
            // hot loop with bit_extract_block_table_unrolled<2>(x, bt)
            break;
        }
        case BitExtractMode::kBlockTableUnrolled4:
        {
            auto const bt = bit_extract_block_table_preprocess(mask);
            (void) bt;
            // hot loop with bit_extract_block_table_unrolled<4>(x, bt)
            break;
        }
        case BitExtractMode::kBlockTableUnrolled8:
        {
            auto const bt = bit_extract_block_table_preprocess(mask);
            (void) bt;
            // hot loop with bit_extract_block_table_unrolled<8>(x, bt)
            break;
        }
        case BitExtractMode::kButterflyTable:
        {
            auto const bf = bit_extract_butterfly_table_preprocess(mask);
            (void) bf;
            // hot loop with bit_extract_butterfly_table(x, bf)
            break;
        }
    }
}

template<class Callback>
inline void run_bit_extract_mode(
    BitExtractMode mode,
    BitExtractMask mask,
    Callback&& cb
)
{
    switch (mode) {
        #if defined(FISK_HAS_BMI2)
        case BitExtractMode::kPext:
            cb([&](std::uint64_t x) noexcept {
                return bit_extract_pext(x, mask);
            });
            break;
        #endif

        case BitExtractMode::kBlockTable:
        {
            auto const bt = bit_extract_block_table_preprocess(mask.mask);
            cb([&](std::uint64_t x) noexcept {
                return bit_extract_block_table_unrolled<1>(x, bt);
            });
            break;
        }

        case BitExtractMode::kBlockTableUnrolled2:
        {
            auto const bt = bit_extract_block_table_preprocess(mask.mask);
            cb([&](std::uint64_t x) noexcept {
                return bit_extract_block_table_unrolled<2>(x, bt);
            });
            break;
        }

        case BitExtractMode::kBlockTableUnrolled4:
        {
            auto const bt = bit_extract_block_table_preprocess(mask.mask);
            cb([&](std::uint64_t x) noexcept {
                return bit_extract_block_table_unrolled<4>(x, bt);
            });
            break;
        }

        case BitExtractMode::kBlockTableUnrolled8:
        {
            auto const bt = bit_extract_block_table_preprocess(mask.mask);
            cb([&](std::uint64_t x) noexcept {
                return bit_extract_block_table_unrolled<8>(x, bt);
            });
            break;
        }

        case BitExtractMode::kButterflyTable:
        {
            auto const bf = bit_extract_butterfly_table_preprocess(mask.mask);
            cb([&](std::uint64_t x) noexcept {
                return bit_extract_butterfly_table(x, bf);
            });
            break;
        }
    }
}
