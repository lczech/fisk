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
#include "utils.hpp"

// =================================================================================================
//     Adaptive Bit Extract Helper
// =================================================================================================

/**
 * @brief Helper to adaptively select the fastest bit extract implementation for a given fixed mask.
 *
 * Upon construction, a benchmark is run that tests hardware PEXT and several flavors of software
 * bit extract implementations, and selects the most performant one for the given mask. Then, use
 * `operator()` to apply this mask to a value.
 *
 * This helper is intended for usage with a single fixed mask (or few masks, in separate
 * instanciations of this class). When the mask is not fixed, this strategy is not beneficial,
 * as precomputing lookup tables specific for the mask does not make sense then.
 *
 * There is an overhead for the calling here, as we are using a function pointer to dispatch.
 * In the very hot loop of k-mer extraction, this can be a bit too slow; we hence to not use
 * this class in performance-critical code. It could be adapted to just benchmark the algorithm,
 * and return the best mode. Then, the better solution for high performance is to build the full
 * k-mer extraction loop templated, and use a `switch` to instanciate it for each case of algorithm
 * present here. Left as future work, as this is mostly relevant for actual applications, but not
 * our benchmark here.
 */
class AdaptiveBitExtract
{
public:

    // -------------------------------------------------------------------------
    //     Typedefs and Enums
    // -------------------------------------------------------------------------

    /**
     * @brief Bit extract on a given value, assuming a fixed mask.
     */
    using BitExtractFunction = std::uint64_t (AdaptiveBitExtract::*)(std::uint64_t) const;

    /**
     * @brief Mode, either hardware or our fastest software implementation using a block table.
     *
     * If mode `kAutomatic` is selected in the constructor, a small benchmark is run,
     * which selects the fastest algorithm, and sets it for usage.
     */
    enum class ExtractMode : int
    {
        kAutomatic,
        kPext,
        kByteTable,
        kBlockTable,
        kBlockTableUnrolled2,
        kBlockTableUnrolled4,
        kBlockTableUnrolled8,
        kButterflyTable,
    };

    // -------------------------------------------------------------------------
    //     Constructors and Rule of Five
    // -------------------------------------------------------------------------

    /**
     * @brief Default constructed instance; cannot be used.
     */
    AdaptiveBitExtract()
    {
        // Set the functor to a dummy, so that we get a proper exception instead of
        // a nullptr dereference, if operator() is called on a default constructed instance.
        bit_extr_func_ = &AdaptiveBitExtract::bit_extract_dummy_;
    }

    /**
     * @brief Construct an instance for a given mask.
     *
     * This sets the mask and the mode, by default using auto-tuning to the fastest bit extract
     * implementation, by running a small benchmark internally.
     */
    AdaptiveBitExtract( std::uint64_t mask, ExtractMode mode = ExtractMode::kAutomatic )
        : mode_(mode)
        , mask_(mask)
    {
        // Prepare the masks. We do not need block tables if we hard-set pext mode.
        if( mode_ != ExtractMode::kPext ) {
            block_table_ = bit_extract_block_table_preprocess(mask);
            butterfly_table_ = bit_extract_butterfly_table_preprocess(mask);
        }

        // Find the fastest mode if requested, or hard-set the extractor function.
        if( mode_ == ExtractMode::kAutomatic ) {
            tune_to_fastest_mode_();
        } else {
            set_bit_extr_func_( mode_ );
        }
    }

    ~AdaptiveBitExtract() = default;

    AdaptiveBitExtract( AdaptiveBitExtract const& ) = default;
    AdaptiveBitExtract( AdaptiveBitExtract&& )      = default;

    AdaptiveBitExtract& operator= ( AdaptiveBitExtract const& ) = default;
    AdaptiveBitExtract& operator= ( AdaptiveBitExtract&& )      = default;

    // -------------------------------------------------------------------------
    //     Getters and Operators
    // -------------------------------------------------------------------------

    /**
     * @brief Bit extract a given @p value, by applying the mask through the given mode.
     */
    inline std::uint64_t operator() ( std::uint64_t value ) const
    {
        return (this->*bit_extr_func_)(value);
    }

    /**
     * @brief Obtain the currently set bit extract function.
     *
     * This might be useful to apply it externally. See `operator()` though to apply it to a value.
     */
    inline BitExtractFunction bit_extract_function() const
    {
        return bit_extr_func_;
    }

    /**
     * @brief Get the mode used by this instance.
     */
    ExtractMode mode() const
    {
        return mode_;
    }

    /**
     * @brief Get the mode used by this instance as a printable string.
     */
    std::string mode_name() const
    {
        return mode_name(mode_);
    }

    /**
     * @brief Get the mode as a printable string for a given ExtractMode
     */
    static std::string mode_name( ExtractMode mode )
    {
        switch(mode) {
            case ExtractMode::kAutomatic: {
                return "Automatic";
            }
            case ExtractMode::kPext: {
                return "Pext";
            }
            case ExtractMode::kByteTable: {
                return "ByteTable";
            }
            case ExtractMode::kBlockTable: {
                return "BlockTable";
            }
            case ExtractMode::kBlockTableUnrolled2: {
                return "BlockTableUnrolled2";
            }
            case ExtractMode::kBlockTableUnrolled4: {
                return "BlockTableUnrolled4";
            }
            case ExtractMode::kBlockTableUnrolled8: {
                return "BlockTableUnrolled8";
            }
            case ExtractMode::kButterflyTable: {
                return "ButterflyTable";
            }
            default: {
                throw std::invalid_argument(
                    "Invalid ExtractMode in AdaptiveBitExtract: " +
                    std::to_string(static_cast<int>(mode))
                );
            }
        }
    }

    /**
     * @brief Number of different modes offered here, including `ExtractMode::kAutomatic`.
     */
    static size_t mode_count()
    {
        // Ensure that this is set up correctly. Still some room for error, but less likely.
        static const size_t num = 8;
        static_assert( static_cast<int>(ExtractMode::kButterflyTable) == num - 1 );
        return num;
    }

    // -------------------------------------------------------------------------
    //     Internal Functions
    // -------------------------------------------------------------------------

private:

    void tune_to_fastest_mode_()
    {
        // Prepare some random data that is long enough for a meaningful test,
        // but does usually not exceed cache space, to avoid testing ram speed
        // instead of actual algorithm performance.
        size_t const n = ( 2 << 16 );
        std::vector<std::uint64_t> data;
        data.reserve(n);
        Splitmix64 rng{};
        for( size_t i = 0; i < n; ++i ) {
            data.push_back( rng.get_uint64() );
        }

        // Find the fastest mode for the given mask.
        // We store and check the accumulated value, to avoid that
        // the compiler optimizes out the whole loop below.
        auto best_mode = ExtractMode::kAutomatic;
        auto best_time = std::numeric_limits<double>::infinity();
        auto best_accu = static_cast<std::uint64_t>(0);

        // Static assert that the enum values are as expected, to remind us
        // about them should more algorithms be added in the future.
        static_assert( static_cast<int>(ExtractMode::kPext) == 1 );
        static_assert( static_cast<int>(ExtractMode::kByteTable) == 2 );
        static_assert( static_cast<int>(ExtractMode::kButterflyTable) == 7 );

        // Depending on hardware, we might not want to test hardware pext, if not available.
        #if defined(HAVE_BMI2)
            auto const first_mode = ExtractMode::kPext;
        #else
            auto const first_mode = ExtractMode::kByteTable;
        #endif
        auto const last_mode = ExtractMode::kButterflyTable;

        // Try all algorithms, benchmarking which one is the fastest.
        for(
            ExtractMode m = first_mode;
            m <= last_mode;
            m = static_cast<ExtractMode>(static_cast<int>(m) + 1)
        ) {
            set_bit_extr_func_( m );

            // Run the mode for all test data, summing up the masked values.
            std::uint64_t accu = 0;
            auto const time_b = std::chrono::steady_clock::now();
            for( auto const in : data ) {
                accu += static_cast<std::uint64_t>( operator()(in) );
            }
            auto const time_e = std::chrono::steady_clock::now();
            double const time = std::chrono::duration<double>(time_e - time_b).count();

            // Check the accumulated value, to make sure the compiler cannot optimize this out.
            if( best_accu == 0 ) {
                best_accu = accu;
            }
            if( best_accu != accu ) {
                // This only happens if our implementation is wrong, and should never trigger.
                throw std::runtime_error(
                    "Bit extract benchmarking with inconsistent results for different methods."
                );
            }

            // Update the mode if it was faster.
            if( time < best_time ) {
                best_mode = m;
                best_time = time;
                best_accu = accu;
            }
        }

        // Finally, set the fastest algorithm.
        mode_ = best_mode;
        set_bit_extr_func_( best_mode );
    }

    void set_bit_extr_func_( ExtractMode mode )
    {
        // Set the function pointer to one of the wrapper functions below, which are needed
        // to capture the mask and block table, and forward them to the actual call.
        switch(mode) {
            case ExtractMode::kAutomatic: {
                // We only use automatic to indicate that we want to run benchmarking.
                // This function here is hence only ever called for actual algorithms,
                // such that this exception here should never trigger.
                throw std::invalid_argument(
                    "Cannot set adaptive bit extract function to automatic."
                );
            }
            case ExtractMode::kPext: {
                bit_extr_func_ = &AdaptiveBitExtract::bit_extract_pext_;
                break;
            }
            case ExtractMode::kByteTable: {
                bit_extr_func_ = &AdaptiveBitExtract::bit_extract_byte_table_;
                break;
            }
            case ExtractMode::kBlockTable: {
                bit_extr_func_ = &AdaptiveBitExtract::bit_extract_block_table_;
                break;
            }
            case ExtractMode::kBlockTableUnrolled2: {
                bit_extr_func_ = &AdaptiveBitExtract::bit_extract_block_table_unrolled2_;
                break;
            }
            case ExtractMode::kBlockTableUnrolled4: {
                bit_extr_func_ = &AdaptiveBitExtract::bit_extract_block_table_unrolled4_;
                break;
            }
            case ExtractMode::kBlockTableUnrolled8: {
                bit_extr_func_ = &AdaptiveBitExtract::bit_extract_block_table_unrolled8_;
                break;
            }
            case ExtractMode::kButterflyTable: {
                bit_extr_func_ = &AdaptiveBitExtract::bit_extract_butterfly_table_;
                break;
            }
            default: {
                // User error.
                throw std::invalid_argument(
                    "Invalid ExtractMode in AdaptiveBitExtract: " + std::to_string(static_cast<int>(mode))
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    //     Wrappers
    // -------------------------------------------------------------------------

    // These member functions are simple wrappers that call the underlying implementations
    // with the masks and preprocessed tables stored here.

    inline std::uint64_t bit_extract_pext_( std::uint64_t value ) const
    {
        #if defined(HAVE_BMI2)
            return bit_extract_pext( value, mask_ );
        #else
            // This function will be called if ExtractMode::kPext is set at construction,
            // which is invalid if not available on the given hardware.
            // In automatic mode, it will not be tested or set.
            (void) value;
            throw std::runtime_error(
                "Invalid use of hardware PEXT mode on architecture without BMI2 instructions. "
                "Cannot use AdaptiveBitExtract::ExtractMode::kPext on given hardware."
            );
        #endif
    }

    inline std::uint64_t bit_extract_byte_table_( std::uint64_t value ) const
    {
        return bit_extract_byte_table( value, mask_ );
    }

    inline std::uint64_t bit_extract_block_table_( std::uint64_t value ) const
    {
        return bit_extract_block_table( value, block_table_ );
    }

    inline std::uint64_t bit_extract_block_table_unrolled2_( std::uint64_t value ) const
    {
        return bit_extract_block_table_unrolled<2>( value, block_table_ );
    }

    inline std::uint64_t bit_extract_block_table_unrolled4_( std::uint64_t value ) const
    {
        return bit_extract_block_table_unrolled<4>( value, block_table_ );
    }

    inline std::uint64_t bit_extract_block_table_unrolled8_( std::uint64_t value ) const
    {
        return bit_extract_block_table_unrolled<8>( value, block_table_ );
    }

    inline std::uint64_t bit_extract_butterfly_table_( std::uint64_t value ) const
    {
        return bit_extract_butterfly_table( value, butterfly_table_ );
    }

    inline std::uint64_t bit_extract_dummy_( std::uint64_t ) const
    {
        throw std::invalid_argument(
            "Invalid call to operator() on default-constructed AdaptiveBitExtract instance"
        );
    }

    // -------------------------------------------------------------------------
    //     Member variables
    // -------------------------------------------------------------------------

private:

    // Extraction mode and function pointer
    ExtractMode mode_;
    BitExtractFunction bit_extr_func_ = nullptr;

    // Bit extract mask and block table for the block algorithm
    BitExtractMask mask_;
    BitExtractBlockTable block_table_;
    BitExtractButterflyTable butterfly_table_;
};
