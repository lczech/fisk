#pragma once

#include <cmath>
#include <cstdint>

// =================================================================================================
//     Random numbers
// =================================================================================================

/**
 * @brief Fast deterministic random bit generator via the splitmix64 algorithm.
 *
 * Splitmix64 is a pseudo-random number generator, which uses a fairly simple algorithm that,
 * though it is considered to be poor for cryptographic purposes, is very fast to calculate,
 * and is "good enough" for many random number needs. It passes several fairly rigorous PRNG
 * "fitness" tests that some more complex algorithms fail.
 *
 * Here, we implement a fixed-increment version of it. If you need a full implementation,
 * see for instance https://gist.github.com/imneme/6179748664e88ef3c34860f44309fc71
 *
 * @see http://dx.doi.org/10.1145/2714064.2660195
 */
class Splitmix64
{
public:

    // -------------------------------------------------
    //     Constructors and Rule of Five
    // -------------------------------------------------

    Splitmix64() = default;
    Splitmix64( std::uint64_t seed )
        : state_(seed)
    {}

    // -------------------------------------------------
    //     Member Functions
    // -------------------------------------------------

    inline void set_seed( std::uint64_t seed )
    {
        state_ = seed;
    }

    inline std::uint64_t get_uint64()
    {
        state_ += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = state_;
        z = ( z ^ ( z >> 30 ) ) * 0xbf58476d1ce4e5b9ULL;
        z = ( z ^ ( z >> 27 ) ) * 0x94d049bb133111ebULL;
        return z ^ ( z >> 31 );
    }

    inline double get_double()
    {
        return static_cast<double>( get_uint64() ) / two_pow_64_;
    }

    // -------------------------------------------------
    //     Private Members
    // -------------------------------------------------

private:
    std::uint64_t state_ = 0;
    double const two_pow_64_ = std::pow( 2.0, 64 );
};
