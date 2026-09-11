#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.hpp"
#include "fisk/kmer_extract/packed.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "fisk/core/seq_enc.hpp"
#include "microbench.hpp"

// =================================================================================================
//     Sum Hashing
// =================================================================================================

// Benchmark sinks for the for_each_kmer_packed_*() functions (kmer_extract/packed.hpp): sum every
// emitted k-mer into `hash`.

template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_narrow_blockwise(TwoBitSequence<Order> const& seq, std::size_t k)
{
    std::uint64_t hash = 0;
    for_each_kmer_packed_narrow_blockwise(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_narrow_fixed_k(TwoBitSequence<Order> const& seq, std::size_t k)
{
    std::uint64_t hash = 0;
    for_each_kmer_packed_narrow_fixed_k(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_narrow_rolling(
    TwoBitSequence<Order> const& seq, std::size_t k
) {
    std::uint64_t hash = 0;
    for_each_kmer_packed_narrow_rolling(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_wide_rolling(
    TwoBitSequence<Order> const& seq, std::size_t k
) {
    std::uint64_t hash = 0;
    for_each_kmer_packed_wide_rolling(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_wide_blockwise(TwoBitSequence<Order> const& seq, std::size_t k)
{
    std::uint64_t hash = 0;
    for_each_kmer_packed_wide_blockwise(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

// for_each_kmer_packed_wide_128() only exists where `unsigned __int128` does -- see the
// __SIZEOF_INT128__ guard around its definition in kmer_extract/packed.hpp.
#ifdef __SIZEOF_INT128__
template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_wide_128(TwoBitSequence<Order> const& seq, std::size_t k)
{
    std::uint64_t hash = 0;
    for_each_kmer_packed_wide_128(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}
#endif // __SIZEOF_INT128__

template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_wide_hybrid(
    TwoBitSequence<Order> const& seq, std::size_t k
) {
    std::uint64_t hash = 0;
    for_each_kmer_packed_wide_hybrid(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_wide_hoisted(
    TwoBitSequence<Order> const& seq, std::size_t k
) {
    std::uint64_t hash = 0;
    for_each_kmer_packed_wide_hoisted(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_wide_fixed_k(TwoBitSequence<Order> const& seq, std::size_t k)
{
    std::uint64_t hash = 0;
    for_each_kmer_packed_wide_fixed_k(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_wide_hybrid_hoisted(
    TwoBitSequence<Order> const& seq, std::size_t k
) {
    std::uint64_t hash = 0;
    for_each_kmer_packed_wide_hybrid_hoisted(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

// Experimental aligned-window variants use the same sum sink as the existing implementations.
template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_narrow_aligned(
    TwoBitSequence<Order> const& seq, std::size_t k
) {
    std::uint64_t hash = 0;
    for_each_kmer_packed_narrow_aligned(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_wide_aligned(
    TwoBitSequence<Order> const& seq, std::size_t k
) {
    std::uint64_t hash = 0;
    for_each_kmer_packed_wide_aligned(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

template <BitOrder Order>
inline std::uint64_t compute_kmer_hash_packed_wide_split_k(
    TwoBitSequence<Order> const& seq, std::size_t k
) {
    std::uint64_t hash = 0;
    for_each_kmer_packed_wide_split_k(seq, k, [&](std::uint64_t v) { hash += v; });
    return hash;
}

/**
 * @brief Benchmark k-mer extraction directly from a packed TwoBitSequence (kmer_extract/packed.hpp),
 * across both BitOrder conventions and both accumulator widths (narrow: single 64-bit register,
 * k<=29; wide: 128-bit-equivalent, k<=32).
 *
 * Packing is done once per sequence, outside the timed region: real usage would already have `seq`
 * packed ahead of time, so packing cost should not count against extraction cost here. See
 * bench_kmer_extract() (bench_kmer_extract.hpp) for the ASCII-input baseline to compare this
 * against, in the separate "kmer_extract" CSV suite.
 */
inline void bench_kmer_extract_packed(
    std::vector<std::string> const& sequences,
    std::size_t k_min,
    std::size_t k_max,
    std::ostream& csv_os
) {
    if (k_min < 1 || k_min > 32 || k_max < 1 || k_max > 32) {
        throw std::runtime_error("Invalid k outside of [1, 32]");
    }
    if (k_min > k_max) {
        throw std::runtime_error("Invalid k_min > k_max");
    }

    std::size_t const rounds = 8;
    std::size_t const repeats = 8;

    std::string const suite_title = "kmer_extract_packed";
    std::cout << "\n=== k-mer extract (packed) ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    std::vector<TwoBitSequence<BitOrder::Msb>> packed_msb;
    std::vector<TwoBitSequence<BitOrder::Lsb>> packed_lsb;
    packed_msb.reserve(sequences.size());
    packed_lsb.reserve(sequences.size());
    for (auto const& seq : sequences) {
        packed_msb.push_back(pack_sequence(seq, EncodeAcgt8ButterflyMsb{}));
        packed_lsb.push_back(pack_sequence(seq, EncodeAcgt8ButterflyLsb{}));
    }

    std::size_t const narrow_k_max = std::min<std::size_t>(k_max, 29);

    // -----------------------------------------------------------------------
    //     order=msb, k in [k_min, min(k_max, 29)]: narrow and wide
    // -----------------------------------------------------------------------
    for (std::size_t k = k_min; k <= narrow_k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rorder=msb k " << std::setw(2) << k << std::flush;
        }
        Microbench<TwoBitSequence<BitOrder::Msb>> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](TwoBitSequence<BitOrder::Msb> const& seq) {
                return static_cast<double>(seq.length - k + 1);
            }
        );
        auto results = suite.run(
            packed_msb,
            bench("narrow_blockwise", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_narrow_blockwise(seq, k);
            }),
            bench("narrow_fixed_k", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_narrow_fixed_k(seq, k);
            }),
            bench("narrow_aligned", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_narrow_aligned(seq, k);
            }),
            bench("narrow_rolling", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_narrow_rolling(seq, k);
            }),
            bench("wide_rolling", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_rolling(seq, k);
            }),
            bench("wide_blockwise", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_blockwise(seq, k);
            }),
#ifdef __SIZEOF_INT128__
            bench("wide_128", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_128(seq, k);
            }),
#endif // __SIZEOF_INT128__
            bench("wide_hybrid", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_hybrid(seq, k);
            }),
            bench("wide_hoisted", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_hoisted(seq, k);
            }),
            bench("wide_hybrid_hoisted", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_hybrid_hoisted(seq, k);
            }),
            bench("wide_fixed_k", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_fixed_k(seq, k);
            }),
            bench("wide_aligned", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_aligned(seq, k);
            }),
            bench("wide_split_k", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_split_k(seq, k);
            })
        );
        write_csv_rows(csv_os, suite_title, "order=msb;k=" + std::to_string(k), results);
    }

    // -----------------------------------------------------------------------
    //     order=msb, k in [30, k_max]: wide only
    // -----------------------------------------------------------------------
    for (std::size_t k = std::max<std::size_t>(k_min, 30); k <= k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rorder=msb k " << std::setw(2) << k << std::flush;
        }
        Microbench<TwoBitSequence<BitOrder::Msb>> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](TwoBitSequence<BitOrder::Msb> const& seq) {
                return static_cast<double>(seq.length - k + 1);
            }
        );
        auto results = suite.run(
            packed_msb,
            bench("wide_rolling", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_rolling(seq, k);
            }),
            bench("wide_blockwise", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_blockwise(seq, k);
            }),
#ifdef __SIZEOF_INT128__
            bench("wide_128", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_128(seq, k);
            }),
#endif // __SIZEOF_INT128__
            bench("wide_hybrid", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_hybrid(seq, k);
            }),
            bench("wide_hoisted", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_hoisted(seq, k);
            }),
            bench("wide_hybrid_hoisted", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_hybrid_hoisted(seq, k);
            }),
            bench("wide_fixed_k", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_fixed_k(seq, k);
            }),
            bench("wide_aligned", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_aligned(seq, k);
            }),
            bench("wide_split_k", [&](TwoBitSequence<BitOrder::Msb> const& seq) {
                return compute_kmer_hash_packed_wide_split_k(seq, k);
            })
        );
        write_csv_rows(csv_os, suite_title, "order=msb;k=" + std::to_string(k), results);
    }

    // -----------------------------------------------------------------------
    //     order=lsb, k in [k_min, min(k_max, 29)]: narrow and wide
    // -----------------------------------------------------------------------
    for (std::size_t k = k_min; k <= narrow_k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rorder=lsb k=" << std::setw(2) << k << std::flush;
        }
        Microbench<TwoBitSequence<BitOrder::Lsb>> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return static_cast<double>(seq.length - k + 1);
            }
        );
        auto results = suite.run(
            packed_lsb,
            bench("narrow_blockwise", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_narrow_blockwise(seq, k);
            }),
            bench("narrow_fixed_k", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_narrow_fixed_k(seq, k);
            }),
            bench("narrow_aligned", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_narrow_aligned(seq, k);
            }),
            bench("narrow_rolling", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_narrow_rolling(seq, k);
            }),
            bench("wide_rolling", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_rolling(seq, k);
            }),
            bench("wide_blockwise", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_blockwise(seq, k);
            }),
#ifdef __SIZEOF_INT128__
            bench("wide_128", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_128(seq, k);
            }),
#endif // __SIZEOF_INT128__
            bench("wide_hybrid", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_hybrid(seq, k);
            }),
            bench("wide_hoisted", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_hoisted(seq, k);
            }),
            bench("wide_hybrid_hoisted", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_hybrid_hoisted(seq, k);
            }),
            bench("wide_fixed_k", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_fixed_k(seq, k);
            }),
            bench("wide_aligned", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_aligned(seq, k);
            }),
            bench("wide_split_k", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_split_k(seq, k);
            })
        );
        write_csv_rows(csv_os, suite_title, "order=lsb;k=" + std::to_string(k), results);
    }

    // -----------------------------------------------------------------------
    //     order=lsb, k in [30, k_max]: wide only
    // -----------------------------------------------------------------------
    for (std::size_t k = std::max<std::size_t>(k_min, 30); k <= k_max; ++k) {
        if (stdout_is_terminal()) {
            std::cout << "\rorder=lsb k=" << std::setw(2) << k << std::flush;
        }
        Microbench<TwoBitSequence<BitOrder::Lsb>> suite(suite_title);
        suite.rounds(rounds).repeats(repeats).units_fn(
            [k](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return static_cast<double>(seq.length - k + 1);
            }
        );
        auto results = suite.run(
            packed_lsb,
            bench("wide_rolling", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_rolling(seq, k);
            }),
            bench("wide_blockwise", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_blockwise(seq, k);
            }),
#ifdef __SIZEOF_INT128__
            bench("wide_128", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_128(seq, k);
            }),
#endif // __SIZEOF_INT128__
            bench("wide_hybrid", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_hybrid(seq, k);
            }),
            bench("wide_hoisted", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_hoisted(seq, k);
            }),
            bench("wide_hybrid_hoisted", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_hybrid_hoisted(seq, k);
            }),
            bench("wide_fixed_k", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_fixed_k(seq, k);
            }),
            bench("wide_aligned", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_aligned(seq, k);
            }),
            bench("wide_split_k", [&](TwoBitSequence<BitOrder::Lsb> const& seq) {
                return compute_kmer_hash_packed_wide_split_k(seq, k);
            })
        );
        write_csv_rows(csv_os, suite_title, "order=lsb;k=" + std::to_string(k), results);
    }

    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}

inline void bench_kmer_extract_packed(
    std::vector<std::string> const& sequences,
    std::ostream& csv_os
) {
    bench_kmer_extract_packed(sequences, 1, 32, csv_os);
}
