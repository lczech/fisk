#pragma once

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "seq_data.hpp"

/**
 * @brief Benchmark the original CLARK implementation vs our improved one.
 */
void bench_kmer_clark(
    std::vector<InputSequence> const& sequences,
    std::ostream& csv_os
);

// Kernels compared above, each defined in its own translation unit
// (var_original.cpp, var_improved.cpp).
std::uint64_t run_var_original(InputSequence const& seq);
std::uint64_t run_var_improved(InputSequence const& seq);
