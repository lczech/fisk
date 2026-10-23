#include <algorithm>
#include <bit>
#include <deque>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "kmer_spaced_packed_checked_multi/bench.hpp"
#include "microbench.hpp"
#include "utils.hpp"

using namespace fisk;

// Run the benchmarks of all techniques for the masks in Layout `L`, on the sequences packed in
// that Layout, in their own suite.run(), as Microbench::run() cross-validates the sinks of all
// benches within one call, and the Layouts emit different values.
template <Layout L>
static void bench_kmer_spaced_packed_checked_multi_layout_(
    std::vector<PackedSequenceWithValidity<Encoding::kACGT, L>> const& sequences,
    std::vector<std::string> const& masks,
    std::vector<std::uint64_t>& sink_buffer,
    std::ostream& csv_os,
    std::string const& suite_title,
    std::string const& case_label,
    int rounds,
    int repeats
) {
    using Packed = PackedSequenceWithValidity<Encoding::kACGT, L>;
    std::string const prefix = (L == Layout::kMSB) ? "msb" : "lsb";

    // Benchmark names are string views, so their strings need to outlive the run.
    std::deque<std::string> names;
    auto const name = [&](std::string_view technique) -> std::string_view {
        return names.emplace_back(prefix + "_" + std::string(technique));
    };

    // Prepare a benchmark with repititions
    std::size_t const k = masks[0].size();
    double const mask_count = static_cast<double>(masks.size());
    Microbench<Packed> suite(suite_title);
    suite
        .rounds(rounds)
        .repeats(repeats)
        .units_fn([k, mask_count](Packed const& seq) {
            // 1 unit per original (unspaced) k-mer and per mask
            return static_cast<double>(seq.length() - k + 1) * mask_count;
        });

    // Prepare masks for all implementations as needed
    auto const bit_ext_masks = make_spaced_kmer_masks<L>(masks);
    auto const bit_ext_block_masks = make_spaced_kmer_block_tables(bit_ext_masks);
    auto const bit_ext_butterfly_tables = make_spaced_kmer_butterfly_tables(bit_ext_masks);

    // Run the benchmark for all algorithms
    auto results = suite.run(
        sequences,
        #if defined(FISK_HAS_BMI2)
        bench(
            name("pext"),
            [&](Packed const& seq) {
                return run_var_pext(seq, bit_ext_masks, sink_buffer);
            }
        ),
        #endif
        bench(
            name("bitloop"),
            [&](Packed const& seq) {
                return run_var_bitloop(seq, bit_ext_masks, sink_buffer);
            }
        ),
        bench(
            name("byte_table"),
            [&](Packed const& seq) {
                return run_var_byte_table(seq, bit_ext_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table"),
            [&](Packed const& seq) {
                return run_var_block_table(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table_unrolled2"),
            [&](Packed const& seq) {
                return run_var_block_table_unrolled2(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table_unrolled4"),
            [&](Packed const& seq) {
                return run_var_block_table_unrolled4(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("block_table_unrolled8"),
            [&](Packed const& seq) {
                return run_var_block_table_unrolled8(seq, bit_ext_block_masks, sink_buffer);
            }
        ),
        bench(
            name("butterfly_table"),
            [&](Packed const& seq) {
                return run_var_butterfly_table(seq, bit_ext_butterfly_tables, sink_buffer);
            }
        )
    );
    write_csv_rows(csv_os, suite_title, case_label, results, kSinkName);
}

void bench_kmer_spaced_packed_checked_multi(
    std::vector<std::string> const& sequences,
    std::vector<std::vector<std::string>> const& multi_masks,
    std::ostream& csv_os
) {
    // Check masks for consistent value of k
    for (auto const& masks : multi_masks) {
        if (masks.size() == 0) {
            throw std::invalid_argument("Mask set is empty");
        }
        std::size_t k = masks[0].size();
        for (auto const& mask : masks) {
            if (mask.size() != k) {
                throw std::invalid_argument("Inconsistent mask lengths in mask set");
            }
        }
    }

    int const rounds = 1;
    int const repeats = 4;

    // User output
    std::string const suite_title = "kmer_spaced_packed_checked_multi";
    std::cout << "\n=== spaced k-mer extract from packed sequences multi masks (checked) ===\n";
    std::cout << "rounds=" << rounds << ", repeats=" << repeats << "\n";

    write_csv_header(csv_os);

    // Pack the sequences once per Layout, outside of the timed region.
    std::vector<CheckedMsb> packed_msb;
    std::vector<CheckedLsb> packed_lsb;
    packed_msb.reserve(sequences.size());
    packed_lsb.reserve(sequences.size());
    for (auto const& seq : sequences) {
        packed_msb.push_back(
            pack_sequence_with_validity(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kMSB>{})
        );
        packed_lsb.push_back(
            pack_sequence_with_validity(seq, WordEncoderButterfly<Encoding::kACGT, Layout::kLSB>{})
        );
    }

    // Backing storage for the Write sink (see sink.hpp), sized to the longest sequence actually
    // benchmarked, times the largest mask set, rounded up to a power of two so that no call here
    // ever wraps it, as in bench_kmer_spaced_multi().
    std::size_t max_seq_len = 0;
    for (auto const& seq : sequences) {
        max_seq_len = std::max(max_seq_len, seq.size());
    }
    std::size_t max_masks = 1;
    for (auto const& masks : multi_masks) {
        max_masks = std::max(max_masks, masks.size());
    }
    std::vector<std::uint64_t> sink_buffer(
        std::bit_ceil(std::max<std::size_t>(max_seq_len * max_masks, 1)), 0
    );

    // Run a benchmark for each mask set, in both Layouts
    for (std::size_t m = 0; m < multi_masks.size(); ++m) {
        if (stdout_is_terminal()) {
            std::cout << "\rmask set ";
            std::cout << std::setw(2) << (m + 1) << " / " << multi_masks.size() << "\n";
        }
        std::string const case_label = "mask_set=" + std::to_string(m);
        bench_kmer_spaced_packed_checked_multi_layout_<Layout::kMSB>(
            packed_msb, multi_masks[m], sink_buffer, csv_os, suite_title, case_label,
            rounds, repeats
        );
        bench_kmer_spaced_packed_checked_multi_layout_<Layout::kLSB>(
            packed_lsb, multi_masks[m], sink_buffer, csv_os, suite_title, case_label,
            rounds, repeats
        );
    }
    if (stdout_is_terminal()) {
        std::cout << "\n";
    }
}
