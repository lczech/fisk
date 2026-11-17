#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "for_each_scaling/bench.hpp"
#include "microbench.hpp"
#include "seq_data.hpp"
#include "sink.hpp"

void bench_for_each_scaling(
    std::size_t max_length,
    double n_prob,
    std::uint64_t seed,
    std::ostream& csv_os
) {
    // The mask of the spaced k-mers, W22L31, the first of masks/default.txt. Its span is k, so that
    // both have the same number of k-mers per sequence.
    constexpr std::size_t k = kScalingK;
    ScalingMaskForms const mask_forms(std::vector<std::string>{ "1111011101110010111001011011111" });
    constexpr std::size_t min_length = 64;

    // User output
    std::string const suite_title = "for_each_scaling";
    std::cout << "\n=== for_each scaling ===\n";

    // Prepare csv output file with benchmark results
    write_csv_header(csv_os);
    if (max_length < min_length) {
        std::cout << "Skipped, as the maximum length is below " << min_length << "\n";
        return;
    }

    // The input of all lengths. With the seed and N probability of the other suites, its start is
    // their input.
    std::cout << "Generating input sequence of length " << max_length << "\n";
    InputSequence const input = random_acgt(max_length, n_prob, seed);

    // The buffer of the Write sink, see sink.hpp; Sum and Barrier ignore it.
    auto sink_buffer = make_sink_buffer();

    // The benchmark names: the algorithms of the estimate. The callback is in the case, as the
    // same algorithm can run for both.
    std::string const name_kmer_checked = name_var_kmer_checked();
    std::string const name_kmer_checked_vectors = name_var_kmer_checked_vectors();
    std::string const name_kmer_unchecked = name_var_kmer_unchecked();
    std::string const name_kmer_unchecked_vectors = name_var_kmer_unchecked_vectors();
    std::string const name_spaced_kmer_checked = name_var_spaced_kmer_checked(mask_forms);
    std::string const name_spaced_kmer_checked_vectors =
        name_var_spaced_kmer_checked_vectors(mask_forms);
    std::string const name_spaced_kmer_unchecked = name_var_spaced_kmer_unchecked(mask_forms);
    std::string const name_spaced_kmer_unchecked_vectors =
        name_var_spaced_kmer_unchecked_vectors(mask_forms);
    std::cout << "Estimate, one k-mer at a time:\n"
              << "  " << name_kmer_checked << "\n"
              << "  " << name_kmer_unchecked << "\n"
              << "  " << name_spaced_kmer_checked << "\n"
              << "  " << name_spaced_kmer_unchecked << "\n"
              << "Estimate, with vectors:\n"
              << "  " << name_kmer_checked_vectors << "\n"
              << "  " << name_kmer_unchecked_vectors << "\n"
              << "  " << name_spaced_kmer_checked_vectors << "\n"
              << "  " << name_spaced_kmer_unchecked_vectors << "\n";

    // Time per k-mer, or per spaced k-mer of the one mask.
    auto const units = [](std::string_view const& seq) {
        return static_cast<double>(seq.size() - k + 1);
    };

    // The variants differ in their sinks, so these are not compared.
    for (std::size_t length = min_length; length <= max_length; length *= 2) {
        std::vector<std::string_view> const inputs = { std::string_view(input).substr(0, length) };
        std::string const length_label = "length=" + std::to_string(length);
        std::cout << "\nLength " << length << "\n";

        Microbench<std::string_view> single(suite_title);
        single.units_fn(units).check_sinks(false);
        auto const single_results = single.run(
            inputs,
            bench(name_kmer_checked, [&](std::string_view seq) {
                return run_var_kmer_checked(seq, sink_buffer);
            }),
            bench(name_kmer_unchecked, [&](std::string_view seq) {
                return run_var_kmer_unchecked(seq, sink_buffer);
            }),
            bench(name_spaced_kmer_checked, [&](std::string_view seq) {
                return run_var_spaced_kmer_checked(seq, mask_forms, sink_buffer);
            }),
            bench(name_spaced_kmer_unchecked, [&](std::string_view seq) {
                return run_var_spaced_kmer_unchecked(seq, mask_forms, sink_buffer);
            })
        );
        write_csv_rows(
            csv_os, suite_title, "callback=single;" + length_label, single_results, kSinkName
        );

        Microbench<std::string_view> vectors(suite_title);
        vectors.units_fn(units).check_sinks(false);
        auto const vectors_results = vectors.run(
            inputs,
            bench(name_kmer_checked_vectors, [&](std::string_view seq) {
                return run_var_kmer_checked_vectors(seq, sink_buffer);
            }),
            bench(name_kmer_unchecked_vectors, [&](std::string_view seq) {
                return run_var_kmer_unchecked_vectors(seq, sink_buffer);
            }),
            bench(name_spaced_kmer_checked_vectors, [&](std::string_view seq) {
                return run_var_spaced_kmer_checked_vectors(seq, mask_forms, sink_buffer);
            }),
            bench(name_spaced_kmer_unchecked_vectors, [&](std::string_view seq) {
                return run_var_spaced_kmer_unchecked_vectors(seq, mask_forms, sink_buffer);
            })
        );
        write_csv_rows(
            csv_os, suite_title, "callback=vectors;" + length_label, vectors_results, kSinkName
        );
    }
}
