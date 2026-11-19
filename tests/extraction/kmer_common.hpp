#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>
#include <type_traits>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extraction/kmer_estimate.hpp"
#include "extraction/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

// =================================================================================================
//     K-mer Candidate Checks
// =================================================================================================

// Every candidate of the estimate of for_each_kmer() that this CPU can run, run directly, with a
// callback of single k-mers, and with one that also takes vectors.
template <fisk::InputFormat I, fisk::InputCheck C, fisk::Encoding E, fisk::Layout L>
inline void check_kmer_candidates()
{
    using namespace fisk;
    using Candidates = typename detail::KmerEstimate<I, C, E, L>::type;
    auto const seqs = extraction_test_sequences<C>();

    for (std::size_t k : std::initializer_list<std::size_t>{1, 7, 15, 31, 32}) {
        std::size_t count = 0;
        for_each_available_candidate(Candidates{}, [&](auto const& candidate) {
            ++count;
            auto const name = std::remove_cvref_t<decltype(candidate)>::name();

            for (auto const& seq : seqs) {
                auto const expected = extraction_test_expected<C, E, L>(seq, k);
                auto const input = extraction_test_input<I, C, E, L>(seq);

                KmerRecorder single;
                candidate.run(k, input, single);
                check_emitted_kmers(expected, k, single.values, &single.positions);

                // The SIMD algorithms hand their vectors to a callback that takes them.
                KmerRecorder unpacked;
                std::size_t vectors = 0;
                auto both = callback_overload{
                    [&](std::size_t pos, Kmer<E, L> kmer) { unpacked(pos, kmer); },
                    [&](std::size_t pos, auto kmers, auto valid) {
                        ++vectors;
                        for_each_valid_lane(pos, kmers, valid, unpacked);
                    }
                };
                candidate.run(k, input, both);
                check_emitted_kmers(expected, k, unpacked.values, &unpacked.positions);
                bool const simd = name.find("simd") != std::string::npos;
                EXPECT_TRUE(simd ? (vectors > 0 || expected.empty()) : vectors == 0);
            }
        });
        EXPECT_NE(count, 0u);
    }
}
