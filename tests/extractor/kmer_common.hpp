#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>

#include "fisk/core/kmer_callback.hpp"
#include "fisk/extractor/kmer_extractor.hpp"
#include "extractor/common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

// =================================================================================================
//     K-mer Extractor Checks
// =================================================================================================

// Every algorithm of the extractor, run with a callback of single k-mers, and with one that also
// takes vectors.
template <fisk::InputFormat I, fisk::InputCheck C, fisk::Encoding E, fisk::Layout L>
inline void check_kmer_extractor_algorithms()
{
    using namespace fisk;
    using Extractor = KmerExtractor<I, C, E, L>;
    auto const seqs = extractor_test_sequences<C>();

    for (std::size_t k : std::initializer_list<std::size_t>{1, 7, 15, 31, 32}) {
        Extractor extractor(k);
        auto const names = extractor.algorithm_names();
        EXPECT_FALSE(names.empty());

        for (auto const& name : names) {
            extractor.set_algorithm(name);
            EXPECT_EQ(extractor.algorithm_name(), name);

            for (auto const& seq : seqs) {
                auto const expected = extractor_test_expected<Extractor>(seq, k);
                auto const input = extractor_test_input<Extractor>(seq);

                KmerRecorder single;
                extractor.for_each(input, single);
                check_emitted_kmers(expected, k, single.values, &single.positions);

                // The SIMD algorithms hand their vectors to a callback that takes them.
                KmerRecorder unpacked;
                std::size_t vectors = 0;
                auto const both = callback_overload{
                    [&](std::size_t pos, Kmer<E, L> kmer) { unpacked(pos, kmer); },
                    [&](std::size_t pos, auto kmers, auto valid) {
                        ++vectors;
                        for_each_valid_lane(pos, kmers, valid, unpacked);
                    }
                };
                extractor.for_each(input, both);
                check_emitted_kmers(expected, k, unpacked.values, &unpacked.positions);
                bool const simd = name.find("simd") != std::string::npos;
                EXPECT_TRUE(simd ? (vectors > 0 || expected.empty()) : vectors == 0);
            }
        }
    }
}
