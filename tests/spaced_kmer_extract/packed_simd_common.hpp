#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/util/random.hpp"
#include "fisk/simd/avx2.hpp"
#include "fisk/simd/avx512.hpp"
#include "fisk/simd/neon.hpp"
#include "fisk/simd/scalar.hpp"
#include "fisk/simd/sse2.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "simd_lanes.hpp"
#include "testing.hpp"

// =================================================================================================
//     Spaced K-mer Extraction from Packed Sequences, SIMD: Shared Test Helpers
// =================================================================================================

// The helpers of the tests of packed_bytewise_simd.hpp, which are split into
// test_packed_bytewise_simd_unchecked.cpp and test_packed_bytewise_simd_checked.cpp, so that each
// compiles with less memory.

// Sequence lengths 0..100, which covers sequences shorter than the span, the last step, and the
// main loop for every span and arch, plus two longer ones, each with random per-position content.
inline std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        fisk::Splitmix64 rng(8008);
        for (std::size_t len = 0; len <= 100; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        out.push_back(random_sequence(rng, 257));
        out.push_back(random_sequence(rng, 511));
        return out;
    }();
    return seqs;
}

// Random sequences of the lengths 0..100 with lower and upper case nucleotides, without invalid
// characters, with scattered ones, and with many, plus longer ones with runs of N, as in
// assemblies, for the checked extraction.
inline std::vector<std::string> const& checked_test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        fisk::Splitmix64 rng(9119);
        for (std::size_t len = 0; len <= 100; ++len) {
            std::string seq = random_sequence(rng, len);
            double const rates[] = {0.0, 0.02, 0.2};
            inject_invalid(seq, rng, rates[len % 3], all_invalid_bytes());
            out.push_back(std::move(seq));
        }
        for (std::size_t len : {257u, 511u}) {
            std::string seq = random_sequence(rng, len);
            inject_invalid(seq, rng, 0.01, kHandPickedInvalid);
            for (std::size_t r = 0; r < 3; ++r) {
                std::size_t const run = 1 + rng.get_uint64() % 60;
                std::size_t const at = rng.get_uint64() % (seq.size() - run);
                seq.replace(at, run, run, 'N');
            }
            out.push_back(std::move(seq));
        }
        return out;
    }();
    return seqs;
}

// Random masks of the given span, which keep the first and last position.
template <std::size_t N>
inline std::array<std::string, N> random_masks(fisk::Splitmix64& rng, std::size_t span_k)
{
    std::array<std::string, N> masks;
    for (auto& mask : masks) {
        mask.assign(span_k, '0');
        for (auto& c : mask) {
            c = (rng.get_uint64() & 1u) ? '1' : '0';
        }
        mask.front() = '1';
        mask.back() = '1';
    }
    return masks;
}

template <typename Kernel, fisk::Layout L, std::size_t N>
inline std::array<fisk::SpacedKmerMask<L, Kernel>, N> make_kernels(
    std::array<std::string, N> const& masks
) {
    std::array<fisk::SpacedKmerMask<L, Kernel>, N> kernels;
    for (std::size_t m = 0; m < N; ++m) {
        kernels[m] = fisk::make_spaced_kmer_kernel<Kernel, L>(masks[m]);
    }
    return kernels;
}

// Runs the extraction with a vector callback, and returns its spaced k-mers as events, taken out
// of the vectors with for_each_valid_lane(), in the order of the scalar callbacks. Checks the
// structure of the vectors on the way: consecutive positions per mask, and the masks in order per
// vector of positions. With `Shorthand`, the callback is the single-mask form without mask_idx.
template <typename Simd, fisk::Encoding E, fisk::Layout L, bool Shorthand = false, typename Extract>
inline std::vector<SpacedEvent> collect_vector_events(std::size_t mask_count, Extract&& extract)
{
    using V = fisk::SpacedKmerVector<E, L, Simd>;
    constexpr std::size_t lanes = Simd::lanes;
    std::vector<SpacedEvent> events;
    std::vector<std::size_t> next_pos(mask_count, 0);
    std::size_t expected_mask = 0;
    // Not inlined into each step of the unrolled loops, as that multiplies the compile time.
    auto const on_vector = [&](
        std::size_t pos, std::size_t mask_idx, V spaced_kmers, std::size_t valid_count
    ) FISK_NOINLINE_LAMBDA {
        EXPECT_EQ(mask_idx, expected_mask);
        expected_mask = (expected_mask + 1) % mask_count;
        EXPECT_TRUE(mask_idx < mask_count);
        if (mask_idx >= mask_count) {
            return;
        }
        EXPECT_EQ(pos, next_pos[mask_idx]);
        next_pos[mask_idx] = pos + valid_count;
        EXPECT_TRUE(valid_count >= 1 && valid_count <= lanes);
        fisk::for_each_valid_lane(
            pos, spaced_kmers, std::min(valid_count, lanes),
            [&](std::size_t p, fisk::SpacedKmer<E, L> const& spaced_kmer) {
                events.push_back({p, mask_idx, fisk::kmer_value(spaced_kmer)});
            }
        );
    };
    if constexpr (Shorthand) {
        extract([&](std::size_t pos, V spaced_kmers, std::size_t valid_count) {
            on_vector(pos, 0, spaced_kmers, valid_count);
        });
    } else {
        extract(on_vector);
    }
    EXPECT_EQ(expected_mask, std::size_t{0});
    std::sort(events.begin(), events.end(), [](auto const& lhs, auto const& rhs) {
        return std::tie(lhs.pos, lhs.mask_idx) < std::tie(rhs.pos, rhs.mask_idx);
    });
    return events;
}
