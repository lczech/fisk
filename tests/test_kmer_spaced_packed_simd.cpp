#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/random.hpp"
#include "fisk/core/simd.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "fisk/kmer_spaced/packed_simd.hpp"
#include "fisk/kmer_spaced/simd.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// Sequence lengths 0..100, which covers sequences shorter than the span, the tail, and the main
// loop for every span and arch, plus two longer ones, each with random per-position content.
static std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(8008);
        for (std::size_t len = 0; len <= 100; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        out.push_back(random_sequence(rng, 257));
        out.push_back(random_sequence(rng, 511));
        return out;
    }();
    return seqs;
}

// Random masks of the given span, which keep the first and last position.
template <std::size_t N>
static std::array<std::string, N> random_masks(Splitmix64& rng, std::size_t span_k)
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

template <typename Kernel, Layout L, std::size_t N>
static std::array<SpacedKmerMask<L, Kernel>, N> make_kernels(
    std::array<std::string, N> const& masks
) {
    std::array<SpacedKmerMask<L, Kernel>, N> kernels;
    for (std::size_t m = 0; m < N; ++m) {
        kernels[m] = make_spaced_kmer_kernel<Kernel, L>(masks[m]);
    }
    return kernels;
}

// Runs the extraction with a vector callback, and returns its spaced k-mers as events, in the
// order of the scalar callbacks. Checks the structure of the vectors on the way: consecutive
// positions per mask, the masks in order per vector of positions, and zero lanes past the valid
// ones.
template <typename Simd, typename Extract>
static std::vector<SpacedEvent> collect_vector_events(std::size_t mask_count, Extract&& extract)
{
    constexpr std::size_t lanes = Simd::lanes;
    std::vector<SpacedEvent> events;
    std::vector<std::size_t> next_pos(mask_count, 0);
    std::size_t expected_mask = 0;
    extract([&](
        std::size_t pos, std::size_t mask_idx, typename Simd::simd_vector vec,
        std::size_t valid_count
    ) {
        EXPECT_EQ(mask_idx, expected_mask);
        expected_mask = (expected_mask + 1) % mask_count;
        EXPECT_TRUE(mask_idx < mask_count);
        if (mask_idx >= mask_count) {
            return;
        }
        EXPECT_EQ(pos, next_pos[mask_idx]);
        next_pos[mask_idx] = pos + valid_count;
        EXPECT_TRUE(valid_count >= 1 && valid_count <= lanes);

        alignas(64) std::uint64_t buf[lanes];
        Simd::store_aligned(vec, buf);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            if (lane < valid_count) {
                events.push_back({pos + lane, mask_idx, buf[lane]});
            } else {
                EXPECT_EQ(buf[lane], std::uint64_t{0});
            }
        }
    });
    EXPECT_EQ(expected_mask, std::size_t{0});
    std::stable_sort(events.begin(), events.end(), [](auto const& lhs, auto const& rhs) {
        return lhs.pos < rhs.pos;
    });
    return events;
}

// =================================================================================================
//     Extraction
// =================================================================================================

// Checks one kernel type for one set of masks, through the array overload, and, for a single mask,
// also through the single-kernel overload, in all callback forms, against the oracle.
template <typename Kernel, Encoding E, Layout L, std::size_t N>
static void check_kernel(
    std::array<std::string, N> const& mask_strings, PackedSequence<E, L> const& packed,
    std::vector<SpacedEvent> const& expected
) {
    using Simd = typename Kernel::simd_arch;
    auto const kernels = make_kernels<Kernel, L>(mask_strings);

    auto const runs = collect_spaced_callback_forms<L, N == 1>([&](auto const& callback) {
        for_each_spaced_kmer_packed_simd(packed, kernels, callback);
    });
    for (auto const& got : runs) {
        check_spaced_events(got, expected);
    }
    check_spaced_events(collect_vector_events<Simd>(N, [&](auto const& callback) {
        for_each_spaced_kmer_packed_simd(packed, kernels, callback);
    }), expected);

    if constexpr (N == 1) {
        auto const single_runs = collect_spaced_callback_forms<L, true>(
            [&](auto const& callback) {
                for_each_spaced_kmer_packed_simd(packed, kernels[0], callback);
            }
        );
        for (auto const& got : single_runs) {
            check_spaced_events(got, expected);
        }
        check_spaced_events(collect_vector_events<Simd>(1, [&](auto const& callback) {
            for_each_spaced_kmer_packed_simd(packed, kernels[0], callback);
        }), expected);
    }
}

// Checks the butterfly and block table kernels of every arch of the build for one set of masks.
template <Encoding E, Layout L, std::size_t N>
static void check_all_kernels(std::array<std::string, N> const& mask_strings)
{
    WordEncoderButterfly<E, L> const encoder;
    std::vector<std::string> const mask_vector(mask_strings.begin(), mask_strings.end());
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence(seq, encoder);
        auto const expected = oracle_spaced_events<E, L>(seq, mask_vector);
        auto const check_arch = [&]<typename Simd>() {
            check_kernel<BitExtractKernelButterfly<Simd>>(mask_strings, packed, expected);
            check_kernel<BitExtractKernelBlock<Simd>>(mask_strings, packed, expected);
        };
        check_arch.template operator()<SimdScalar>();
        #if defined(FISK_HAS_SSE2)
        check_arch.template operator()<SimdSSE2>();
        #endif
        #if defined(FISK_HAS_AVX2)
        check_arch.template operator()<SimdAVX2>();
        #endif
        #if defined(FISK_HAS_AVX512)
        check_arch.template operator()<SimdAVX512>();
        #endif
        #if defined(FISK_HAS_NEON)
        check_arch.template operator()<SimdNEON>();
        #endif
    }
}

// For every span in [1, 32], a single random mask, and a set of three random masks, plus masks
// that keep all positions, or only the first and the last.
template <Encoding E, Layout L>
static void check_extraction()
{
    Splitmix64 rng(9009);
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_all_kernels<E, L>(random_masks<1>(rng, span_k));
        check_all_kernels<E, L>(random_masks<3>(rng, span_k));
    }
    check_all_kernels<E, L, 1>({ std::string(32, '1') });
    check_all_kernels<E, L, 2>({ std::string(29, '1'), "1" + std::string(27, '0') + "1" });
    check_all_kernels<E, L, 1>({ "1" + std::string(30, '0') + "1" });
}

// =================================================================================================
//     Error Contract
// =================================================================================================

struct NoopSpacedCallback
{
    template <typename K>
    void operator()(std::size_t, std::size_t, K const&) const {}
};

// The kernels need to have the Layout of the sequence, and to be built on a SIMD arch.
template <typename Seq, typename Kernel>
concept CanExtractPackedSimd = requires(Seq const& seq, Kernel const& kernel) {
    for_each_spaced_kmer_packed_simd(seq, kernel, NoopSpacedCallback{});
};
static_assert(CanExtractPackedSimd<
    PackedSequence<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelButterfly<SimdScalar>>
>);
static_assert(CanExtractPackedSimd<
    PackedSequence<Encoding::kACTG, Layout::kLSB>,
    SpacedKmerMask<Layout::kLSB, BitExtractKernelBlock<SimdScalar>>
>);
static_assert(!CanExtractPackedSimd<
    PackedSequence<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kLSB, BitExtractKernelButterfly<SimdScalar>>
>);
static_assert(!CanExtractPackedSimd<
    PackedSequence<Encoding::kACGT, Layout::kMSB>, BitExtractKernelButterfly<SimdScalar>
>);
#if defined(FISK_HAS_BMI2)
static_assert(!CanExtractPackedSimd<
    PackedSequence<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelPEXT<4>>
>);
#endif

// Spans outside of [1, 32] and sets of masks with different spans are rejected, also for sequences
// shorter than the span.
template <Layout L>
static void check_error_contract()
{
    using Kernel = BitExtractKernelButterfly<SimdScalar>;
    WordEncoderButterfly<Encoding::kACGT, L> const encoder;
    for (std::string const& seq : {std::string{}, std::string(129, 'T')}) {
        auto const packed = pack_sequence(seq, encoder);
        for (std::size_t const span_k : {std::size_t{0}, std::size_t{33}}) {
            SpacedKmerMask<L, Kernel> kernel;
            static_cast<Kernel&>(kernel) = Kernel(3);
            kernel.span_k = span_k;
            EXPECT_THROW(
                for_each_spaced_kmer_packed_simd(packed, kernel, NoopSpacedCallback{}),
                std::invalid_argument
            );
        }

        auto const mixed = std::array<SpacedKmerMask<L, Kernel>, 2>{
            make_spaced_kmer_kernel<Kernel, L>("101"), make_spaced_kmer_kernel<Kernel, L>("1001")
        };
        EXPECT_THROW(
            for_each_spaced_kmer_packed_simd(packed, mixed, NoopSpacedCallback{}),
            std::invalid_argument
        );
    }
}

// =================================================================================================
//     Tests
// =================================================================================================

TEST(KmerSpacedPackedSimd, AcgtMsb)
{
    check_extraction<Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerSpacedPackedSimd, AcgtLsb)
{
    check_extraction<Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerSpacedPackedSimd, ActgMsb)
{
    check_extraction<Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerSpacedPackedSimd, ActgLsb)
{
    check_extraction<Encoding::kACTG, Layout::kLSB>();
}

TEST(KmerSpacedPackedSimd, ErrorContract)
{
    check_error_contract<Layout::kMSB>();
    check_error_contract<Layout::kLSB>();
}
