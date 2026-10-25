#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/core/random.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_simd.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "simd_lanes.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// Sequence lengths 0..100, which covers sequences shorter than the span, the last step, and the
// main loop for every span and arch, plus two longer ones, each with random per-position content.
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

// Random sequences of the lengths 0..100 with lower and upper case nucleotides, without invalid
// characters, with scattered ones, and with many, plus longer ones with runs of N, as in
// assemblies, for the checked extraction.
static std::vector<std::string> const& checked_test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(9119);
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

// Runs the extraction with a vector callback, and returns its spaced k-mers as events, taken out
// of the vectors with for_each_valid_lane(), in the order of the scalar callbacks. Checks the
// structure of the vectors on the way: consecutive positions per mask, and the masks in order per
// vector of positions. With `Shorthand`, the callback is the single-mask form without mask_idx.
template <typename Simd, Encoding E, Layout L, bool Shorthand = false, typename Extract>
static std::vector<SpacedEvent> collect_vector_events(std::size_t mask_count, Extract&& extract)
{
    using V = SpacedKmerVector<E, L, Simd>;
    constexpr std::size_t lanes = Simd::lanes;
    std::vector<SpacedEvent> events;
    std::vector<std::size_t> next_pos(mask_count, 0);
    std::size_t expected_mask = 0;
    auto const on_vector = [&](
        std::size_t pos, std::size_t mask_idx, V spaced_kmers, std::size_t valid_count
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
        for_each_valid_lane(
            pos, spaced_kmers, std::min(valid_count, lanes),
            [&](std::size_t p, SpacedKmer<E, L> const& spaced_kmer) {
                events.push_back({p, mask_idx, kmer_value(spaced_kmer)});
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

// =================================================================================================
//     Extraction
// =================================================================================================

// Checks one kernel type for one set of masks, through the array overload, and, for a single mask,
// also through the single-kernel overload and the shorthand without mask_idx, against the oracle.
template <typename Kernel, Encoding E, Layout L, std::size_t N>
static void check_kernel(
    std::array<std::string, N> const& mask_strings, PackedSequence<E, L> const& packed,
    std::vector<SpacedEvent> const& expected
) {
    using Simd = typename Kernel::simd_arch;
    auto const kernels = make_kernels<Kernel, L>(mask_strings);
    auto const extract_array = [&](auto const& callback) {
        for_each_spaced_kmer_packed_unchecked_bytewise_simd(packed, kernels, callback);
    };
    check_spaced_events(collect_vector_events<Simd, E, L>(N, extract_array), expected);

    if constexpr (N == 1) {
        auto const extract_single = [&](auto const& callback) {
            for_each_spaced_kmer_packed_unchecked_bytewise_simd(packed, kernels[0], callback);
        };
        check_spaced_events(collect_vector_events<Simd, E, L>(1, extract_single), expected);
        check_spaced_events(
            collect_vector_events<Simd, E, L, true>(1, extract_single), expected
        );
        check_spaced_events(
            collect_vector_events<Simd, E, L, true>(1, extract_array), expected
        );
    }
}

// As check_kernel(), for the checked extraction, whose vectors come with their valid lanes.
template <typename Kernel, Encoding E, Layout L, std::size_t N>
static void check_checked_kernel(
    std::array<std::string, N> const& mask_strings, PackedSequenceWithValidity<E, L> const& packed,
    std::vector<SpacedEvent> const& expected
) {
    using Simd = typename Kernel::simd_arch;
    std::size_t const span_k = mask_strings.front().size();
    std::size_t const positions = packed.length() < span_k ? 0 : packed.length() - span_k + 1;
    auto const kernels = make_kernels<Kernel, L>(mask_strings);
    auto const collect = [&]<bool Shorthand>(std::size_t mask_count, auto const& extract) {
        return collect_valid_lanes_events<Simd, E, L, Shorthand, true>(
            mask_count, positions, extract
        );
    };
    auto const extract_array = [&](auto const& callback) {
        for_each_spaced_kmer_packed_checked_bytewise_simd(packed, kernels, callback);
    };
    check_spaced_events(collect.template operator()<false>(N, extract_array), expected);

    if constexpr (N == 1) {
        auto const extract_single = [&](auto const& callback) {
            for_each_spaced_kmer_packed_checked_bytewise_simd(packed, kernels[0], callback);
        };
        check_spaced_events(collect.template operator()<false>(1, extract_single), expected);
        check_spaced_events(collect.template operator()<true>(1, extract_single), expected);
        check_spaced_events(collect.template operator()<true>(1, extract_array), expected);
    }
}

// Checks the butterfly and block table kernels of every arch of the build, and the PEXT kernel, for
// one set of masks.
template <Encoding E, Layout L, std::size_t N>
static void check_all_kernels(std::array<std::string, N> const& mask_strings)
{
    WordEncoderButterfly<E, L> const encoder;
    std::vector<std::string> const mask_vector(mask_strings.begin(), mask_strings.end());
    for (auto const& seq : test_sequences()) {
        auto const packed = pack_sequence_unchecked(seq, encoder);
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
        #if defined(FISK_HAS_BMI2)
        check_kernel<BitExtractKernelPEXT>(mask_strings, packed, expected);
        #endif
    }
}

// As check_all_kernels(), for the checked extraction, on sequences with invalid characters.
template <Encoding E, Layout L, std::size_t N>
static void check_all_checked_kernels(std::array<std::string, N> const& mask_strings)
{
    WordEncoderButterfly<E, L> const encoder;
    std::vector<std::string> const mask_vector(mask_strings.begin(), mask_strings.end());
    for (auto const& seq : checked_test_sequences()) {
        auto const packed = pack_sequence_checked(seq, encoder);
        auto const expected = oracle_spaced_events<E, L>(seq, mask_vector);
        auto const check_arch = [&]<typename Simd>() {
            check_checked_kernel<BitExtractKernelButterfly<Simd>>(mask_strings, packed, expected);
            check_checked_kernel<BitExtractKernelBlock<Simd>>(mask_strings, packed, expected);
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
        #if defined(FISK_HAS_BMI2)
        check_checked_kernel<BitExtractKernelPEXT>(mask_strings, packed, expected);
        #endif
    }
}

// As check_extraction(), for the checked extraction.
template <Encoding E, Layout L>
static void check_checked_extraction()
{
    Splitmix64 rng(9229);
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_all_checked_kernels<E, L>(random_masks<1>(rng, span_k));
        check_all_checked_kernels<E, L>(random_masks<3>(rng, span_k));
    }
    check_all_checked_kernels<E, L, 1>({ std::string(32, '1') });
    check_all_checked_kernels<E, L, 1>({ std::string("1").append(30, '0').append("1") });
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
    check_all_kernels<E, L, 2>(
        { std::string(29, '1'), std::string("1").append(27, '0').append("1") }
    );
    check_all_kernels<E, L, 1>({ std::string("1").append(30, '0').append("1") });
}

// =================================================================================================
//     Error Contract
// =================================================================================================

struct NoopSpacedVectorCallback
{
    template <typename V>
    void operator()(std::size_t, std::size_t, V const&, std::size_t) const {}
};

struct NoopSpacedValidLanesCallback
{
    template <typename V, typename Lanes>
    void operator()(std::size_t, std::size_t, V const&, Lanes) const {}
};

// The kernels need to have the Layout of the sequence.
template <typename Seq, typename Kernel>
concept CanExtractPackedSimd = requires(Seq const& seq, Kernel const& kernel) {
    for_each_spaced_kmer_packed_unchecked_bytewise_simd(seq, kernel, NoopSpacedVectorCallback{});
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

static_assert(CanExtractPackedSimd<
    PackedSequence<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelPEXT>
>);

#endif

// The same for the checked extraction, which also needs the validity bits.
template <typename Seq, typename Kernel>
concept CanExtractPackedCheckedSimd = requires(Seq const& seq, Kernel const& kernel) {
    for_each_spaced_kmer_packed_checked_bytewise_simd(seq, kernel, NoopSpacedValidLanesCallback{});
};

static_assert(CanExtractPackedCheckedSimd<
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelButterfly<SimdScalar>>
>);
static_assert(!CanExtractPackedCheckedSimd<
    PackedSequenceWithValidity<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kLSB, BitExtractKernelButterfly<SimdScalar>>
>);
static_assert(!CanExtractPackedCheckedSimd<
    PackedSequence<Encoding::kACGT, Layout::kMSB>,
    SpacedKmerMask<Layout::kMSB, BitExtractKernelButterfly<SimdScalar>>
>);

// Spans outside of [1, 32] and sets of masks with different spans are rejected, also for sequences
// shorter than the span.
template <Layout L>
static void check_error_contract()
{
    using Kernel = BitExtractKernelButterfly<SimdScalar>;
    WordEncoderButterfly<Encoding::kACGT, L> const encoder;
    for (std::string const& seq : {std::string{}, std::string(129, 'T')}) {
        auto const packed = pack_sequence_unchecked(seq, encoder);
        auto const checked = pack_sequence_checked(seq, encoder);
        for (std::size_t const span_k : {std::size_t{0}, std::size_t{33}}) {
            SpacedKmerMask<L, Kernel> kernel;
            static_cast<Kernel&>(kernel) = Kernel(3);
            kernel.span_k = span_k;
            EXPECT_THROW(
                for_each_spaced_kmer_packed_unchecked_bytewise_simd(
                    packed, kernel, NoopSpacedVectorCallback{}
                ),
                std::invalid_argument
            );
            EXPECT_THROW(
                for_each_spaced_kmer_packed_checked_bytewise_simd(
                    checked, kernel, NoopSpacedValidLanesCallback{}
                ),
                std::invalid_argument
            );
        }

        auto const mixed = std::array<SpacedKmerMask<L, Kernel>, 2>{
            make_spaced_kmer_kernel<Kernel, L>("101"), make_spaced_kmer_kernel<Kernel, L>("1001")
        };
        EXPECT_THROW(
            for_each_spaced_kmer_packed_unchecked_bytewise_simd(
                packed, mixed, NoopSpacedVectorCallback{}
            ),
            std::invalid_argument
        );
        EXPECT_THROW(
            for_each_spaced_kmer_packed_checked_bytewise_simd(
                checked, mixed, NoopSpacedValidLanesCallback{}
            ),
            std::invalid_argument
        );
    }
}

// =================================================================================================
//     Tests
// =================================================================================================

TEST(SpacedKmerExtractPackedBytewiseSimd, UncheckedAcgtMsb)
{
    check_extraction<Encoding::kACGT, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, UncheckedAcgtLsb)
{
    check_extraction<Encoding::kACGT, Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, UncheckedActgMsb)
{
    check_extraction<Encoding::kACTG, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, UncheckedActgLsb)
{
    check_extraction<Encoding::kACTG, Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, CheckedAcgtMsb)
{
    check_checked_extraction<Encoding::kACGT, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, CheckedAcgtLsb)
{
    check_checked_extraction<Encoding::kACGT, Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, CheckedActgMsb)
{
    check_checked_extraction<Encoding::kACTG, Layout::kMSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, CheckedActgLsb)
{
    check_checked_extraction<Encoding::kACTG, Layout::kLSB>();
}

TEST(SpacedKmerExtractPackedBytewiseSimd, ErrorContract)
{
    check_error_contract<Layout::kMSB>();
    check_error_contract<Layout::kLSB>();
}
