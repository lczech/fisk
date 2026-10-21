#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/core/kmer_hash.hpp"
#include "fisk/core/cast.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "fisk/kmer_spaced/selector.hpp"
#include "fisk/kmer_spaced/simd.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "simd_lanes.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

static bool event_less(SpacedEvent const& lhs, SpacedEvent const& rhs)
{
    if (lhs.pos != rhs.pos) {
        return lhs.pos < rhs.pos;
    }
    if (lhs.mask_idx != rhs.mask_idx) {
        return lhs.mask_idx < rhs.mask_idx;
    }
    return lhs.value < rhs.value;
}

static void check_event_multiset(
    std::vector<SpacedEvent> got, std::vector<SpacedEvent> expected
) {
    std::sort(got.begin(), got.end(), event_less);
    std::sort(expected.begin(), expected.end(), event_less);
    check_spaced_events(got, expected);
}

static std::vector<SpacedEvent> events_for_mask(
    std::vector<SpacedEvent> const& events, std::size_t mask_idx
) {
    std::vector<SpacedEvent> out;
    for (auto const& event : events) {
        if (event.mask_idx == mask_idx) {
            out.push_back(event);
        }
    }
    return out;
}

static void check_by_mask_events(
    std::vector<SpacedEvent> const& got,
    std::vector<SpacedEvent> const& expected,
    std::size_t mask_count
) {
    check_event_multiset(got, expected);
    for (std::size_t m = 0; m < mask_count; ++m) {
        auto const got_mask = events_for_mask(got, m);
        auto const expected_mask = events_for_mask(expected, m);
        check_spaced_events(got_mask, expected_mask);
        for (std::size_t i = 1; i < got_mask.size(); ++i) {
            EXPECT_TRUE(got_mask[i - 1].pos <= got_mask[i].pos);
        }
    }
}

// Sequences around the span and the SIMD block width: empty, one short of a span, and a span plus
// up to two blocks, with random content (rather than a periodic pattern, under which a k-mer taken
// from the wrong window could still have the right value). Then invalid characters: hand-picked
// ones at the start, around the end of the first window, and in runs, and random bytes from the
// full range at two densities.
static std::vector<std::string> sequence_corpus(std::size_t span_k, std::size_t lanes)
{
    Splitmix64 rng(1000 * span_k + lanes);
    std::size_t marker_idx = 0;
    auto marker = [&] { return kHandPickedInvalid[marker_idx++ % kHandPickedInvalid.size()]; };

    std::vector<std::string> sequences;
    sequences.push_back("");
    if (span_k > 1) {
        sequences.push_back(random_sequence(rng, span_k - 1));
    }
    for (std::size_t extra = 0; extra <= 2 * lanes; ++extra) {
        sequences.push_back(random_sequence(rng, span_k + extra));
    }

    std::string invalid = random_sequence(rng, span_k + 2 * lanes + 5);
    invalid[0] = marker();
    if (span_k > 2) {
        invalid[1] = marker();
    }
    invalid[span_k - 1] = marker();
    invalid[span_k] = marker();
    sequences.push_back(invalid);

    // N at index 1 is skipped by 10101, while N at index 2 is kept and suppresses position 0.
    for (std::string seq : {"ANCGTACGTACGT", "ACNGTACGTACGT", "ACGTNNNACGTACGT"}) {
        std::replace(seq.begin(), seq.end(), 'N', marker());
        sequences.push_back(seq);
    }

    for (double const rate : {0.05, 0.2}) {
        for (std::size_t i = 0; i < 4; ++i) {
            std::string seq = random_sequence(rng, span_k + 3 * lanes + i);
            inject_invalid(seq, rng, rate, all_invalid_bytes());
            sequences.push_back(std::move(seq));
        }
    }
    return sequences;
}

// A mask with arbitrary bits and span, bypassing make_spaced_kmer_mask(), to test invalid masks.
static SpacedKmerMask<Layout::kMSB, BitExtractMask> raw_spaced_kmer_mask(
    std::uint64_t bits, std::size_t span_k
) {
    SpacedKmerMask<Layout::kMSB, BitExtractMask> mask;
    static_cast<BitExtractMask&>(mask) = BitExtractMask(bits);
    mask.span_k = span_k;
    return mask;
}

template <std::size_t N>
static std::array<std::string, N> generated_masks()
{
    std::array<std::string, N> masks{};
    for (std::size_t m = 0; m < N; ++m) {
        std::string mask(9, '0');
        mask.front() = '1';
        mask.back() = '1';
        for (std::size_t i = 1; i + 1 < mask.size(); ++i) {
            if (((i + m) % 3) != 0) {
                mask[i] = '1';
            }
        }
        masks[m] = std::move(mask);
    }
    if constexpr (N >= 3) {
        masks[N - 1] = masks[0];
    }
    return masks;
}

template <std::size_t N>
static std::vector<std::string> to_vector(std::array<std::string, N> const& masks)
{
    return std::vector<std::string>(masks.begin(), masks.end());
}

// Keep both endpoints and vary retained interior positions across every legal span.
static std::string mask_for_span(std::size_t span_k)
{
    std::string mask(span_k, '0');
    mask.front() = '1';
    mask.back() = '1';
    for (std::size_t i = 1; i + 1 < span_k; ++i) {
        if ((i % 3) != 0) {
            mask[i] = '1';
        }
    }
    return mask;
}

// =================================================================================================
//     Scalar Extraction
// =================================================================================================

// Checks one scalar bit extractor for every sequence in the corpus, through the overload for a
// vector of masks, and, for a single mask, also through the single-mask overload.
template <typename Enc, typename Mask, typename BitExtract>
static void check_scalar_variant(
    std::vector<std::string> const& mask_strings,
    std::vector<Mask> const& masks,
    Enc const& enc,
    BitExtract&& bit_extract
) {
    std::size_t const span_k = mask_strings.front().size();
    for (auto const& seq : sequence_corpus(span_k, 8)) {
        auto const expected = oracle_spaced_events<Enc::encoding, Mask::layout>(seq, mask_strings);
        auto const runs = collect_spaced_callback_forms<Mask::layout, false>(
            [&](auto const& callback) {
                for_each_spaced_kmer(seq, masks, enc, bit_extract, callback);
            }
        );
        for (auto const& got : runs) {
            check_spaced_events(got, expected);
        }

        if (masks.size() == 1) {
            auto const single_runs = collect_spaced_callback_forms<Mask::layout, true>(
                [&](auto const& callback) {
                    for_each_spaced_kmer(seq, masks.front(), enc, bit_extract, callback);
                }
            );
            for (auto const& got : single_runs) {
                check_spaced_events(got, expected);
            }
        }
    }
}

template <Layout L, typename Enc>
static void check_all_scalar_variants(
    std::vector<std::string> const& mask_strings, Enc const& enc
) {
    auto const direct_masks = make_spaced_kmer_masks<L>(mask_strings);
    auto const block_masks = make_spaced_kmer_block_tables(direct_masks);
    auto const butterfly_masks = make_spaced_kmer_butterfly_tables(direct_masks);

    check_scalar_variant(
        mask_strings, direct_masks, enc,
        [](std::uint64_t x, BitExtractMask const& mask) { return bit_extract_bitloop(x, mask); }
    );
    check_scalar_variant(
        mask_strings, direct_masks, enc,
        [](std::uint64_t x, BitExtractMask const& mask) { return bit_extract_byte_table(x, mask); }
    );
    check_scalar_variant(
        mask_strings, block_masks, enc,
        [](std::uint64_t x, BitExtractBlockTable const& mask) {
            return bit_extract_block_table(x, mask);
        }
    );
    check_scalar_variant(
        mask_strings, block_masks, enc,
        [](std::uint64_t x, BitExtractBlockTable const& mask) {
            return bit_extract_block_table_unrolled<2>(x, mask);
        }
    );
    check_scalar_variant(
        mask_strings, block_masks, enc,
        [](std::uint64_t x, BitExtractBlockTable const& mask) {
            return bit_extract_block_table_unrolled<4>(x, mask);
        }
    );
    check_scalar_variant(
        mask_strings, block_masks, enc,
        [](std::uint64_t x, BitExtractBlockTable const& mask) {
            return bit_extract_block_table_unrolled<8>(x, mask);
        }
    );
    check_scalar_variant(
        mask_strings, butterfly_masks, enc,
        [](std::uint64_t x, BitExtractButterflyTable const& mask) {
            return bit_extract_butterfly_table(x, mask);
        }
    );

    #if defined(FISK_HAS_BMI2)
    check_scalar_variant(
        mask_strings, direct_masks, enc,
        [](std::uint64_t x, BitExtractMask const& mask) { return bit_extract_pext(x, mask); }
    );
    #endif
}

template <Layout L, typename Enc>
static void check_scalar_masks(Enc const& enc)
{
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_all_scalar_variants<L>(std::vector<std::string>{mask_for_span(span_k)}, enc);
    }
    check_all_scalar_variants<L>({"1"}, enc);
    check_all_scalar_variants<L>({"10101", "11111", "10001"}, enc);
    check_all_scalar_variants<L>({"11011", "11101", "10111"}, enc);
    check_all_scalar_variants<L>({"10000000000000000000000000000001"}, enc);
}

// =================================================================================================
//     SIMD Extraction
// =================================================================================================

template <typename Kernel, Layout L, std::size_t N>
static std::array<SpacedKmerMask<L, Kernel>, N> make_kernels(
    std::array<std::string, N> const& masks
) {
    std::array<SpacedKmerMask<L, Kernel>, N> kernels{};
    for (std::size_t i = 0; i < N; ++i) {
        kernels[i] = make_spaced_kmer_kernel<Kernel, L>(masks[i]);
    }
    return kernels;
}

// Checks both SIMD emission orders for one set of masks, for every sequence in the corpus: exact
// order for by_position, and per-mask order for by_mask (see check_by_mask_events()). For a single
// mask, also through the single-kernel overloads.
template <typename Kernel, Layout L, std::size_t N, typename Enc>
static void check_simd_set(std::array<std::string, N> const& mask_array, Enc const& enc)
{
    std::vector<std::string> const masks = to_vector(mask_array);
    auto const kernels = make_kernels<Kernel, L>(mask_array);
    std::size_t const span_k = masks.front().size();

    for (auto const& seq : sequence_corpus(span_k, Kernel::lanes)) {
        auto const expected = oracle_spaced_events<Enc::encoding, L>(seq, masks);

        auto const by_position = collect_spaced_callback_forms<L, N == 1>(
            [&](auto const& callback) {
                for_each_spaced_kmer_simd_by_position(seq, kernels, enc, callback);
            }
        );
        for (auto const& got : by_position) {
            check_spaced_events(got, expected);
        }
        auto const by_mask = collect_spaced_callback_forms<L, N == 1>([&](auto const& callback) {
            for_each_spaced_kmer_simd_by_mask(seq, kernels, enc, callback);
        });
        for (auto const& got : by_mask) {
            check_by_mask_events(got, expected, N);
        }

        if constexpr (N == 1) {
            auto const single_by_position = collect_spaced_callback_forms<L, true>(
                [&](auto const& callback) {
                    for_each_spaced_kmer_simd_by_position(seq, kernels[0], enc, callback);
                }
            );
            for (auto const& got : single_by_position) {
                check_spaced_events(got, expected);
            }
            auto const single_by_mask = collect_spaced_callback_forms<L, true>(
                [&](auto const& callback) {
                    for_each_spaced_kmer_simd_by_mask(seq, kernels[0], enc, callback);
                }
            );
            for (auto const& got : single_by_mask) {
                check_by_mask_events(got, expected, N);
            }
        }
    }
}

template <typename Kernel, Layout L>
static void check_simd_kernel_in_layout()
{
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_simd_set<Kernel, L>(
            std::array<std::string, 1>{mask_for_span(span_k)},
            CharEncoderTable<Encoding::kACGT>{}
        );
    }
    check_simd_set<Kernel, L>(generated_masks<3>(), CharEncoderTable<Encoding::kACGT>{});
}

template <typename Kernel>
static void check_simd_kernel()
{
    check_simd_kernel_in_layout<Kernel, Layout::kMSB>();
    check_simd_kernel_in_layout<Kernel, Layout::kLSB>();
}

template <typename Kernel, Layout L>
static void check_simd_dispatcher_in_layout(std::vector<std::string> const& masks)
{
    BitExtractKernelDispatcher<SpacedKmerMask<L, Kernel>> const dispatcher(
        make_spaced_kmer_kernels<Kernel>(make_spaced_kmer_masks<L>(masks))
    );
    std::size_t const span_k = masks.front().size();

    for (auto const& seq : sequence_corpus(span_k, Kernel::lanes)) {
        auto const expected = oracle_spaced_events<Encoding::kACGT, L>(seq, masks);
        std::vector<SpacedEvent> by_position;
        dispatcher.run([&](auto const& kernels) {
            for_each_spaced_kmer_simd_by_position(
                seq, kernels, CharEncoderTable<Encoding::kACGT>{},
                [&](std::size_t pos, std::size_t mask_idx, auto const& value) {
                    by_position.push_back({pos, mask_idx, kmer_value(value)});
                }
            );
        });
        check_spaced_events(by_position, expected);

        std::vector<SpacedEvent> by_mask;
        dispatcher.run([&](auto const& kernels) {
            for_each_spaced_kmer_simd_by_mask(
                seq, kernels, CharEncoderTable<Encoding::kACGT>{},
                [&](std::size_t pos, std::size_t mask_idx, auto const& value) {
                    by_mask.push_back({pos, mask_idx, kmer_value(value)});
                }
            );
        });
        check_by_mask_events(by_mask, expected, masks.size());
    }
}

template <typename Kernel>
static void check_simd_dispatcher(std::vector<std::string> const& masks)
{
    check_simd_dispatcher_in_layout<Kernel, Layout::kMSB>(masks);
    check_simd_dispatcher_in_layout<Kernel, Layout::kLSB>(masks);
}

// =================================================================================================
//     Spaced K-mer Type
// =================================================================================================

template <typename K>
concept CanReverse = requires(K kmer) { reverse(kmer, 3); };

template <typename K>
concept CanReverseComplement = requires(K kmer) { reverse_complement(kmer, 3); };

template <typename K>
concept CanCanonical = requires(K kmer) { canonical(kmer, 3); };

static_assert(KmerType<SpacedKmerAcgtMsb>);
static_assert(!ContiguousKmerType<SpacedKmerAcgtMsb>);
static_assert(ContiguousKmerType<KmerAcgtMsb>);
static_assert(SpacedKmerType<SpacedKmerActgLsb>);
static_assert(!SpacedKmerType<KmerAcgtMsb>);
static_assert(!SpacedKmerType<std::uint64_t>);
static_assert(std::is_same_v<SpacedKmerActgLsb, SpacedKmer<Encoding::kACTG, Layout::kLSB>>);

static_assert(CanReverse<KmerAcgtMsb>);
static_assert(CanReverseComplement<KmerAcgtMsb>);
static_assert(CanCanonical<KmerAcgtMsb>);

// Spaced k-mer extraction only accepts masks that are tagged with their Layout.
template <typename Mask>
concept CanExtractWith = requires(std::string_view seq, Mask const& mask) {
    for_each_spaced_kmer(
        seq, mask, CharEncoderTable<Encoding::kACGT>{}, bit_extract_bitloop,
        [](std::size_t, auto const&) {}
    );
};

static_assert(SpacedKmerMaskType<SpacedKmerMask<Layout::kMSB, BitExtractMask>>);
static_assert(
    SpacedKmerMaskType<SpacedKmerMask<Layout::kLSB, BitExtractKernelButterfly<SimdScalar>>>
);
static_assert(!SpacedKmerMaskType<BitExtractMask>);
static_assert(!SpacedKmerMaskType<BitExtractKernelButterfly<SimdScalar>>);
static_assert(!std::is_aggregate_v<SpacedKmerMask<Layout::kMSB, BitExtractMask>>);
static_assert(CanExtractWith<SpacedKmerMask<Layout::kMSB, BitExtractMask>>);
static_assert(!CanExtractWith<BitExtractMask>);
static_assert(decltype(make_spaced_kmer_mask("1"))::layout == Layout::kMSB);
static_assert(decltype(make_spaced_kmer_mask<Layout::kLSB>("1"))::layout == Layout::kLSB);
static_assert(std::is_same_v<
    decltype(make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>, Layout::kLSB>("1")),
    SpacedKmerMask<Layout::kLSB, BitExtractKernelButterfly<SimdScalar>>
>);
static_assert(!CanReverse<SpacedKmerAcgtMsb> && !CanReverseComplement<SpacedKmerAcgtMsb>);
static_assert(!CanCanonical<SpacedKmerAcgtMsb>);

namespace {
    constexpr auto spaced_acg = spaced_kmer_cast<Encoding::kACGT, Layout::kMSB>(0x06, 3);
}
static_assert(spaced_kmer_cast_unchecked<Encoding::kACGT, Layout::kMSB>(0x06) == spaced_acg);
static_assert(kmer_cast_unchecked<Encoding::kACGT, Layout::kMSB>(0x06) == KmerAcgtMsb{0x06});
static_assert(kmer_value(spaced_acg) == 0x06);
static_assert(base_at(spaced_acg, 0, 3) == 0 && base_at(spaced_acg, 2, 3) == 2);
static_assert(complement(spaced_acg, 3) == SpacedKmerAcgtMsb{0x39});
static_assert(kmer_convert<Layout::kLSB>(spaced_acg, 3) == SpacedKmerAcgtLsb{0x24});
static_assert(kmer_convert<Encoding::kACTG>(spaced_acg, 3) == SpacedKmerActgMsb{0x07});
static_assert(std::is_same_v<
    decltype(kmer_convert<Encoding::kACTG, Layout::kLSB>(spaced_acg, 3)), SpacedKmerActgLsb
>);

// Bases kept by a spaced mask carry the same meaning as the contiguous k-mer of the same bases, so
// decoding and the convention-only operations must agree with the contiguous results.
TEST(SpacedKmer, AgreesWithContiguousOnBaseOps)
{
    std::size_t const width = 9;
    for (auto const& seq : random_seqs(width, 32, 4242)) {
        auto const contiguous = oracle_kmer<KmerAcgtMsb>(seq);
        auto const spaced = spaced_kmer_cast<Encoding::kACGT, Layout::kMSB>(
            kmer_value(contiguous), width
        );
        EXPECT_EQ(kmer_decode(spaced, width), seq);
        EXPECT_EQ(
            kmer_value(complement(spaced, width)),
            kmer_value(complement(contiguous, width))
        );
        EXPECT_EQ(
            kmer_value(kmer_convert<Layout::kLSB>(spaced, width)),
            kmer_value(kmer_convert<Layout::kLSB>(contiguous, width))
        );
    }
}

// See check_vector_cast() in simd_lanes.hpp, run once per vector ISA compiled in.
TEST(SpacedKmer, CastVector)
{
    std::size_t const width = 10;
    for_each_lane_vector(width, 919190, [&](auto, auto vec, auto const& seqs) {
        check_vector_cast<SpacedKmerAcgtMsb>(vec, seqs, width);
    });
}

TEST(SpacedKmer, Hashing)
{
    auto const probe = spaced_kmer_cast<Encoding::kACGT, Layout::kMSB>(0x1B, 3);
    EXPECT_EQ(std::hash<SpacedKmerAcgtMsb>{}(probe), KmerHashMix{}(probe));
    EXPECT_NE(std::hash<SpacedKmerAcgtMsb>{}(probe), KmerHashIdentity{}(probe));
}

TEST(SpacedKmer, UnorderedContainers)
{
    std::size_t const width = 12;
    auto const seqs = random_seqs(width, 64, 135790);

    std::unordered_set<SpacedKmerAcgtMsb> spaced_set;
    std::unordered_set<KmerAcgtMsb> contiguous_set;
    for (auto const& seq : seqs) {
        auto const contiguous = oracle_kmer<KmerAcgtMsb>(seq);
        auto const spaced = spaced_kmer_cast<Encoding::kACGT, Layout::kMSB>(
            kmer_value(contiguous), width
        );
        spaced_set.insert(spaced);
        contiguous_set.insert(contiguous);
        EXPECT_EQ(spaced_set.count(spaced), std::size_t{1});
    }
    EXPECT_EQ(spaced_set.size(), contiguous_set.size());
}

TEST(SpacedKmer, Comparison)
{
    auto const a = SpacedKmerAcgtMsb{7};
    auto const b = SpacedKmerAcgtMsb{9};
    EXPECT_TRUE(a == a);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= a);
}

TEST(SpacedKmer, ConvertRoundTripAndComposition)
{
    for (std::size_t width = 1; width <= 32; ++width) {
        for (auto const& seq : random_seqs(width, 4, 888888 + width)) {
            auto const kmer = spaced_kmer_cast<Encoding::kACGT, Layout::kMSB>(
                kmer_value(oracle_kmer<KmerAcgtMsb>(seq)), width
            );

            EXPECT_EQ(
                kmer_convert<Encoding::kACGT>(kmer_convert<Encoding::kACTG>(kmer, width), width),
                kmer
            );
            EXPECT_EQ(
                kmer_convert<Layout::kMSB>(kmer_convert<Layout::kLSB>(kmer, width), width), kmer
            );
            EXPECT_EQ(
                (kmer_convert<Encoding::kACGT, Layout::kMSB>(
                    kmer_convert<Encoding::kACTG, Layout::kLSB>(kmer, width), width
                )),
                kmer
            );

            auto const combined = kmer_convert<Encoding::kACTG, Layout::kLSB>(kmer, width);
            auto const encoding_first =
                kmer_convert<Layout::kLSB>(kmer_convert<Encoding::kACTG>(kmer, width), width);
            auto const layout_first =
                kmer_convert<Encoding::kACTG>(kmer_convert<Layout::kLSB>(kmer, width), width);
            EXPECT_EQ(combined, encoding_first);
            EXPECT_EQ(combined, layout_first);
        }
    }
}

// =================================================================================================
//     Spaced K-mer Encoding
// =================================================================================================

// Encoding a window with a mask keeps exactly the bases at the mask's '1' positions, in order, and
// must agree with the oracle under every convention.
template <typename K>
static void check_spaced_encode(std::string const& mask)
{
    std::size_t const weight = spaced_kmer_weight(mask);
    for (auto const& window : random_seqs(mask.size(), 8, 525252 + mask.size())) {
        std::string kept;
        std::string span;
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == '1') {
                kept.push_back(window[i]);
            }
            span.push_back(mask[i] == '1' ? window[i] : '-');
        }
        auto const spaced = spaced_kmer_encode<K>(window, mask);
        EXPECT_EQ(spaced, oracle_kmer<K>(kept));
        EXPECT_EQ(kmer_decode(spaced, weight), kept);
        EXPECT_EQ(spaced_kmer_decode(spaced, mask), span);
    }
}

TEST(SpacedKmer, EncodeWithMask)
{
    for (std::string const& mask : std::vector<std::string>{
        "1", "11", "101", "1*011", "110100111", mask_for_span(31), std::string(32, '1')
    }) {
        check_spaced_encode<SpacedKmerAcgtMsb>(mask);
        check_spaced_encode<SpacedKmerAcgtLsb>(mask);
        check_spaced_encode<SpacedKmerActgMsb>(mask);
        check_spaced_encode<SpacedKmerActgLsb>(mask);
    }
}

// The result type is named as for the contiguous kmer_encode(), and defaults to ACGT/MSB. Only
// spaced k-mers take a mask, for encoding and decoding.
template <typename K>
concept EncodableWithMask = requires(std::string_view window, std::string const& mask) {
    spaced_kmer_encode<K>(window, mask);
};

static_assert(std::is_same_v<decltype(spaced_kmer_encode("ACGT", "1011")), SpacedKmerAcgtMsb>);
static_assert(std::is_same_v<
    decltype(spaced_kmer_encode<SpacedKmerActgLsb>("A", "1")), SpacedKmerActgLsb
>);
static_assert(EncodableWithMask<SpacedKmerAcgtMsb>);
static_assert(!EncodableWithMask<KmerAcgtMsb>);

template <typename K>
concept DecodableWithMask = requires(K const& kmer, std::string const& mask) {
    spaced_kmer_decode(kmer, mask);
};

static_assert(DecodableWithMask<SpacedKmerActgLsb>);

// With a tagged mask, the Layout of the result is the one of the mask, as during extraction.
static_assert(std::is_same_v<
    decltype(spaced_kmer_encode("A", make_spaced_kmer_mask("1"))), SpacedKmerAcgtMsb
>);
static_assert(std::is_same_v<
    decltype(spaced_kmer_encode<Encoding::kACTG>("A", make_spaced_kmer_mask("1"))),
    SpacedKmerActgMsb
>);
static_assert(std::is_same_v<
    decltype(spaced_kmer_encode("A", make_spaced_kmer_mask<Layout::kLSB>("1"))),
    SpacedKmerAcgtLsb
>);
static_assert(!DecodableWithMask<KmerAcgtMsb>);

TEST(SpacedKmer, EncodeWithMaskDefaults)
{
    // ACGT under 1011 keeps A, G, T: codes 0, 2, 3, earliest in the high bits.
    EXPECT_EQ(kmer_value(spaced_kmer_encode("ACGT", "1011")), std::uint64_t{0x0B});
    EXPECT_EQ(spaced_kmer_encode("acgt", "1011"), spaced_kmer_encode("ACGT", "1011"));
}

TEST(SpacedKmer, DecodeWithMask)
{
    EXPECT_EQ(spaced_kmer_decode(spaced_kmer_encode("ACGTA", "10101"), "10101"), "A-G-A");
    EXPECT_EQ(spaced_kmer_decode(spaced_kmer_encode("acgta", "1*101"), "1*101"), "A-G-A");
    EXPECT_EQ(spaced_kmer_decode(spaced_kmer_encode("T", "1"), "1"), "T");

    // Tagged masks, in any representation, behave as their mask string.
    auto const kmer = spaced_kmer_encode("ACGTA", "10101");
    auto const mask = make_spaced_kmer_mask("1*101");
    auto const kernel = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>(mask);
    EXPECT_EQ(spaced_kmer_encode("ACGTA", mask), kmer);
    EXPECT_EQ(spaced_kmer_encode("ACGTA", make_spaced_kmer_block_table(mask)), kmer);
    EXPECT_EQ(spaced_kmer_encode("ACGTA", kernel), kmer);
    EXPECT_EQ(spaced_kmer_decode(kmer, mask), "A-G-A");
    EXPECT_EQ(spaced_kmer_decode(kmer, make_spaced_kmer_butterfly_table(mask)), "A-G-A");
    EXPECT_EQ(spaced_kmer_decode(kmer, kernel), "A-G-A");
    EXPECT_EQ(
        spaced_kmer_encode<Encoding::kACTG>("ACGTA", mask),
        spaced_kmer_encode<SpacedKmerActgMsb>("ACGTA", "10101")
    );

    // An LSB mask encodes to an LSB spaced k-mer, but shows the same span.
    auto const lsb_kmer = spaced_kmer_encode<SpacedKmerAcgtLsb>("ACGTT", "11001");
    auto const lsb_mask = make_spaced_kmer_mask<Layout::kLSB>("11001");
    auto const lsb_kernel
        = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>(lsb_mask)
    ;
    EXPECT_EQ(spaced_kmer_encode("ACGTT", lsb_mask), lsb_kmer);
    EXPECT_EQ(spaced_kmer_encode("ACGTT", lsb_kernel), lsb_kmer);
    EXPECT_EQ(spaced_kmer_decode(lsb_kmer, lsb_mask), "AC--T");
    EXPECT_EQ(spaced_kmer_decode(lsb_kmer, lsb_kernel), "AC--T");
    EXPECT_EQ(kmer_value(lsb_kmer), std::uint64_t{0x34});

    EXPECT_THROW((void) spaced_kmer_decode(kmer, ""), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_decode(kmer, "01101"), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_decode(kmer, "10x01"), std::invalid_argument);
}

// Invalid masks, mismatched lengths and non-nucleotides at kept positions throw. Positions the mask
// skips are not inspected, as during extraction.
TEST(SpacedKmer, EncodeWithMaskInvalidInput)
{
    EXPECT_THROW((void) spaced_kmer_encode("ACG", "1011"), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_encode("ACGTA", "1011"), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_encode("", ""), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_encode("ACGT", "0111"), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_encode("ACGT", "1110"), std::invalid_argument);
    EXPECT_THROW((void) spaced_kmer_encode("ACGT", "1x11"), std::invalid_argument);
    EXPECT_THROW(
        (void) spaced_kmer_encode(std::string(33, 'A'), std::string(33, '1')), std::invalid_argument
    );

    for (char const c : all_invalid_bytes()) {
        std::string window = "ACGT";
        window[2] = c;
        EXPECT_THROW((void) spaced_kmer_encode(window, "1111"), std::invalid_argument);
        EXPECT_EQ(spaced_kmer_encode(window, "1101"), spaced_kmer_encode("ACAT", "1101"));
        EXPECT_EQ(spaced_kmer_encode(window, "11*1"), spaced_kmer_encode("ACAT", "1101"));
    }
}

// =================================================================================================
//     Mask Helpers
// =================================================================================================

TEST(KmerSpaced, Weight)
{
    EXPECT_EQ(spaced_kmer_weight(std::string("1")), std::size_t{1});
    EXPECT_EQ(spaced_kmer_weight(std::string("1011")), std::size_t{3});
    EXPECT_EQ(spaced_kmer_weight(std::string("1*1")), std::size_t{2});
    EXPECT_EQ(spaced_kmer_weight(std::string(32, '1')), std::size_t{32});

    EXPECT_EQ(spaced_kmer_weight(std::uint64_t{0xCF}), std::size_t{3});
    EXPECT_EQ(spaced_kmer_weight(std::uint64_t{0x33}), std::size_t{2});
    EXPECT_EQ(spaced_kmer_weight(~std::uint64_t{0}), std::size_t{32});

    EXPECT_EQ(spaced_kmer_weight(make_spaced_kmer_mask("1011")), std::size_t{3});
    EXPECT_EQ(spaced_kmer_weight(make_spaced_kmer_butterfly_table("1011")), std::size_t{3});
    EXPECT_EQ(spaced_kmer_weight(make_spaced_kmer_block_table("1011")), std::size_t{3});
    EXPECT_EQ(
        spaced_kmer_weight(make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>("1011")),
        std::size_t{3}
    );

    EXPECT_ANY_THROW(spaced_kmer_weight(std::string("")));
    EXPECT_ANY_THROW(spaced_kmer_weight(std::string("010")));
}

// Emitted values are typed k-mers under the encoder's encoding and the mask's layout. Decoding one
// with the weight of its mask must give exactly the bases kept by that mask, with no bits above the
// weight, and it must equal spaced_kmer_encode() of the same window and mask, given as string or as
// the tagged mask used for extraction. Decoding it with its mask gives the window, with gaps.
template <Layout L>
static void check_emitted_kmers()
{
    std::vector<std::string> const mask_strings = {"1011", "1001"};
    std::size_t const span_k = 4;
    auto const masks = make_spaced_kmer_masks<L>(mask_strings);

    std::size_t emitted = 0;
    for (auto const& seq : sequence_corpus(span_k, 8)) {
        for_each_spaced_kmer(
            std::string_view(seq), masks, CharEncoderTable<Encoding::kACGT>{},
            bit_extract_bitloop,
            [&](std::size_t pos, std::size_t mask_idx, auto const& kmer) {
                using K = SpacedKmer<Encoding::kACGT, L>;
                static_assert(std::is_same_v<std::remove_cvref_t<decltype(kmer)>, K>);
                auto const window = seq.substr(pos, span_k);
                EXPECT_EQ(kmer, spaced_kmer_encode<K>(window, mask_strings[mask_idx]));
                EXPECT_EQ(kmer, spaced_kmer_encode(window, masks[mask_idx]));

                auto const weight = spaced_kmer_weight(mask_strings[mask_idx]);
                EXPECT_EQ(kmer_value(kmer) >> (2 * weight), std::uint64_t{0});

                std::string expected;
                std::string expected_span;
                for (std::size_t p = 0; p < span_k; ++p) {
                    char const base = static_cast<char>(
                        std::toupper(static_cast<unsigned char>(seq[pos + p]))
                    );
                    if (mask_strings[mask_idx][p] == '1') {
                        expected += base;
                    }
                    expected_span += (mask_strings[mask_idx][p] == '1') ? base : '-';
                }
                EXPECT_EQ(kmer_decode(kmer, weight), expected);
                EXPECT_EQ(spaced_kmer_decode(kmer, mask_strings[mask_idx]), expected_span);
                EXPECT_EQ(spaced_kmer_decode(kmer, masks[mask_idx]), expected_span);
                ++emitted;
            }
        );
    }
    EXPECT_TRUE(emitted > 0);
}

TEST(KmerSpaced, EmittedKmersAreTypedAndDecodeToKeptBases)
{
    check_emitted_kmers<Layout::kMSB>();
    check_emitted_kmers<Layout::kLSB>();
}

// The same extraction in both layouts emits at the same positions, and each LSB spaced k-mer is
// the layout conversion of the MSB one. The mask bits of both layouts only agree for palindromic
// masks, while the values differ for any mask of weight above 1.
TEST(KmerSpaced, LayoutsAgreeViaConversion)
{
    std::vector<std::string> const mask_strings = {"11101", "10101", "11011", "10001", "1100111"};
    for (auto const& mask_string : mask_strings) {
        auto const msb_mask = make_spaced_kmer_mask<Layout::kMSB>(mask_string);
        auto const lsb_mask = make_spaced_kmer_mask<Layout::kLSB>(mask_string);
        std::size_t const weight = spaced_kmer_weight(mask_string);
        std::string const reversed(mask_string.rbegin(), mask_string.rend());
        EXPECT_EQ(msb_mask.mask == lsb_mask.mask, reversed == mask_string);

        bool any_differ = false;
        for (auto const& seq : sequence_corpus(mask_string.size(), 8)) {
            std::vector<std::pair<std::size_t, SpacedKmerAcgtMsb>> msb;
            std::vector<std::pair<std::size_t, SpacedKmerAcgtLsb>> lsb;
            for_each_spaced_kmer(
                std::string_view(seq), msb_mask, CharEncoderTable<Encoding::kACGT>{},
                bit_extract_bitloop,
                [&](std::size_t pos, auto const& kmer) { msb.emplace_back(pos, kmer); }
            );
            for_each_spaced_kmer(
                std::string_view(seq), lsb_mask, CharEncoderTable<Encoding::kACGT>{},
                bit_extract_bitloop,
                [&](std::size_t pos, auto const& kmer) { lsb.emplace_back(pos, kmer); }
            );

            ASSERT_EQ(msb.size(), lsb.size());
            for (std::size_t i = 0; i < msb.size(); ++i) {
                EXPECT_EQ(msb[i].first, lsb[i].first);
                EXPECT_EQ(kmer_convert<Layout::kLSB>(msb[i].second, weight), lsb[i].second);
                any_differ |= (kmer_value(msb[i].second) != kmer_value(lsb[i].second));
            }
        }
        EXPECT_TRUE(any_differ);
    }
}

// A mask that keeps every position is the contiguous k-mer, so the emitted spaced k-mers must match
// the contiguous extraction exactly, in order.
TEST(KmerSpaced, AllOnesMaskMatchesContiguousKmers)
{
    for (std::size_t const k : std::vector<std::size_t>{1, 4, 12, 31, 32}) {
        auto const mask = make_spaced_kmer_mask(std::string(k, '1'));
        for (auto const& seq : sequence_corpus(k, 4)) {
            std::vector<std::uint64_t> spaced_values;
            for_each_spaced_kmer(
                std::string_view(seq), mask, CharEncoderTable<Encoding::kACGT>{},
                bit_extract_bitloop,
                [&](std::size_t /*pos*/, std::size_t /*mask_idx*/, auto const& kmer) {
                    spaced_values.push_back(kmer_value(kmer));
                }
            );

            std::vector<std::uint64_t> contiguous_values;
            for_each_kmer_rolling(
                std::string_view(seq), k, CharEncoderTable<Encoding::kACGT>{},
                [&](auto const& kmer) { contiguous_values.push_back(kmer_value(kmer)); }
            );
            EXPECT_EQ(spaced_values, contiguous_values);
        }
    }
}

TEST(KmerSpaced, MaskHelpers)
{
    EXPECT_EQ(spaced_kmer_mask_positions("1"), (std::vector<std::size_t>{0}));
    EXPECT_EQ(spaced_kmer_mask_positions("10*11"), (std::vector<std::size_t>{0, 3, 4}));

    EXPECT_EQ(make_spaced_kmer_mask("1011").mask, std::uint64_t{0xCF});
    EXPECT_EQ(make_spaced_kmer_mask("1011").span_k, std::size_t{4});
    EXPECT_EQ(make_spaced_kmer_mask("1*1").mask, std::uint64_t{0x33});
    EXPECT_EQ(make_spaced_kmer_mask(std::string(32, '1')).mask, ~std::uint64_t{0});
    EXPECT_EQ(make_spaced_kmer_mask(std::string(32, '1')).span_k, std::size_t{32});

    // LSB masks are mirrored, with the first position of the span in the lowest bits.
    EXPECT_EQ(make_spaced_kmer_mask<Layout::kLSB>("1011").mask, std::uint64_t{0xF3});
    EXPECT_EQ(make_spaced_kmer_mask<Layout::kLSB>("1011").span_k, std::size_t{4});
    EXPECT_EQ(make_spaced_kmer_mask<Layout::kLSB>("1101").mask, std::uint64_t{0xCF});
    EXPECT_EQ(make_spaced_kmer_mask<Layout::kLSB>("1*1").mask, std::uint64_t{0x33});
    EXPECT_EQ(make_spaced_kmer_mask<Layout::kLSB>(std::string(32, '1')).mask, ~std::uint64_t{0});
    EXPECT_ANY_THROW(make_spaced_kmer_mask<Layout::kLSB>("010"));
    EXPECT_ANY_THROW(make_spaced_kmer_mask<Layout::kLSB>("1x1"));

    EXPECT_ANY_THROW(spaced_kmer_mask_positions(""));
    EXPECT_ANY_THROW(spaced_kmer_mask_positions(std::string(33, '1')));
    EXPECT_ANY_THROW(spaced_kmer_mask_positions("010"));
    EXPECT_ANY_THROW(spaced_kmer_mask_positions("1010"));
    EXPECT_ANY_THROW(spaced_kmer_mask_positions("1x1"));

    EXPECT_ANY_THROW(make_spaced_kmer_mask(""));
    EXPECT_ANY_THROW(make_spaced_kmer_mask(std::string(33, '1')));
    EXPECT_ANY_THROW(make_spaced_kmer_mask("010"));
    EXPECT_ANY_THROW(make_spaced_kmer_mask("1010"));
    EXPECT_ANY_THROW(make_spaced_kmer_mask("1x1"));
}

// All representations of a mask carry the same Layout, span and bits, and the vector overloads
// create the same masks as the single ones.
template <Layout L>
static void check_mask_factories()
{
    std::vector<std::string> const strings = {"1011", "1*0011"};
    auto const masks = make_spaced_kmer_masks<L>(strings);
    auto const butterfly_tables = make_spaced_kmer_butterfly_tables(masks);
    auto const block_tables = make_spaced_kmer_block_tables(masks);
    auto const kernels = make_spaced_kmer_kernels<BitExtractKernelBlock<SimdScalar>>(masks);
    ASSERT_EQ(masks.size(), strings.size());

    for (std::size_t i = 0; i < strings.size(); ++i) {
        auto const& mask = masks[i];
        EXPECT_EQ(mask.mask, make_spaced_kmer_mask<L>(strings[i]).mask);
        EXPECT_EQ(mask.span_k, strings[i].size());

        auto const butterfly_table = make_spaced_kmer_butterfly_table<L>(strings[i]);
        auto const block_table = make_spaced_kmer_block_table<L>(strings[i]);
        auto const kernel
            = make_spaced_kmer_kernel<BitExtractKernelBlock<SimdScalar>, L>(strings[i])
        ;
        static_assert(decltype(butterfly_table)::layout == L);
        static_assert(decltype(block_table)::layout == L);
        static_assert(decltype(kernel)::layout == L);
        static_assert(std::remove_cvref_t<decltype(butterfly_tables[i])>::layout == L);
        static_assert(std::remove_cvref_t<decltype(block_tables[i])>::layout == L);
        static_assert(std::remove_cvref_t<decltype(kernels[i])>::layout == L);
        EXPECT_EQ(butterfly_table.span_k, mask.span_k);
        EXPECT_EQ(block_table.span_k, mask.span_k);
        EXPECT_EQ(kernel.span_k, mask.span_k);
        EXPECT_EQ(butterfly_tables[i].span_k, mask.span_k);
        EXPECT_EQ(block_tables[i].span_k, mask.span_k);
        EXPECT_EQ(kernels[i].span_k, mask.span_k);

        // The tables and kernels extract the same bits as the plain mask.
        for (std::uint64_t const x : {0x0ull, ~0x0ull, 0x0123456789ABCDEFull}) {
            auto const expected = bit_extract_bitloop(x, mask);
            EXPECT_EQ(bit_extract_butterfly_table(x, butterfly_table), expected);
            EXPECT_EQ(bit_extract_block_table(x, block_table), expected);
            EXPECT_EQ(kernel.bit_extract(x), expected);
            EXPECT_EQ(bit_extract_butterfly_table(x, butterfly_tables[i]), expected);
            EXPECT_EQ(bit_extract_block_table(x, block_tables[i]), expected);
            EXPECT_EQ(kernels[i].bit_extract(x), expected);
        }
    }
}

TEST(KmerSpaced, MaskFactories)
{
    check_mask_factories<Layout::kMSB>();
    check_mask_factories<Layout::kLSB>();
}

// Valid masks are valid and print as their string in either layout.
template <Layout L>
static void check_mask_formatting()
{
    std::vector<std::string> const valid_masks = {
        "1", "101", "1*01", "1101", "11111", "10001",
        "10000000000000000000000000000001", std::string(32, '1')
    };
    for (auto const& mask_string : valid_masks) {
        auto const mask = make_spaced_kmer_mask<L>(mask_string);
        EXPECT_TRUE(is_valid_spaced_kmer_mask(mask));
        EXPECT_TRUE(is_valid_spaced_kmer_mask(mask.mask, mask_string.size()));

        std::string normalized = mask_string;
        std::replace(normalized.begin(), normalized.end(), '*', '0');
        EXPECT_EQ(spaced_kmer_mask_string(mask), normalized);
        EXPECT_EQ(spaced_kmer_mask_string(make_spaced_kmer_butterfly_table(mask)), normalized);
        EXPECT_EQ(
            spaced_kmer_mask_string(
                make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>(mask)
            ),
            normalized
        );
    }

    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        std::string const mask_string = mask_for_span(span_k);
        auto const mask = make_spaced_kmer_mask<L>(mask_string);
        EXPECT_TRUE(is_valid_spaced_kmer_mask(mask));
        EXPECT_EQ(spaced_kmer_mask_string(mask), mask_string);
    }
}

TEST(KmerSpaced, MaskValidationAndFormatting)
{
    check_mask_formatting<Layout::kMSB>();
    check_mask_formatting<Layout::kLSB>();

    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x3}, 2));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0xD}, 2));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x3F}, 2));

    // A middle position that uses only one of its two bits, with both ends kept.
    EXPECT_TRUE(is_valid_spaced_kmer_mask(std::uint64_t{0x33}, 3));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x37}, 3));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x3B}, 3));

    // Spans outside of [1, 32] are invalid for any mask.
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0}, 0));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x3}, 0));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(~std::uint64_t{0}, 33));

    EXPECT_TRUE(!is_valid_spaced_kmer_mask(raw_spaced_kmer_mask(0x3, 2)));
    EXPECT_EQ(spaced_kmer_mask_string(raw_spaced_kmer_mask(0x3, 2)), "01");
    EXPECT_ANY_THROW(spaced_kmer_mask_string(raw_spaced_kmer_mask(0x1, 1)));
    EXPECT_ANY_THROW(spaced_kmer_mask_string(raw_spaced_kmer_mask(0x3F, 2)));
}

// =================================================================================================
//     Scalar Tests
// =================================================================================================

TEST(KmerSpaced, ScalarAcgt)
{
    check_scalar_masks<Layout::kMSB>(CharEncoderTable<Encoding::kACGT>{});
    check_scalar_masks<Layout::kLSB>(CharEncoderTable<Encoding::kACGT>{});
}

TEST(KmerSpaced, ScalarActg)
{
    check_scalar_masks<Layout::kMSB>(CharEncoderTable<Encoding::kACTG>{});
    check_scalar_masks<Layout::kLSB>(CharEncoderTable<Encoding::kACTG>{});
}

// The span is set by make_spaced_kmer_mask(), so an invalid one is only possible by setting it by
// hand, and still has to be rejected.
TEST(KmerSpaced, InvalidSpan)
{
    auto const extract = [](std::uint64_t x, BitExtractMask const& m) {
        return bit_extract_bitloop(x, m);
    };
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        auto mask = make_spaced_kmer_mask("1");
        mask.span_k = k;
        for_each_spaced_kmer(
            seq, mask, CharEncoderTable<Encoding::kACGT>{}, extract,
            [](std::size_t, auto const&) {}
        );
    });

    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        auto kernel = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>("1");
        kernel.span_k = k;
        for_each_spaced_kmer_simd_by_position(
            seq, kernel, CharEncoderTable<Encoding::kACGT>{},
            [](std::size_t, auto const&) {}
        );
    });
    check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
        auto kernel = make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>("1");
        kernel.span_k = k;
        for_each_spaced_kmer_simd_by_mask(
            seq, kernel, CharEncoderTable<Encoding::kACGT>{},
            [](std::size_t, auto const&) {}
        );
    });
}

// All masks of one extraction share one rolling window, and hence need the same span.
TEST(KmerSpaced, MixedSpans)
{
    std::string const seq = "ACGTACGTACGT";
    auto const callback = [](std::size_t, std::size_t, auto const&) {};

    auto const masks = make_spaced_kmer_masks({"101", "10001"});
    EXPECT_THROW(
        for_each_spaced_kmer(
            seq, masks, CharEncoderTable<Encoding::kACGT>{}, bit_extract_bitloop, callback
        ),
        std::invalid_argument
    );

    using Kernel = SpacedKmerMask<Layout::kMSB, BitExtractKernelButterfly<SimdScalar>>;
    std::array<Kernel, 2> const kernels{{
        make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>("101"),
        make_spaced_kmer_kernel<BitExtractKernelButterfly<SimdScalar>>("10001")
    }};
    EXPECT_THROW(
        for_each_spaced_kmer_simd_by_position(
            seq, kernels, CharEncoderTable<Encoding::kACGT>{}, callback
        ),
        std::invalid_argument
    );
    EXPECT_THROW(
        for_each_spaced_kmer_simd_by_mask(
            seq, kernels, CharEncoderTable<Encoding::kACGT>{}, callback
        ),
        std::invalid_argument
    );
    EXPECT_THROW((void) spaced_kmer_selector(masks), std::invalid_argument);
}

// An empty set of masks emits nothing, for any sequence.
TEST(KmerSpaced, EmptyMasks)
{
    std::vector<SpacedKmerMask<Layout::kMSB, BitExtractMask>> const masks;
    for (std::string const& seq : {std::string{}, std::string("ACGTACGTACGT")}) {
        std::size_t count = 0;
        for_each_spaced_kmer(
            seq, masks, CharEncoderTable<Encoding::kACGT>{}, bit_extract_bitloop,
            [&](std::size_t, std::size_t, auto const&) { ++count; }
        );
        EXPECT_EQ(count, 0u);
    }
}

// =================================================================================================
//     SIMD Tests
// =================================================================================================

TEST(KmerSpacedSimd, ButterflyScalar)
{
    check_simd_kernel<BitExtractKernelButterfly<SimdScalar>>();
}

TEST(KmerSpacedSimd, BlockScalar)
{
    check_simd_kernel<BitExtractKernelBlock<SimdScalar>>();
}

#if defined(FISK_HAS_SSE2)
TEST(KmerSpacedSimd, ButterflySse2)
{
    check_simd_kernel<BitExtractKernelButterfly<SimdSSE2>>();
}

TEST(KmerSpacedSimd, BlockSse2)
{
    check_simd_kernel<BitExtractKernelBlock<SimdSSE2>>();
}
#endif

#if defined(FISK_HAS_AVX2)
TEST(KmerSpacedSimd, ButterflyAvx2)
{
    check_simd_kernel<BitExtractKernelButterfly<SimdAVX2>>();
}

TEST(KmerSpacedSimd, BlockAvx2)
{
    check_simd_kernel<BitExtractKernelBlock<SimdAVX2>>();
}
#endif

#if defined(FISK_HAS_AVX512)
TEST(KmerSpacedSimd, ButterflyAvx512)
{
    check_simd_kernel<BitExtractKernelButterfly<SimdAVX512>>();
}

TEST(KmerSpacedSimd, BlockAvx512)
{
    check_simd_kernel<BitExtractKernelBlock<SimdAVX512>>();
}
#endif

#if defined(FISK_HAS_NEON)
TEST(KmerSpacedSimd, ButterflyNeon)
{
    check_simd_kernel<BitExtractKernelButterfly<SimdNEON>>();
}

TEST(KmerSpacedSimd, BlockNeon)
{
    check_simd_kernel<BitExtractKernelBlock<SimdNEON>>();
}
#endif

#if defined(FISK_HAS_BMI2)
TEST(KmerSpacedSimd, Pext)
{
    check_simd_kernel<BitExtractKernelPEXT<>>();
}
#endif

TEST(KmerSpacedSimd, DispatcherMaskCounts)
{
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_simd_dispatcher<BitExtractKernelButterfly<SimdScalar>>({mask_for_span(span_k)});
    }
    check_simd_dispatcher<BitExtractKernelButterfly<SimdScalar>>(to_vector(generated_masks<1>()));
    check_simd_dispatcher<BitExtractKernelButterfly<SimdScalar>>(to_vector(generated_masks<2>()));
    check_simd_dispatcher<BitExtractKernelButterfly<SimdScalar>>(to_vector(generated_masks<3>()));
    check_simd_dispatcher<BitExtractKernelButterfly<SimdScalar>>(to_vector(generated_masks<9>()));
    check_simd_dispatcher<BitExtractKernelButterfly<SimdScalar>>(to_vector(generated_masks<16>()));

    #if defined(FISK_HAS_BMI2)
    // Eight PEXT lanes and 9+ masks exercise the two-word position-presence bitset.
    check_simd_dispatcher<BitExtractKernelPEXT<>>(to_vector(generated_masks<9>()));
    check_simd_dispatcher<BitExtractKernelPEXT<>>(to_vector(generated_masks<16>()));
    #elif defined(FISK_HAS_AVX512)
    // AVX-512 likewise has eight lanes, reaching the two-word presence path above eight masks.
    check_simd_dispatcher<BitExtractKernelButterfly<SimdAVX512>>(to_vector(generated_masks<9>()));
    check_simd_dispatcher<BitExtractKernelButterfly<SimdAVX512>>(to_vector(generated_masks<16>()));
    #endif
}

// One representative SIMD path uses ACTG too. Scalar tests above cover every scalar extractor
// under both encodings; this verifies the generic SIMD outer loop forwards its encoder unchanged.
TEST(KmerSpacedSimd, ActgForwarding)
{
    std::array<std::string, 3> const masks{"10101", "11101", "10001"};
    check_simd_set<BitExtractKernelButterfly<SimdScalar>, Layout::kMSB>(
        masks, CharEncoderTable<Encoding::kACTG>{}
    );
    check_simd_set<BitExtractKernelButterfly<SimdScalar>, Layout::kLSB>(
        masks, CharEncoderTable<Encoding::kACTG>{}
    );
}

// =================================================================================================
//     Selector
// =================================================================================================

static bool mode_matches_axis(SpacedKmerMode mode, SpacedKmerAxis axis)
{
    using Mode = SpacedKmerMode;
    switch (mode) {
        case Mode::kPext:
        case Mode::kButterflyTable:
            return true;

        case Mode::kButterflyTableSSE2ByMask:
        case Mode::kButterflyTableAVX2ByMask:
        case Mode::kButterflyTableAVX512ByMask:
        case Mode::kButterflyTableNeonByMask:
            return axis != SpacedKmerAxis::kByPosition;

        case Mode::kButterflyTableSSE2ByPosition:
        case Mode::kButterflyTableAVX2ByPosition:
        case Mode::kButterflyTableAVX512ByPosition:
        case Mode::kButterflyTableNeonByPosition:
            return axis != SpacedKmerAxis::kByMask;
    }
    return false;
}

TEST(KmerSpacedSelector, ValidationAndAxes)
{
    auto const masks = make_spaced_kmer_masks({"10101", "11111"});
    for (auto const axis : {
        SpacedKmerAxis::kByMask, SpacedKmerAxis::kByPosition, SpacedKmerAxis::kEither
    }) {
        auto const mode = spaced_kmer_selector(masks, axis, 256);
        EXPECT_TRUE(mode_matches_axis(mode, axis));
    }
    auto const lsb_masks = make_spaced_kmer_masks<Layout::kLSB>({"11101", "11111"});
    for (auto const axis : {
        SpacedKmerAxis::kByMask, SpacedKmerAxis::kByPosition, SpacedKmerAxis::kEither
    }) {
        EXPECT_TRUE(mode_matches_axis(spaced_kmer_selector(lsb_masks, axis, 256), axis));
    }

    auto const mask = make_spaced_kmer_mask("10101");
    EXPECT_TRUE(mode_matches_axis(
        spaced_kmer_selector(mask, SpacedKmerAxis::kByPosition, 256),
        SpacedKmerAxis::kByPosition
    ));
    EXPECT_TRUE(mode_matches_axis(
        spaced_kmer_selector(mask, SpacedKmerAxis::kByMask, 256),
        SpacedKmerAxis::kByMask
    ));
    EXPECT_TRUE(mode_matches_axis(
        spaced_kmer_selector(
            make_spaced_kmer_mask(std::string(32, '1')), SpacedKmerAxis::kEither, 256
        ),
        SpacedKmerAxis::kEither
    ));

    auto with_span = [&](std::size_t span_k) {
        auto result = mask;
        result.span_k = span_k;
        return result;
    };
    EXPECT_ANY_THROW(spaced_kmer_selector(
        std::vector<SpacedKmerMask<Layout::kMSB, BitExtractMask>>{}, SpacedKmerAxis::kEither, 256
    ));
    EXPECT_ANY_THROW(spaced_kmer_selector(with_span(0), SpacedKmerAxis::kEither, 256));
    EXPECT_ANY_THROW(spaced_kmer_selector(with_span(33), SpacedKmerAxis::kEither, 256));
    EXPECT_ANY_THROW(spaced_kmer_selector(mask, SpacedKmerAxis::kEither, 4));
    EXPECT_ANY_THROW(
        spaced_kmer_selector(raw_spaced_kmer_mask(0x3, 5), SpacedKmerAxis::kEither, 256)
    );
}
