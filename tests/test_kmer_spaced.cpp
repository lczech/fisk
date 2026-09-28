#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/bit_extract/simd.hpp"
#include "fisk/core/char_encoder.hpp"
#include "fisk/kmer_spaced/kmer_spaced.hpp"
#include "fisk/kmer_spaced/selector.hpp"
#include "fisk/kmer_spaced/simd.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Independent Oracle
// =================================================================================================

struct SpacedEvent
{
    std::size_t pos;
    std::size_t mask_idx;
    std::uint64_t value;
};

static int code_acgt(char c)
{
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return -1;
    }
}

static int code_actg(char c)
{
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'T': case 't': return 2;
        case 'G': case 'g': return 3;
        default:            return -1;
    }
}

// Builds the public spaced-k-mer contract from characters, without any rolling word or bit-extract
// helper: each output event is (full-span start position, mask index, packed selected bases).
template <typename CodeFn>
static std::vector<SpacedEvent> oracle_events(
    std::string const& seq, std::vector<std::string> const& masks, CodeFn&& code_of
) {
    std::vector<SpacedEvent> out;
    if (masks.empty() || seq.size() < masks.front().size()) {
        return out;
    }

    std::size_t const span_k = masks.front().size();
    for (std::size_t pos = 0; pos + span_k <= seq.size(); ++pos) {
        for (std::size_t mask_idx = 0; mask_idx < masks.size(); ++mask_idx) {
            std::uint64_t value = 0;
            bool valid = true;
            for (std::size_t i = 0; i < span_k; ++i) {
                if (masks[mask_idx][i] != '1') {
                    continue;
                }
                int const code = code_of(seq[pos + i]);
                if (code < 0) {
                    valid = false;
                    break;
                }
                value = (value << 2) | static_cast<std::uint64_t>(code);
            }
            if (valid) {
                out.push_back({pos, mask_idx, value});
            }
        }
    }
    return out;
}

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

static void check_events(
    std::vector<SpacedEvent> const& got, std::vector<SpacedEvent> const& expected
) {
    EXPECT_EQ(got.size(), expected.size());
    std::size_t const n = std::min(got.size(), expected.size());
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(got[i].pos, expected[i].pos);
        EXPECT_EQ(got[i].mask_idx, expected[i].mask_idx);
        EXPECT_EQ(got[i].value, expected[i].value);
    }
}

static void check_event_multiset(
    std::vector<SpacedEvent> got, std::vector<SpacedEvent> expected
) {
    std::sort(got.begin(), got.end(), event_less);
    std::sort(expected.begin(), expected.end(), event_less);
    check_events(got, expected);
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
        check_events(got_mask, expected_mask);
        for (std::size_t i = 1; i < got_mask.size(); ++i) {
            EXPECT_TRUE(got_mask[i - 1].pos <= got_mask[i].pos);
        }
    }
}

static std::string patterned_sequence(std::size_t length)
{
    static char const bases[] = "ACGTacgt";
    std::string seq;
    seq.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        seq += bases[i % 8];
    }
    return seq;
}

static std::vector<std::string> sequence_corpus(std::size_t span_k, std::size_t lanes)
{
    std::vector<std::string> sequences;
    sequences.push_back("");
    if (span_k > 1) {
        sequences.push_back(patterned_sequence(span_k - 1));
    }
    for (std::size_t extra = 0; extra <= 2 * lanes; ++extra) {
        sequences.push_back(patterned_sequence(span_k + extra));
    }

    std::string invalid = patterned_sequence(span_k + 2 * lanes + 5);
    invalid[0] = 'N';
    if (span_k > 2) {
        invalid[1] = 'N';
    }
    invalid[span_k - 1] = 'N';
    invalid[span_k] = 'N';
    sequences.push_back(invalid);

    // N at index 1 is skipped by 10101, while N at index 2 is kept and suppresses position 0.
    sequences.push_back("ANCGTACGTACGT");
    sequences.push_back("ACNGTACGTACGT");
    sequences.push_back("ACGTNNNACGTACGT");
    return sequences;
}

static std::vector<std::uint64_t> raw_masks(std::vector<std::string> const& masks)
{
    std::vector<std::uint64_t> out;
    out.reserve(masks.size());
    for (auto const& mask : masks) {
        out.push_back(prepare_spaced_kmer_bit_extract_mask(mask));
    }
    return out;
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

template <typename Mask, typename Enc, typename BitExtract>
static std::vector<SpacedEvent> collect_scalar(
    std::string const& seq,
    std::size_t span_k,
    std::vector<Mask> const& masks,
    Enc const& enc,
    BitExtract&& bit_extract
) {
    std::vector<SpacedEvent> out;
    for_each_spaced_kmer(
        seq, span_k, masks, enc, std::forward<BitExtract>(bit_extract),
        [&](std::size_t pos, std::size_t mask_idx, std::uint64_t value) {
            out.push_back({pos, mask_idx, value});
        }
    );
    return out;
}

template <typename Enc, typename Mask, typename BitExtract>
static void check_scalar_variant(
    std::vector<std::string> const& mask_strings,
    std::vector<Mask> const& masks,
    Enc const& enc,
    BitExtract&& bit_extract
) {
    std::size_t const span_k = mask_strings.front().size();
    for (auto const& seq : sequence_corpus(span_k, 8)) {
        check_events(
            collect_scalar(seq, span_k, masks, enc, bit_extract),
            oracle_events(
                seq, mask_strings,
                [](char c) { return Enc::encoding == Encoding::kACGT ? code_acgt(c) : code_actg(c); }
            )
        );
    }
}

template <typename Enc>
static void check_all_scalar_variants(
    std::vector<std::string> const& mask_strings, Enc const& enc
) {
    std::vector<BitExtractMask> direct_masks;
    std::vector<BitExtractBlockTable> block_masks;
    std::vector<BitExtractButterflyTable> butterfly_masks;
    for (auto const raw : raw_masks(mask_strings)) {
        direct_masks.emplace_back(raw);
        block_masks.push_back(bit_extract_block_table_preprocess(raw));
        butterfly_masks.push_back(bit_extract_butterfly_table_preprocess(raw));
    }

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

template <typename Enc>
static void check_scalar_all_spans(Enc const& enc)
{
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_all_scalar_variants(std::vector<std::string>{mask_for_span(span_k)}, enc);
    }
}

static void check_scalar_callback_forms()
{
    std::string const seq = "ACGTNACGTACGT";
    std::string const mask_string = "10101";
    BitExtractMask const mask(prepare_spaced_kmer_bit_extract_mask(mask_string));

    std::vector<SpacedEvent> full;
    for_each_spaced_kmer(
        seq, mask_string.size(), mask, CharEncoderTable<Encoding::kACGT>{},
        [](std::uint64_t x, BitExtractMask const& m) { return bit_extract_bitloop(x, m); },
        [&](std::size_t pos, std::size_t mask_idx, std::uint64_t value) {
            full.push_back({pos, mask_idx, value});
        }
    );

    std::vector<SpacedEvent> shorthand;
    for_each_spaced_kmer(
        seq, mask_string.size(), mask, CharEncoderTable<Encoding::kACGT>{},
        [](std::uint64_t x, BitExtractMask const& m) { return bit_extract_bitloop(x, m); },
        [&](std::size_t pos, std::uint64_t value) {
            shorthand.push_back({pos, 0, value});
        }
    );

    check_events(full, shorthand);
    check_events(full, oracle_events(seq, {mask_string}, code_acgt));
}

// =================================================================================================
//     SIMD Extraction
// =================================================================================================

template <typename Kernel, std::size_t N>
static std::array<Kernel, N> make_kernels(std::array<std::string, N> const& masks)
{
    std::array<Kernel, N> kernels{};
    for (std::size_t i = 0; i < N; ++i) {
        kernels[i] = Kernel(prepare_spaced_kmer_bit_extract_mask(masks[i]));
    }
    return kernels;
}

template <typename Kernel, std::size_t N, typename Enc>
static std::vector<SpacedEvent> collect_simd_by_position(
    std::string const& seq,
    std::size_t span_k,
    std::array<Kernel, N> const& kernels,
    Enc const& enc
) {
    std::vector<SpacedEvent> out;
    for_each_spaced_kmer_simd_by_position(
        seq, span_k, kernels, enc,
        [&](std::size_t pos, std::size_t mask_idx, std::uint64_t value) {
            out.push_back({pos, mask_idx, value});
        }
    );
    return out;
}

template <typename Kernel, std::size_t N, typename Enc>
static std::vector<SpacedEvent> collect_simd_by_mask(
    std::string const& seq,
    std::size_t span_k,
    std::array<Kernel, N> const& kernels,
    Enc const& enc
) {
    std::vector<SpacedEvent> out;
    for_each_spaced_kmer_simd_by_mask(
        seq, span_k, kernels, enc,
        [&](std::size_t pos, std::size_t mask_idx, std::uint64_t value) {
            out.push_back({pos, mask_idx, value});
        }
    );
    return out;
}

template <typename Kernel, std::size_t N, typename Enc>
static void check_simd_set(std::array<std::string, N> const& mask_array, Enc const& enc)
{
    std::vector<std::string> const masks = to_vector(mask_array);
    auto const kernels = make_kernels<Kernel>(mask_array);
    std::size_t const span_k = masks.front().size();

    for (auto const& seq : sequence_corpus(span_k, Kernel::lanes)) {
        auto const expected = oracle_events(
            seq, masks,
            [](char c) { return Enc::encoding == Encoding::kACGT ? code_acgt(c) : code_actg(c); }
        );
        check_events(collect_simd_by_position(seq, span_k, kernels, enc), expected);
        check_by_mask_events(collect_simd_by_mask(seq, span_k, kernels, enc), expected, N);
    }
}

template <typename Kernel>
static void check_simd_callback_forms()
{
    std::string const seq = "ACGTNACGTACGT";
    std::string const mask_string = "10101";
    std::size_t const span_k = mask_string.size();
    Kernel const kernel(prepare_spaced_kmer_bit_extract_mask(mask_string));
    std::array<Kernel, 1> const kernels{{kernel}};

    std::vector<SpacedEvent> expected = oracle_events(seq, {mask_string}, code_acgt);
    std::vector<SpacedEvent> got;


    for_each_spaced_kmer_simd_by_mask(
        seq, span_k, kernel, CharEncoderTable<Encoding::kACGT>{},
        [&](std::size_t pos, std::size_t mask_idx, std::uint64_t value) {
            got.push_back({pos, mask_idx, value});
        }
    );
    check_events(got, expected);

    got.clear();
    for_each_spaced_kmer_simd_by_position(
        seq, span_k, kernel, CharEncoderTable<Encoding::kACGT>{},
        [&](std::size_t pos, std::size_t mask_idx, std::uint64_t value) {
            got.push_back({pos, mask_idx, value});
        }
    );
    check_events(got, expected);

    got.clear();
    for_each_spaced_kmer_simd_by_mask(
        seq, span_k, kernel, CharEncoderTable<Encoding::kACGT>{},
        [&](std::size_t pos, std::uint64_t value) { got.push_back({pos, 0, value}); }
    );
    check_events(got, expected);

    got.clear();
    for_each_spaced_kmer_simd_by_mask(
        seq, span_k, kernels, CharEncoderTable<Encoding::kACGT>{},
        [&](std::size_t pos, std::uint64_t value) { got.push_back({pos, 0, value}); }
    );
    check_events(got, expected);

    got.clear();
    for_each_spaced_kmer_simd_by_position(
        seq, span_k, kernel, CharEncoderTable<Encoding::kACGT>{},
        [&](std::size_t pos, std::uint64_t value) { got.push_back({pos, 0, value}); }
    );
    check_events(got, expected);

    got.clear();
    for_each_spaced_kmer_simd_by_position(
        seq, span_k, kernels, CharEncoderTable<Encoding::kACGT>{},
        [&](std::size_t pos, std::uint64_t value) { got.push_back({pos, 0, value}); }
    );
    check_events(got, expected);
}

template <typename Kernel>
static void check_simd_kernel()
{
    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        check_simd_set<Kernel>(
            std::array<std::string, 1>{mask_for_span(span_k)},
            CharEncoderTable<Encoding::kACGT>{}
        );
    }
    check_simd_set<Kernel>(generated_masks<3>(), CharEncoderTable<Encoding::kACGT>{});
    check_simd_callback_forms<Kernel>();
}

template <typename Kernel>
static void check_simd_dispatcher(std::vector<std::string> const& masks)
{
    BitExtractKernelDispatcher<Kernel> const dispatcher(raw_masks(masks));
    std::size_t const span_k = masks.front().size();

    for (auto const& seq : sequence_corpus(span_k, Kernel::lanes)) {
        auto const expected = oracle_events(seq, masks, code_acgt);
        std::vector<SpacedEvent> by_position;
        dispatcher.run([&](auto const& kernels) {
            for_each_spaced_kmer_simd_by_position(
                seq, span_k, kernels, CharEncoderTable<Encoding::kACGT>{},
                [&](std::size_t pos, std::size_t mask_idx, std::uint64_t value) {
                    by_position.push_back({pos, mask_idx, value});
                }
            );
        });
        check_events(by_position, expected);

        std::vector<SpacedEvent> by_mask;
        dispatcher.run([&](auto const& kernels) {
            for_each_spaced_kmer_simd_by_mask(
                seq, span_k, kernels, CharEncoderTable<Encoding::kACGT>{},
                [&](std::size_t pos, std::size_t mask_idx, std::uint64_t value) {
                    by_mask.push_back({pos, mask_idx, value});
                }
            );
        });
        check_by_mask_events(by_mask, expected, masks.size());
    }
}

// =================================================================================================
//     Mask Helpers
// =================================================================================================

TEST(KmerSpaced, MaskHelpers)
{
    EXPECT_EQ(
        prepare_spaced_kmer_position_mask("1"),
        (std::vector<std::size_t>{0})
    );
    EXPECT_EQ(
        prepare_spaced_kmer_position_mask("10*11"),
        (std::vector<std::size_t>{0, 3, 4})
    );
    EXPECT_EQ(
        prepare_spaced_kmer_position_masks({"101", "1*1"}),
        (std::vector<std::vector<std::size_t>>{{0, 2}, {0, 2}})
    );

    EXPECT_EQ(prepare_spaced_kmer_bit_extract_mask("1011"), std::uint64_t{0xCF});
    EXPECT_EQ(prepare_spaced_kmer_bit_extract_mask("1*1"), std::uint64_t{0x33});
    EXPECT_EQ(prepare_spaced_kmer_bit_extract_mask(std::string(32, '1')), ~std::uint64_t{0});

    EXPECT_ANY_THROW(prepare_spaced_kmer_position_mask(""));
    EXPECT_ANY_THROW(prepare_spaced_kmer_position_mask(std::string(33, '1')));
    EXPECT_ANY_THROW(prepare_spaced_kmer_position_mask("010"));
    EXPECT_ANY_THROW(prepare_spaced_kmer_position_mask("1010"));
    EXPECT_ANY_THROW(prepare_spaced_kmer_position_mask("1x1"));

    EXPECT_ANY_THROW(prepare_spaced_kmer_bit_extract_mask(""));
    EXPECT_ANY_THROW(prepare_spaced_kmer_bit_extract_mask(std::string(33, '1')));
    EXPECT_ANY_THROW(prepare_spaced_kmer_bit_extract_mask("010"));
    EXPECT_ANY_THROW(prepare_spaced_kmer_bit_extract_mask("1010"));
    EXPECT_ANY_THROW(prepare_spaced_kmer_bit_extract_mask("1x1"));
}

TEST(KmerSpaced, MaskValidationAndFormatting)
{
    std::vector<std::string> const valid_masks = {
        "1", "101", "1*01", "11111", "10001",
        "10000000000000000000000000000001", std::string(32, '1')
    };
    for (auto const& mask_string : valid_masks) {
        std::uint64_t const mask = prepare_spaced_kmer_bit_extract_mask(mask_string);
        EXPECT_TRUE(is_valid_spaced_kmer_mask(mask, mask_string.size()));

        std::string normalized = mask_string;
        std::replace(normalized.begin(), normalized.end(), '*', '0');
        EXPECT_EQ(
            bit_extract_mask_to_spaced_kmer_mask_string(mask, mask_string.size()),
            normalized
        );
    }

    for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
        std::string const mask_string = mask_for_span(span_k);
        std::uint64_t const mask = prepare_spaced_kmer_bit_extract_mask(mask_string);
        EXPECT_TRUE(is_valid_spaced_kmer_mask(mask, span_k));
        EXPECT_EQ(bit_extract_mask_to_spaced_kmer_mask_string(mask, span_k), mask_string);
    }

    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x3}, 2));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0xD}, 2));
    EXPECT_TRUE(!is_valid_spaced_kmer_mask(std::uint64_t{0x3F}, 2));
    EXPECT_ANY_THROW(is_valid_spaced_kmer_mask(std::uint64_t{0}, 0));
    EXPECT_ANY_THROW(is_valid_spaced_kmer_mask(std::uint64_t{0}, 33));

    EXPECT_EQ(bit_extract_mask_to_spaced_kmer_mask_string(std::uint64_t{0x3}, 2), "01");
    EXPECT_ANY_THROW(bit_extract_mask_to_spaced_kmer_mask_string(std::uint64_t{0x1}, 1));
    EXPECT_ANY_THROW(bit_extract_mask_to_spaced_kmer_mask_string(std::uint64_t{0x3F}, 2));
}

// =================================================================================================
//     Scalar Tests
// =================================================================================================

TEST(KmerSpaced, ScalarAcgt)
{
    check_scalar_callback_forms();
    check_scalar_all_spans(CharEncoderTable<Encoding::kACGT>{});
    check_all_scalar_variants(
        {"1"},
        CharEncoderTable<Encoding::kACGT>{}
    );
    check_all_scalar_variants(
        {"10101", "11111", "10001"},
        CharEncoderTable<Encoding::kACGT>{}
    );
    check_all_scalar_variants(
        {"10000000000000000000000000000001"},
        CharEncoderTable<Encoding::kACGT>{}
    );
}

TEST(KmerSpaced, ScalarActg)
{
    check_scalar_all_spans(CharEncoderTable<Encoding::kACTG>{});
    check_all_scalar_variants(
        {"1"},
        CharEncoderTable<Encoding::kACTG>{}
    );
    check_all_scalar_variants(
        {"10101", "11111", "10001"},
        CharEncoderTable<Encoding::kACTG>{}
    );
    check_all_scalar_variants(
        {"10000000000000000000000000000001"},
        CharEncoderTable<Encoding::kACTG>{}
    );
}

TEST(KmerSpaced, InvalidSpan)
{
    BitExtractMask const mask(prepare_spaced_kmer_bit_extract_mask("1"));
    auto const extract = [](std::uint64_t x, BitExtractMask const& m) {
        return bit_extract_bitloop(x, m);
    };

    EXPECT_THROW(
        for_each_spaced_kmer(
            "A", 0, mask, CharEncoderTable<Encoding::kACGT>{}, extract,
            [](std::size_t, std::uint64_t) {}
        ),
        std::invalid_argument
    );
    EXPECT_THROW(
        for_each_spaced_kmer(
            "A", 33, mask, CharEncoderTable<Encoding::kACGT>{}, extract,
            [](std::size_t, std::uint64_t) {}
        ),
        std::invalid_argument
    );

    using Kernel = BitExtractKernelButterflyScalar;
    Kernel const kernel(mask.mask);
    EXPECT_THROW(
        for_each_spaced_kmer_simd_by_position(
            "A", 0, kernel, CharEncoderTable<Encoding::kACGT>{},
            [](std::size_t, std::uint64_t) {}
        ),
        std::invalid_argument
    );
    EXPECT_THROW(
        for_each_spaced_kmer_simd_by_mask(
            "A", 33, kernel, CharEncoderTable<Encoding::kACGT>{},
            [](std::size_t, std::uint64_t) {}
        ),
        std::invalid_argument
    );
}

// =================================================================================================
//     SIMD Tests
// =================================================================================================

TEST(KmerSpacedSimd, ButterflyScalar)
{
    check_simd_kernel<BitExtractKernelButterflyScalar>();
}

TEST(KmerSpacedSimd, BlockScalar)
{
    check_simd_kernel<BitExtractKernelBlockScalar<>>();
}

#if defined(FISK_HAS_SSE2)
TEST(KmerSpacedSimd, ButterflySse2)
{
    check_simd_kernel<BitExtractKernelButterflySSE2>();
}

TEST(KmerSpacedSimd, BlockSse2)
{
    check_simd_kernel<BitExtractKernelBlockSSE2<>>();
}
#endif

#if defined(FISK_HAS_AVX2)
TEST(KmerSpacedSimd, ButterflyAvx2)
{
    check_simd_kernel<BitExtractKernelButterflyAVX2>();
}

TEST(KmerSpacedSimd, BlockAvx2)
{
    check_simd_kernel<BitExtractKernelBlockAVX2<>>();
}
#endif

#if defined(FISK_HAS_AVX512)
TEST(KmerSpacedSimd, ButterflyAvx512)
{
    check_simd_kernel<BitExtractKernelButterflyAVX512>();
}

TEST(KmerSpacedSimd, BlockAvx512)
{
    check_simd_kernel<BitExtractKernelBlockAVX512<>>();
}
#endif

#if defined(FISK_HAS_NEON)
TEST(KmerSpacedSimd, ButterflyNeon)
{
    check_simd_kernel<BitExtractKernelButterflyNEON>();
}

TEST(KmerSpacedSimd, BlockNeon)
{
    check_simd_kernel<BitExtractKernelBlockNEON<>>();
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
        check_simd_dispatcher<BitExtractKernelButterflyScalar>({mask_for_span(span_k)});
    }
    check_simd_dispatcher<BitExtractKernelButterflyScalar>(to_vector(generated_masks<1>()));
    check_simd_dispatcher<BitExtractKernelButterflyScalar>(to_vector(generated_masks<2>()));
    check_simd_dispatcher<BitExtractKernelButterflyScalar>(to_vector(generated_masks<3>()));
    check_simd_dispatcher<BitExtractKernelButterflyScalar>(to_vector(generated_masks<9>()));
    check_simd_dispatcher<BitExtractKernelButterflyScalar>(to_vector(generated_masks<16>()));

    #if defined(FISK_HAS_BMI2)
    // Eight PEXT lanes and 9+ masks exercise the two-word position-presence bitset.
    check_simd_dispatcher<BitExtractKernelPEXT<>>(to_vector(generated_masks<9>()));
    check_simd_dispatcher<BitExtractKernelPEXT<>>(to_vector(generated_masks<16>()));
    #elif defined(FISK_HAS_AVX512)
    // AVX-512 likewise has eight lanes, reaching the two-word presence path above eight masks.
    check_simd_dispatcher<BitExtractKernelButterflyAVX512>(to_vector(generated_masks<9>()));
    check_simd_dispatcher<BitExtractKernelButterflyAVX512>(to_vector(generated_masks<16>()));
    #endif
}

// One representative SIMD path uses ACTG too. Scalar tests above cover every scalar extractor
// under both encodings; this verifies the generic SIMD outer loop forwards its encoder unchanged.
TEST(KmerSpacedSimd, ActgForwarding)
{
    check_simd_set<BitExtractKernelButterflyScalar>(
        std::array<std::string, 3>{"10101", "11111", "10001"},
        CharEncoderTable<Encoding::kACTG>{}
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
    std::vector<BitExtractMask> const masks = {
        BitExtractMask(prepare_spaced_kmer_bit_extract_mask("10101")),
        BitExtractMask(prepare_spaced_kmer_bit_extract_mask("11111"))
    };

    for (auto const axis : {
        SpacedKmerAxis::kByMask, SpacedKmerAxis::kByPosition, SpacedKmerAxis::kEither
    }) {
        auto const mode = spaced_kmer_selector(masks, 5, axis, 256);
        EXPECT_TRUE(mode_matches_axis(mode, axis));
    }

    std::uint64_t const raw = prepare_spaced_kmer_bit_extract_mask("10101");
    EXPECT_TRUE(mode_matches_axis(
        spaced_kmer_selector(BitExtractMask(raw), 5, SpacedKmerAxis::kByPosition, 256),
        SpacedKmerAxis::kByPosition
    ));
    EXPECT_TRUE(mode_matches_axis(
        spaced_kmer_selector(raw, 5, SpacedKmerAxis::kByMask, 256),
        SpacedKmerAxis::kByMask
    ));
    EXPECT_TRUE(mode_matches_axis(
        spaced_kmer_selector(std::vector<std::uint64_t>{raw}, 5, SpacedKmerAxis::kEither, 256),
        SpacedKmerAxis::kEither
    ));

    std::uint64_t const full32 = prepare_spaced_kmer_bit_extract_mask(std::string(32, '1'));
    EXPECT_TRUE(mode_matches_axis(
        spaced_kmer_selector(full32, 32, SpacedKmerAxis::kByPosition, 256),
        SpacedKmerAxis::kByPosition
    ));

    EXPECT_ANY_THROW(spaced_kmer_selector(std::vector<BitExtractMask>{}, 5, SpacedKmerAxis::kEither, 256));
    EXPECT_ANY_THROW(spaced_kmer_selector(BitExtractMask(raw), 0, SpacedKmerAxis::kEither, 256));
    EXPECT_ANY_THROW(spaced_kmer_selector(BitExtractMask(raw), 33, SpacedKmerAxis::kEither, 256));
    EXPECT_ANY_THROW(spaced_kmer_selector(BitExtractMask(raw), 5, SpacedKmerAxis::kEither, 4));
    EXPECT_ANY_THROW(spaced_kmer_selector(BitExtractMask(std::uint64_t{0x3}), 5, SpacedKmerAxis::kEither, 256));
}
