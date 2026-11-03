#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fisk/bit_extract/bit_extract.hpp"
#include "fisk/core/types.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "fisk/spaced_kmer_extract/packed_bytewise_scalar.hpp"
#include "fisk/util/random.hpp"
#include "fisk/util/static_span.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "spaced_kmer_extract/common.hpp"
#include "testing.hpp"

// =================================================================================================
//     Spaced K-mer Extraction from Packed Sequences, Scalar: Shared Test Helpers
// =================================================================================================

// The checks of the tests of packed_bytewise_scalar.hpp, which are split into
// test_packed_bytewise_scalar_unchecked.cpp and _checked.cpp for spans up to 32, and
// test_packed_bytewise_scalar_wide_unchecked.cpp and _wide_checked.cpp for spans above 32, so that
// each compiles with less memory.

// Maximum span of the masks of the tests of spans up to 32, see SpacedKmerMask: masks that can
// have more pieces would also compile the loops for them, which these tests do not run, see the
// checks of spans above 32 below for those.
inline constexpr std::size_t kNarrowMaxSpan = 32;
template <fisk::Layout L, typename M>
using NarrowSpacedKmerMask = fisk::SpacedKmerMask<L, M, kNarrowMaxSpan>;

// Sequence lengths 0..100, which covers sequences shorter than the span, the last step, and the
// main loop for every span, plus two longer ones, each with random per-position content. The
// oracle (oracle.hpp) works on the ASCII input, independent of the packing and the extraction.
inline std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        fisk::Splitmix64 rng(6006);
        for (std::size_t len = 0; len <= 100; ++len) {
            out.push_back(random_sequence(rng, len));
        }
        out.push_back(random_sequence(rng, 257));
        out.push_back(random_sequence(rng, 511));
        return out;
    }();
    return seqs;
}

// Random sequences of the lengths 0..160 with lower and upper case nucleotides, without invalid
// characters, with scattered ones, and with many, plus longer ones around blocks of 32 positions,
// and with runs of N of up to 100, as in assemblies, for the checked extraction.
inline std::vector<std::string> const& checked_test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        fisk::Splitmix64 rng(8008);
        for (std::size_t len = 0; len <= 160; ++len) {
            std::string seq = random_sequence(rng, len);
            double const rates[] = {0.0, 0.02, 0.2};
            inject_invalid(seq, rng, rates[len % 3], all_invalid_bytes());
            out.push_back(std::move(seq));
        }
        for (std::size_t len : {511u, 512u, 513u, 1023u, 1024u, 1025u}) {
            std::string seq = random_sequence(rng, len);
            inject_invalid(seq, rng, 0.005, kHandPickedInvalid);
            out.push_back(std::move(seq));
        }
        for (std::size_t i = 0; i < 4; ++i) {
            std::string seq = random_sequence(rng, 1000);
            for (std::size_t r = 0; r < 5; ++r) {
                std::size_t const run = 1 + rng.get_uint64() % 100;
                std::size_t const at = rng.get_uint64() % (seq.size() - run);
                seq.replace(at, run, run, 'N');
            }
            out.push_back(std::move(seq));
        }
        return out;
    }();
    return seqs;
}

// For every span in [1, 32], a single random mask, and a set of three random masks, plus masks
// that keep all positions, or only the first and the last.
inline std::vector<std::vector<std::string>> const& test_mask_sets()
{
    static std::vector<std::vector<std::string>> const sets = [] {
        fisk::Splitmix64 rng(7007);
        auto random_mask = [&](std::size_t span_k) {
            std::string mask(span_k, '0');
            for (auto& c : mask) {
                c = (rng.get_uint64() & 1u) ? '1' : '0';
            }
            mask.front() = '1';
            mask.back() = '1';
            return mask;
        };

        std::vector<std::vector<std::string>> out;
        for (std::size_t span_k = 1; span_k <= 32; ++span_k) {
            out.push_back({ random_mask(span_k) });
            out.push_back({ random_mask(span_k), random_mask(span_k), random_mask(span_k) });
        }
        out.push_back({ std::string(32, '1') });
        out.push_back({ std::string(29, '1'), std::string("1").append(27, '0').append("1") });
        out.push_back({ std::string("1").append(30, '0').append("1") });
        return out;
    }();
    return sets;
}

// Checks `extract(masks, callback)` against `expected`, in all callback forms, with the vector of
// masks as a span of fixed size, via dispatch_static_span(), and for a single mask also
// `extract(mask, callback)`, with the mask itself.
template <fisk::Layout L, typename Masks, typename Extract>
inline void check_callback_forms(
    Masks const& masks, std::vector<SpacedEvent> const& expected, Extract&& extract
) {
    auto const runs = collect_spaced_callback_forms<L, false>([&](auto const& callback) {
        fisk::dispatch_static_span<kMaxTestMasks>(masks, [&](auto mask_span) {
            extract(mask_span, callback);
        });
    });
    for (auto const& got : runs) {
        check_spaced_events(got, expected);
    }
    if (masks.size() == 1) {
        auto const single_runs = collect_spaced_callback_forms<L, true>(
            [&](auto const& callback) {
                extract(masks.front(), callback);
            }
        );
        for (auto const& got : single_runs) {
            check_spaced_events(got, expected);
        }
    }
}

// Checks that spans outside of [1, MaxSpan] and sets of masks with different spans are rejected,
// unchecked and checked, through the single-mask overload and a set of masks, also for sequences
// shorter than the span; with spans above 32 among the different ones where MaxSpan allows them.
template <fisk::Layout L, std::size_t MaxSpan>
inline void check_error_contract()
{
    using Mask = fisk::SpacedKmerMask<L, fisk::BitExtractMask, MaxSpan>;
    fisk::WordEncoderButterfly<fisk::Encoding::kACGT, L> const encoder;
    auto const noop = [](std::size_t, std::size_t, auto const&) {};
    for (std::string const& seq : {std::string{}, std::string(2 * MaxSpan + 1, 'T')}) {
        auto const unchecked = fisk::pack_sequence_unchecked(seq, encoder);
        auto const checked = fisk::pack_sequence_checked(seq, encoder);
        auto const expect_throw = [&](auto const& mask_or_masks) {
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
                    unchecked, mask_or_masks, fisk::bit_extract_bit_loop, noop
                ),
                std::invalid_argument
            );
            EXPECT_THROW(
                fisk::for_each_spaced_kmer_packed_checked_bytewise_scalar(
                    checked, mask_or_masks, fisk::bit_extract_bit_loop, noop
                ),
                std::invalid_argument
            );
        };

        // A mask with an invalid span, which make_spaced_kmer_mask() would reject.
        for (std::size_t const span_k : {std::size_t{0}, MaxSpan + 1}) {
            Mask mask;
            mask.mask_pieces[0] = fisk::BitExtractMask(3);
            mask.span_k = static_cast<std::uint16_t>(span_k);
            expect_throw(mask);
            expect_throw(std::span<Mask const, 1>(&mask, 1));
        }

        std::vector<std::vector<std::string>> mixed_sets = {{"101", "1001"}};
        if constexpr (MaxSpan > 32) {
            mixed_sets.push_back({
                std::string("1").append(38, '0').append("1"),
                std::string("1").append(39, '0').append("1")
            });
        }
        for (auto const& mask_strings : mixed_sets) {
            auto const mixed = fisk::make_spaced_kmer_masks<L, MaxSpan>(mask_strings);
            fisk::dispatch_static_span<kMaxTestMasks>(mixed, [&](auto mask_span) {
                expect_throw(mask_span);
            });
        }
    }
}

// Checks the largest set of masks that dispatch_static_span() hands on by default, of
// kDefaultMaxMasks masks, as a span of that fixed size, with one bit extractor, unchecked or
// checked: the other checks dispatch only up to kMaxTestMasks masks.
template <fisk::Layout L, bool Checked>
inline void check_default_max_masks()
{
    constexpr fisk::Encoding E = fisk::Encoding::kACGT;
    fisk::WordEncoderButterfly<E, L> const encoder;
    fisk::Splitmix64 rng(7008);
    std::vector<std::string> mask_strings;
    for (std::size_t i = 0; i < fisk::kDefaultMaxMasks; ++i) {
        std::string mask(21, '0');
        for (auto& c : mask) {
            c = (rng.get_uint64() & 1u) ? '1' : '0';
        }
        mask.front() = '1';
        mask.back() = '1';
        mask_strings.push_back(std::move(mask));
    }
    auto const masks = fisk::make_spaced_kmer_masks<L, kNarrowMaxSpan>(mask_strings);
    std::span<NarrowSpacedKmerMask<L, fisk::BitExtractMask> const, fisk::kDefaultMaxMasks> const
        mask_span(masks.data(), fisk::kDefaultMaxMasks);

    for (auto const& seq : Checked ? checked_test_sequences() : test_sequences()) {
        std::vector<SpacedEvent> events;
        auto const collect = [&](
            std::size_t pos, std::size_t mask_idx, fisk::SpacedKmer<E, L> const& spaced_kmer
        ) {
            events.push_back({pos, mask_idx, fisk::kmer_value(spaced_kmer)});
        };
        if constexpr (Checked) {
            fisk::for_each_spaced_kmer_packed_checked_bytewise_scalar(
                fisk::pack_sequence_checked(seq, encoder), mask_span, fisk::bit_extract_bit_loop,
                collect
            );
        } else {
            fisk::for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
                fisk::pack_sequence_unchecked(seq, encoder), mask_span, fisk::bit_extract_bit_loop,
                collect
            );
        }
        check_spaced_events(events, oracle_spaced_events<E, L>(seq, mask_strings));
    }
}

// =================================================================================================
//     Spans above 32
// =================================================================================================

// For the spans of one piece at a maximum span above 32, and for spans above 32, around the
// boundaries of their pieces, see SpacedKmerMask, a single mask and a set of three masks, each with
// random kept positions, at most 32 of them. Plus single masks that keep only the first and the
// last position, so that the pieces between them keep none, and masks that keep 32 positions
// spread over all pieces.
inline std::vector<std::vector<std::string>> const& wide_test_mask_sets()
{
    static std::vector<std::vector<std::string>> const sets = [] {
        fisk::Splitmix64 rng(7009);
        auto random_mask = [&](std::size_t span_k) {
            std::string mask(span_k, '0');
            std::size_t const weight = 2 + rng.get_uint64() % 31;
            for (std::size_t i = 0; i < weight; ++i) {
                mask[rng.get_uint64() % span_k] = '1';
            }
            mask.front() = '1';
            mask.back() = '1';
            return mask;
        };
        std::vector<std::vector<std::string>> out;
        for (std::size_t span_k : {
            29u, 30u, 31u, 32u, 33u, 34u, 56u, 57u, 58u, 60u, 64u, 84u, 85u, 86u, 100u, 113u,
            114u, 128u
        }) {
            out.push_back({ random_mask(span_k) });
            out.push_back({ random_mask(span_k), random_mask(span_k), random_mask(span_k) });
        }
        for (std::size_t span_k : {60u, 86u, 128u}) {
            out.push_back({ std::string("1").append(span_k - 2, '0').append("1") });
        }
        for (std::size_t span_k : {64u, 100u, 128u}) {
            std::string mask(span_k, '0');
            for (std::size_t i = 0; i < 32; ++i) {
                mask[i * (span_k - 1) / 31] = '1';
            }
            out.push_back({ mask });
        }
        return out;
    }();
    return sets;
}

// Checks masks with spans above 32, unchecked or checked, against the oracle: with the default
// `MaxCompiledPieces`, through the overload for a set of masks and, for a single mask, also the
// single-mask overload, in all callback forms, with three bit extractors; and with a copy of the
// loop for each piece count (`MaxCompiledPieces` 5) and for one piece only (1), with the bit loop.
// The Encoding does not change the loop, so that ACGT suffices here.
template <fisk::Layout L, bool Checked>
inline void check_wide_extraction()
{
    constexpr fisk::Encoding E = fisk::Encoding::kACGT;
    fisk::WordEncoderButterfly<E, L> const encoder;
    auto const bit_loop = [](std::uint64_t x, fisk::BitExtractMask const& piece) {
        return fisk::bit_extract_bit_loop(x, piece);
    };
    for (auto const& mask_strings : wide_test_mask_sets()) {
        auto const direct_masks = fisk::make_spaced_kmer_masks<L, 128>(mask_strings);
        auto const butterfly_masks = fisk::make_spaced_kmer_butterfly_masks(direct_masks);

        auto const check = [&](std::string const& seq, auto const& packed) {
            auto const expected = oracle_spaced_events<E, L>(seq, mask_strings);
            auto const extract = [&]<std::size_t MaxCompiledPieces>(
                auto const& mask_or_masks, auto const& bit_ext, auto const& callback
            ) {
                if constexpr (Checked) {
                    fisk::for_each_spaced_kmer_packed_checked_bytewise_scalar<MaxCompiledPieces>(
                        packed, mask_or_masks, bit_ext, callback
                    );
                } else {
                    fisk::for_each_spaced_kmer_packed_unchecked_bytewise_scalar<MaxCompiledPieces>(
                        packed, mask_or_masks, bit_ext, callback
                    );
                }
            };
            auto const run = [&](auto const& masks, auto const& bit_ext) {
                check_callback_forms<L>(
                    masks, expected, [&](auto const& mask_or_masks, auto const& callback) {
                        extract.template operator()<fisk::kDefaultMaxCompiledPieces>(
                            mask_or_masks, bit_ext, callback
                        );
                    }
                );
            };
            run(direct_masks, bit_loop);
            run(butterfly_masks, [](std::uint64_t x, fisk::BitExtractButterflyMask const& piece) {
                return fisk::bit_extract_butterfly(x, piece);
            });
            #if defined(FISK_HAS_BMI2)
            run(direct_masks, [](std::uint64_t x, fisk::BitExtractMask const& piece) {
                return fisk::bit_extract_pext(x, piece);
            });
            #endif

            // The single mask itself, or the set of three masks as a span of that size.
            auto const check_cap = [&]<std::size_t MaxCompiledPieces>() {
                std::vector<SpacedEvent> events;
                auto const collect = [&](
                    std::size_t pos, std::size_t mask_idx, auto const& spaced_kmer
                ) {
                    events.push_back({pos, mask_idx, fisk::kmer_value(spaced_kmer)});
                };
                if (direct_masks.size() == 1) {
                    extract.template operator()<MaxCompiledPieces>(
                        direct_masks.front(), bit_loop, collect
                    );
                } else {
                    using Mask = typename decltype(direct_masks)::value_type;
                    extract.template operator()<MaxCompiledPieces>(
                        std::span<Mask const, 3>(direct_masks.data(), 3), bit_loop, collect
                    );
                }
                check_spaced_events(events, expected);
            };
            check_cap.template operator()<1>();
            check_cap.template operator()<5>();
        };
        if constexpr (Checked) {
            for (auto const& seq : checked_test_sequences()) {
                check(seq, fisk::pack_sequence_checked(seq, encoder));
            }
        } else {
            for (auto const& seq : test_sequences()) {
                check(seq, fisk::pack_sequence_unchecked(seq, encoder));
            }
        }
    }
}

// Checks masks of the default maximum span, see SpacedKmerMask, which the other tests leave out,
// at spans on both sides of 32 and of the boundaries of the pieces, unchecked or checked, with one
// bit extractor and callback form.
template <fisk::Layout L, bool Checked>
inline void check_default_max_span()
{
    constexpr fisk::Encoding E = fisk::Encoding::kACGT;
    fisk::WordEncoderButterfly<E, L> const encoder;
    fisk::Splitmix64 rng(7010);
    for (std::size_t span_k : {1u, 29u, 30u, 31u, 32u, 33u, 57u, 58u, 64u}) {
        std::string mask_string(span_k, '0');
        std::size_t const weight = 1 + rng.get_uint64() % std::min<std::size_t>(span_k, 32);
        for (std::size_t i = 0; i < weight; ++i) {
            mask_string[rng.get_uint64() % span_k] = '1';
        }
        mask_string.front() = '1';
        mask_string.back() = '1';
        auto const mask = fisk::make_spaced_kmer_mask<L>(mask_string);

        auto const check = [&](std::string const& seq, auto const& packed) {
            std::vector<SpacedEvent> events;
            auto const collect = [&](
                std::size_t pos, std::size_t mask_idx, auto const& spaced_kmer
            ) {
                events.push_back({pos, mask_idx, fisk::kmer_value(spaced_kmer)});
            };
            if constexpr (Checked) {
                fisk::for_each_spaced_kmer_packed_checked_bytewise_scalar(
                    packed, mask, fisk::bit_extract_bit_loop, collect
                );
            } else {
                fisk::for_each_spaced_kmer_packed_unchecked_bytewise_scalar(
                    packed, mask, fisk::bit_extract_bit_loop, collect
                );
            }
            std::vector<std::string> const mask_strings{mask_string};
            check_spaced_events(events, oracle_spaced_events<E, L>(seq, mask_strings));
        };
        if constexpr (Checked) {
            for (auto const& seq : checked_test_sequences()) {
                check(seq, fisk::pack_sequence_checked(seq, encoder));
            }
        } else {
            for (auto const& seq : test_sequences()) {
                check(seq, fisk::pack_sequence_unchecked(seq, encoder));
            }
        }
    }
}
