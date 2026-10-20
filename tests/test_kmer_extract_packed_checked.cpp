#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/kmer_extract/packed.hpp"
#include "fisk/kmer_extract/packed_simd.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "fisk/core/simd_scalar.hpp"
#include "fisk/core/simd_sse2.hpp"
#include "fisk/core/simd_avx2.hpp"
#include "fisk/core/simd_avx512.hpp"
#include "fisk/core/simd_neon.hpp"
#include "corpus.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

// =================================================================================================
//     Helpers
// =================================================================================================

// Random sequences of the lengths 0..300 with lower and upper case nucleotides, without invalid
// characters, with scattered ones, and with many. Plus longer ones around blocks of 32 and 64
// positions, and with runs of N of up to 100, as in assemblies.
static std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        Splitmix64 rng(7007);
        for (std::size_t len = 0; len <= 300; ++len) {
            std::string seq = random_sequence(rng, len);
            double const rates[] = {0.0, 0.02, 0.2};
            inject_invalid(seq, rng, rates[len % 3], all_invalid_bytes());
            out.push_back(std::move(seq));
        }
        std::size_t const long_lengths[] = {511, 512, 513, 1000, 1023, 1024, 1025};
        for (std::size_t len : long_lengths) {
            std::string seq = random_sequence(rng, len);
            inject_invalid(seq, rng, 0.005, kHandPickedInvalid);
            out.push_back(std::move(seq));
        }
        for (std::size_t i = 0; i < 8; ++i) {
            std::string seq = random_sequence(rng, 2000);
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

// Runs `extract`, called as `extract(callback)`, with a `callback(pos, vec, valid_lanes)`, and
// checks the k-mers of the valid lanes against `expected`. The vectors have to cover the positions
// in order, `Simd::lanes` at a time, and lanes beyond the last k-mer must not be valid.
template <SimdArch Simd, typename Extract>
static void check_valid_lanes_callback(
    std::vector<ExpectedKmer> const& expected, std::size_t k, std::size_t num_kmers,
    Extract&& extract
) {
    using simd_vector = typename Simd::simd_vector;
    constexpr std::size_t lanes = Simd::lanes;

    std::vector<std::size_t> positions;
    std::vector<std::uint64_t> values;
    std::size_t next_pos = 0;
    extract([&](std::size_t pos, simd_vector vec, KmerVectorValidLanes<Simd> valid_lanes) {
        EXPECT_EQ(pos, next_pos);
        next_pos = pos + lanes;

        alignas(64) std::uint64_t buf[lanes];
        Simd::store_aligned(vec, buf);
        std::uint64_t const bits = valid_lanes.bits();
        EXPECT_EQ(bits >> lanes, std::uint64_t{0});
        for (std::size_t j = 0; j < lanes; ++j) {
            if ((bits >> j) & 1) {
                EXPECT_TRUE(pos + j < num_kmers);
                positions.push_back(pos + j);
                values.push_back(buf[j]);
            }
        }
    });
    EXPECT_TRUE(next_pos >= num_kmers);
    check_emitted_kmers(expected, k, values, &positions);
}

// Checks the checked extractors under the conventions of E and L on `seq`, packed with its
// validity, for all k, against `expected_of(seq, k)`.
template <Encoding E, Layout L, typename ExpectedOf>
static void check_checked_extractors(
    PackedSequenceWithValidity<E, L> const& packed, ExpectedOf&& expected_of
) {
    for (std::size_t k = 1; k <= 32; ++k) {
        auto const expected = expected_of(k);
        std::size_t const num_kmers = packed.length >= k ? packed.length - k + 1 : 0;

        check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
            for_each_kmer_packed_checked_aligned(packed, k, callback);
        });
        check_kmer_callbacks<E, L>(expected, k, [&](auto const& callback) {
            for_each_kmer_packed_checked_rolling(packed, k, callback);
        });

        auto const check_simd = [&]<typename Simd>() {
            check_valid_lanes_callback<Simd>(expected, k, num_kmers, [&](auto const& callback) {
                for_each_kmer_packed_checked_simd<Simd>(packed, k, callback);
            });
        };
        check_simd.template operator()<SimdScalar>();
        #if defined(FISK_HAS_SSE2)
        check_simd.template operator()<SimdSSE2>();
        #endif
        #if defined(FISK_HAS_AVX2)
        check_simd.template operator()<SimdAVX2>();
        #endif
        #if defined(FISK_HAS_AVX512)
        check_simd.template operator()<SimdAVX512>();
        #endif
        #if defined(FISK_HAS_NEON)
        check_simd.template operator()<SimdNEON>();
        #endif
    }
}

// Checks the checked extractors on `seq` against the k-mers of its windows without invalid
// characters.
template <Encoding E, Layout L>
static void check_sequence(std::string const& seq)
{
    auto const packed = pack_sequence_with_validity(seq, WordEncoderButterfly<E, L>{});
    check_checked_extractors(packed, [&](std::size_t k) {
        return oracle_kmers_skipping_invalid<E, L>(seq, k);
    });
}

// =================================================================================================
//     Checks
// =================================================================================================

template <Encoding E, Layout L>
static void check_random_sequences()
{
    for (auto const& seq : test_sequences()) {
        check_sequence<E, L>(seq);
    }
}

// One N at each position, on lengths that span two blocks of 32 plus the k - 1 nucleotides after
// them: its first and last nucleotides, and the edges of blocks and of the windows of their test.
template <Encoding E, Layout L>
static void check_single_invalid()
{
    Splitmix64 rng(8008);
    std::size_t const lengths[] = {64, 97, 130};
    for (std::size_t len : lengths) {
        std::string const base = random_sequence(rng, len);
        for (std::size_t at = 0; at < len; ++at) {
            std::string seq = base;
            seq[at] = 'N';
            check_sequence<E, L>(seq);
        }
    }
}

// With all validity bits set, the checked extractors emit the same k-mers as the unchecked ones,
// whatever the characters were.
template <Encoding E, Layout L>
static void check_all_valid_equals_unchecked()
{
    for (auto const& seq : test_sequences()) {
        auto packed = pack_sequence_with_validity(seq, WordEncoderButterfly<E, L>{});
        for (std::size_t i = 0; i < packed.length; ++i) {
            packed.valid[i / 8] = static_cast<std::uint8_t>(packed.valid[i / 8] | (1u << (i % 8)));
        }
        PackedSequence<E, L> const& unchecked = packed;
        check_checked_extractors(packed, [&](std::size_t k) {
            std::vector<ExpectedKmer> expected;
            for_each_kmer_packed_aligned(unchecked, k, [&](std::size_t pos, Kmer<E, L> kmer) {
                expected.push_back({pos, kmer_value(kmer)});
            });
            return expected;
        });
    }
}

// A nucleotide that is invalid only by its validity bit is invalid for the extractors: they never
// look at the codes for validity.
template <Encoding E, Layout L>
static void check_invalid_in_validity_only()
{
    Splitmix64 rng(9009);
    std::string const seq = random_sequence(rng, 200);
    std::size_t const invalid_at[] = {0, 1, 31, 32, 63, 64, 95, 96, 150, 199};
    for (std::size_t at : invalid_at) {
        auto packed = pack_sequence_with_validity(seq, WordEncoderButterfly<E, L>{});
        packed.valid[at / 8] = static_cast<std::uint8_t>(packed.valid[at / 8] & ~(1u << (at % 8)));
        std::string marked = seq;
        marked[at] = 'N';
        check_checked_extractors(packed, [&](std::size_t k) {
            return oracle_kmers_skipping_invalid<E, L>(marked, k);
        });
    }
}

// Validity bits that do not match the length of the sequence, and an invalid k, throw.
template <Encoding E, Layout L>
static void check_throws()
{
    auto const check_extractor = [](auto const& extract) {
        std::size_t const lengths[] = {0, 1, 8, 9, 100};
        for (std::size_t len : lengths) {
            std::string const seq(len, 'A');
            auto packed = pack_sequence_with_validity(seq, WordEncoderButterfly<E, L>{});
            packed.valid.push_back(0);
            EXPECT_THROW(extract(packed, 1), std::invalid_argument);
            packed.valid.pop_back();
            if (!packed.valid.empty()) {
                packed.valid.pop_back();
                EXPECT_THROW(extract(packed, 1), std::invalid_argument);
            }
        }
        check_invalid_k_throws(32, [&](std::string const& seq, std::size_t k) {
            extract(pack_sequence_with_validity(seq, WordEncoderButterfly<E, L>{}), k);
        });
    };
    check_extractor([](auto const& seq, std::size_t k) {
        for_each_kmer_packed_checked_aligned(seq, k, [](Kmer<E, L>) {});
    });
    check_extractor([](auto const& seq, std::size_t k) {
        for_each_kmer_packed_checked_rolling(seq, k, [](Kmer<E, L>) {});
    });
    check_extractor([](auto const& seq, std::size_t k) {
        for_each_kmer_packed_checked_simd<SimdScalar>(seq, k, [](std::size_t, auto, auto) {});
    });
}

// =================================================================================================
//     Tests
// =================================================================================================

TEST(KmerExtractPackedChecked, RandomAcgtMsb)
{
    check_random_sequences<Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedChecked, RandomAcgtLsb)
{
    check_random_sequences<Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedChecked, RandomActgMsb)
{
    check_random_sequences<Encoding::kACTG, Layout::kMSB>();
}

TEST(KmerExtractPackedChecked, RandomActgLsb)
{
    check_random_sequences<Encoding::kACTG, Layout::kLSB>();
}

TEST(KmerExtractPackedChecked, SingleInvalidMsb)
{
    check_single_invalid<Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedChecked, SingleInvalidLsb)
{
    check_single_invalid<Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedChecked, AllValidEqualsUncheckedMsb)
{
    check_all_valid_equals_unchecked<Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedChecked, AllValidEqualsUncheckedLsb)
{
    check_all_valid_equals_unchecked<Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedChecked, InvalidInValidityOnlyMsb)
{
    check_invalid_in_validity_only<Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedChecked, InvalidInValidityOnlyLsb)
{
    check_invalid_in_validity_only<Encoding::kACGT, Layout::kLSB>();
}

TEST(KmerExtractPackedChecked, ThrowsMsb)
{
    check_throws<Encoding::kACGT, Layout::kMSB>();
}

TEST(KmerExtractPackedChecked, ThrowsLsb)
{
    check_throws<Encoding::kACGT, Layout::kLSB>();
}
