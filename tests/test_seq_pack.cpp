#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "fisk/core/random.hpp"
#include "fisk/seq_pack/seq_pack.hpp"
#include "fisk/seq_pack/simd.hpp"
#include "testing.hpp"

// =================================================================================================
//     Helpers and Oracle
// =================================================================================================

// Ground truth two-bit codes, independent of the encoders under test.

static int expected_acgt(char c)
{
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:            return -1;
    }
}

static int expected_actg(char c)
{
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'T': case 't': return 2;
        case 'G': case 'g': return 3;
        default:            return -1;
    }
}

// Decode base `i` per TwoBitSequence's documented contract (see BitOrder in core/seq_enc.hpp),
// independent of pack_sequence()/pack_sequence_simd()'s internals.
template <BitOrder Order>
static int decode_base(TwoBitSequence<Order> const& s, std::size_t i)
{
    std::uint8_t const byte = s.data[i / 4];
    std::size_t const in_byte = i % 4;
    if constexpr (Order == BitOrder::Msb) {
        return (byte >> (6 - 2 * in_byte)) & 0x3;
    } else {
        return (byte >> (2 * in_byte)) & 0x3;
    }
}

// Sequence lengths 0..65 plus a batch of longer random lengths, each with random per-position
// content, so bit- and byte-order correctness are both exercised at every length.
static std::vector<std::string> const& test_sequences()
{
    static std::vector<std::string> const seqs = [] {
        std::vector<std::string> out;
        char const bases[] = "ACGTacgt";
        Splitmix64 rng(3003);

        auto random_seq = [&](std::size_t len) {
            std::string s;
            s.reserve(len);
            for (std::size_t k = 0; k < len; ++k) {
                s += bases[rng.get_uint64() % 8];
            }
            return s;
        };

        for (std::size_t len = 0; len <= 65; ++len) {
            out.push_back(random_seq(len));
        }
        for (int i = 0; i < 30; ++i) {
            std::size_t const len = static_cast<std::size_t>(rng.get_uint64() % 200);
            out.push_back(random_seq(len));
        }
        return out;
    }();
    return seqs;
}

// Checks a packed TwoBitSequence against `seq` and its oracle: length, data size, the sentinel
// bytes, and every base's code.
template <BitOrder Order, typename OracleFn>
static void check_packed(
    std::string const& seq, TwoBitSequence<Order> const& packed, OracleFn&& oracle
) {
    EXPECT_EQ(packed.length, seq.size());

    std::size_t const expected_content_bytes = (seq.size() + 3) / 4;
    EXPECT_EQ(packed.data.size(), expected_content_bytes + 8);

    for (std::size_t k = 0; k < 8; ++k) {
        EXPECT_EQ(static_cast<int>(packed.data[expected_content_bytes + k]), 0);
    }

    for (std::size_t i = 0; i < seq.size(); ++i) {
        EXPECT_EQ(decode_base(packed, i), oracle(seq[i]));
    }
}

// Checks pack_sequence() -- both overloads, and reuse of one TwoBitSequence across calls -- for a
// given scalar Extractor and its matching oracle.
template <typename Extractor, typename OracleFn>
static void check_scalar_pack(OracleFn&& oracle)
{
    Extractor extract{};
    TwoBitSequence<Extractor::order> out;

    for (auto const& seq : test_sequences()) {
        pack_sequence(seq, extract, out);
        check_packed(seq, out, oracle);

        auto const out2 = pack_sequence(seq, extract);
        check_packed(seq, out2, oracle);
    }

    // Reuse must not leak stale bytes from a previous, larger pack.
    pack_sequence(std::string(100, 'T'), extract, out);
    pack_sequence(std::string("AC"), extract, out);
    check_packed(std::string("AC"), out, oracle);
}

#if defined(FISK_HAS_SSE2)   || \
    defined(FISK_HAS_AVX2)   || \
    defined(FISK_HAS_AVX512) || \
    defined(FISK_HAS_NEON)

// Same as check_scalar_pack(), but for pack_sequence_simd() and a SIMD Extractor.
template <typename Extractor, typename OracleFn>
static void check_simd_pack(OracleFn&& oracle)
{
    Extractor extract{};
    TwoBitSequence<Extractor::order> out;

    for (auto const& seq : test_sequences()) {
        pack_sequence_simd(seq, extract, out);
        check_packed(seq, out, oracle);

        auto const out2 = pack_sequence_simd(seq, extract);
        check_packed(seq, out2, oracle);
    }

    pack_sequence_simd(std::string(100, 'T'), extract, out);
    pack_sequence_simd(std::string("AC"), extract, out);
    check_packed(std::string("AC"), out, oracle);
}

#endif

// =================================================================================================
//     Scalar pack_sequence()
// =================================================================================================

// -----------------------------------------------------------------------------
//     Butterfly Table
// -----------------------------------------------------------------------------

TEST(SeqPack, ScalarButterflyActgLsb)
{
    check_scalar_pack<EncodeActg8ButterflyLsb>(expected_actg);
}

TEST(SeqPack, ScalarButterflyActgMsb)
{
    check_scalar_pack<EncodeActg8ButterflyMsb>(expected_actg);
}

TEST(SeqPack, ScalarButterflyAcgtLsb)
{
    check_scalar_pack<EncodeAcgt8ButterflyLsb>(expected_acgt);
}

TEST(SeqPack, ScalarButterflyAcgtMsb)
{
    check_scalar_pack<EncodeAcgt8ButterflyMsb>(expected_acgt);
}

// -----------------------------------------------------------------------------
//     PEXT
// -----------------------------------------------------------------------------

#if defined(FISK_HAS_BMI2)

TEST(SeqPack, ScalarPextActgLsb)
{
    check_scalar_pack<EncodeActg8PextLsb>(expected_actg);
}

TEST(SeqPack, ScalarPextActgMsb)
{
    check_scalar_pack<EncodeActg8PextMsb>(expected_actg);
}

TEST(SeqPack, ScalarPextAcgtLsb)
{
    check_scalar_pack<EncodeAcgt8PextLsb>(expected_acgt);
}

TEST(SeqPack, ScalarPextAcgtMsb)
{
    check_scalar_pack<EncodeAcgt8PextMsb>(expected_acgt);
}

#endif // FISK_HAS_BMI2

// =================================================================================================
//     SIMD pack_sequence_simd()
// =================================================================================================

#if defined(FISK_HAS_SSE2)

TEST(SeqPackSimd, Sse2ActgLsb) { check_simd_pack<EncodeActgButterflySse2Lsb>(expected_actg); }
TEST(SeqPackSimd, Sse2ActgMsb) { check_simd_pack<EncodeActgButterflySse2Msb>(expected_actg); }
TEST(SeqPackSimd, Sse2AcgtLsb) { check_simd_pack<EncodeAcgtButterflySse2Lsb>(expected_acgt); }
TEST(SeqPackSimd, Sse2AcgtMsb) { check_simd_pack<EncodeAcgtButterflySse2Msb>(expected_acgt); }

#endif // FISK_HAS_SSE2

#if defined(FISK_HAS_AVX2)

TEST(SeqPackSimd, Avx2ActgLsb) { check_simd_pack<EncodeActgButterflyAvx2Lsb>(expected_actg); }
TEST(SeqPackSimd, Avx2ActgMsb) { check_simd_pack<EncodeActgButterflyAvx2Msb>(expected_actg); }
TEST(SeqPackSimd, Avx2AcgtLsb) { check_simd_pack<EncodeAcgtButterflyAvx2Lsb>(expected_acgt); }
TEST(SeqPackSimd, Avx2AcgtMsb) { check_simd_pack<EncodeAcgtButterflyAvx2Msb>(expected_acgt); }

#endif // FISK_HAS_AVX2

#if defined(FISK_HAS_AVX512)

TEST(SeqPackSimd, Avx512ActgLsb) { check_simd_pack<EncodeActgButterflyAvx512Lsb>(expected_actg); }
TEST(SeqPackSimd, Avx512ActgMsb) { check_simd_pack<EncodeActgButterflyAvx512Msb>(expected_actg); }
TEST(SeqPackSimd, Avx512AcgtLsb) { check_simd_pack<EncodeAcgtButterflyAvx512Lsb>(expected_acgt); }
TEST(SeqPackSimd, Avx512AcgtMsb) { check_simd_pack<EncodeAcgtButterflyAvx512Msb>(expected_acgt); }

#endif // FISK_HAS_AVX512

#if defined(FISK_HAS_NEON)

TEST(SeqPackSimd, NeonActgLsb) { check_simd_pack<EncodeActgButterflyNeonLsb>(expected_actg); }
TEST(SeqPackSimd, NeonActgMsb) { check_simd_pack<EncodeActgButterflyNeonMsb>(expected_actg); }
TEST(SeqPackSimd, NeonAcgtLsb) { check_simd_pack<EncodeAcgtButterflyNeonLsb>(expected_acgt); }
TEST(SeqPackSimd, NeonAcgtMsb) { check_simd_pack<EncodeAcgtButterflyNeonMsb>(expected_acgt); }

#endif // FISK_HAS_NEON
