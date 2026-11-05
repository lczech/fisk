// The k-mer loops forced inline, as programs that use fisk get them. The test suite otherwise
// defines FISK_ALWAYS_INLINE_FOR_EACH as empty (see CMakeLists.txt), as forcing all of its many
// instantiations inline costs too much compile time. Here, it is undefined before any fisk header,
// so that util/intrinsics.hpp defines it as for any other program, and each loop is compiled and
// run once with it: the algorithms of the extractors, and the loops that are none of them.
#undef FISK_ALWAYS_INLINE_FOR_EACH

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "fisk/core/char_encoder.hpp"
#include "fisk/core/kmer_callback.hpp"
#include "fisk/kmer_extract/ascii_reextract_scalar.hpp"
#include "fisk/kmer_extract/ascii_rolling_scalar.hpp"
#include "fisk/kmer_extract/ascii_rolling_simd.hpp"
#include "fisk/kmer_extract/packed_rolling_scalar.hpp"
#include "fisk/sequence_pack/pack_sequence.hpp"
#include "fisk/sequence_pack/word_encoder.hpp"
#include "fisk/spaced_kmer_extract/ascii_rolling_scalar.hpp"
#include "fisk/spaced_kmer_extract/ascii_rolling_simd.hpp"
#include "fisk/spaced_kmer_extract/mask.hpp"
#include "extractor/common.hpp"
#include "extractor/kmer_common.hpp"
#include "extractor/spaced_common.hpp"
#include "oracle.hpp"
#include "testing.hpp"

using namespace fisk;

constexpr auto kAcgt = Encoding::kACGT;
constexpr auto kMsb = Layout::kMSB;
constexpr auto kAscii = InputFormat::kAscii;
constexpr auto kPacked = InputFormat::kPacked;
constexpr auto kChecked = InputCheck::kChecked;
constexpr auto kUnchecked = InputCheck::kUnchecked;

// Calls `f` with each SIMD arch that is compiled in.
template <typename F>
static void for_each_compiled_simd_arch(F const& f)
{
    f.template operator()<SimdScalar>();
    #if defined(FISK_HAS_SSE2)
    f.template operator()<SimdSSE2>();
    #endif
    #if defined(FISK_HAS_AVX2)
    f.template operator()<SimdAVX2>();
    #endif
    #if defined(FISK_HAS_AVX512)
    f.template operator()<SimdAVX512>();
    #endif
    #if defined(FISK_HAS_NEON)
    f.template operator()<SimdNEON>();
    #endif
}

// =================================================================================================
//     Extractors
// =================================================================================================

TEST(ForcedInline, KmerExtractor)
{
    check_kmer_extractor_algorithms<kAscii, kChecked, kAcgt, kMsb>();
    check_kmer_extractor_algorithms<kAscii, kUnchecked, kAcgt, kMsb>();
    check_kmer_extractor_algorithms<kPacked, kChecked, kAcgt, kMsb>();
    check_kmer_extractor_algorithms<kPacked, kUnchecked, kAcgt, kMsb>();
}

TEST(ForcedInline, SpacedKmerExtractor)
{
    check_spaced_kmer_extractor_algorithms<kAscii, kChecked, kAcgt, kMsb>();
    check_spaced_kmer_extractor_algorithms<kAscii, kUnchecked, kAcgt, kMsb>();
    check_spaced_kmer_extractor_algorithms<kPacked, kChecked, kAcgt, kMsb>();
    check_spaced_kmer_extractor_algorithms<kPacked, kUnchecked, kAcgt, kMsb>();
}

// =================================================================================================
//     Other Loops
// =================================================================================================

TEST(ForcedInline, KmerLoops)
{
    std::size_t const k = 15;
    CharEncoderTable<kAcgt> const enc{};
    WordEncoderButterfly<kAcgt, kMsb> const encoder{};

    for (auto const& seq : extractor_test_sequences<kChecked>()) {
        auto const expected = oracle_kmers_skipping_invalid<kAcgt, kMsb>(seq, k);
        auto const check = [&](auto const& extract) {
            KmerRecorder result;
            extract(result);
            check_emitted_kmers(expected, k, result.values, &result.positions);
        };

        check([&](auto& callback) {
            for_each_kmer_ascii_checked_reextract_scalar<kMsb>(seq, k, enc, callback);
        });
        check([&](auto& callback) {
            for_each_kmer_ascii_checked_rolling_scalar<kMsb>(seq, k, enc, callback);
        });
        for_each_compiled_simd_arch([&]<SimdArch Simd>() {
            check([&](auto& callback) {
                for_each_kmer_ascii_checked_rolling_simd<Simd, kAcgt, kMsb>(seq, k, callback);
            });
        });
        auto const packed = pack_sequence_checked(seq, encoder);
        check([&](auto& callback) {
            for_each_kmer_packed_checked_rolling_scalar(packed, k, callback);
        });
    }

    for (auto const& seq : extractor_test_sequences<kUnchecked>()) {
        auto const expected = oracle_kmers_all_windows<kAcgt, kMsb>(seq, k);
        auto const packed = pack_sequence_unchecked(seq, encoder);
        KmerRecorder result;
        for_each_kmer_packed_unchecked_rolling_scalar(packed, k, result);
        check_emitted_kmers(expected, k, result.values, &result.positions);
    }
}

TEST(ForcedInline, SpacedKmerLoops)
{
    std::vector<std::string> const masks = {"1101101"};
    std::array<SpacedKmerMask<kMsb, BitExtractMask>, 1> const mask_array{
        make_spaced_kmer_mask<kMsb>(masks[0])
    };
    CharEncoderTable<kAcgt> const enc{};

    for (auto const& seq : extractor_test_sequences<kChecked>()) {
        auto const expected = by_mask(oracle_spaced_events<kAcgt, kMsb>(seq, masks));

        SpacedKmerRecorder scalar;
        for_each_spaced_kmer_ascii_checked_rolling_scalar(
            seq, std::span{mask_array}, enc, bit_extract_bit_loop, scalar
        );
        check_spaced_events(by_mask(scalar.events), expected);

        for_each_compiled_simd_arch([&]<SimdArch Simd>() {
            std::array<SpacedKmerMask<kMsb, BitExtractKernelButterfly<Simd>>, 1> const kernels{
                make_spaced_kmer_kernel<BitExtractKernelButterfly<Simd>, kMsb>(masks[0])
            };
            SpacedKmerRecorder simd;
            for_each_spaced_kmer_ascii_checked_rolling_simd(
                seq, std::span{kernels}, enc,
                [&](std::size_t pos, std::size_t mask_idx, auto kmers, auto valid) {
                    for_each_valid_lane(
                        pos, kmers, valid, [&](std::size_t lane_pos, auto const& kmer) {
                            simd(lane_pos, mask_idx, kmer);
                        }
                    );
                }
            );
            check_spaced_events(by_mask(simd.events), expected);
        });
    }
}
