// See forced_inline_common.hpp: undefined before any fisk header.
#undef FISK_ALWAYS_INLINE_FOR_EACH

#include "extractor/forced_inline_common.hpp"

// =================================================================================================
//     Spaced K-mers from ASCII
// =================================================================================================

TEST(ForcedInline, SpacedKmerExtractorAscii)
{
    check_spaced_kmer_extractor_algorithms<kAscii, kChecked, kAcgt, kMsb>();
    check_spaced_kmer_extractor_algorithms<kAscii, kUnchecked, kAcgt, kMsb>();
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
