// See forced_inline_common.hpp: undefined before any fisk header.
#undef FISK_ALWAYS_INLINE_FOR_EACH

#include "extractor/forced_inline_common.hpp"

// =================================================================================================
//     K-mers
// =================================================================================================

TEST(ForcedInline, KmerExtractor)
{
    check_kmer_extractor_algorithms<kAscii, kChecked, kAcgt, kMsb>();
    check_kmer_extractor_algorithms<kAscii, kUnchecked, kAcgt, kMsb>();
    check_kmer_extractor_algorithms<kPacked, kChecked, kAcgt, kMsb>();
    check_kmer_extractor_algorithms<kPacked, kUnchecked, kAcgt, kMsb>();
}

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
