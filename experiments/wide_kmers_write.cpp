// Check whether a sum consumer gives misleading timings for contiguous wide k-mers.
// Output storage is allocated before timing and verified after every measured call.

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "fisk/core/char_encoder.hpp"

namespace {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

#if defined(__GNUC__) || defined(__clang__)
#define FISK_NOINLINE __attribute__((noinline))
#else
#define FISK_NOINLINE
#endif

struct Wide2 {
  // Numeric limb order: words[0] is least significant.
  std::array<u64, 2> words{};

  bool operator==(Wide2 const &) const = default;
};

struct Wide2Msw {
  // Sequence-oriented alternative for MSB layout: words[0] is most significant.
  std::array<u64, 2> words{};
};

volatile u64 benchmark_sink = 0;

inline std::uint8_t encode(char c) noexcept {
  return fisk::CharEncoderTable<fisk::Encoding::kACGT>{}(c);
}

inline u64 high_mask(std::size_t k) noexcept {
  unsigned const high_bits = static_cast<unsigned>(2 * k - 64);
  return high_bits == 64 ? ~u64{0} : ((u64{1} << high_bits) - 1);
}

inline void shift_in(Wide2 &value, u64 code, u64 hi_mask) noexcept {
  value.words[1] = ((value.words[1] << 2) | (value.words[0] >> 62)) & hi_mask;
  value.words[0] = (value.words[0] << 2) | code;
}

std::string make_sequence(std::size_t size) {
  std::mt19937_64 rng(123456789);
  constexpr std::string_view bases = "ACGT";
  std::string seq(size, 'A');
  for (char &c : seq) {
    c = bases[rng() & 3u];
  }
  return seq;
}

FISK_NOINLINE u64 array2_sum(std::string_view seq, std::size_t k) {
  Wide2 value;
  u64 const mask = high_mask(k);
  u64 sum = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    shift_in(value, encode(seq[i]), mask);
    if (i + 1 >= k) {
      sum += value.words[0] + std::rotl(value.words[1], 17);
    }
  }
  return sum;
}

FISK_NOINLINE std::size_t array2_write(std::string_view seq, std::size_t k,
                                       Wide2 *out) {
  Wide2 value;
  u64 const mask = high_mask(k);
  std::size_t count = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    shift_in(value, encode(seq[i]), mask);
    if (i + 1 >= k) {
      out[count++] = value;
    }
  }
  return count;
}

FISK_NOINLINE u64 array2_low_first_sum(std::string_view seq, std::size_t k) {
  Wide2 value;
  u64 const mask = high_mask(k);
  u64 sum = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const carry = value.words[0] >> 62;
    value.words[0] = (value.words[0] << 2) | encode(seq[i]);
    value.words[1] = ((value.words[1] << 2) | carry) & mask;
    if (i + 1 >= k) {
      sum += value.words[0] + std::rotl(value.words[1], 17);
    }
  }
  return sum;
}

FISK_NOINLINE std::size_t array2_low_first_write(std::string_view seq,
                                                 std::size_t k, Wide2 *out) {
  Wide2 value;
  u64 const mask = high_mask(k);
  std::size_t count = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const carry = value.words[0] >> 62;
    value.words[0] = (value.words[0] << 2) | encode(seq[i]);
    value.words[1] = ((value.words[1] << 2) | carry) & mask;
    if (i + 1 >= k) {
      out[count++] = value;
    }
  }
  return count;
}

FISK_NOINLINE u64 array2_ssa_sum(std::string_view seq, std::size_t k) {
  Wide2 value;
  u64 const mask = high_mask(k);
  u64 sum = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const low = value.words[0];
    u64 const high = value.words[1];
    u64 const code = encode(seq[i]);
    value = Wide2{{(low << 2) | code, ((high << 2) | (low >> 62)) & mask}};
    if (i + 1 >= k) {
      sum += value.words[0] + std::rotl(value.words[1], 17);
    }
  }
  return sum;
}

FISK_NOINLINE std::size_t array2_ssa_write(std::string_view seq,
                                           std::size_t k, Wide2 *out) {
  Wide2 value;
  u64 const mask = high_mask(k);
  std::size_t count = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const low = value.words[0];
    u64 const high = value.words[1];
    u64 const code = encode(seq[i]);
    value = Wide2{{(low << 2) | code, ((high << 2) | (low >> 62)) & mask}};
    if (i + 1 >= k) {
      out[count++] = value;
    }
  }
  return count;
}

FISK_NOINLINE u64 msb_msw_sum(std::string_view seq, std::size_t k) {
  Wide2Msw value;
  u64 const mask = high_mask(k);
  u64 sum = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const carry = value.words[1] >> 62;
    value.words[1] = (value.words[1] << 2) | encode(seq[i]);
    value.words[0] = ((value.words[0] << 2) | carry) & mask;
    if (i + 1 >= k) {
      sum += value.words[1] + std::rotl(value.words[0], 17);
    }
  }
  return sum;
}

FISK_NOINLINE std::size_t msb_msw_write(std::string_view seq, std::size_t k,
                                        Wide2Msw *out) {
  Wide2Msw value;
  u64 const mask = high_mask(k);
  std::size_t count = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const carry = value.words[1] >> 62;
    value.words[1] = (value.words[1] << 2) | encode(seq[i]);
    value.words[0] = ((value.words[0] << 2) | carry) & mask;
    if (i + 1 >= k) {
      out[count++] = value;
    }
  }
  return count;
}

FISK_NOINLINE u64 lsb_lsw_sum(std::string_view seq, std::size_t k) {
  Wide2 value;
  unsigned const insert_shift = static_cast<unsigned>(2 * (k - 33));
  u64 sum = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const high = value.words[1];
    value.words[0] = (value.words[0] >> 2) | (high << 62);
    value.words[1] = (high >> 2) | (u64{encode(seq[i])} << insert_shift);
    if (i + 1 >= k) {
      sum += value.words[0] + std::rotl(value.words[1], 17);
    }
  }
  return sum;
}

FISK_NOINLINE std::size_t lsb_lsw_write(std::string_view seq, std::size_t k,
                                        Wide2 *out) {
  Wide2 value;
  unsigned const insert_shift = static_cast<unsigned>(2 * (k - 33));
  std::size_t count = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const high = value.words[1];
    value.words[0] = (value.words[0] >> 2) | (high << 62);
    value.words[1] = (high >> 2) | (u64{encode(seq[i])} << insert_shift);
    if (i + 1 >= k) {
      out[count++] = value;
    }
  }
  return count;
}

FISK_NOINLINE u64 lsb_msw_sum(std::string_view seq, std::size_t k) {
  Wide2Msw value;
  unsigned const insert_shift = static_cast<unsigned>(2 * (k - 33));
  u64 sum = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const high = value.words[0];
    value.words[1] = (value.words[1] >> 2) | (high << 62);
    value.words[0] = (high >> 2) | (u64{encode(seq[i])} << insert_shift);
    if (i + 1 >= k) {
      sum += value.words[1] + std::rotl(value.words[0], 17);
    }
  }
  return sum;
}

FISK_NOINLINE std::size_t lsb_msw_write(std::string_view seq, std::size_t k,
                                        Wide2Msw *out) {
  Wide2Msw value;
  unsigned const insert_shift = static_cast<unsigned>(2 * (k - 33));
  std::size_t count = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const high = value.words[0];
    value.words[1] = (value.words[1] >> 2) | (high << 62);
    value.words[0] = (high >> 2) | (u64{encode(seq[i])} << insert_shift);
    if (i + 1 >= k) {
      out[count++] = value;
    }
  }
  return count;
}

FISK_NOINLINE u64 uint128_sum(std::string_view seq, std::size_t k) {
  u128 value = 0;
  u128 const mask = k == 64 ? ~u128{0} : ((u128{1} << (2 * k)) - 1);
  u64 sum = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    value = ((value << 2) | encode(seq[i])) & mask;
    if (i + 1 >= k) {
      sum += static_cast<u64>(value) +
             std::rotl(static_cast<u64>(value >> 64), 17);
    }
  }
  return sum;
}

FISK_NOINLINE std::size_t uint128_write(std::string_view seq, std::size_t k,
                                        Wide2 *out) {
  u128 value = 0;
  u128 const mask = k == 64 ? ~u128{0} : ((u128{1} << (2 * k)) - 1);
  std::size_t count = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    value = ((value << 2) | encode(seq[i])) & mask;
    if (i + 1 >= k) {
      out[count++] = Wide2{{static_cast<u64>(value), static_cast<u64>(value >> 64)}};
    }
  }
  return count;
}

FISK_NOINLINE u64 vector2_sum(std::string_view seq, std::size_t k) {
  std::vector<u64> words(2);
  u64 const mask = high_mask(k);
  u64 sum = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const next = words[0] >> 62;
    words[0] = (words[0] << 2) | encode(seq[i]);
    words[1] = ((words[1] << 2) | next) & mask;
    if (i + 1 >= k) {
      sum += words[0] + std::rotl(words[1], 17);
    }
  }
  return sum;
}

FISK_NOINLINE std::size_t vector2_write(std::string_view seq, std::size_t k,
                                        Wide2 *out) {
  std::vector<u64> words(2);
  u64 const mask = high_mask(k);
  std::size_t count = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 const next = words[0] >> 62;
    words[0] = (words[0] << 2) | encode(seq[i]);
    words[1] = ((words[1] << 2) | next) & mask;
    if (i + 1 >= k) {
      out[count++] = Wide2{{words[0], words[1]}};
    }
  }
  return count;
}

FISK_NOINLINE u64 scalar_sum(std::string_view seq, std::size_t k) {
  u64 value = 0;
  u64 const mask = k == 32 ? ~u64{0} : ((u64{1} << (2 * k)) - 1);
  u64 sum = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    value = ((value << 2) | encode(seq[i])) & mask;
    if (i + 1 >= k) {
      sum += value;
    }
  }
  return sum;
}

FISK_NOINLINE std::size_t scalar_write(std::string_view seq, std::size_t k,
                                       u64 *out) {
  u64 value = 0;
  u64 const mask = k == 32 ? ~u64{0} : ((u64{1} << (2 * k)) - 1);
  std::size_t count = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    value = ((value << 2) | encode(seq[i])) & mask;
    if (i + 1 >= k) {
      out[count++] = value;
    }
  }
  return count;
}

template <std::size_t Words>
FISK_NOINLINE u64 array_sum(std::string_view seq, std::size_t k) {
  std::array<u64, Words> words{};
  unsigned const high_bits = static_cast<unsigned>(2 * k - 64 * (Words - 1));
  u64 const mask = high_bits == 64 ? ~u64{0} : ((u64{1} << high_bits) - 1);
  u64 sum = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 carry = encode(seq[i]);
    for (u64 &word : words) {
      u64 const next = word >> 62;
      word = (word << 2) | carry;
      carry = next;
    }
    words.back() &= mask;
    if (i + 1 >= k) {
      sum += words[0];
    }
  }
  return sum;
}

template <std::size_t Words>
FISK_NOINLINE std::size_t array_write(std::string_view seq, std::size_t k,
                                      u64 *out) {
  std::array<u64, Words> words{};
  unsigned const high_bits = static_cast<unsigned>(2 * k - 64 * (Words - 1));
  u64 const mask = high_bits == 64 ? ~u64{0} : ((u64{1} << high_bits) - 1);
  std::size_t count = 0;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    u64 carry = encode(seq[i]);
    for (u64 &word : words) {
      u64 const next = word >> 62;
      word = (word << 2) | carry;
      carry = next;
    }
    words.back() &= mask;
    if (i + 1 >= k) {
      out[count++] = words[0];
    }
  }
  return count;
}

FISK_NOINLINE u64 checksum(Wide2 const *values, std::size_t count) {
  u64 sum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    sum += values[i].words[0] + std::rotl(values[i].words[1], 17);
  }
  return sum;
}

FISK_NOINLINE u64 checksum(Wide2Msw const *values, std::size_t count) {
  u64 sum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    sum += values[i].words[1] + std::rotl(values[i].words[0], 17);
  }
  return sum;
}

FISK_NOINLINE u64 checksum(u64 const *values, std::size_t count) {
  u64 sum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    sum += values[i];
  }
  return sum;
}

template <typename Function>
double benchmark_sum(Function &&function, u64 expected, std::size_t count) {
  double best = std::numeric_limits<double>::infinity();
  for (int repeat = 0; repeat < 9; ++repeat) {
    auto const start = std::chrono::steady_clock::now();
    u64 const got = function();
    auto const stop = std::chrono::steady_clock::now();
    if (got != expected) std::abort();
    benchmark_sink ^= got;
    best = std::min(best, std::chrono::duration<double, std::nano>(stop - start).count());
  }
  return best / static_cast<double>(count);
}

template <typename Function, typename Value>
double benchmark_write(Function &&function, std::vector<Value> &out,
                       u64 expected, std::size_t count) {
  double best = std::numeric_limits<double>::infinity();
  for (int repeat = 0; repeat < 9; ++repeat) {
    auto const start = std::chrono::steady_clock::now();
    std::size_t const got_count = function(out.data());
    auto const stop = std::chrono::steady_clock::now();
    if (got_count != count || checksum(out.data(), got_count) != expected) std::abort();
    benchmark_sink ^= expected;
    best = std::min(best, std::chrono::duration<double, std::nano>(stop - start).count());
  }
  return best / static_cast<double>(count);
}

void print(std::size_t k, std::string_view representation, std::string_view sink,
           double ns) {
  std::cout << k << ',' << representation << ',' << sink << ',' << std::fixed
            << std::setprecision(4) << ns << '\n';
}

}  // namespace

int main() {
  std::string const seq = make_sequence(1u << 20);
  std::cout << "k,representation,sink,ns_per_kmer\n";

  for (std::size_t k : {33u, 48u, 64u}) {
    std::size_t const count = seq.size() - k + 1;
    std::vector<Wide2> out(count);
    std::vector<Wide2Msw> out_msw(count);
    u64 const expected = array2_sum(seq, k);
    if (array2_low_first_sum(seq, k) != expected || array2_ssa_sum(seq, k) != expected ||
        uint128_sum(seq, k) != expected || vector2_sum(seq, k) != expected) {
      std::abort();
    }

    print(k, "array2", "sum",
          benchmark_sum([&] { return array2_sum(seq, k); }, expected, count));
    print(k, "array2", "write",
          benchmark_write([&](Wide2 *p) { return array2_write(seq, k, p); },
                          out, expected, count));
    print(k, "array2_low_first", "sum",
          benchmark_sum([&] { return array2_low_first_sum(seq, k); }, expected,
                        count));
    print(k, "array2_low_first", "write",
          benchmark_write(
              [&](Wide2 *p) { return array2_low_first_write(seq, k, p); }, out,
              expected, count));
    print(k, "array2_ssa", "sum",
          benchmark_sum([&] { return array2_ssa_sum(seq, k); }, expected,
                        count));
    print(k, "array2_ssa", "write",
          benchmark_write([&](Wide2 *p) { return array2_ssa_write(seq, k, p); },
                          out, expected, count));
    print(k, "msb_msw_words", "sum",
          benchmark_sum([&] { return msb_msw_sum(seq, k); }, expected, count));
    print(k, "msb_msw_words", "write",
          benchmark_write([&](Wide2Msw *p) { return msb_msw_write(seq, k, p); },
                          out_msw, expected, count));
    print(k, "uint128", "sum",
          benchmark_sum([&] { return uint128_sum(seq, k); }, expected, count));
    print(k, "uint128", "write",
          benchmark_write([&](Wide2 *p) { return uint128_write(seq, k, p); },
                          out, expected, count));
    print(k, "vector2", "sum",
          benchmark_sum([&] { return vector2_sum(seq, k); }, expected, count));
    print(k, "vector2", "write",
          benchmark_write([&](Wide2 *p) { return vector2_write(seq, k, p); },
                          out, expected, count));

    u64 const lsb_expected = lsb_lsw_sum(seq, k);
    if (lsb_msw_sum(seq, k) != lsb_expected) std::abort();
    print(k, "lsb_lsw_words", "sum",
          benchmark_sum([&] { return lsb_lsw_sum(seq, k); }, lsb_expected,
                        count));
    print(k, "lsb_lsw_words", "write",
          benchmark_write([&](Wide2 *p) { return lsb_lsw_write(seq, k, p); },
                          out, lsb_expected, count));
    print(k, "lsb_msw_words", "sum",
          benchmark_sum([&] { return lsb_msw_sum(seq, k); }, lsb_expected,
                        count));
    print(k, "lsb_msw_words", "write",
          benchmark_write([&](Wide2Msw *p) { return lsb_msw_write(seq, k, p); },
                          out_msw, lsb_expected, count));
  }

  {
    std::size_t constexpr k = 31;
    std::size_t const count = seq.size() - k + 1;
    std::vector<u64> out(count);
    u64 const expected = scalar_sum(seq, k);
    if (array_sum<1>(seq, k) != expected) std::abort();
    print(k, "scalar", "sum",
          benchmark_sum([&] { return scalar_sum(seq, k); }, expected, count));
    print(k, "scalar", "write",
          benchmark_write([&](u64 *p) { return scalar_write(seq, k, p); }, out,
                          expected, count));
    print(k, "array1", "sum",
          benchmark_sum([&] { return array_sum<1>(seq, k); }, expected, count));
    print(k, "array1", "write",
          benchmark_write([&](u64 *p) { return array_write<1>(seq, k, p); }, out,
                          expected, count));
  }

  return benchmark_sink == 0x9e3779b97f4a7c15ull ? 1 : 0;
}
