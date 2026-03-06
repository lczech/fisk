#include "seq_enc.hpp"

constexpr std::array<std::uint8_t,256> NucleotideEncoder::table = NucleotideEncoder::make_table();
