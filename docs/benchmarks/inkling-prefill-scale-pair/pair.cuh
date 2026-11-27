template<int Bits, unsigned Base>
__device__ __forceinline__ unsigned expand_expert_scale_fields(unsigned fields) {
    constexpr unsigned mask = (1u << Bits) - 1;
    return Base * 0x01010101u | (fields & mask)
        | (((fields >> Bits) & mask) << 8)
        | (((fields >> (2 * Bits)) & mask) << 16)
        | (((fields >> (3 * Bits)) & mask) << 24);
}

__device__ __forceinline__ uint2 nvfp4_scale_pair(const unsigned* scales, unsigned index) {
    const unsigned first = (index * 3u) >> 2;
    const unsigned long long bits = scales[first] |
        (static_cast<unsigned long long>(scales[first + 1]) << 32);
    const unsigned long long pair = bits >> ((index & 2u) * 8u);
    return make_uint2(expand_expert_scale_fields<6, 64>(static_cast<unsigned>(pair)),
                      expand_expert_scale_fields<6, 64>(static_cast<unsigned>(pair >> 24)));
}
