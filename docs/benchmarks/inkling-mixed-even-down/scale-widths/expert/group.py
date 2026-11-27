"""Prototype an expert-uniform scale descriptor in the fused gate/up kernel."""
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
beg=s.index('    const uint32_t sc0 = expert_scale_word<PackedScales, 6, 64>(s13_u32, sc_off + (u0 >> 1));')
end=s.index('    __syncthreads();',beg)
orig=s[beg:end]
assert 'const uint32_t sc3' in orig and 'sc4 = expert_scale_word' in orig
s=s[:beg]+'''    uint32_t sc0, sc1, sc2, sc3, sc4 = 0;
    if constexpr (PackedScales && MixedEven) {
        const unsigned tag = s13_u32[ex];
        const unsigned* payload = s13_u32 + 4u * (tag & 0x7fffffu);
        const unsigned base4 = (64u + ((tag >> 23) & 63u)) * 0x01010101u;
        const unsigned local_scale = (2u * channel + (lane >> 4)) * 36u;
        auto load_scales = [&]<int Width>() {
            auto read = [&](int u) {
                return base4 + expert_scale_word<true, Width, 0>(payload, local_scale + (u >> 1));
            };
            sc0 = read(u0); sc1 = read(u1); sc2 = read(u2); sc3 = read(u3);
            if (has4) sc4 = read(u4);
        };
        if ((tag >> 29) == 5) load_scales.template operator()<5>();
        else load_scales.template operator()<6>();
    } else {
'''+orig.replace('const uint32_t sc','sc').replace('uint32_t sc4 = 0u;', 'sc4 = 0u;')+'    }\n'+s[end:]
# C++17 uses a type tag for compile-time specialization.
s=s.replace('auto load_scales = [&]<int Width>() {','auto load_scales = [&](auto width) {\n            constexpr int Width = decltype(width)::value;').replace('load_scales.template operator()<5>()','load_scales(std::integral_constant<int, 5>{})').replace('load_scales.template operator()<6>()','load_scales(std::integral_constant<int, 6>{})')
Path('grouped.cu').write_text('#include <type_traits>\n'+s)
