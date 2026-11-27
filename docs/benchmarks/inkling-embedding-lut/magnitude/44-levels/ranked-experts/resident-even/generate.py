from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
first=s.index('template<bool PackedScales, class Weight>\n__device__ __forceinline__ void fused_gate_up_q8')
last=s.index('INK2_SCALE_KERNEL_LB(ink2_moe_ternary_dp4a_up_half_q8',first)
template=s[first:last]
body=template[:template.index('INK2_SCALE_KERNEL_LB(')]
entry=template[len(body):]
body=body.replace('class Weight>','class Weight, bool MixedEven = false>')
body=body.replace('    const Weight* weights = w13_u128 + row * 72u;', '''    const Weight* weights;
    const unsigned* compact = nullptr;
    unsigned local = (2u * channel + (lane >> 4)) * 72u;
    if constexpr (MixedEven) {
        const unsigned* frame = reinterpret_cast<const unsigned*>(w13_u128);
        const unsigned descriptor = frame[ex];
        const unsigned* data = frame + (descriptor & 0x7fffffffu);
        weights = reinterpret_cast<const Weight*>(data);
        if (descriptor >> 31) compact = data;
    } else {
        weights = w13_u128 + row * 72u;
    }
    auto load_weight = [&](unsigned u) -> Weight {
        if constexpr (MixedEven) {
            const unsigned i = local + u;
            if (!compact) return weights[i];
            const unsigned* p = compact + (i >> 5) * 96u + (i & 31u);
            const unsigned a = p[0], b = p[32], c = p[64];
            constexpr unsigned bit = 0x11111111u;
            return make_uint4(a & ~bit, b & ~bit, c & ~bit,
                              ((a & bit) << 1) | ((b & bit) << 2) | ((c & bit) << 3));
        } else {
            return weights[u];
        }
    };''')
assert body.count('weights[u')==6
for i in range(5):body=body.replace(f'weights[u{i}]',f'load_weight(u{i})')
extra=entry.replace('ink2_moe_nvfp4_dp4a_up_half_q8','ink2_moe_mixed_even_dp4a_up_half_q8').replace('fused_gate_up_q8<PackedScales>','fused_gate_up_q8<PackedScales, uint4, true>')
Path('candidate.cu').write_text(s[:first]+body+entry+extra+s[last:])
