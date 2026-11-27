"""Generate a standalone specialization of the pinned fused kernel for a controlled measurement."""
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
# Include the unmodified TU first so both modes use its identical inline math helpers.
start=s.index('INK2_SCALE_KERNEL_LB(ink2_moe_nvfp4_dp4a_up_half_q8')
end=s.index('// Copy contiguous spans',start)
k=s[start:end]
args=k.index('    (w13_u128,')
body=k[k.index(' {',args):]
body=body.replace('const uint4* weights','const uint2* weights').replace('const uint4 pk','const uint2 pk').replace('uint4 pk4 = make_uint4(0u, 0u, 0u, 0u);','uint2 pk4 = make_uint2(0u, 0u);').replace('accum_nvfp4_dp4a_u128_step','accum_ternary_step').replace('expert_scale_word<PackedScales,','expert_scale_word<true,')
assert 'uint4 pk' not in body and body.count('accum_ternary_step')==5
start=s.index('__device__ __forceinline__ float accum_nvfp4_dp4a_u128_step')
step=s[start:s.index('\n}',start)+2].replace('accum_nvfp4_dp4a_u128_step','accum_ternary_step').replace('uint4 pk16','uint2 pk16')
start=step.index('    int d0 =');end=step.index('    acc =',start) if '    acc =' in step[start:] else step.index('    return',start)
step=step[:start]+'''    int d0 = ternary_block_dp4a(pk16.x, xq0);
    int d1 = ternary_block_dp4a(pk16.y, xq1);
'''+step[end:]
header='''extern "C" __global__ __launch_bounds__(512,2) void ink2_moe_ternary_dp4a_up_half_q8_packed_scales(
    const uint2* __restrict__ w13_u128, const uint32_t* __restrict__ s13_u32,
    const float* __restrict__ a13, const uint4* __restrict__ x_q8_u128,
    const float* __restrict__ x_sc16, const int32_t* __restrict__ sel_ids,
    uint32_t* __restrict__ out_q8_u32, float* __restrict__ out_sc16)'''
# INK2_SCALE_KERNEL_LB waits on PDL in the wrapper and the existing body.
body=body.replace(' {\n',' {\n    pdl_wait();\n',1)
Path('trial.cu').write_text('#include "inkling2_sm120.cu"\n'+step+'\n'+header+body)
