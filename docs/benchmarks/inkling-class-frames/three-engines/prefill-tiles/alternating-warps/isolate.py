from pathlib import Path
p=Path('kernels/inkling2_sm120.cu');s=p.read_text()
end=s.index('extern "C" __global__ __launch_bounds__(512, 1) void ink2_moe_nvfp4_tc_up_situ(')
start=s.rindex('template <bool GateUp',0,s.index('void grouped_nvfp4_tc_impl('))
core=s[start:end]
core=core.replace('bool MixedEven = false>','bool MixedEven = false, bool AlternateTokenGroups = false>')
anchor='    const int g = lane >> 2;'
assert core.count(anchor)==1
core=core.replace('wt * 64 + nt * 8','first_token + nt * token_step').replace('wt * 64 < live','first_token < live')
core=core.replace(anchor,'''    // Interleave eight-token MMA groups between the two token warps. A
    // partially filled expert can then use both warps within the same CTA.
    const int first_token = wt * (AlternateTokenGroups ? 8 : 64);
    constexpr int token_step = AlternateTokenGroups ? 16 : 8;
'''+anchor)
assert core.count('first_token + nt * token_step')==4 and core.count('first_token < live')==2
core=core.replace('void grouped_nvfp4_tc_impl(', 'void grouped_nvfp4_tc_alternating_impl(')
s=s[:end]+core+s[end:]
old='    grouped_nvfp4_tc_impl<true, true>('
assert s.count(old)==1
p.write_text(s.replace(old,'    grouped_nvfp4_tc_alternating_impl<true, true, false, false, true>('))
