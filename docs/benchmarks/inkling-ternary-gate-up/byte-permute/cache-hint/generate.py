from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text()
old='''        (void)is_last_tile;
        (void)evict_pol;
        const uint2 codes = reinterpret_cast<const uint2*>(weights)[group];'''
new='''        const uint2* src = reinterpret_cast<const uint2*>(weights) + group;
        uint2 codes;
        if (is_last_tile) {
            asm volatile("ld.global.cg.L2::cache_hint.v2.u32 {%0, %1}, [%2], %3;"
                         : "=r"(codes.x), "=r"(codes.y) : "l"(src), "l"(evict_pol) : "memory");
        } else {
            asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];"
                         : "=r"(codes.x), "=r"(codes.y) : "l"(src) : "memory");
        }'''
assert s.count(old)==1 and s.count('!Ternary && (first +')==2 and s.count('Ternary ? 0ull : make_l2_evict_first_policy()')==2
p.write_text(s.replace(old,new).replace('!Ternary && (first +','(first +').replace('Ternary ? 0ull : make_l2_evict_first_policy()','make_l2_evict_first_policy()'))
