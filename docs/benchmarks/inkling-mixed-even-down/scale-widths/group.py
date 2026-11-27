from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
a='    if constexpr (!PackedScales) return scales[index];\n'
assert s.count(a)==1
code='''    if constexpr (PackedScales && Bits == 6 && Base == 64) {
        // index addresses a four-byte word of consecutive scale codes.
        const unsigned tag = scales[index >> 5];
        const unsigned width = tag >> 29;
        const unsigned base4 = (64u + ((tag >> 23) & 63u)) * 0x01010101u;
        if (width == 0) return base4;
        const unsigned bit = (index & 31u) * (4u * width);
        const unsigned pos = (tag & 0x7fffffu) * 4u + (bit >> 5);
        const unsigned bits = __funnelshift_r(scales[pos], scales[pos + 1], bit & 31u);
        const unsigned mask = (1u << width) - 1;
        return base4 + ((bits & mask) | (((bits >> width) & mask) << 8)
            | (((bits >> (width * 2)) & mask) << 16) | (((bits >> (width * 3)) & mask) << 24));
    }
'''
Path('grouped.cu').write_text(s.replace(a,a+code))
