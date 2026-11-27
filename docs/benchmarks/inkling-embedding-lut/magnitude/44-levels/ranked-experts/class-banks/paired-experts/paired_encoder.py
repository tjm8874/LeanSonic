"""Keep the existing exact encoder and search only complete groups."""
from pathlib import Path
import sys
old=Path(sys.argv[1]).read_text()
a=old.index('    lower=0;upper=None;n=guess')
b=old.index('    groups=schedule(lower)',a)
old=old[:a]+'''    import bisect
    ends=r['legal_ends']
    group_lo=0
    group_hi=len(ends)-1
    index=max(1,bisect.bisect_right(ends,guess)-1)
    while group_lo<group_hi:
        cost=size_for(ends[index],pool)
        if cost<=cap-4096:
            group_lo=index
        else:
            group_hi=index-1
        index=(group_lo+group_hi+1)//2
    lower=ends[group_lo]
    assert lower>0
'''+old[b:]
Path('assemble_pairs.py').write_text(old)
