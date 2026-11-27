from pathlib import Path
import json,sqlite3,sys
results={}
for filename in sorted(Path('.').glob('*.sqlite')):
    if not filename.with_suffix('.nsys-rep').exists():continue
    with sqlite3.connect(f'file:{filename}?mode=ro',uri=True) as db:
        tables={x[0] for x in db.execute("select name from sqlite_master where type='table'")}
        if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables:continue
        data=db.execute('select s.value,count(*),sum(k.end-k.start) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName group by s.value order by 3 desc').fetchall()
        if not data:continue
        assert all(n>0 and duration>0 for _,n,duration in data)
        bounds=db.execute('select min(start),max(end) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
        memcpy=[]
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            memcpy=db.execute('select copyKind,count(*),sum(bytes),sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind').fetchall()
        results[filename.stem]=dict(kernel_span_ms=(bounds[1]-bounds[0])/1e6,kernel_sum_ms=sum(d for _,_,d in data)/1e6,kernel_count=sum(n for _,n,_ in data),kernels=[dict(name=k,calls=n,total_ms=t/1e6) for k,n,t in data],copies=[dict(kind=int(kind),calls=n,bytes=b,total_ms=t/1e6) for kind,n,b,t in memcpy])
Path('traces.json').write_text(json.dumps(results,indent=2)+'\n')
for k,v in results.items():print(k,v['kernel_span_ms'],v['kernel_count'],v['kernels'][:3],v['copies'])
