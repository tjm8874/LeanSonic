"""Create a text-only comparison pack for the pinned MXFP8 Inkling Basalt layout."""
from pathlib import Path
import sys,json
from model_file import ModelFile,ModelWriter
source,dest=map(Path,sys.argv[1:])
if dest.exists():raise ValueError('use a new output path')
m=ModelFile(source);d=json.loads(m.read('meta'))
assert d['arch']=='inkling2' and set(m.names())=={'tokenizer/chat_template.jinja','tokenizer/merges.txt','tokenizer/token_type.json','tokenizer/tokenizer.json','tokenizer/vocab.json','experts','experts.layout','gguf','vision/mmproj','meta'}
tensors=d['tensors']
core={k:v for k,v in tensors.items() if not k.startswith('dspark.')}
side={k:v for k,v in tensors.items() if k.startswith('dspark.')}
end=max(v['offset']+v['bytes'] for v in core.values())
assert len(side)==54 and len(core)==656 and end<=min(v['offset'] for v in side.values())
d['tensors']=core
d['dspark']=None;d['vision']=None
d['gguf_kv']={k:v for k,v in d['gguf_kv'].items() if '.dspark.' not in k and not k.startswith('dspark.')}
w=ModelWriter(str(dest))
for name in m.names():
 offset,n=m.section(name)
 if name=='vision/mmproj':continue
 if name=='meta':w.add_bytes(name,json.dumps(d,separators=(',',':'),ensure_ascii=False).encode())
 else:w.add_file(name,str(source),offset,end if name=='gguf' else n)
w.close()
result=ModelFile(dest)
assert set(result.names())==set(m.names())-{'vision/mmproj'}
for k in result.names():
 if k not in ('gguf','meta'):
  assert result.sections[k][2]==m.sections[k][2]
assert result.section('gguf')[1]==end
print(json.dumps(dict(original_bytes=source.stat().st_size,text_bytes=dest.stat().st_size,core_tensors=len(core),drafter_tensors=len(side),dense_before=m.section('gguf')[1],dense_after=end,vision_bytes=m.section('vision/mmproj')[1],sections={k:result.section(k)[1] for k in result.names()}),indent=2))
