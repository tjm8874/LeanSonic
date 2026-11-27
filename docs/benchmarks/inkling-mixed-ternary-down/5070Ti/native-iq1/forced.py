from pathlib import Path
import ctypes as ct,json,os,re,subprocess,sys,time
cycle=int(sys.argv[1]);out=Path('llama-runs');out.mkdir(exist_ok=True)
root=Path.cwd()
inp=[root/'inkling-readme1024.ids',root/'inkling-continuation32.ids']
image=Path('model-path').read_text().strip()
nv=ct.CDLL('libnvidia-ml.so.1');assert nv.nvmlInit_v2()==0
dev=ct.c_void_p();assert nv.nvmlDeviceGetHandleByUUID(os.environ['CUDA_VISIBLE_DEVICES'].encode(),ct.byref(dev))==0
class Proc(ct.Structure):
 _fields_=[('pid',ct.c_uint),('memory',ct.c_ulonglong),('instance',ct.c_uint),('compute',ct.c_uint)]
processes=(Proc*64)()
for i,arm in enumerate(('lean','llama','llama','lean') if cycle%2 else ('llama','lean','lean','llama')):
 stem=out/f'{cycle}-{i}-{arm}';cmd=(['./forced-lean',image] if arm=='lean' else ['./forced-native',Path('iq').read_text().strip()])+inp
 with stem.with_suffix('.out').open('w') as stdout,stem.with_suffix('.log').open('w') as stderr:
  env=os.environ.copy()

  proc=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=env);peak=obs=0
  while proc.poll() is None:
   count=ct.c_uint(64);assert nv.nvmlDeviceGetComputeRunningProcesses_v2(dev,ct.byref(count),processes)==0
   for row in processes[:count.value]:
    if row.pid==proc.pid:peak=max(peak,row.memory);obs+=1
   time.sleep(.004)
  assert proc.returncode==0 and 0<peak<2**60 and obs>100,(proc.returncode,peak)
 log=stem.with_suffix(".log").read_text()
 if arm=="llama": assert "offloaded 25/25 layers to GPU" in log
 else: assert "Selected 19/20" in log and "24/24 (Spilled: 0)" in log
 legs=[]
 for line in stem.with_suffix('.out').read_text().splitlines():
  m=re.fullmatch(r'(\d+) ([\d.]+) ([\d.]+) (\d+) (\d+)',line.strip())
  if m:
   r,p,d,n,first=m.groups()
   assert 0<=int(n)<=32
   legs.append(dict(run=int(r),prefill_ms=float(p),decode32_ms=float(d),matches=int(n),first=int(first)))
 assert len(legs)==5,(stem,legs)
 if legs:
  expected=list(map(int,inp[1].read_text().strip().split(',')))
  seqs=[[int(t) for t in l.split()[2:]] for l in stem.with_suffix('.out').read_text().splitlines() if l.startswith('P ')]
  assert len(seqs)==5
  for record,seq in zip(legs,seqs):
   assert len(seq)==len(expected)==32 and sum(a==b for a,b in zip(seq,expected))==record['matches']
   record['predicted_ids']=seq
 stem.with_suffix('.json').write_text(json.dumps(dict(cycle=cycle,arm=arm,leg=i,peak=peak,requests=legs))+'\n')
