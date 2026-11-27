import ctypes as ct
import json,os,subprocess,sys,time
from pathlib import Path
stem=Path(sys.argv[1]);arm=sys.argv[2]
api=ct.CDLL('libnvidia-ml.so.1');assert api.nvmlInit_v2()==0
gpu=ct.c_void_p();assert api.nvmlDeviceGetHandleByUUID(os.environ['CUDA_VISIBLE_DEVICES'].encode(),ct.byref(gpu))==0
class Process(ct.Structure):
    _fields_=[('pid',ct.c_uint),('used',ct.c_ulonglong),('instance',ct.c_uint),('compute',ct.c_uint)]
buf=(Process*64)();peak=observed=0
env=dict(os.environ)
with stem.with_suffix('.tmp').open('w') as out,stem.with_suffix('.log').open('w') as err:
    with subprocess.Popen(['./bench','plain' if arm=='0' else 'spec',arm],env=env,stdout=out,stderr=err) as proc:
        while proc.poll() is None:
            n=ct.c_uint(len(buf));assert api.nvmlDeviceGetComputeRunningProcesses_v2(gpu,ct.byref(n),buf)==0
            for row in buf[:n.value]:
                if row.pid==proc.pid and 0<row.used<2**60:
                    peak=max(peak,row.used);observed+=1
            time.sleep(.004)
        assert proc.returncode==0,(str(stem),proc.returncode)
assert peak>0 and observed>100
stem.with_suffix('.peak').write_text(json.dumps(dict(bytes=peak,polls=observed))+'\n')
stem.with_suffix('.tmp').rename(stem.with_suffix('.csv'))
