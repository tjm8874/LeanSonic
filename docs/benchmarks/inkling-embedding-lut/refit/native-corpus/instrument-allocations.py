from pathlib import Path
def replace(text,before,after):
 assert text.count(before)==1,before
 return text.replace(before,after)
p=Path('crates/leansonic-core/src/cuda_ffi.rs')
original=p.read_text()
i=original.index('impl DeviceBuf {');j=original.index('impl Drop for DeviceBuf {',i)
s=original[i:j]
for method in ('alloc_zeroed','from_host_bytes'):
 s=replace(s,'    pub fn '+method+'(','    #[track_caller]\n    pub fn '+method+'(')
s=replace(s,'        let buf = Self {','        eprintln!("MEMTRACE alloc {} {} {}", ptr, alloc_len, std::panic::Location::caller());\n        let buf = Self {')
rest=original[j:]
rest=replace(rest,'                let _ = (self.ctx.api.cuMemFree_v2)(self.ptr);','                eprintln!("MEMTRACE free {}", self.ptr);\n                let _ = (self.ctx.api.cuMemFree_v2)(self.ptr);')
p.write_text(original[:i]+s+rest)
p=Path('crates/leansonic-core/examples/inkling_fixed_decode.rs')
s=p.read_text()
s=replace(s,'    for run in 0..5 {','    eprintln!("MEMTRACE phase loaded");\n    for run in 0..5 {')
s=replace(s,'        let prefill_end = Instant::now();','        let prefill_end = Instant::now();\n        eprintln!("MEMTRACE phase prefill_{run}");')
s=replace(s,'        let decode_ms =','        eprintln!("MEMTRACE phase decode_{run}");\n        let decode_ms =')
p.write_text(s)
