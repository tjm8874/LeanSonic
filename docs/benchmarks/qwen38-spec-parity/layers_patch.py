from pathlib import Path
p=Path("crates/leansonic-core/src/qwen38_gpu.rs")
s=p.read_text()
mark="    fn enqueue_all_48_layers("
i=s.index(mark)
s=s[:i]+'''    fn dump_spec_layer(&self, ptr:u64, ids:u64, stage:u32, layer:usize) -> Result<(),String> {
        if self.host_pos < 19 {return Ok(());}
        let Some(path)=std::env::var_os("Q38_LAYER_TRACE") else {return Ok(())};
        self.ctx.sync_compute()?;
        let mut out=vec![0u8; Q38_HC_D*4+40];
        unsafe {
            check_cu((self.ctx.api.cuMemcpyDtoH_v2)(out.as_mut_ptr().cast(),ptr,Q38_HC_D*4),"trace residual")?;
            if stage==1 {check_cu((self.ctx.api.cuMemcpyDtoH_v2)(out.as_mut_ptr().add(Q38_HC_D*4).cast(),ids,40),"trace routes")?;}
        }
        use std::io::Write;
        let mut f=std::fs::OpenOptions::new().append(true).create(true).open(path).map_err(|e| e.to_string())?;
        for n in [self.host_pos as u32,layer as u32,stage] {f.write_all(&n.to_le_bytes()).map_err(|e|e.to_string())?;}
        f.write_all(&out).map_err(|e|e.to_string())
    }

'''+s[i:]
for anchor,ptr,ids in [('self.launch_hc_post(stream)?;', 'self.d_hc.ptr', 'self.d_sel_ids10.ptr'),('self.launch_hc_post_batch(stream, bb, b_count)?;', 'bb.d_b_hc.ptr','bb.d_b_sel_ids10.ptr')]:
 assert s.count(anchor)==2
 i=s.rindex(anchor)+len(anchor)
 s=s[:i]+f'\n            self.dump_spec_layer({ptr},{ids},1,l_idx)?;'+s[i:]
 i=s.index(anchor)+len(anchor)
 s=s[:i]+f'\n            self.dump_spec_layer({ptr},{ids},0,l_idx)?;'+s[i:]
p.write_text(s)
