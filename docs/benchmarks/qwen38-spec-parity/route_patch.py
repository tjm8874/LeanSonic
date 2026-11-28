from pathlib import Path
p=Path("crates/leansonic-core/src/qwen38_gpu.rs");s=p.read_text()
i=s.index("    fn enqueue_all_48_layers(")
s=s[:i]+'''    fn check_route(&self, ids:u64, logits:u64, mode:&str) -> Result<(),String> {
        let mut top=[0i32;10]; let mut scores=[0f32;512];
        self.ctx.sync_compute()?;
        unsafe {
            check_cu((self.ctx.api.cuMemcpyDtoH_v2)(top.as_mut_ptr().cast(),ids,40),"route read")?;
            check_cu((self.ctx.api.cuMemcpyDtoH_v2)(scores.as_mut_ptr().cast(),logits,2048),"score read")?;
        }
        let rank=top[9]; eprintln!("route {mode} tenth={rank} score217={} score334={}",scores[217],scores[334]);
        if std::env::var("Q38_ROUTE_SWAP").as_deref()==Ok(mode) {
            let replace=match (mode,rank) {("serial",217)=>334i32,("verify",334)=>217i32,_=>return Err("unexpected route".into())};
            unsafe {check_cu((self.ctx.api.cuMemcpyHtoD_v2)(ids+36,(&replace as *const i32).cast(),4),"replace single route")?;}
        }
        Ok(())
    }

'''+s[i:]
i=s.index('            let (mut sh_gu_w',s.index('    fn enqueue_all_48_layers('))
s=s[:i]+'''            if self.host_pos==95 && l_idx==3 {self.check_route(self.d_sel_ids10.ptr,self.d_router_logits.ptr,"serial")?;}
'''+s[i:]
i=s.index('            let (base_ptr, z24_bases)',s.index('    fn enqueue_batch_48_layers('))
s=s[:i]+'''            if self.host_pos==95 && l_idx==3 && b_count==1 {self.check_route(sids,rlog,"verify")?;}
'''+s[i:]
p.write_text(s)
