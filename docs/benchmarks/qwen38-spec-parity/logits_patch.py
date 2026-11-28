from pathlib import Path
p=Path("crates/leansonic-core/src/qwen38_gpu.rs")
s=p.read_text()
marker="    pub fn read_last_logits(&self)"
a=s.index(marker)
s=s[:a]+'''    fn log_probe(&self, ptr: u64, pos: usize, input: u32, label: &str) -> Result<(), String> {
        let mut logits = vec![0f32; Q38_VOCAB];
        self.ctx.sync_compute()?;
        unsafe { check_cu((self.ctx.api.cuMemcpyDtoH_v2)(logits.as_mut_ptr().cast(), ptr, logits.len()*4), "read diagnostic logits")?; }
        let top = logits.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0;
        eprintln!("logits {label} pos={pos} input={input} top={top} newline={} start={}", logits[271], logits[248045]);
        Ok(())
    }

'''+s[a:]
mark='        self.host_pos += b_count;'
assert s.count(mark)==1
s=s.replace(mark,'''        if mode == BatchLogitMode::AllVerifyLogits && (self.host_pos..self.host_pos+b_count).contains(&98) {
            let row = 98-self.host_pos;
            self.log_probe(bb.d_b_logits.ptr+(row*Q38_VOCAB*4) as u64,98,tokens[row],"verify")?;
        }
'''+mark)
mark='        self.host_pos += 1;'
assert s.count(mark)==1
s=s.replace(mark,'''        if compute_logits && self.host_pos == 98 {
            self.log_probe(self.d_logits.ptr,98,tok,"serial")?;
        }
'''+mark)
p.write_text(s)
