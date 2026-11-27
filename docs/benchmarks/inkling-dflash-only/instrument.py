from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
marker='    pub fn read_last_logits(&self) -> Result<Vec<f32>, String> {'
assert s.count(marker)==1
p.write_text(s.replace(marker,'''    pub fn read_scalar_drafter_taps(&self) -> Result<Vec<u8>, String> {
        self.ctx.sync_compute()?;
        let taps=self.prefill.d_taps.as_ref().ok_or("missing tap buffer for diagnostic")?;
        let mut raw=vec![0u8;4*D*4];
        for tap in 0..4 {
            unsafe {check_cu((self.ctx.api.cuMemcpyDtoH_v2)(
                raw[tap*D*4..].as_mut_ptr().cast(),
                taps.ptr+(tap*self.prefill.capacity*D*4) as u64,D*4
            ),"read scalar tap")?;}
        }
        Ok(raw)
    }

'''+marker))
