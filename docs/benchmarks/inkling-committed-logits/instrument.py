from pathlib import Path
p=Path('src/crates/leansonic-core/src/inkling2_gpu.rs');s=p.read_text()
m='            cur_tok = preds[accepted];'
assert s.count(m)==1
s=s.replace(m,m+'''
            if steps_done == n_gen && stop_tok.is_none() {
                // Diagnostic binary only: preserve the stale-buffer fingerprint
                // before copying the row whose argmax was actually returned.
                let prior = self.read_last_logits()?;
                let prior_max = prior.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).unwrap().0;
                let prior_hash = prior.iter().fold(14695981039346656037u64, |h,x| (h ^ u64::from(x.to_bits())).wrapping_mul(1099511628211));
                eprintln!("verified-row,{accepted},{b_len},{cur_tok},{prior_max},{prior_hash:016x}");
                unsafe {
                    check_cu((self.ctx.api.cuMemcpyDtoDAsync_v2)(
                        self.d_logits.ptr,
                        self.d_spec_logits.ptr + (accepted * VOCAB * 4) as u64,
                        VOCAB * 4, self.ctx.compute_stream,
                    ), "retain actual final verification row")?;
                }
            }''')
p.write_text(s)
