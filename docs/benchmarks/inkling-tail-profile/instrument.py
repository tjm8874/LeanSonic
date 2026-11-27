from pathlib import Path
for arm in ('prior','candidate'):
    p=Path(arm)/'crates/leansonic-core/src/inkling2_gpu.rs';s=p.read_text()
    a=s.index('let head_mode = if is_last',s.index('pub fn prefill_tokens'))
    b=s.index('            self.enqueue_drafter_context(chunk_len)?;',a)
    k=s.index('            self.prefill_tc_tail =',a,b) if arm=='candidate' else s.index('            self.enqueue_batch_step_impl(',a,b)
    s=s[:b]+'''            if profiling {
                self.ctx.sync_compute()?;
                if unsafe { cuProfilerStop() } != 0 { return Err("cuProfilerStop".into()); }
            }
'''+s[b:]
    s=s[:k]+'''            #[link(name="cuda")]
            unsafe extern "C" { fn cuProfilerStart() -> i32; fn cuProfilerStop() -> i32; }
            let profiling=offset != 0 && is_last && std::env::var_os("INKLING_PROFILE_TAIL").is_some();
            if profiling {
                self.ctx.sync_compute()?;
                if unsafe { cuProfilerStart() } != 0 { return Err("cuProfilerStart".into()); }
            }
'''+s[k:]
    p.write_text(s)
