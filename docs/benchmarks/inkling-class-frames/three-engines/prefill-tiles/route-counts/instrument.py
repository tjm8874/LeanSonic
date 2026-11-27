from pathlib import Path
p=Path('src/crates/leansonic-core/src/inkling2_gpu.rs')
text=p.read_text()
needle='        let mut moe_wrote_f16 = false;'
assert text.count(needle)==1
capture='''        if b_len == 1024 {
            let mut counts = [0i32; N_EXP];
            self.ctx.sync_stream(stream)?;
            unsafe {
                check_cu(
                    (self.ctx.api.cuMemcpyDtoH_v2)(
                        counts.as_mut_ptr().cast(), self.d_b_exp_counts.ptr,
                        std::mem::size_of_val(&counts),
                    ),
                    "diagnostic copy of prefill expert counts",
                )?;
            }
            assert!(counts.iter().all(|v| (0..=1024).contains(v)));
            assert_eq!(counts.iter().sum::<i32>(), 8192);
            let encoding = match &l.expert_scales {
                ExpertScales::Nvfp4 { .. } => "nvfp4",
                _ => "other",
            };
            eprintln!("ROUTES {il} {encoding} {counts:?}");
        }
'''
p.write_text(text.replace(needle,capture+needle))
