from pathlib import Path
import shutil
p=Path('src/crates/leansonic-core/src/inkling2_gpu.rs')
code=p.read_text()
key='            let mut hits = [0u32; N_EXP];'
assert code.count(key)==1
code=code.replace(key,'''            let mut hidden = vec![0f32; b_len * D];
            unsafe {
                check_cu((self.ctx.api.cuMemcpyDtoH_v2)(hidden.as_mut_ptr().cast(),
                    self.prefill.layer_workspace.mlp_hidden, hidden.len() * 4), "capture normalized expert input")?;
            }
            train_moments::observe(il, &ids, &weights, &hidden)?;
'''+key)
p.write_text(code+'\n#[path = "train_moments.rs"]\nmod train_moments;\n')
shutil.copyfile('moments.rs',p.parent/'train_moments.rs')
