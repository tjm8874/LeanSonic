from pathlib import Path
p=Path('src/crates/leansonic-core/src/inkling2_gpu.rs');s=p.read_text();m='            self.spec_accepted_total += accepted;';assert s.count(m)==1
s=s.replace(m,'            println!("round,{steps_done},{k_len},{accepted},{drafts:?}");\n'+m);p.write_text(s)
