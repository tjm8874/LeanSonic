from pathlib import Path
p=Path('src/crates/leansonic-core/src/inkling2_gpu.rs');s=p.read_text()
def replace(old,new):
 global s
 assert s.count(old)==1,old
 s=s.replace(old,new)
replace('        let mut price = DraftPrice::default();','''        let mut trace = Vec::with_capacity(n_gen);
        let mut price = DraftPrice::default();''')
replace('            let price_skip = max_draft', '''            let round_start = Instant::now();
            let from_copy = !copied.is_empty();
            let prior_pause = price.skip_left;
            let price_skip = max_draft''')
replace('            if drafts.is_empty() {\n                let draft_ns', '''            let proposal_ns = round_start.elapsed().as_nanos();
            let action = if max_draft == 0 { "tail" } else if paused { "legacy" } else if price_skip { "price" } else if from_copy { "copy" } else { "neural" };
            if drafts.is_empty() {
                let draft_ns''')
replace('                last_verify_row = None;\n                continue;','''                trace.push((steps_done-1,1,action,prior_pause,price.skip_left,0,0,proposal_ns,round_start.elapsed().as_nanos(),price.scalar_ns));
                last_verify_row = None;
                continue;''')
replace('            let emitted = verified - usize::from(hit_stop);','''            let emitted = verified - usize::from(hit_stop);
            trace.push((steps_done, emitted, action, prior_pause, price.skip_left, k_len, accepted, proposal_ns, round_start.elapsed().as_nanos(), price.scalar_ns));''')
replace('        // A verify batch writes all rows','''        for (pos,n,mode,prior,after,k,a,propose,cost,scalar) in trace {
            eprintln!("TRACE,{pos},{n},{mode},{prior},{after},{k},{a},{propose},{cost},{scalar:.0}");
        }
        // A verify batch writes all rows''')
p.write_text(s)
