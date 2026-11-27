"""Observe the same Rust scorer rows after scoring, without changing GPU calls or reductions."""
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
marker='    for chunk in start..start + chunks {'
assert s.count(marker)==1
s=s.replace(marker,'''    use std::io::Write as _;
    let mut diag = std::io::BufWriter::new(std::fs::OpenOptions::new().create_new(true).write(true).open(std::env::var("DIAG_ROWS")?)?);
'''+marker)
marker='            if s.kld > worst.0 {'
assert s.count(marker)==1
s=s.replace(marker,'''            let fingerprint = logits.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ u64::from(x.to_bits())).wrapping_mul(1099511628211));
            let top = logits.iter().enumerate().fold(0, |best, (i, &v)| if v > logits[best] {i} else {best});
            writeln!(diag, "{},{},{},{},{:.12},{:.12},{},{:016x}", chunk,pos,target,top,s.kld,s.nll,s.same_top,fingerprint)?;
'''+marker)
p.write_text(s)
