from pathlib import Path
p=Path('crates/leansonic-core/src/inkling_mxfp8/mean_scales.rs')
s=p.read_text()
a=s.index('    for edges in [&left, &above]')
b=s.index('fn read_edges(',a)
encoder=s[a:b]
encoder=encoder.replace('for edges in [&left, &above]', 'for edges in [left, above]').replace('MeanContext::new(&left, &above, width)','MeanContext::new(left, above, width)')
fn='''pub fn pack_custom(raw: &[u8], width: usize, left: &[u16], above: &[u16], mut output: impl Write) -> io::Result<usize> {
    if !valid_shape(raw.len(),width) || [left,above].iter().any(|e| !(1..=11).contains(&e.len()) || e.windows(2).any(|v|v[0]>=v[1]) || usize::from(*e.last().unwrap())>MAX) {
        return Err(io::Error::other("invalid feature boundaries"));
    }
'''+encoder+'''
pub fn sample_mean_features(raw: &[u8], width: usize, mut out: impl Write) -> io::Result<()> {
    for (i,tile) in raw.chunks(TILE).enumerate() {
        for p in ((i*53)%32..tile.len()).step_by(32) {
            let [x,y] = features(tile,p,i*TILE,width);
            out.write_all(&(x as u16).to_le_bytes())?;
            out.write_all(&(y as u16).to_le_bytes())?;
            out.write_all(&[tile[p]])?;
        }
    }
    Ok(())
}
'''
p.write_text(s[:b]+fn+s[b:])
p=Path('crates/leansonic-core/src/inkling_mxfp8.rs');s=p.read_text();p.write_text(s.replace('pub use mean_scales::pack_mean_scales;', 'pub use mean_scales::{pack_mean_scales, pack_custom, sample_mean_features};'))
