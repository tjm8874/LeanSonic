use std::fs;
fn encode(raw: &[u8], group: usize) -> (Vec<u32>, [usize; 7]) {
    assert!(raw.len().is_multiple_of(group));
    let descriptors = raw.len()/group;
    let mut words = vec![0u32; descriptors.next_multiple_of(4)];
    let mut counts = [0usize;7];
    for (i, tile) in raw.chunks_exact(group).enumerate() {
        let lo = *tile.iter().min().unwrap();
        let hi = *tile.iter().max().unwrap();
        assert!(lo>=64 && hi<128);
        let width = (u8::BITS - (hi-lo).leading_zeros()) as usize;
        counts[width]+=1;
        let start=words.len();
        let offset=start/4;
        assert!(start%4==0 && offset < 1<<23);
        words[i]=offset as u32 | (u32::from(lo-64)<<23) | ((width as u32)<<29);
        let size = (group*width).div_ceil(32);
        words.resize(start+size.next_multiple_of(4),0);
        if width==0 {continue}
        for (j, &code) in tile.iter().enumerate() {
            let bit=j*width; let pos=start+bit/32; let shift=bit%32;
            let val=u32::from(code-lo);
            words[pos]|=val<<shift;
            if shift+width>32 { words[pos+1]|=val>>(32-shift); }
        }
    }
    words.push(0);
    (words, counts)
}
fn verify(raw: &[u8], words:&[u32], group:usize) {
    for (i,&v) in raw.iter().enumerate() {
        let tag=words[i/group]; let base=((tag>>23)&63)+64; let width=tag>>29;
        let bit=(i%group)*width as usize;
        let pos=(tag as usize&0x7fffff)*4+bit/32;
        let got=if width==0 {base} else {
            let data=u64::from(words[pos]) | (u64::from(words[pos+1])<<32);
            base+((data>>(bit%32)) as u32 & ((1u32<<width)-1))
        };
        assert_eq!(got,u32::from(v))
    }
}
fn bytes(w:&[u32])->Vec<u8>{w.iter().flat_map(|v|v.to_le_bytes()).collect()}
fn main() {
    let raw=fs::read("scales.bin").unwrap();
    let (words,_)=encode(&raw,128);
    verify(&raw,&words,128);
    fs::write("scales-128.bin",bytes(&words)).unwrap();
}
