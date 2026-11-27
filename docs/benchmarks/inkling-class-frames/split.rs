use leansonic_core::inkling_mxfp8::{pack_four_scale_banks,unpack_four_scale_banks,unpack_mean8_scale_codes};
use std::{env,error::Error,fs::{self,File},io::{Read,Seek,SeekFrom}};

fn take(f:&mut File,off:&str,n:&str)->Result<Vec<u8>,Box<dyn Error>>{
    f.seek(SeekFrom::Start(off.parse()?))?;
    let mut out=vec![0;n.parse()?];f.read_exact(&mut out)?;Ok(out)
}
fn member(mask:&[u8], e:usize)->usize{usize::from((mask[e/8]>>(e%8))&1)}
fn pack_class(raw:&[u8],scale:Vec<u8>)->Result<Vec<u8>,Box<dyn Error>>{
    assert!(scale.len().is_multiple_of(128) && raw.len()==8*scale.len());
    let size=scale.len()/128;let mut base=[0;128];
    for (e,row) in scale.chunks_exact(size).enumerate(){
        let mut hist=[0usize;128];for &s in row {hist[s as usize]+=1;}
        let mut n=0;let mut best=(size,0);
        for (i,&f) in hist.iter().enumerate().take(127) {n+=f;best=best.min(((2*n).abs_diff(size),i+1));}
        base[e]=best.1 as u8;
    }
    let mut result=Vec::new();pack_four_scale_banks(raw,scale,&base,&mut result)?;Ok(result)
}
fn decode(frame:&[u8], scales:&[u8])->Result<Vec<u8>,Box<dyn Error>>{
    if frame.len()<40 || &frame[..8]!=b"INKCL2V1" || !scales.len().is_multiple_of(128) {return Err("invalid class frame".into());}
    let mask=&frame[8..24];let stride=scales.len()/128;
    let lengths=[24,32].map(|i|usize::try_from(u64::from_le_bytes(frame[i..i+8].try_into().unwrap()))).into_iter().collect::<Result<Vec<_>,_>>()?;
    if lengths.iter().try_fold(40usize,|a,b|a.checked_add(*b))!=Some(frame.len()) {return Err("invalid class lengths".into());}
    let mut output=vec![0u8;scales.len()*8]; let mut offset=40;
    for class in 0..2 {
        let ids:Vec<_>=(0..128).filter(|&e|member(mask,e)==class).collect();
        if ids.is_empty(){return Err("empty expert class".into());}
        let mut values=Vec::with_capacity(ids.len()*stride);
        for &e in &ids {values.extend_from_slice(&scales[e*stride..(e+1)*stride]);}
        let end=offset+lengths[class];
        let decoded=unpack_four_scale_banks(&frame[offset..end],values)?;
        if decoded.len()!=ids.len()*8*stride{return Err("invalid decoded class length".into());}
        for (&e,data) in ids.iter().zip(decoded.chunks_exact(8*stride)) {output[e*8*stride..(e+1)*8*stride].copy_from_slice(data);}
        offset=end;
    }
    Ok(output)
}
fn main()->Result<(),Box<dyn Error>>{
    let args:Vec<_>=env::args().skip(1).collect();
    let [path,w,n,s,m,width,ids,out]=args.as_slice() else{return Err("expected shard, tensor offsets, width, restored experts and frame path".into());};
    let mut f=File::open(path)?;let original=take(&mut f,w,n)?;let sf=take(&mut f,s,m)?;
    let width:usize=width.parse()?;let stride=2304*width;let scales=unpack_mean8_scale_codes(&sf,128*stride,width)?;
    let raw=unpack_four_scale_banks(&original,scales.clone())?;
    let mut mask=[0u8;16];for id in ids.split(','){let e=id.parse::<usize>()?;assert!(e<128 && member(&mask,e)==0);mask[e/8]|=1<<(e%8);}
    let mut parts=Vec::new();
    for class in 0..2 {
        let mut weights=Vec::new();let mut sc=Vec::new();
        for e in 0..128{if member(&mask,e)==class {
            weights.extend_from_slice(&raw[e*8*stride..(e+1)*8*stride]);sc.extend_from_slice(&scales[e*stride..(e+1)*stride]);
        }}
        parts.push(pack_class(&weights,sc)?);
    }
    let mut frame=Vec::new();frame.extend_from_slice(b"INKCL2V1");frame.extend_from_slice(&mask);
    for part in &parts{frame.extend_from_slice(&(part.len() as u64).to_le_bytes());}
    for part in parts{frame.extend_from_slice(&part);}
    assert!(decode(&frame,&scales)?==raw);
    let choose=frame.len()<original.len();
    if choose {fs::write(out,&frame)?;}
    println!("original {}",original.len());println!("separate {}",frame.len());
    println!("selected {} {} raw {}",if choose{"separate"}else{"original"},frame.len().min(original.len()),raw.len());
    Ok(())
}
