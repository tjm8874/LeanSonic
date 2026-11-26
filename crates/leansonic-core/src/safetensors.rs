//! Zero-Dependency `safetensors` Memory-Mapped Reader (`safetensors.rs`).
//!
//! Maps multi-gigabyte checkpoints (e.g. `19.5 GB` `Inkling-2-Mini` `model-00001-of-00001.safetensors`)
//! directly from OS page cache via POSIX `mmap` with zero third-party Rust crates.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

pub struct TensorView<'a> {
    pub dtype: &'a str,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn as_u16_slice(&self) -> &'a [u16] {
        assert_eq!(self.data.len() % 2, 0, "unaligned u16/bf16 byte length");
        let (pre, mid, post) = unsafe { self.data.align_to::<u16>() };
        assert!(
            pre.is_empty() && post.is_empty(),
            "safetensors u16 alignment error"
        );
        mid
    }

    pub fn as_f32_slice(&self) -> &'a [f32] {
        assert_eq!(self.data.len() % 4, 0, "unaligned f32 byte length");
        let (pre, mid, post) = unsafe { self.data.align_to::<f32>() };
        assert!(
            pre.is_empty() && post.is_empty(),
            "safetensors f32 alignment error"
        );
        mid
    }
}

#[cfg(unix)]
mod os_mmap {
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;

    const PROT_READ: i32 = 1;
    const MAP_PRIVATE: i32 = 2;

    unsafe extern "C" {
        fn mmap(
            addr: *mut core::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut core::ffi::c_void;
        fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
    }

    pub struct RawMmap {
        ptr: *mut u8,
        len: usize,
    }

    unsafe impl Send for RawMmap {}
    unsafe impl Sync for RawMmap {}

    impl RawMmap {
        pub fn map_read_only(file: &File) -> io::Result<Self> {
            let len = file.metadata()?.len() as usize;
            if len == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "empty safetensors file",
                ));
            }
            let raw = unsafe {
                mmap(
                    core::ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if raw as isize == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(Self {
                ptr: raw.cast::<u8>(),
                len,
            })
        }

        pub fn as_slice(&self) -> &[u8] {
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    impl Drop for RawMmap {
        fn drop(&mut self) {
            if !self.ptr.is_null() && self.len > 0 {
                unsafe {
                    let _ = munmap(self.ptr.cast(), self.len);
                }
            }
        }
    }
}

#[cfg(not(unix))]
mod os_mmap {
    use std::fs::File;
    use std::io::{self, Read};

    pub struct RawMmap {
        buf: Vec<u8>,
    }

    impl RawMmap {
        pub fn map_read_only(mut file: &File) -> io::Result<Self> {
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            Ok(Self { buf })
        }

        pub fn as_slice(&self) -> &[u8] {
            &self.buf
        }
    }
}

pub struct MappedSafeTensors {
    mmap: os_mmap::RawMmap,
    data_base: usize,
    tensors: HashMap<String, TensorInfo>,
}

impl MappedSafeTensors {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = os_mmap::RawMmap::map_read_only(&file)?;
        let bytes = mmap.as_slice();
        if bytes.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors header < 8B",
            ));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let data_base = 8usize.saturating_add(header_len);
        if data_base > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors header exceeds file length",
            ));
        }
        let header_str = std::str::from_utf8(&bytes[8..data_base])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let tensors = parse_safetensors_header(header_str, bytes.len() - data_base)?;
        Ok(Self {
            mmap,
            data_base,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Option<TensorView<'_>> {
        let info = self.tensors.get(name)?;
        let raw = self.mmap.as_slice();
        Some(TensorView {
            dtype: &info.dtype,
            shape: &info.shape,
            data: &raw[self.data_base + info.data_start..self.data_base + info.data_end],
        })
    }

    pub fn require(
        &self,
        name: &str,
        expected_dtype: &str,
        expected_shape: &[usize],
    ) -> TensorView<'_> {
        let tv = self
            .get(name)
            .unwrap_or_else(|| panic!("missing required tensor `{name}`"));
        assert_eq!(
            tv.dtype, expected_dtype,
            "tensor `{name}` dtype mismatch: expected {expected_dtype}, got {}",
            tv.dtype
        );
        assert_eq!(
            tv.shape, expected_shape,
            "tensor `{name}` shape mismatch: expected {expected_shape:?}, got {:?}",
            tv.shape
        );
        tv
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

// Minimal recursive-free scanner for top-level `safetensors` JSON header entries.
fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    i
}

fn parse_str(b: &[u8], mut i: usize) -> io::Result<(String, usize)> {
    i = skip_ws(b, i);
    if b.get(i) != Some(&b'"') {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "expected '\"'"));
    }
    i += 1;
    let mut s = Vec::new();
    while i < b.len() {
        match b[i] {
            b'"' => {
                let text = String::from_utf8(s)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                return Ok((text, i + 1));
            }
            b'\\' => {
                i += 1;
                if i < b.len() {
                    s.push(b[i]);
                }
            }
            ch => s.push(ch),
        }
        i += 1;
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "unterminated string",
    ))
}

fn skip_json_value(b: &[u8], mut i: usize) -> io::Result<usize> {
    i = skip_ws(b, i);
    match b.get(i) {
        Some(&b'"') => Ok(parse_str(b, i)?.1),
        Some(&b'{') => {
            let mut depth = 1usize;
            i += 1;
            while i < b.len() && depth > 0 {
                match b[i] {
                    b'"' => {
                        i = parse_str(b, i)?.1;
                        continue;
                    }
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            Ok(i)
        }
        Some(&b'[') => {
            let mut depth = 1usize;
            i += 1;
            while i < b.len() && depth > 0 {
                match b[i] {
                    b'"' => {
                        i = parse_str(b, i)?.1;
                        continue;
                    }
                    b'[' => depth += 1,
                    b']' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            Ok(i)
        }
        Some(_) => {
            while i < b.len() && !matches!(b[i], b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t')
            {
                i += 1;
            }
            Ok(i)
        }
        None => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unexpected EOF in JSON",
        )),
    }
}

fn parse_usize_list(b: &[u8], mut i: usize) -> io::Result<(Vec<usize>, usize)> {
    i = skip_ws(b, i);
    if b.get(i) != Some(&b'[') {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "expected '['"));
    }
    i += 1;
    let mut vals = Vec::new();
    loop {
        i = skip_ws(b, i);
        if b.get(i) == Some(&b']') {
            return Ok((vals, i + 1));
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let num_str = std::str::from_utf8(&b[start..i])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let val: usize = num_str
            .parse()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        vals.push(val);
        i = skip_ws(b, i);
        if b.get(i) == Some(&b',') {
            i += 1;
        }
    }
}

fn parse_safetensors_header(
    header: &str,
    payload_len: usize,
) -> io::Result<HashMap<String, TensorInfo>> {
    let b = header.as_bytes();
    let mut i = skip_ws(b, 0);
    if b.get(i) != Some(&b'{') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "safetensors header must start with '{'",
        ));
    }
    i += 1;
    let mut map = HashMap::new();
    loop {
        i = skip_ws(b, i);
        if b.get(i) == Some(&b'}') {
            break;
        }
        let (key, next_i) = parse_str(b, i)?;
        i = skip_ws(b, next_i);
        if b.get(i) != Some(&b':') {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "expected ':'"));
        }
        i = skip_ws(b, i + 1);

        if key == "__metadata__" {
            i = skip_json_value(b, i)?;
        } else {
            if b.get(i) != Some(&b'{') {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "expected tensor object '{'",
                ));
            }
            i += 1;
            let mut dtype = String::new();
            let mut shape = Vec::new();
            let mut offsets = Vec::new();
            loop {
                i = skip_ws(b, i);
                if b.get(i) == Some(&b'}') {
                    i += 1;
                    break;
                }
                let (field, ni) = parse_str(b, i)?;
                i = skip_ws(b, ni);
                if b.get(i) != Some(&b':') {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "expected ':' in tensor object",
                    ));
                }
                i = skip_ws(b, i + 1);
                match field.as_str() {
                    "dtype" => {
                        let (dt, ni2) = parse_str(b, i)?;
                        dtype = dt;
                        i = ni2;
                    }
                    "shape" => {
                        let (sh, ni2) = parse_usize_list(b, i)?;
                        shape = sh;
                        i = ni2;
                    }
                    "data_offsets" => {
                        let (ofs, ni2) = parse_usize_list(b, i)?;
                        offsets = ofs;
                        i = ni2;
                    }
                    _ => {
                        i = skip_json_value(b, i)?;
                    }
                }
                i = skip_ws(b, i);
                if b.get(i) == Some(&b',') {
                    i += 1;
                }
            }
            if offsets.len() != 2 || offsets[1] > payload_len || offsets[0] > offsets[1] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid data_offsets for `{key}`"),
                ));
            }
            map.insert(
                key,
                TensorInfo {
                    dtype,
                    shape,
                    data_start: offsets[0],
                    data_end: offsets[1],
                },
            );
        }

        i = skip_ws(b, i);
        if b.get(i) == Some(&b',') {
            i += 1;
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_safetensors_header_with_metadata() {
        let hdr = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]}}"#;
        let parsed = parse_safetensors_header(hdr, 16).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed["w"].dtype, "BF16");
        assert_eq!(parsed["w"].shape, vec![2, 4]);
        assert_eq!((parsed["w"].data_start, parsed["w"].data_end), (0, 16));
    }
}
