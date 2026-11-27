//! Decode one JSON string and return the next byte offset.
use std::io;

fn parse_hex4(b: &[u8], i: usize) -> io::Result<u16> {
    if i + 4 > b.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated \\u escape",
        ));
    }
    let s = std::str::from_utf8(&b[i..i + 4])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if !s.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid hexadecimal escape",
        ));
    }
    u16::from_str_radix(s, 16).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub(crate) fn quoted(b: &[u8], mut i: usize) -> io::Result<(String, usize)> {
    if b.get(i) != Some(&b'"') {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "expected '\"'"));
    }
    i += 1;
    let mut out = String::new();
    while i < b.len() {
        match b[i] {
            b'"' => return Ok((out, i + 1)),
            b'\\' => {
                i += 1;
                if i >= b.len() {
                    break;
                }
                match b[i] {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let hi = parse_hex4(b, i + 1)?;
                        i += 4;
                        let scalar = if (0xD800..=0xDBFF).contains(&hi) {
                            if !b[i + 1..].starts_with(br"\u") {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "missing low surrogate",
                                ));
                            }
                            let lo = parse_hex4(b, i + 3)?;
                            if !(0xDC00..=0xDFFF).contains(&lo) {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "invalid low surrogate",
                                ));
                            }
                            i += 6;
                            0x10000 + ((u32::from(hi) - 0xD800) << 10) + (u32::from(lo) - 0xDC00)
                        } else {
                            u32::from(hi)
                        };
                        out.push(char::from_u32(scalar).ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "invalid Unicode scalar")
                        })?);
                    }
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "invalid JSON escape",
                        ))
                    }
                }
                i += 1;
            }
            _ => {
                let start = i;
                while i < b.len() && b[i] != b'"' && b[i] != b'\\' {
                    i += 1;
                }
                let seg = std::str::from_utf8(&b[start..i])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                if seg.as_bytes().iter().any(|&byte| byte < 0x20) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unescaped control character",
                    ));
                }
                out.push_str(seg);
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "unterminated JSON string",
    ))
}
