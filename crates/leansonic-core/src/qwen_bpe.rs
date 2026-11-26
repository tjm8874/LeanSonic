//! Zero-Dependency HuggingFace `tokenizer.json` GPT-2 Byte-Level BPE Tokenizer (`qwen_bpe.rs`)
//! for `Qwen3.8-Flash-Next` / `Qwen4Exp` (`248,044` BPE vocab + `33` special tokens).

use std::collections::HashMap;
use std::io;
use std::path::Path;

pub const QWEN_TOK_ENDOFTEXT: u32 = 248_044;
pub const QWEN_TOK_IM_START: u32 = 248_045;
pub const QWEN_TOK_IM_END: u32 = 248_046;
pub const QWEN_TOK_THINK_OPEN: u32 = 248_068;
pub const QWEN_TOK_THINK_CLOSE: u32 = 248_069;

pub struct QwenBpeTokenizer {
    vocab: HashMap<String, u32>,
    id_to_bytes: Vec<Vec<u8>>,
    merges: HashMap<(String, String), u32>,
    added_special: Vec<(String, u32)>,
    byte_to_char: [char; 256],
}

impl QwenBpeTokenizer {
    pub fn load_from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        Self::from_tokenizer_json(&raw)
    }

    pub fn from_tokenizer_json(raw: &str) -> io::Result<Self> {
        let (byte_to_char, char_to_byte) = build_gpt2_byte_tables();
        let b = raw.as_bytes();
        let mut i = skip_ws(b, 0);
        if b.get(i) != Some(&b'{') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tokenizer.json must start with '{'",
            ));
        }
        i += 1;

        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut merges: HashMap<(String, String), u32> = HashMap::new();
        let mut added_special: Vec<(String, u32)> = Vec::new();
        let mut max_id = 0u32;

        loop {
            i = skip_ws(b, i);
            if b.get(i) == Some(&b'}') {
                break;
            }
            let (key, ni) = parse_json_str(b, i)?;
            i = expect_colon(b, ni)?;
            match key.as_str() {
                "added_tokens" => {
                    if b.get(i) != Some(&b'[') {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "added_tokens must be '['",
                        ));
                    }
                    i += 1;
                    loop {
                        i = skip_ws(b, i);
                        if b.get(i) == Some(&b']') {
                            i += 1;
                            break;
                        }
                        if b.get(i) != Some(&b'{') {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "added_token entry must be '{'",
                            ));
                        }
                        i += 1;
                        let mut tid = 0u32;
                        let mut content = String::new();
                        loop {
                            i = skip_ws(b, i);
                            if b.get(i) == Some(&b'}') {
                                i += 1;
                                break;
                            }
                            let (fk, fni) = parse_json_str(b, i)?;
                            i = expect_colon(b, fni)?;
                            match fk.as_str() {
                                "id" => {
                                    let (v, ni2) = parse_u32(b, i)?;
                                    tid = v;
                                    i = ni2;
                                }
                                "content" => {
                                    let (s, ni2) = parse_json_str(b, i)?;
                                    content = s;
                                    i = ni2;
                                }
                                _ => {
                                    i = skip_json_val(b, i)?;
                                }
                            }
                            i = consume_comma(b, i);
                        }
                        if !content.is_empty() {
                            if tid > max_id {
                                max_id = tid;
                            }
                            added_special.push((content, tid));
                        }
                        i = consume_comma(b, i);
                    }
                }
                "model" => {
                    if b.get(i) != Some(&b'{') {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "model must be '{'",
                        ));
                    }
                    i += 1;
                    loop {
                        i = skip_ws(b, i);
                        if b.get(i) == Some(&b'}') {
                            i += 1;
                            break;
                        }
                        let (mk, mni) = parse_json_str(b, i)?;
                        i = expect_colon(b, mni)?;
                        match mk.as_str() {
                            "vocab" => {
                                if b.get(i) != Some(&b'{') {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "vocab must be '{'",
                                    ));
                                }
                                i += 1;
                                loop {
                                    i = skip_ws(b, i);
                                    if b.get(i) == Some(&b'}') {
                                        i += 1;
                                        break;
                                    }
                                    let (tok_s, ni2) = parse_json_str(b, i)?;
                                    i = expect_colon(b, ni2)?;
                                    let (tid, ni3) = parse_u32(b, i)?;
                                    if tid > max_id {
                                        max_id = tid;
                                    }
                                    vocab.insert(tok_s, tid);
                                    i = consume_comma(b, ni3);
                                }
                            }
                            "merges" => {
                                if b.get(i) != Some(&b'[') {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "merges must be '['",
                                    ));
                                }
                                i += 1;
                                let mut rank = 0u32;
                                loop {
                                    i = skip_ws(b, i);
                                    if b.get(i) == Some(&b']') {
                                        i += 1;
                                        break;
                                    }
                                    if b.get(i) == Some(&b'[') {
                                        i += 1;
                                        let (p0, ni2) = parse_json_str(b, i)?;
                                        i = consume_comma(b, ni2);
                                        let (p1, ni3) = parse_json_str(b, i)?;
                                        i = skip_ws(b, ni3);
                                        if b.get(i) != Some(&b']') {
                                            return Err(io::Error::new(
                                                io::ErrorKind::InvalidData,
                                                "expected ']' in merge pair",
                                            ));
                                        }
                                        i += 1;
                                        merges.insert((p0, p1), rank);
                                    } else {
                                        let (pair_s, ni2) = parse_json_str(b, i)?;
                                        i = ni2;
                                        if let Some((p0, p1)) = pair_s.split_once(' ') {
                                            merges.insert((p0.to_string(), p1.to_string()), rank);
                                        }
                                    }
                                    rank += 1;
                                    i = consume_comma(b, i);
                                }
                            }
                            _ => {
                                i = skip_json_val(b, i)?;
                            }
                        }
                        i = consume_comma(b, i);
                    }
                }
                _ => {
                    i = skip_json_val(b, i)?;
                }
            }
            i = consume_comma(b, i);
        }

        // Sort special tokens longest-first for greedy exact matching
        added_special.sort_by(|a, b_item| b_item.0.len().cmp(&a.0.len()));

        let mut id_to_bytes = vec![Vec::new(); (max_id as usize) + 1];
        for (tok_s, &tid) in &vocab {
            let mut decoded = Vec::with_capacity(tok_s.len());
            for ch in tok_s.chars() {
                if let Some(&byte_val) = char_to_byte.get(&ch) {
                    decoded.push(byte_val);
                } else {
                    let mut u8_buf = [0u8; 4];
                    decoded.extend_from_slice(ch.encode_utf8(&mut u8_buf).as_bytes());
                }
            }
            id_to_bytes[tid as usize] = decoded;
        }
        for (sp_str, tid) in &added_special {
            id_to_bytes[*tid as usize] = sp_str.as_bytes().to_vec();
        }

        Ok(Self {
            vocab,
            id_to_bytes,
            merges,
            added_special,
            byte_to_char,
        })
    }

    pub fn decode_token_bytes(&self, id: u32) -> &[u8] {
        self.id_to_bytes
            .get(id as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn is_special_token(&self, id: u32) -> bool {
        id >= QWEN_TOK_ENDOFTEXT
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut cursor = 0usize;
        while cursor < text.len() {
            // Find earliest matching special token at or after `cursor`
            let mut best_match: Option<(usize, usize, u32)> = None;
            for (sp_str, sp_id) in &self.added_special {
                if let Some(rel) = text[cursor..].find(sp_str.as_str()) {
                    let pos = cursor + rel;
                    match best_match {
                        None => best_match = Some((pos, sp_str.len(), *sp_id)),
                        Some((b_pos, b_len, _)) => {
                            if pos < b_pos || (pos == b_pos && sp_str.len() > b_len) {
                                best_match = Some((pos, sp_str.len(), *sp_id));
                            }
                        }
                    }
                }
            }

            if let Some((pos, sp_len, sp_id)) = best_match {
                if pos > cursor {
                    self.encode_ordinary(&text[cursor..pos], &mut out);
                }
                out.push(sp_id);
                cursor = pos + sp_len;
            } else {
                self.encode_ordinary(&text[cursor..], &mut out);
                break;
            }
        }
        out
    }

    /// Formats a user message using `Qwen4Exp`'s official ChatML + `<think>` template.
    pub fn encode_chat_prompt(&self, user_prompt: &str, effort: &str) -> Vec<u32> {
        let think_suffix = if effort == "none" {
            "<think>\n\n</think>\n\n"
        } else {
            "<think>\n"
        };
        let formatted = format!(
            "<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n{think_suffix}"
        );
        self.encode(&formatted)
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        for piece in split_qwen_pretokenize(text) {
            self.encode_piece(piece.as_bytes(), out);
        }
    }

    fn encode_piece(&self, piece_bytes: &[u8], out: &mut Vec<u32>) {
        if piece_bytes.is_empty() {
            return;
        }
        let mut symbols: Vec<String> = piece_bytes
            .iter()
            .map(|&b| self.byte_to_char[b as usize].to_string())
            .collect();

        if symbols.len() == 1 {
            if let Some(&id) = self.vocab.get(&symbols[0]) {
                out.push(id);
            }
            return;
        }

        loop {
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;
            for idx in 0..symbols.len() - 1 {
                let key = (symbols[idx].clone(), symbols[idx + 1].clone());
                if let Some(&r) = self.merges.get(&key) {
                    if r < best_rank {
                        best_rank = r;
                        best_idx = idx;
                    }
                }
            }
            if best_idx == usize::MAX {
                break;
            }
            let mut i = 0usize;
            let target_a = symbols[best_idx].clone();
            let target_b = symbols[best_idx + 1].clone();
            let merged_tok = format!("{target_a}{target_b}");
            let mut next_syms = Vec::with_capacity(symbols.len() - 1);
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == target_a && symbols[i + 1] == target_b {
                    next_syms.push(merged_tok.clone());
                    i += 2;
                } else {
                    next_syms.push(std::mem::take(&mut symbols[i]));
                    i += 1;
                }
            }
            symbols = next_syms;
            if symbols.len() == 1 {
                break;
            }
        }

        for sym in &symbols {
            if let Some(&id) = self.vocab.get(sym) {
                out.push(id);
            }
        }
    }
}

/// Qwen/GPT-4 style pre-tokenizer: splits digits individually (`\p{N}` single-digit in Qwen!),
/// optional leading space + letters, optional leading space + punctuation, and newline/space runs.
fn split_qwen_pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut pieces = Vec::new();
    let mut k = 0usize;
    while k < chars.len() {
        let start_byte = chars[k].0;
        let ch = chars[k].1;

        if ch == '\r' || ch == '\n' {
            k += 1;
            while k < chars.len() && (chars[k].1 == '\r' || chars[k].1 == '\n') {
                k += 1;
            }
        } else if ch.is_numeric() {
            // Qwen split_digits = individual digits!
            k += 1;
        } else if ch.is_alphabetic() {
            k += 1;
            while k < chars.len() && chars[k].1.is_alphabetic() {
                k += 1;
            }
        } else if ch == ' ' && k + 1 < chars.len() && chars[k + 1].1.is_alphabetic() {
            k += 2;
            while k < chars.len() && chars[k].1.is_alphabetic() {
                k += 1;
            }
        } else if ch == ' '
            && k + 1 < chars.len()
            && !chars[k + 1].1.is_whitespace()
            && !chars[k + 1].1.is_numeric()
        {
            k += 2;
            while k < chars.len()
                && !chars[k].1.is_whitespace()
                && !chars[k].1.is_alphabetic()
                && !chars[k].1.is_numeric()
            {
                k += 1;
            }
        } else if ch.is_whitespace() {
            k += 1;
            while k < chars.len()
                && chars[k].1.is_whitespace()
                && chars[k].1 != '\r'
                && chars[k].1 != '\n'
            {
                // Leave trailing single space if followed by letter/punct so ` word` attaches!
                if chars[k].1 == ' '
                    && k + 1 < chars.len()
                    && !chars[k + 1].1.is_whitespace()
                    && !chars[k + 1].1.is_numeric()
                {
                    break;
                }
                k += 1;
            }
        } else {
            k += 1;
            while k < chars.len()
                && !chars[k].1.is_whitespace()
                && !chars[k].1.is_alphabetic()
                && !chars[k].1.is_numeric()
            {
                k += 1;
            }
        }

        let end_byte = if k < chars.len() {
            chars[k].0
        } else {
            text.len()
        };
        pieces.push(&text[start_byte..end_byte]);
    }
    pieces
}

fn build_gpt2_byte_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    let mut next_extra = 256u32;
    for (b, slot) in byte_to_char.iter_mut().enumerate() {
        let is_direct =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let ch = if is_direct {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(next_extra).unwrap();
            next_extra += 1;
            c
        };
        *slot = ch;
        char_to_byte.insert(ch, b as u8);
    }
    (byte_to_char, char_to_byte)
}

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    i
}

fn expect_colon(b: &[u8], i: usize) -> io::Result<usize> {
    let i = skip_ws(b, i);
    if b.get(i) != Some(&b':') {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "expected ':'"));
    }
    Ok(skip_ws(b, i + 1))
}

fn consume_comma(b: &[u8], i: usize) -> usize {
    let i = skip_ws(b, i);
    if b.get(i) == Some(&b',') {
        skip_ws(b, i + 1)
    } else {
        i
    }
}

fn parse_u32(b: &[u8], i: usize) -> io::Result<(u32, usize)> {
    let mut j = skip_ws(b, i);
    let start = j;
    while j < b.len() && b[j].is_ascii_digit() {
        j += 1;
    }
    let s = std::str::from_utf8(&b[start..j])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let v = s
        .parse::<u32>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok((v, j))
}

fn parse_hex4(b: &[u8], i: usize) -> io::Result<u16> {
    if i + 4 > b.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated \\u escape",
        ));
    }
    let s = std::str::from_utf8(&b[i..i + 4])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    u16::from_str_radix(s, 16).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn parse_json_str(b: &[u8], mut i: usize) -> io::Result<(String, usize)> {
    i = skip_ws(b, i);
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
                        if (0xD800..=0xDBFF).contains(&hi)
                            && i + 6 < b.len()
                            && b[i + 1] == b'\\'
                            && b[i + 2] == b'u'
                        {
                            let lo = parse_hex4(b, i + 3)?;
                            i += 6;
                            let cp =
                                0x10000 + (((hi as u32 - 0xD800) << 10) | (lo as u32 - 0xDC00));
                            if let Some(ch) = char::from_u32(cp) {
                                out.push(ch);
                            }
                        } else if let Some(ch) = char::from_u32(hi as u32) {
                            out.push(ch);
                        }
                    }
                    other => out.push(other as char),
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
                out.push_str(seg);
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "unterminated JSON string",
    ))
}

fn skip_json_val(b: &[u8], mut i: usize) -> io::Result<usize> {
    i = skip_ws(b, i);
    match b.get(i) {
        Some(&b'"') => Ok(parse_json_str(b, i)?.1),
        Some(&b'{') => {
            let mut depth = 1usize;
            i += 1;
            while i < b.len() && depth > 0 {
                match b[i] {
                    b'"' => {
                        i = parse_json_str(b, i)?.1;
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
                        i = parse_json_str(b, i)?.1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_byte_bijection_and_bpe_roundtrip() {
        let sample = r#"{
            "added_tokens": [{"id": 248045, "content": "<|im_start|>"}, {"id": 248046, "content": "<|im_end|>"}],
            "model": {
                "type": "BPE",
                "vocab": {"H": 0, "i": 1, "Hi": 2, "\u0120": 3, "1": 4, "7": 5},
                "merges": ["H i"]
            }
        }"#;
        let tok = QwenBpeTokenizer::from_tokenizer_json(sample).unwrap();
        let ids = tok.encode("<|im_start|>Hi 17<|im_end|>");
        assert_eq!(ids, vec![248045, 2, 3, 4, 5, 248046]);
        let mut bytes = Vec::new();
        for id in ids {
            bytes.extend_from_slice(tok.decode_token_bytes(id));
        }
        assert_eq!(
            std::str::from_utf8(&bytes).unwrap(),
            "<|im_start|>Hi 17<|im_end|>"
        );
    }
}
