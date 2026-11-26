//! Zero-Dependency `tiktoken` BPE Encoder/Decoder & `Inkling-2-Mini` Chat Formatter.
//!
//! Reads `<model_dir>/tiktoken/tokenizer.model` (`<base64> <rank>` lines, 199,998 BPE merges)
//! with zero external crates and formats chat messages identically to `chat_template.jinja`.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

pub const TOK_ENDOFTEXT: u32 = 199_999;
pub const TOK_MSG_USER: u32 = 200_000;
pub const TOK_MSG_MODEL: u32 = 200_001;
pub const TOK_MSG_SYSTEM: u32 = 200_002;
pub const TOK_MSG_TOOL: u32 = 200_003;
pub const TOK_CONTENT_TEXT: u32 = 200_004;
pub const TOK_END_SAMPLING: u32 = 200_006;
pub const TOK_CONTENT_THINKING: u32 = 200_008;
pub const TOK_END_MESSAGE: u32 = 200_010;

pub struct TiktokenBpe {
    encoder: HashMap<Vec<u8>, u32>,
    decoder: Vec<Vec<u8>>,
}

impl TiktokenBpe {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = model_dir.as_ref();
        let path = if dir.join("tiktoken/tokenizer.model").exists() {
            dir.join("tiktoken/tokenizer.model")
        } else {
            dir.join("tokenizer.model")
        };
        let raw = fs::read_to_string(path)?;
        Self::from_tiktoken_str(&raw)
    }

    pub fn from_tiktoken_str(content: &str) -> io::Result<Self> {
        let mut encoder = HashMap::with_capacity(200_100);
        let mut max_rank = 0u32;
        let mut entries = Vec::with_capacity(200_000);

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some((b64, rank_str)) = line.split_once(' ') else {
                continue;
            };
            let bytes = decode_base64(b64.trim())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "bad base64"))?;
            let rank: u32 = rank_str
                .trim()
                .parse()
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            max_rank = max_rank.max(rank);
            entries.push((bytes, rank));
        }

        let mut decoder = vec![Vec::new(); (max_rank as usize) + 1];
        for (bytes, rank) in entries {
            decoder[rank as usize] = bytes.clone();
            encoder.insert(bytes, rank);
        }

        Ok(Self { encoder, decoder })
    }

    /// Standard byte-pair merge over pre-segmented chunks (whitespace/word/punct pieces).
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for piece in split_tiktoken_pieces(text) {
            self.bpe_merge_piece(piece.as_bytes(), &mut out);
        }
        out
    }

    fn bpe_merge_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        if let Some(&id) = self.encoder.get(piece) {
            out.push(id);
            return;
        }
        let mut parts: Vec<(usize, usize)> = (0..piece.len()).map(|i| (i, i + 1)).collect();
        while parts.len() > 1 {
            let mut best_rank = u32::MAX;
            let mut best_idx = usize::MAX;
            for idx in 0..(parts.len() - 1) {
                let merged = &piece[parts[idx].0..parts[idx + 1].1];
                if let Some(&rank) = self.encoder.get(merged) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = idx;
                    }
                }
            }
            if best_idx == usize::MAX {
                break;
            }
            parts[best_idx].1 = parts[best_idx + 1].1;
            parts.remove(best_idx + 1);
        }
        for (s, e) in parts {
            if let Some(&id) = self.encoder.get(&piece[s..e]) {
                out.push(id);
            }
        }
    }

    /// Build `Inkling-2-Mini` single-turn or multi-turn chat prompt token IDs.
    pub fn format_inkling2_chat(&self, user_text: &str, effort: &str) -> Vec<u32> {
        let effort_val = match effort {
            "none" | "0" | "0.0" => "0",
            "minimal" => "0.1",
            "low" => "0.2",
            "medium" => "0.7",
            "xhigh" | "max" => "0.99",
            _ => "0.9",
        };
        let mut ids = Vec::with_capacity(64 + user_text.len() / 2);
        ids.extend_from_slice(&[TOK_MSG_SYSTEM, TOK_CONTENT_TEXT]);
        ids.extend(self.encode_ordinary(&format!("Thinking effort level: {effort_val}")));
        ids.push(TOK_END_MESSAGE);

        ids.extend_from_slice(&[TOK_MSG_USER, TOK_CONTENT_TEXT]);
        ids.extend(self.encode_ordinary(user_text));
        ids.push(TOK_END_MESSAGE);
        ids
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for &t in tokens {
            let idx = t as usize;
            if idx < self.decoder.len() && !self.decoder[idx].is_empty() {
                bytes.extend_from_slice(&self.decoder[idx]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn decode_token_bytes(&self, token: u32) -> &[u8] {
        let idx = token as usize;
        if idx < self.decoder.len() {
            &self.decoder[idx]
        } else {
            &[]
        }
    }
}

/// Lightweight GPT-4 / o200k-compatible piece splitter (`[ \t]?[A-Za-z]+`, digits, non-alnum, newlines).
fn split_tiktoken_pieces(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut i = 0usize;

    while i < n {
        let start = chars[i].0;
        let c = chars[i].1;
        if c == ' '
            && i + 1 < n
            && (chars[i + 1].1.is_alphabetic() || !chars[i + 1].1.is_whitespace())
        {
            i += 1;
            let c2 = chars[i].1;
            if c2.is_alphabetic() {
                while i < n && chars[i].1.is_alphabetic() {
                    i += 1;
                }
            } else if c2.is_ascii_digit() {
                let mut d = 0;
                while i < n && chars[i].1.is_ascii_digit() && d < 3 {
                    i += 1;
                    d += 1;
                }
            } else {
                i += 1;
            }
        } else if c.is_alphabetic() {
            while i < n && chars[i].1.is_alphabetic() {
                i += 1;
            }
        } else if c.is_ascii_digit() {
            let mut d = 0;
            while i < n && chars[i].1.is_ascii_digit() && d < 3 {
                i += 1;
                d += 1;
            }
        } else if c.is_whitespace() {
            while i < n && chars[i].1.is_whitespace() {
                i += 1;
            }
        } else {
            i += 1;
        }
        let end = if i < n { chars[i].0 } else { text.len() };
        pieces.push(&text[start..end]);
    }
    pieces
}

fn decode_base64(input: &str) -> Option<Vec<u8>> {
    fn val(b: u8) -> Option<u8> {
        match b {
            b'A'..=b'Z' => Some(b - b'A'),
            b'a'..=b'z' => Some(b - b'a' + 26),
            b'0'..=b'9' => Some(b - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes = input.as_bytes();
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut idx = 0;
    while idx + 4 <= bytes.len() {
        let chunk = &bytes[idx..idx + 4];
        idx += 4;
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        let c = if chunk[2] == b'=' { 0 } else { val(chunk[2])? };
        let d = if chunk[3] == b'=' { 0 } else { val(chunk[3])? };
        out.push((a << 2) | (b >> 4));
        if chunk[2] != b'=' {
            out.push((b << 4) | (c >> 2));
        }
        if chunk[3] != b'=' {
            out.push((c << 6) | d);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_and_bpe_roundtrip() {
        let sample = "IQ== 0\nSGVs 1\nbG8= 2\nSGVsbG8= 3\nIA== 4\n";
        let bpe = TiktokenBpe::from_tiktoken_str(sample).unwrap();
        assert_eq!(bpe.encode_ordinary("Hello!"), vec![3, 0]);
        assert_eq!(bpe.decode_tokens(&[3, 0]), "Hello!");
    }
}
