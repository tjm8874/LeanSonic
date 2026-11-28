//! Zero-Dependency `tiktoken` BPE Encoder/Decoder & `Inkling-2-Mini` Chat Formatter.
//!
//! Reads `<model_dir>/tiktoken/tokenizer.model` (`<base64> <rank>` lines, 199,998 BPE merges)
//! with zero external crates and formats chat messages identically to `chat_template.jinja`.

mod pieces;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
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
    encoder: HashMap<Box<[u8]>, u32>,
    decoder: Vec<std::ops::Range<usize>>,
    token_bytes: Vec<u8>,
}

impl TiktokenBpe {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = model_dir.as_ref();
        let mut candidates = vec![
            dir.join("tiktoken/tokenizer.model"),
            dir.join("tokenizer.model"),
        ];
        for sub in ["32g", "16g"] {
            candidates.push(dir.join(sub).join("tiktoken/tokenizer.model"));
            candidates.push(dir.join(sub).join("tokenizer.model"));
        }
        if matches!(
            dir.file_name().and_then(|s| s.to_str()),
            Some("16g" | "32g")
        ) {
            if let Some(parent) = dir.parent() {
                candidates.push(parent.join("tiktoken/tokenizer.model"));
                candidates.push(parent.join("tokenizer.model"));
            }
        }
        let path = candidates
            .into_iter()
            .find(|p| p.exists())
            .unwrap_or_else(|| dir.join("tokenizer.model"));
        let raw = fs::read_to_string(path)?;
        Self::from_tiktoken_str(&raw)
    }

    pub fn from_tiktoken_str(content: &str) -> io::Result<Self> {
        // The official vocabulary has just under 200k entries. Bound the hint for small files.
        let count = (content.len() / 8).min(200_100);
        let mut encoder = HashMap::with_capacity(count);
        let mut decoder = Vec::with_capacity(count);
        let mut token_bytes = Vec::new();

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
            let start = token_bytes.len();
            token_bytes.extend_from_slice(&bytes);
            let index = rank as usize;
            if index >= decoder.len() {
                decoder.resize(index + 1, 0..0);
            }
            decoder[index] = start..token_bytes.len();
            encoder.insert(bytes.into_boxed_slice(), rank);
        }

        Ok(Self {
            encoder,
            decoder,
            token_bytes,
        })
    }

    /// Standard byte-pair merge over pre-segmented chunks (whitespace/word/punct pieces).
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for piece in pieces::split(text) {
            self.bpe_merge_piece(piece.as_bytes(), &mut out);
        }
        out
    }

    fn bpe_merge_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        if let Some(&id) = self.encoder.get(piece) {
            out.push(id);
            return;
        }
        // Re-scanning every remaining pair after each merge is cheap for short
        // words, but quadratic for long punctuation, whitespace and letter runs.
        if piece.len() >= 128 {
            self.merge_long_piece(piece, out);
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

    fn merge_long_piece(&self, piece: &[u8], out: &mut Vec<u32>) {
        // Each live token spans [start, next[start]). Offsets never move.
        // Order candidates by (rank, start) to break ties at the leftmost pair.
        let n = piece.len();
        let mut next: Vec<usize> = (1..=n).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| i.saturating_sub(1)).collect();
        let candidate = |left: usize, mid: usize, end: usize| {
            self.encoder
                .get(&piece[left..end])
                .map(|&rank| Reverse((rank, left, mid, end)))
        };
        let initial: Vec<_> = (0..n - 1)
            .filter_map(|i| candidate(i, i + 1, i + 2))
            .collect();
        let mut queue = BinaryHeap::from(initial);
        while let Some(Reverse((_, left, mid, end))) = queue.pop() {
            if next[left] != mid || next[mid] != end {
                continue;
            }
            next[left] = end;
            next[mid] = mid; // invalidate all queued pairs involving the removed token
            if end < n {
                prev[end] = left;
                queue.extend(candidate(left, end, next[end]));
            }
            if left > 0 {
                queue.extend(candidate(prev[left], left, end));
            }
        }
        let mut start = 0;
        while start < n {
            let end = next[start];
            if let Some(&id) = self.encoder.get(&piece[start..end]) {
                out.push(id);
            }
            start = end;
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
            bytes.extend_from_slice(self.decode_token_bytes(t));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn decode_token_bytes(&self, token: u32) -> &[u8] {
        self.decoder
            .get(token as usize)
            .map_or(&[], |range| &self.token_bytes[range.clone()])
    }
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
    let padding =
        usize::from(bytes[bytes.len() - 1] == b'=') + usize::from(bytes[bytes.len() - 2] == b'=');
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3 - padding);
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
    fn decode_sparse_unordered_ranks_and_split_utf8() {
        let bpe = TiktokenBpe::from_tiktoken_str("gg== 8\n4w== 2\ngQ== 5\n").unwrap();
        assert_eq!(bpe.decode_tokens(&[2, 5, 8]), "あ");
        assert!(bpe.decode_token_bytes(3).is_empty());
        assert!(bpe.decode_token_bytes(99).is_empty());
    }

    #[test]
    fn official_ids_for_punctuation_before_a_newline() {
        // Every entry from tokenizer.model that is a substring of this input.
        let ranks = "IQ== 0\nLw== 14\nYQ== 64\nYg== 65\nYw== 66\nCg== 198\nYWI= 332\n\
                     YmM= 31491\nYWJj 33782\nLy8v 69984\nISE= 70042\nISEh 84950\nLy8= 145055\n";
        let bpe = TiktokenBpe::from_tiktoken_str(ranks).unwrap();
        assert_eq!(
            bpe.encode_ordinary("abc!!!///\n"),
            [33782, 84950, 69984, 198]
        );
    }

    #[test]
    fn long_merges_match_leftmost_rank_order() {
        let bpe = TiktokenBpe::from_tiktoken_str(
            "YQ== 0\nYg== 1\nYWI= 2\nYWE= 3\nYmE= 4\nYWJh 5\nYWFhYQ== 6\n",
        )
        .unwrap();
        for n in 2..=12 {
            for mask in 0..(1 << n) {
                let bytes: Vec<u8> = (0..n)
                    .map(|bit| if mask & (1 << bit) == 0 { b'a' } else { b'b' })
                    .collect();
                if bpe.encoder.contains_key(bytes.as_slice()) {
                    continue; // whole-piece vocabulary hits bypass both merge algorithms
                }
                let mut want = Vec::new();
                bpe.bpe_merge_piece(&bytes, &mut want);
                let mut result = Vec::new();
                bpe.merge_long_piece(&bytes, &mut result);
                assert_eq!(result, want);
            }
        }
        assert_eq!(
            bpe.encode_ordinary(&"a".repeat(257)),
            [vec![6; 64], vec![0]].concat()
        );
    }

    #[test]
    fn base64_and_bpe_roundtrip() {
        let sample = "IQ== 0\nSGVs 1\nbG8= 2\nSGVsbG8= 3\nIA== 4\n";
        let bpe = TiktokenBpe::from_tiktoken_str(sample).unwrap();
        assert_eq!(bpe.encode_ordinary("Hello!"), vec![3, 0]);
        assert_eq!(bpe.decode_tokens(&[3, 0]), "Hello!");
    }
}
