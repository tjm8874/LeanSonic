//! Storage for the Qwen3.8 host Engram (PLE n-gram tables, ~29.8 GiB).
//!
//! The tables are looked up row by row on the host for every token. Served from the model's
//! file mapping, rows that are not in the page cache fault in from disk, and other processes can
//! evict them mid-run (#261). [`EngramPolicy::Resident`] copies them into owned host memory once
//! at load so lookups never touch the file again.

use crate::safetensors::{MappedTensorRange, ShardedSafeTensors};

/// One Engram tensor: a range of the model mapping, or an owned copy of it.
pub(crate) enum EngramBytes {
    Mapped(MappedTensorRange),
    Owned(Box<[u8]>),
}

impl EngramBytes {
    pub(crate) fn get<'a>(&'a self, st: &'a ShardedSafeTensors) -> &'a [u8] {
        match self {
            Self::Mapped(range) => st.bytes_at(range),
            Self::Owned(bytes) => bytes,
        }
    }
}

/// Where the Engram lives after load (`LEANSONIC_Q38_ENGRAM=resident|mmap`, default `auto`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EngramPolicy {
    Resident,
    Mapped,
}

/// Headroom kept free after a resident copy under `auto`.
const AUTO_HEADROOM: usize = 24 << 30;

impl EngramPolicy {
    /// `auto` copies when `MemAvailable` covers the tables plus [`AUTO_HEADROOM`].
    pub(crate) fn choose(env: Option<&str>, engram_bytes: usize, mem_available: usize) -> Self {
        match env.map(str::trim) {
            Some("resident") => Self::Resident,
            Some("mmap") => Self::Mapped,
            _ if mem_available >= engram_bytes.saturating_add(AUTO_HEADROOM) => Self::Resident,
            _ => Self::Mapped,
        }
    }
}

/// Copies every mapped entry into owned memory, `threads` tensors at a time.
pub(crate) fn make_resident(
    st: &ShardedSafeTensors,
    entries: &mut [&mut EngramBytes],
    threads: usize,
) {
    let threads = threads.max(1);
    let mut rest: &mut [&mut EngramBytes] = entries;
    while !rest.is_empty() {
        let n = threads.min(rest.len());
        let (batch, tail) = rest.split_at_mut(n);
        std::thread::scope(|scope| {
            for e in batch.iter_mut() {
                scope.spawn(move || {
                    if let EngramBytes::Mapped(range) = &**e {
                        let owned: Box<[u8]> = st.bytes_at(range).into();
                        **e = EngramBytes::Owned(owned);
                    }
                });
            }
        });
        rest = tail;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_respects_env_and_headroom() {
        let g = 1usize << 30;
        assert_eq!(
            EngramPolicy::choose(Some("mmap"), 30 * g, 500 * g),
            EngramPolicy::Mapped
        );
        assert_eq!(
            EngramPolicy::choose(Some("resident"), 30 * g, g),
            EngramPolicy::Resident
        );
        assert_eq!(
            EngramPolicy::choose(None, 30 * g, 54 * g),
            EngramPolicy::Resident
        );
        assert_eq!(
            EngramPolicy::choose(None, 30 * g, 53 * g),
            EngramPolicy::Mapped
        );
        assert_eq!(
            EngramPolicy::choose(Some("auto"), 30 * g, 10 * g),
            EngramPolicy::Mapped
        );
    }
}
