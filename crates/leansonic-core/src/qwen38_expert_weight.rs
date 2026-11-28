//! Independently readable lossless Q4 weight records for routed experts.
use super::{EX_W_BYTES, Q38_EXP};
use crate::safetensors::ShardedSafeTensors;
use std::borrow::Cow;

pub(super) enum ExpertWeight<'a> {
    Raw(&'a [u8]),
    #[cfg(feature = "q38-zstd-weights")]
    Packed(Records<'a>),
}

impl<'a> ExpertWeight<'a> {
    pub(super) fn open(
        st: &'a ShardedSafeTensors,
        key: &str,
        shape: &[usize],
    ) -> Result<Self, String> {
        let packed_key = format!("{key}.q38wz1");
        match (st.get(key), st.get(&packed_key)) {
            (Some(t), None)
                if t.dtype == "U32" && t.shape == shape && t.data.len() == Q38_EXP * EX_W_BYTES =>
            {
                Ok(Self::Raw(t.data))
            }
            (None, Some(t)) if t.dtype == "U8" && t.shape == [t.data.len()] => {
                #[cfg(feature = "q38-zstd-weights")]
                {
                    Records::open(t.data)
                        .map(Self::Packed)
                        .map_err(|e| format!("{packed_key}: {e}"))
                }
                #[cfg(not(feature = "q38-zstd-weights"))]
                {
                    Err(format!(
                        "{packed_key} requires the q38-zstd-weights feature"
                    ))
                }
            }
            _ => Err(format!(
                "missing, ambiguous or malformed expert weight {key}"
            )),
        }
    }

    pub(super) fn whole(&self) -> Result<Cow<'a, [u8]>, String> {
        match self {
            Self::Raw(data) => Ok(Cow::Borrowed(data)),
            #[cfg(feature = "q38-zstd-weights")]
            Self::Packed(records) => {
                let mut out = vec![0; Q38_EXP * EX_W_BYTES];
                let workers = std::thread::available_parallelism()
                    .map_or(1, usize::from)
                    .min(8);
                let group = Q38_EXP.div_ceil(workers);
                std::thread::scope(|scope| {
                    let jobs: Vec<_> = out
                        .chunks_mut(group * EX_W_BYTES)
                        .enumerate()
                        .map(|(i, dest)| {
                            scope.spawn(move || {
                                for (j, record) in dest.chunks_exact_mut(EX_W_BYTES).enumerate() {
                                    records.decode(i * group + j, record)?;
                                }
                                Ok::<_, String>(())
                            })
                        })
                        .collect();
                    for job in jobs {
                        job.join().map_err(|_| "expert weight worker panicked")??;
                    }
                    Ok::<_, String>(())
                })?;
                Ok(Cow::Owned(out))
            }
        }
    }

    pub(super) fn copy_record(&self, expert: usize, dest: &mut [u8]) -> Result<(), String> {
        if expert >= Q38_EXP || dest.len() != EX_W_BYTES {
            return Err("invalid expert record range".into());
        }
        match self {
            Self::Raw(data) => {
                dest.copy_from_slice(&data[expert * EX_W_BYTES..(expert + 1) * EX_W_BYTES])
            }
            #[cfg(feature = "q38-zstd-weights")]
            Self::Packed(records) => return records.decode(expert, dest),
        }
        Ok(())
    }
}

#[cfg(feature = "q38-zstd-weights")]
pub(super) struct Records<'a> {
    offsets: &'a [u8],
    payload: &'a [u8],
}

#[cfg(feature = "q38-zstd-weights")]
impl<'a> Records<'a> {
    fn open(data: &'a [u8]) -> Result<Self, String> {
        const END: usize = 16 + 4 * (Q38_EXP + 1);
        if data.len() < END
            || &data[..8] != b"Q38WZ001"
            || u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize != Q38_EXP
            || u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize != EX_W_BYTES
        {
            return Err("invalid expert weight header".into());
        }
        let records = Self {
            offsets: &data[16..END],
            payload: &data[END..],
        };
        if records.offset(0) != 0
            || records.offset(Q38_EXP) != records.payload.len()
            || (0..Q38_EXP).any(|i| {
                !(1..=EX_W_BYTES).contains(&records.offset(i + 1).saturating_sub(records.offset(i)))
            })
        {
            return Err("invalid expert weight record index".into());
        }
        Ok(records)
    }

    fn offset(&self, i: usize) -> usize {
        u32::from_le_bytes(self.offsets[i * 4..i * 4 + 4].try_into().unwrap()) as usize
    }

    fn decode(&self, i: usize, out: &mut [u8]) -> Result<(), String> {
        let src = &self.payload[self.offset(i)..self.offset(i + 1)];
        if src.len() == EX_W_BYTES {
            out.copy_from_slice(src);
        } else {
            #[link(name = "zstd")]
            unsafe extern "C" {
                fn ZSTD_decompress(
                    dst: *mut u8,
                    dst_size: usize,
                    src: *const u8,
                    src_size: usize,
                ) -> usize;
            }
            // Both slices remain alive for this synchronous call. libzstd checks the
            // frame checksum and returns the decoded length or a size_t error code.
            let n =
                unsafe { ZSTD_decompress(out.as_mut_ptr(), out.len(), src.as_ptr(), src.len()) };
            if n != out.len() {
                return Err(format!("invalid Zstandard expert weight record {i}"));
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "q38-zstd-weights"))]
mod tests {
    use super::*;

    fn synthetic() -> Vec<u8> {
        let mut bytes = b"Q38WZ001".to_vec();
        bytes.extend((Q38_EXP as u32).to_le_bytes());
        bytes.extend((EX_W_BYTES as u32).to_le_bytes());
        for offset in 0..=Q38_EXP {
            bytes.extend((offset as u32).to_le_bytes());
        }
        bytes.resize(bytes.len() + Q38_EXP, 0);
        bytes
    }

    #[test]
    fn rejects_bad_index_and_corrupt_record() {
        let data = synthetic();
        let mut out = vec![0; EX_W_BYTES];
        assert!(Records::open(&data).unwrap().decode(511, &mut out).is_err());
        assert!(Records::open(&data[..data.len() - 1]).is_err());
        let mut invalid = data;
        invalid[20..24].copy_from_slice(&0u32.to_le_bytes());
        assert!(Records::open(&invalid).is_err());
    }

    #[test]
    fn raw_fallback_reads_last_record() {
        let mut data = synthetic();
        let end = EX_W_BYTES + Q38_EXP - 1;
        data[16 + Q38_EXP * 4..20 + Q38_EXP * 4].copy_from_slice(&(end as u32).to_le_bytes());
        let start = data.len() - 1;
        data.resize(start + EX_W_BYTES, 91);
        data[start] = 91;
        let mut out = vec![0; EX_W_BYTES];
        Records::open(&data)
            .unwrap()
            .decode(Q38_EXP - 1, &mut out)
            .unwrap();
        assert!(out.iter().all(|&b| b == 91));
    }
}
