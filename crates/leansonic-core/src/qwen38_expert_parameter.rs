//! Direct access to raw and dictionary-coded routed expert scale/bias planes.
use crate::qwen38_pack::Frame;
use crate::safetensors::ShardedSafeTensors;
use std::borrow::Cow;

pub(super) enum ExpertParameter<'a> {
    Raw(&'a [u8]),
    Packed(Frame<'a>),
}

impl<'a> ExpertParameter<'a> {
    pub(super) fn open(
        st: &'a ShardedSafeTensors,
        key: &str,
        shape: &[usize],
    ) -> Result<Self, String> {
        let words = shape.iter().product::<usize>();
        let packed = format!("{key}.q38dict1");
        match (st.get(key), st.get(&packed)) {
            (Some(raw), None)
                if raw.dtype == "BF16" && raw.shape == shape && raw.data.len() == words * 2 =>
            {
                Ok(Self::Raw(raw.data))
            }
            (None, Some(data)) if data.dtype == "U8" && data.shape == [data.data.len()] => {
                Frame::open(data.data, words)
                    .map(Self::Packed)
                    .map_err(|err| format!("{packed}: {err}"))
            }
            _ => Err(format!(
                "missing, ambiguous or malformed expert parameter {key}"
            )),
        }
    }

    /// Borrow raw words or decode just this aligned byte range into a temporary buffer.
    pub(super) fn bytes(&self, offset: usize, len: usize) -> Result<Cow<'a, [u8]>, String> {
        let end = offset.checked_add(len).ok_or("expert range overflow")?;
        if !(offset | len).is_multiple_of(2) {
            return Err("unaligned expert parameter range".into());
        }
        match self {
            Self::Raw(raw) => raw
                .get(offset..end)
                .map(Cow::Borrowed)
                .ok_or_else(|| "expert range out of bounds".into()),
            Self::Packed(frame) => Ok(Cow::Owned(frame.decode_range(offset / 2, len / 2)?)),
        }
    }
}
