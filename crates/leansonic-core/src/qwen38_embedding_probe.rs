//! Optional input embedding tables for comparing precision against the same Q4 engine.
use super::*;

pub(super) struct InputEmbeddingProbe {
    weights: PinnedHostBuf,
    scales: Option<PinnedHostBuf>,
}

impl InputEmbeddingProbe {
    pub(super) fn load(
        ctx: &Arc<CudaContextHandle>,
        st: &ShardedSafeTensors,
    ) -> Result<Option<Self>, String> {
        let prefix = "language_model.model.embed_tokens";
        let bf = st.get(&format!("{prefix}.probe_bf16"));
        let q8 = st.get(&format!("{prefix}.probe_q8"));
        let sc = st.get(&format!("{prefix}.probe_q8_scales"));
        let pinned = |data: &[u8]| -> Result<PinnedHostBuf, String> {
            let buf = PinnedHostBuf::alloc_uninit(ctx, data.len())?;
            parallel_copy_into_pinned(buf.host_ptr, &[(0, data)]);
            Ok(buf)
        };
        match (bf, q8, sc) {
            (None, None, None) => Ok(None),
            (Some(w), None, None)
                if w.dtype == "BF16"
                    && w.shape == [Q38_VOCAB, Q38_D]
                    && w.data.len() == 2 * Q38_VOCAB * Q38_D =>
            {
                Ok(Some(Self {
                    weights: pinned(w.data)?,
                    scales: None,
                }))
            }
            (None, Some(w), Some(s))
                if w.dtype == "I8"
                    && w.shape == [Q38_VOCAB, Q38_D]
                    && w.data.len() == Q38_VOCAB * Q38_D
                    && s.dtype == "BF16"
                    && s.shape == [Q38_VOCAB, Q38_D / 32]
                    && s.data.len() == Q38_VOCAB * Q38_D / 16 =>
            {
                Ok(Some(Self {
                    weights: pinned(w.data)?,
                    scales: Some(pinned(s.data)?),
                }))
            }
            _ => Err("malformed or ambiguous input embedding probe tensors".into()),
        }
    }

    pub(super) fn kernel(&self) -> &'static str {
        if self.scales.is_some() {
            "q38_embed_probe_q8"
        } else {
            "q38_embed_probe_bf16"
        }
    }

    pub(super) fn pointers(&self) -> (CUdeviceptr, CUdeviceptr, CUdeviceptr) {
        (
            self.weights.dev_ptr,
            self.scales.as_ref().map_or(0, |s| s.dev_ptr),
            0,
        )
    }
}
