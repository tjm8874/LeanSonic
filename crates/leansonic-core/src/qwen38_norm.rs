//! In flashnext-norm-zero the converter unfolded 147 target norms, but left
//! the final mixer and all nine drafter norms folded. Each digest below
//! identifies BF16(1 + official weight) for every element of the named tensor.
//! Check each weight independently so partial repairs and repacks work too.

pub(crate) fn offset(name: &str, bytes: &[u8], zero_centered: bool) -> f32 {
    if !zero_centered {
        return 0.0;
    }
    let folded_digest = match name {
        "language_model.model.hyper_connection_mixer.hc_norm.weight" => {
            "8224a458ee9fb93bbf6360f04c7448fb97ca235419fa4a2eae31d5894d919525"
        }
        "language_model.mtp.hyper_connection_mixer.hc_norm.weight" => {
            "6c17016a5bac51d121b7d3fcc450264f572759247bddb263dbeea9ce9106e205"
        }
        "language_model.mtp.layers.0.attn_hyper_connection.hc_norm.weight" => {
            "c7598e20e45060cd09f2a66d1dc963c33a09088a381d9abb8188edb0bb5a968b"
        }
        "language_model.mtp.layers.0.mlp_hyper_connection.hc_norm.weight" => {
            "5fc7c488681dfd8c1ac75a54c7e1b9388732c990c474c2958baa1cae42a9f644"
        }
        "language_model.mtp.layers.0.self_attn.indexer.k_layernorm.weight" => {
            "1df727089de8f687f68ac3809ff8421f7712905a561c19c29dc020b91986673f"
        }
        "language_model.mtp.layers.0.self_attn.indexer.q_layernorm.weight" => {
            "8004a66dbce5305582e1a5f5a9ceecfc63c81ea341468ffa1148ad38c3e666f4"
        }
        "language_model.mtp.layers.0.self_attn.k_norm.weight" => {
            "c705c3576a2a192de15f06b4f1c09506a96177fbc3ee2deb101c93db7c95fd87"
        }
        "language_model.mtp.layers.0.self_attn.q_norm.weight" => {
            "5391d239a59417ed155c004e812f1e9528ec51e0aa35376c55a201e241d03852"
        }
        "language_model.mtp.pre_fc_norm_embedding.weight" => {
            "636cac841613be0a8acdf8cb2d11d302348a62b123c6ab6c1b87a90a8cabc27a"
        }
        "language_model.mtp.pre_fc_norm_hidden.weight" => {
            "b70b2aeda944afbc5447bcdd6d32bdf826b15f1500c91bc4f5a87b74dbdf9782"
        }
        _ => return 1.0,
    };
    if crate::sha256::sha256_hex(bytes) == folded_digest {
        0.0
    } else {
        1.0
    }
}
