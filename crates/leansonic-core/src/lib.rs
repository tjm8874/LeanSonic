//! LeanSonic Core Runtime (Phase 1 Vertical Slice)
//!
//! Zero-dependency minimalist Rust runtime governed by Lean 4 formal specifications
//! (`generated_spec.rs`), unifying Looped MoE (`Inkling-2-mini` / `Inkling-2-Mid`)
//! and Hybrid GDN + QSA + Engram MoE (`Qwen3.8-Flash-Next` / `Qwen4`) on NVIDIA Blackwell.

pub mod bytecode;
pub mod cuda_ffi;
pub mod engine;
pub mod generated_spec;
pub mod inkling2_dflash2;
pub mod inkling2_gpu;
pub mod inkling2_mtp;
pub mod inkling_bf16_planes;
pub mod inkling_embedding;
pub mod inkling_mxfp8;
pub mod inkling_q8_scales;
pub mod inkling_scales;
mod json_string;
pub mod kernels;
pub mod offload;
pub mod qwen38_gpu;
pub mod qwen38_mtp;
pub mod qwen_bpe;
pub mod safetensors;
pub mod serving;
pub mod state_pool;
pub mod tiktoken;
pub mod weights;
