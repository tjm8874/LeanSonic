//! LeanSonic Core Runtime (Phase 1 Vertical Slice)
//!
//! Zero-dependency minimalist Rust runtime governed by Lean 4 formal specifications
//! (`generated_spec.rs`), unifying Looped MoE (`Inkling-2-mini` / `Inkling-2-Mid`)
//! and Hybrid GDN + QSA + Engram MoE (`Qwen3.8-Flash-Next` / `Qwen4`) on NVIDIA Blackwell.

pub mod bytecode;
pub mod engine;
pub mod generated_spec;
pub mod kernels;
pub mod offload;
pub mod serving;
pub mod state_pool;
pub mod weights;
