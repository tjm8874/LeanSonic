//! Bit-Exact Reference & Dual-Regime Kernel Executors (`0-ULP BATCH IDENTICAL`).
//!
//! Mirrors `kernels/moe_dual_regime.cu`, `kernels/attn_blocklist_tc.cu`, and
//! `kernels/loop_residual_blend.cu` with identical IEEE-754 `f32` parenthesization
//! so `C = 1` solo execution and `C = 64` batched execution produce bit-identical `u32` bits.

use crate::generated_spec::{
    E2M1_DOUBLED_INT8_LUT, MAX_BATCH_ROWS, MOE_GROUPED_MMA_MIN_PAIRS, SPEC_VERIFY_WINDOW,
    SWA_WINDOW_TOKENS,
};
use crate::weights::PlanarNvfp4Experts;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeExecutionRegime {
    /// `< 128` routed `(expert, token)` pairs -> `REG:64` `4 CTAs/SM` SIMT (`64 KiB` L1).
    SimtReg64,
    /// `>= 128` routed `(expert, token)` pairs -> `2x E2M1 -> int8` Grouped Tensor Core MMA.
    GroupedMmaInt8,
}

#[inline]
pub fn select_moe_regime(num_routed_pairs: usize) -> MoeExecutionRegime {
    if num_routed_pairs < MOE_GROUPED_MMA_MIN_PAIRS {
        MoeExecutionRegime::SimtReg64
    } else {
        MoeExecutionRegime::GroupedMmaInt8
    }
}

/// Lossless `2x E2M1 -> i8` lookup verified in `LeanSonic.Proofs.e2m1_all_16_codes_exact`.
#[inline]
pub fn decode_e2m1_doubled_s8(nibble: u8) -> i8 {
    E2M1_DOUBLED_INT8_LUT[(nibble & 0x0F) as usize]
}

/// Non-associative IEEE-754 butterfly sum (`(p0 + p1) + (p2 + p3)`).
/// Prevent compiler re-association via `black_box`-guarded `f32` boundaries.
#[inline]
pub fn butterfly_fadd4(p0: f32, p1: f32, p2: f32, p3: f32) -> f32 {
    let lo = std::hint::black_box(p0 + p1);
    let hi = std::hint::black_box(p2 + p3);
    std::hint::black_box(lo + hi)
}

/// Quantize an `f32` vector (`len % 32 == 0`) into Planar `Q8` (`i8` codes + `f32` scales).
pub fn quantize_planar_q8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(x.len() % 32, 0);
    let k_blocks = x.len() / 32;
    let mut codes = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; k_blocks];

    for kb in 0..k_blocks {
        let slice = &x[kb * 32..(kb + 1) * 32];
        let amax = slice.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = amax / 127.0;
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        scales[kb] = scale;
        for i in 0..32 {
            let q = (slice[i] * inv).round().clamp(-127.0, 127.0) as i8;
            codes[kb * 32 + i] = q;
        }
    }
    (codes, scales)
}

/// Execute Dual-Regime Planar-SoA MoE projection (`SimtReg64` vs `GroupedMmaInt8`).
/// Both regimes share the exact `16B x 2` butterfly reduction tree proven in
/// `LeanSonic.Proofs.simt_and_mma_16bx2_0ulp_eq`, guaranteeing `0-ULP` batch invariance!
pub fn moe_dual_regime_forward(
    x_q8_rows: &[Vec<i8>],
    x_scale_rows: &[Vec<f32>],
    weights: &PlanarNvfp4Experts,
    routed_pairs: &[(usize, usize)], // `(expert_id, token_row)`
) -> (MoeExecutionRegime, Vec<Vec<f32>>) {
    let regime = select_moe_regime(routed_pairs.len());
    let mut outputs = Vec::with_capacity(routed_pairs.len());

    for &(expert_id, tok_row) in routed_pairs {
        let x_codes = &x_q8_rows[tok_row];
        let x_scales = &x_scale_rows[tok_row];
        let mut row_out = vec![0.0f32; weights.n_rows];

        for (r, dst) in row_out.iter_mut().enumerate() {
            let mut acc_total = 0.0f32;
            for kb in 0..weights.k_blocks {
                let w_bytes = weights.block_codes(expert_id, r, kb);
                let x_blk = &x_codes[kb * 32..(kb + 1) * 32];

                let mut q_partials = [0.0f32; 4];
                for q in 0..4 {
                    let mut acc8: i32 = 0;
                    for j in 0..4 {
                        let packed = w_bytes[q * 4 + j];
                        let w_lo = decode_e2m1_doubled_s8(packed & 0x0F) as i32;
                        let w_hi = decode_e2m1_doubled_s8(packed >> 4) as i32;
                        acc8 += w_lo * (x_blk[q * 8 + j] as i32);
                        acc8 += w_hi * (x_blk[q * 8 + 4 + j] as i32);
                    }
                    q_partials[q] = std::hint::black_box((acc8 as f32) * 0.5);
                }

                let blk_sum = match regime {
                    MoeExecutionRegime::SimtReg64 => {
                        butterfly_fadd4(q_partials[0], q_partials[1], q_partials[2], q_partials[3])
                    }
                    MoeExecutionRegime::GroupedMmaInt8 => {
                        // `16B x 2` half-block MMA accumulation (`(p0 + p1) + (p2 + p3)`)
                        let half_lo = std::hint::black_box(q_partials[0] + q_partials[1]);
                        let half_hi = std::hint::black_box(q_partials[2] + q_partials[3]);
                        std::hint::black_box(half_lo + half_hi)
                    }
                };
                let scale =
                    std::hint::black_box(x_scales[kb] * weights.block_scale(expert_id, r, kb));
                acc_total = std::hint::black_box(acc_total + std::hint::black_box(blk_sum * scale));
            }
            *dst = acc_total;
        }
        outputs.push(row_out);
    }

    (regime, outputs)
}

/// Pure-Rust IEEE-754 round-to-nearest-even `f32 -> f16 -> f32` roundtrip for 22-bit query splitting.
pub fn round_f32_via_ieee_f16(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        return x;
    }
    let unbiased = exp - 127;
    if unbiased < -24 {
        return f32::from_bits(sign);
    }
    if unbiased > 15 {
        return f32::from_bits(sign | 0x7F80_0000);
    }
    if unbiased < -14 {
        // Subnormal f16
        let shift = (-14 - unbiased) as u32 + 13;
        let full_mant = mant | 0x0080_0000;
        let half_mant = (full_mant + (1 << (shift - 1))) >> shift;
        return (if sign != 0 { -1.0 } else { 1.0 }) * (half_mant as f32) * 2.0f32.powi(-24);
    }
    // Normal f16: round 23-bit mantissa to 10-bit (round-to-nearest-even)
    let round_bit = 1u32 << 12;
    let mask = (1u32 << 13) - 1;
    let half_odd = (mant >> 13) & 1;
    let rounded = mant + round_bit - 1 + half_odd;
    let new_bits = (bits & 0xFF80_0000).wrapping_add(rounded & !mask);
    f32::from_bits(new_bits)
}

/// Split `q_f32` into `(q_hi, q_lo)` in FP16 with `~22-bit` effective precision (`hi + lo / 2048`).
#[inline]
pub fn split_query_22bit_fp16(q_f32: f32) -> (f32, f32) {
    let hi = round_f32_via_ieee_f16(q_f32);
    let lo = round_f32_via_ieee_f16((q_f32 - hi) * 2048.0);
    (hi, lo)
}

/// Hybrid Speculative Window Policy (`W = 16`, Pre-Draft Skip, SWA Horizon Veto Override, Chaining).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DraftDecision {
    /// Batch rows saturated (`active_slots >= MAX_BATCH_ROWS`) -> skip GPU drafter (`0` wasted BW).
    SkipGpuAtBatchSaturation,
    /// Confident CPU suffix match (`len >= 16`) -> skip GPU neural drafter completely.
    UsePureSuffixSkipGpu { tokens: Vec<u32> },
    /// Override low-confidence SWA drafter token-0 veto when suffix match lies beyond `SWA = 1024`.
    OverrideSwaHorizonVetoWithSuffix { tokens: Vec<u32> },
    /// Chain neural draft (`novel_prefix`) into suffix continuation within a single `W = 16` window.
    HybridNeuralThenSuffix { tokens: Vec<u32> },
    /// Pure neural draft window (`<= W`).
    NeuralOnly { tokens: Vec<u32> },
}

pub fn select_speculative_window(
    active_slots: usize,
    neural_tokens: &[u32],
    neural_tok0_conf: f32,
    suffix_tokens: &[u32],
    suffix_match_distance: usize,
    suffix_converged_after_neural_idx: Option<usize>,
) -> DraftDecision {
    if active_slots >= MAX_BATCH_ROWS {
        return DraftDecision::SkipGpuAtBatchSaturation;
    }
    if suffix_tokens.len() >= SPEC_VERIFY_WINDOW {
        return DraftDecision::UsePureSuffixSkipGpu {
            tokens: suffix_tokens[..SPEC_VERIFY_WINDOW].to_vec(),
        };
    }
    if suffix_tokens.len() >= 8
        && suffix_match_distance > SWA_WINDOW_TOKENS
        && neural_tok0_conf < 0.85
    {
        return DraftDecision::OverrideSwaHorizonVetoWithSuffix {
            tokens: suffix_tokens
                .iter()
                .copied()
                .take(SPEC_VERIFY_WINDOW)
                .collect(),
        };
    }
    if let Some(split_idx) = suffix_converged_after_neural_idx {
        if split_idx >= 4 && split_idx < neural_tokens.len() && !suffix_tokens.is_empty() {
            let mut combined = neural_tokens[..split_idx].to_vec();
            for &t in suffix_tokens {
                if combined.len() >= SPEC_VERIFY_WINDOW {
                    break;
                }
                combined.push(t);
            }
            return DraftDecision::HybridNeuralThenSuffix { tokens: combined };
        }
    }
    DraftDecision::NeuralOnly {
        tokens: neural_tokens
            .iter()
            .copied()
            .take(SPEC_VERIFY_WINDOW)
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solo_simt_and_batched_grouped_mma_are_0ulp_bitwise_identical() {
        let weights = PlanarNvfp4Experts::deterministic_synthetic(777, 128, 16, 2);
        let target_x: Vec<f32> = (0..64).map(|i| ((i as f32) - 31.5) * 0.073).collect();
        let (q_codes, q_scales) = quantize_planar_q8(&target_x);

        // 1. Solo execution (`C = 1`, `8` top-k pairs < 128 -> `SimtReg64` regime)
        let solo_rows_c = vec![q_codes.clone()];
        let solo_rows_s = vec![q_scales.clone()];
        let solo_pairs: Vec<(usize, usize)> = (0..8).map(|e| (e * 7 % 128, 0)).collect();
        let (solo_regime, solo_out) =
            moe_dual_regime_forward(&solo_rows_c, &solo_rows_s, &weights, &solo_pairs);
        assert_eq!(solo_regime, MoeExecutionRegime::SimtReg64);

        // 2. High-concurrency execution (`C = 64`, `64 * 8 = 512 >= 128` pairs -> `GroupedMmaInt8`)
        let mut batch_rows_c = Vec::with_capacity(64);
        let mut batch_rows_s = Vec::with_capacity(64);
        for slot in 0..64 {
            if slot == 37 {
                batch_rows_c.push(q_codes.clone());
                batch_rows_s.push(q_scales.clone());
            } else {
                let other: Vec<f32> = (0..64).map(|i| ((i + slot) as f32 * 0.13).sin()).collect();
                let (oc, os) = quantize_planar_q8(&other);
                batch_rows_c.push(oc);
                batch_rows_s.push(os);
            }
        }
        let mut batch_pairs = Vec::with_capacity(64 * 8);
        for slot in 0..64 {
            for e in 0..8 {
                let exp = if slot == 37 {
                    e * 7 % 128
                } else {
                    (slot + e * 13) % 128
                };
                batch_pairs.push((exp, slot));
            }
        }
        let (batch_regime, batch_out) =
            moe_dual_regime_forward(&batch_rows_c, &batch_rows_s, &weights, &batch_pairs);
        assert_eq!(batch_regime, MoeExecutionRegime::GroupedMmaInt8);

        // Verify 0-ULP (`to_bits()` exact equality) for slot 37 across all 8 routed experts!
        for e in 0..8 {
            let solo_vec = &solo_out[e];
            let batch_vec = &batch_out[37 * 8 + e];
            for (a, b) in solo_vec.iter().zip(batch_vec.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "0-ULP violation: solo={} ({:#x}) != batch={} ({:#x})",
                    a,
                    a.to_bits(),
                    b,
                    b.to_bits()
                );
            }
        }
    }

    #[test]
    fn query_22bit_split_recovers_fp32_to_sub_1e6_error() {
        for i in 1..200 {
            let q = (i as f32 * 0.31415927).sin() * 3.7;
            let (hi, lo) = split_query_22bit_fp16(q);
            let recon = hi + lo * (1.0 / 2048.0);
            assert!((q - recon).abs() < 2e-6, "q={q}, recon={recon}");
        }
    }

    #[test]
    fn speculative_policy_handles_saturation_horizon_and_chaining() {
        assert_eq!(
            select_speculative_window(64, &[1, 2], 0.9, &[], 0, None),
            DraftDecision::SkipGpuAtBatchSaturation
        );
        let long_suf: Vec<u32> = (100..116).collect();
        assert!(matches!(
            select_speculative_window(1, &[1, 2], 0.95, &long_suf, 500, None),
            DraftDecision::UsePureSuffixSkipGpu { .. }
        ));
        assert!(matches!(
            select_speculative_window(
                1,
                &[1, 2, 3, 4],
                0.60,
                &[9, 10, 11, 12, 13, 14, 15, 16],
                4096,
                None
            ),
            DraftDecision::OverrideSwaHorizonVetoWithSuffix { .. }
        ));
        if let DraftDecision::HybridNeuralThenSuffix { tokens } = select_speculative_window(
            1,
            &[10, 11, 12, 13, 14, 15],
            0.92,
            &[20, 21, 22, 23, 24],
            200,
            Some(4),
        ) {
            assert_eq!(tokens, vec![10, 11, 12, 13, 20, 21, 22, 23, 24]);
        } else {
            panic!("expected HybridNeuralThenSuffix");
        }
    }
}
