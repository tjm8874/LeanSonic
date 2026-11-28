import LeanSonic.Axioms

namespace LeanSonic.Proofs.Qwen38DenseQ8
open LeanSonic

/-- Static SM120a launch bounds for the measured full Q8 cubin. -/
private def launch (name : String) (threads regs shared ctas l1 : Nat) : KernelResourceSpec :=
  { name := name, threadsPerCta := threads, rawRegsPerThread := regs,
    userSmemPerCtaBytes := shared, smemAlignPadBytes := 0, ctasPerSm := ctas,
    clusterCtas := 1, minRequiredL1KiB := l1, hasStackSpill := false, hasHotLoopLibmCall := false }

def kernels : List KernelResourceSpec := [
  launch "q38_dense_q8_gemm_b4" 256 40 0 6 120,
  launch "q38_dense_q8_gemv" 256 38 0 6 120,
  launch "q38_hc_q8_down_inject" 256 39 0 6 120,
  launch "q38_hc_q8_up_mix" 256 42 0 5 120,
  launch "q38_embed_q8_hc4" 256 14 0 6 120,
  -- Dynamic shared memory for the largest supported shared-expert width, 2,560.
  launch "q38_shared_q8_projection" 256 40 11520 5 64,
  launch "q38_moe_up_swiglu11" 256 46 11552 5 64,
  launch "q38_moe_up_swiglu11_z24" 256 46 11552 5 64,
  launch "q38_moe_up_swiglu11_z24_bn" 256 48 11552 5 64,
  launch "q38_moe_down_combine11" 256 40 31680 2 64,
  launch "q38_moe_down_combine11_z24" 256 64 31680 2 64,
  launch "q38_moe_down_combine11_z24_bn" 256 62 31680 2 64,
  launch "q38_moe_up_grouped_z24" 512 56 46336 2 28,
  launch "q38_moe_down_grouped_z24" 256 96 23040 2 64]

theorem launch_resources_valid : kernels.all checkSm120Kernel = true := by decide

/-- Each selected embedding code word and its BF16 scale belong to the current token row. -/
theorem embedding_offsets (col : Fin 2560) :
    col.val / 4 < 640 ∧ col.val / 32 < 80 := by
  have := col.isLt
  omega

/-- The injection branch follows the 320 nonlinear rows and produces exactly four outputs. -/
theorem injection_offset (row : Nat) (lo : 320 ≤ row) (hi : row < 324) :
    row - 320 < 4 := by omega

/-- Staging reads eight float4 vectors of data and reserves the ninth for the input sum. -/
theorem shared_stage_bounds (group : Fin 80) (vector : Fin 8) :
    group.val * 9 + vector.val < 720 ∧ group.val * 8 + vector.val < 640 := by
  have := group.isLt
  have := vector.isLt
  omega

/-- The four-token grid visits every valid token exactly once, including a short tail. -/
theorem token_tile_coordinates (token batch : Nat) (h : token < batch) :
    token / 4 < (batch + 3) / 4 ∧ token % 4 < 4 ∧
    token / 4 * 4 + token % 4 = token := by omega

end LeanSonic.Proofs.Qwen38DenseQ8
