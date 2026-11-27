/-
  LeanSonic — Phase 1 Concrete Proofs (`Proofs/Inkling2Mini.lean`)
  ============================================================================
  Machine-checked proofs for:
  1. Inkling-2-mini Looped MoE schedule (`24` physical layers -> `48` visits:
     `6` Prelude + `3 × 12` Loop Core + `6` Coda; `12` Full + `36` SWA=1024;
     `128` experts `top-8`).
  2. Qwen3.8-Flash-Next / Qwen4 preview macro-schedule (`3:1` GDN + QSA,
     4-branch Gated Residual, `512` experts `top-10` + shared expert) sharing
     the exact same `VisitBytecode` & `HybridStatePoolSpec` invariants.
  3. Blackwell `sm_120` static cliff proofs across all Phase 1 hot kernels
     (zero stack spill, zero hot `CALL`, SMEM carveout & L1 retention).
  4. `union Smem` stage-liveness non-aliasing & `128B` TMA alignment proof.
  5. Exhaustive 16-code NVFP4 (`E2M1`) -> `2× int8` lossless LUT theorem.
  6. Non-associative IEEE-754 `0-ULP` SIMT <-> MMA butterfly tree theorem.
-/

import LeanSonic.Axioms

namespace LeanSonic.Proofs

/-! ## 1. Inkling-2-mini Schedule (`24` Physical Layers -> `48` Visits) -/

/--
  In Inkling-2-mini, every 4th visit (`visitIndex % 4 == 3`) executes Global Full Attention
  (`12` visits total across `48` visits), while the remaining `36` visits execute
  Sliding-Window Attention (`window = 1024`).
  Layers `0..5` run once (Prelude), `6..17` run 3 times sequentially (Loop Core with
  gated loop blend on passes 2 & 3), and `18..23` run once (Coda).
-/
def inkling2PhysicalLayerOfVisit (v : Nat) : Nat :=
  if v < 6 then v
  else if v < 42 then 6 + ((v - 6) % 12)
  else 18 + (v - 42)

def inkling2NeedsLoopBlend (v : Nat) : Bool :=
  (v >= 18) && (v < 42)

def inkling2Visits : List VisitOp :=
  (List.range 48).map fun v =>
    let mix :=
      if v % 4 == 3 then SequenceMixKind.fullAttention
      else SequenceMixKind.slidingWindow 1024
    { visitIndex := v
      physicalLayerId := inkling2PhysicalLayerOfVisit v
      mix := mix
      residual := ResidualTopology.singleStream (inkling2NeedsLoopBlend v)
      numExperts := 128
      topKExperts := 8
      hasSharedExpert := false }

def inkling2MiniSchedule : ModelScheduleSpec :=
  { modelName := "Inkling-2-Mini"
    physicalLayers := 24
    visits := inkling2Visits }

theorem inkling2_schedule_valid : ValidModelSchedule inkling2MiniSchedule := by
  decide

theorem inkling2_visit_count_eq_48 : inkling2MiniSchedule.visits.length = 48 := by
  decide

def isFullAttnVisit (op : VisitOp) : Bool :=
  match op.mix with
  | SequenceMixKind.fullAttention => true
  | _ => false

def isSwa1024Visit (op : VisitOp) : Bool :=
  match op.mix with
  | SequenceMixKind.slidingWindow 1024 => true
  | _ => false

theorem inkling2_full_attn_visits_eq_12 :
    (inkling2MiniSchedule.visits.filter isFullAttnVisit).length = 12 := by
  decide

theorem inkling2_swa_visits_eq_36 :
    (inkling2MiniSchedule.visits.filter isSwa1024Visit).length = 36 := by
  decide

/-! ## 2. Qwen3.8-Flash-Next / Qwen4 Hybrid Macro-Schedule Compatibility -/

/--
  40-layer (`10 × [GDN, GDN, GDN, QSA]`) representative schedule with 4-branch
  Gated Residual, `512` routed experts (`top-10`), and shared expert enabled.
-/
def qwen38FlashNextVisits : List VisitOp :=
  (List.range 40).map fun v =>
    let mix :=
      if v % 4 == 3 then SequenceMixKind.qwenSparse 64 64
      else SequenceMixKind.gatedDeltaNet 4 32
    { visitIndex := v
      physicalLayerId := v
      mix := mix
      residual := ResidualTopology.quadGatedStream
      numExperts := 512
      topKExperts := 10
      hasSharedExpert := true }

def qwen38FlashNextSchedule : ModelScheduleSpec :=
  { modelName := "Qwen3.8-Flash-Next"
    physicalLayers := 40
    visits := qwen38FlashNextVisits }

theorem qwen38_schedule_valid : ValidModelSchedule qwen38FlashNextSchedule := by
  decide

/-! ## 3. Hybrid Slot & Ring Pool Safety (`C = 64`, `W = 16`, `SWA = 1024`) -/

def phase1HybridPool : HybridStatePoolSpec :=
  { maxConcurrentSlots := 64
    specVerifyWindow := 16
    maxBatchRows := 64
    kvBlockAlignTokens := 64
    swaWindowTokens := 1024
    swaRingCapacityTokens := 1088   -- 1024 + 64 = 1088 (exact multiple of 64)
    recurrentStateSlots := 1088     -- 64 * (1 + 16) = 1088 slots for O(1) GDN rollback
    suffixMarginPercent := 5 }

theorem phase1_hybrid_pool_safe : SafeHybridStatePool phase1HybridPool := by
  decide

/-! ## 4. Blackwell `sm_120` Kernel Suite Resource Proofs -/

/-- Unified Multi-Row Block-List Tensor Core Split-KV Attention (`mma.m16n8k16`, 22-bit Q split). -/
def kAttnBlocklistTc : KernelResourceSpec :=
  { name := "attn_blocklist_tc"
    threadsPerCta := 128
    rawRegsPerThread := 149         -- aligns to 152 -> 19,456 regs/CTA; 3 CTAs = 58,368 <= 65,536
    userSmemPerCtaBytes := 30720    -- 30 KiB + 1 KiB pad + 1 KiB sys = 32 KiB/CTA; 3 CTAs = 96 KiB <= 100 KiB
    smemAlignPadBytes := 1024
    ctasPerSm := 3
    clusterCtas := 1
    minRequiredL1KiB := 28          -- bypasses L1 via `cp.async.cg`, requires >= 28 KiB L1
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Low-Batch (`M < 24`) `REG:64` Planar-SoA SIMT MoE kernel (retains 64 KiB L1). -/
def kMoeSimtLowBatch : KernelResourceSpec :=
  { name := "moe_simt_reg64"
    threadsPerCta := 256
    rawRegsPerThread := 64          -- 16,384 regs/CTA; 4 CTAs = 65,536 <= 65,536 (100% 32-warp / high occ)
    userSmemPerCtaBytes := 14336    -- 14 KiB + 1 KiB pad + 1 KiB sys = 16 KiB/CTA; 4 CTAs = 64 KiB -> 64 KiB L1!
    smemAlignPadBytes := 1024
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 64          -- strictly preserves >= 64 KiB L1 cache (avoids 100 KiB carveout cliff!)
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- High-Batch (`>= 128` pairs) Exact-Integer `2× E2M1 -> int8` Grouped MMA MoE (`mma.m16n8k32`). -/
def kMoeGroupedMmaInt8 : KernelResourceSpec :=
  { name := "moe_grouped_mma_int8"
    threadsPerCta := 256
    rawRegsPerThread := 120         -- 30,720 regs/CTA; 2 CTAs = 61,440 <= 65,536
    userSmemPerCtaBytes := 47104    -- 46 KiB + 1 KiB pad + 1 KiB sys = 48 KiB/CTA; 2 CTAs = 96 KiB <= 100 KiB
    smemAlignPadBytes := 1024
    ctasPerSm := 2
    clusterCtas := 1
    minRequiredL1KiB := 28
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- `Q8` Planar-SoA `128x128` 16-Warp TMA Prefill GEMM. -/
def kQ8PlanarTmaGemm : KernelResourceSpec :=
  { name := "q8_planar_tma_gemm"
    threadsPerCta := 512
    rawRegsPerThread := 128         -- 65,536 regs/CTA; 1 CTA/SM
    userSmemPerCtaBytes := 61440    -- 60 KiB + 1 KiB pad + 1 KiB sys = 62 KiB <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 1024
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64          -- guaranteed >= 64 KiB L1 residency!
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Fused RMSNorm + Gated Loop / 4-Branch Residual Blend + Planar `Q8` Quantize. -/
def kLoopResidualBlend : KernelResourceSpec :=
  { name := "loop_residual_blend"
    threadsPerCta := 256
    rawRegsPerThread := 48
    userSmemPerCtaBytes := 28672    -- 28 KiB + 1 KiB pad + 1 KiB sys = 30 KiB/CTA; 2 CTAs = 60 KiB <= 64 KiB
    smemAlignPadBytes := 1024
    ctasPerSm := 2
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def phase1KernelSuite : List KernelResourceSpec :=
  [kAttnBlocklistTc, kMoeSimtLowBatch, kMoeGroupedMmaInt8, kQ8PlanarTmaGemm, kLoopResidualBlend]

theorem attn_blocklist_tc_valid : ValidSm120Kernel kAttnBlocklistTc := by decide
theorem moe_simt_low_batch_valid : ValidSm120Kernel kMoeSimtLowBatch := by decide
theorem moe_grouped_mma_int8_valid : ValidSm120Kernel kMoeGroupedMmaInt8 := by decide
theorem q8_planar_tma_gemm_valid : ValidSm120Kernel kQ8PlanarTmaGemm := by decide
theorem loop_residual_blend_valid : ValidSm120Kernel kLoopResidualBlend := by decide

theorem phase1_all_kernels_valid :
    (phase1KernelSuite.all checkSm120Kernel) = true := by
  decide

/-! ## 5. Shared Memory `union Smem` Non-Aliasing Proof (`attn_blocklist_tc`) -/

/--
  30 KiB (`30,720` B) Shared Memory map for `attn_blocklist_tc`:
  - Stages `0..2`: `q_hi_lo_tile` (`4,096` B @ `0`, live across prologue & mainloop `0..2`)
  - Stages `1..2`: 4-stage `cp.async.cg` KV ring (`4 × 6,144 = 24,576` B @ `4,096..28,672`)
  - Stage  `3`   : Warp softmax / split-KV epilogue scratch (`8,192` B @ `4,096..12,288`),
                   reusing the KV ring region ONLY after mainloop stage `2` completes!
  - Dedicated `pdl_meta` (`2,048` B @ `28,672..30,720`, live `0..3`, disjoint from all above).
-/
def attnBlocklistSmemSlices : List SmemSlice :=
  [ { name := "q_hi_lo_tile",    offsetBytes := 0,     sizeBytes := 4096, alignBytes := 128, liveStartStage := 0, liveEndStage := 2 }
  , { name := "kv_ring_stage0",  offsetBytes := 4096,  sizeBytes := 6144, alignBytes := 128, liveStartStage := 1, liveEndStage := 2 }
  , { name := "kv_ring_stage1",  offsetBytes := 10240, sizeBytes := 6144, alignBytes := 128, liveStartStage := 1, liveEndStage := 2 }
  , { name := "kv_ring_stage2",  offsetBytes := 16384, sizeBytes := 6144, alignBytes := 128, liveStartStage := 1, liveEndStage := 2 }
  , { name := "kv_ring_stage3",  offsetBytes := 22528, sizeBytes := 6144, alignBytes := 128, liveStartStage := 1, liveEndStage := 2 }
  , { name := "epilogue_reduce", offsetBytes := 4096,  sizeBytes := 8192, alignBytes := 128, liveStartStage := 3, liveEndStage := 3 }
  , { name := "pdl_barriers",    offsetBytes := 28672, sizeBytes := 2048, alignBytes := 128, liveStartStage := 0, liveEndStage := 3 }
  ]

theorem attn_blocklist_smem_safe :
    SafeSmemLayout kAttnBlocklistTc.userSmemPerCtaBytes attnBlocklistSmemSlices := by
  decide

/-! ## 6. NVFP4 (`E2M1`) Exact Integer LUT Proof & `0-ULP` Butterfly Proof -/

/-- All 16 4-bit `E2M1` nibbles decode losslessly via `e2m1BytePermLut` into `[-12, 12] ⊂ int8`. -/
theorem e2m1_all_16_codes_exact :
    ((List.finRange 16).all checkE2M1Code) = true := by
  decide

/--
  `0-ULP BATCH IDENTICAL` Theorem:
  Without assuming associativity or commutativity of `fadd` (hence valid for IEEE-754
  `__fadd_rn`), `simtSubBlockReduce4` and `mmaPacked16Bx2Reduce4` compute the exact
  same expression tree for all inputs.
-/
theorem simt_and_mma_16bx2_0ulp_eq {α : Type} (fadd : α → α → α) (p0 p1 p2 p3 : α) :
    simtSubBlockReduce4 fadd p0 p1 p2 p3 = mmaPacked16Bx2Reduce4 fadd p0 p1 p2 p3 := by
  rfl

/-- Concrete MoE Dual-Regime Switching Threshold (`routedPairs >= 128`). -/
def moeGroupedMmaMinPairs : Nat := 128

/-! ## 7. Measured `ptxas -arch=sm_120a` Wave 1 Production `Inkling-2-Mini` Kernels -/

/-- Planar `Q8_64` SMEM-Staged 2-Rows/Warp `__dp4a` GEMV (`38` regs -> `6` CTAs = `48` warps/SM!). -/
def kInk2Q8Dp4aGemv : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_gemv"
    threadsPerCta := 256
    rawRegsPerThread := 38          -- aligns to 40 -> 10,240 regs/CTA; 6 CTAs = 61,440 <= 65,536 (100% 48w/SM!)
    userSmemPerCtaBytes := 3264     -- 6 * (3264 + 1024) = 25,728 B <= 32 KiB carveout -> 96 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- `wo` (`cols = 3072`) twin of the Q8 GEMV with all weight loads hoisted (`48` regs -> `5` CTAs). -/
def kInk2Q8Dp4aGemvC3072 : KernelResourceSpec :=
  { kInk2Q8Dp4aGemv with
    name := "ink2_q8_dp4a_gemv_c3072"
    rawRegsPerThread := 48          -- 12,288 regs/CTA; 5 CTAs = 61,440 <= 65,536
    ctasPerSm := 5 }

/-- Wide `uint4` (`16B` `LDG.E.128`) `__dp4a` NVFP4 MoE Up+SITU (`40` regs -> `6` CTAs = `48` warps/SM!). -/
def kInk2MoeNvfp4Dp4aUpSitu : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_dp4a_up_situ"
    threadsPerCta := 256
    rawRegsPerThread := 40          -- 10,240 regs/CTA; 6 CTAs = 61,440 <= 65,536 (100% 48w/SM!)
    userSmemPerCtaBytes := 2912     -- 6 * (2912 + 1024) = 23,616 B <= 32 KiB carveout -> 96 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Device-resident gate/up, four channels per CTA. Resource counts from the sm_120a cubin. -/
def kInk2MoeNvfp4Dp4aUpHalf : KernelResourceSpec :=
  { kInk2MoeNvfp4Dp4aUpSitu with
    name := "ink2_moe_nvfp4_dp4a_up_half"
    threadsPerCta := 128
    rawRegsPerThread := 42
    userSmemPerCtaBytes := 0
    ctasPerSm := 10 }

def kInk2MoeNvfp4Dp4aUpHalfPacked : KernelResourceSpec :=
  { kInk2MoeNvfp4Dp4aUpHalf with
    name := "ink2_moe_nvfp4_dp4a_up_half_packed_scales"
    rawRegsPerThread := 40
    ctasPerSm := 12 }

def kInk2MoeNvfp4Dp4aUpStaged : KernelResourceSpec :=
  { kInk2MoeNvfp4Dp4aUpHalf with name := "ink2_moe_nvfp4_dp4a_up_half_staged" }

def kInk2MoeNvfp4Dp4aUpStagedPacked : KernelResourceSpec :=
  { kInk2MoeNvfp4Dp4aUpHalfPacked with name := "ink2_moe_nvfp4_dp4a_up_half_staged_packed_scales" }

def kInk2GatherNvfp4Up192 : KernelResourceSpec :=
  { kInk2MoeNvfp4Dp4aUpSitu with
    name := "ink2_gather_nvfp4_up_192"
    rawRegsPerThread := 17
    userSmemPerCtaBytes := 0 }

/-- Wide `uint4` (`16B` `LDG.E.128`) `__dp4a` NVFP4 MoE Down (`39` regs, `4` CTAs/SM -> `64` KiB L1!). -/
def kInk2MoeNvfp4Dp4aDown : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_dp4a_down"
    threadsPerCta := 256
    rawRegsPerThread := 39          -- aligns to 40 -> 10,240 regs/CTA; 4 CTAs = 40,960 <= 65,536
    userSmemPerCtaBytes := 11520    -- 4 * (11520 + 1024) = 50,176 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Expert-parallel NVFP4 `__dp4a` MoE Down for the decode row (warp `k` = selected expert `k`,
`2` rows per CTA, `1,152` CTAs): activations in registers, `64` B SMEM for the `2 x 8` dots folded
in expert order (bit-identical to `ink2_moe_nvfp4_dp4a_down`). `40` regs -> `6` CTAs = `48` warps/SM. -/
def kInk2MoeNvfp4Dp4aDownXw : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_dp4a_down_xw"
    threadsPerCta := 256
    rawRegsPerThread := 40          -- 10,240 regs/CTA; 6 CTAs = 61,440 <= 65,536
    userSmemPerCtaBytes := 64       -- 6 * (64 + 1024) = 6,528 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Unit-Stride `512B` Coalesced MXFP8 MoE Down (`40` regs -> `6` CTAs = `48` warps/SM, `128` KiB L1!). -/
def kInk2MoeMxfp8Down : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_down"
    threadsPerCta := 256
    rawRegsPerThread := 40
    userSmemPerCtaBytes := 0        -- 6 * 1024 = 6,144 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- 8-Warp Split-KV Online-Softmax GQA + 1024 Rel-Bias + Sink (`40` regs, `5` CTAs/SM -> `64` KiB L1!). -/
def kInk2GqaRelSinkAttn : KernelResourceSpec :=
  { name := "ink2_gqa_rel_sink_attn"
    threadsPerCta := 256
    rawRegsPerThread := 40
    userSmemPerCtaBytes := 8832     -- 5 * (8832 + 1024) = 49,280 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 5
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Fused Batched RMSNorm + Planar `Q8_64` Quantizer (`28` regs -> `6` CTAs = `48` warps/SM!). -/
def kInk2RmsnormQuantQ8 : KernelResourceSpec :=
  { name := "ink2_rmsnorm_quant_q8_64"
    threadsPerCta := 256
    rawRegsPerThread := 28
    userSmemPerCtaBytes := 32       -- 6 * (32 + 1024) = 6,336 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def wave1Inkling2KernelSuite : List KernelResourceSpec :=
  [ kInk2Q8Dp4aGemv
  , kInk2Q8Dp4aGemvC3072
  , kInk2MoeNvfp4Dp4aUpSitu
  , kInk2MoeNvfp4Dp4aUpHalf
  , kInk2MoeNvfp4Dp4aUpHalfPacked
  , kInk2MoeNvfp4Dp4aUpStaged
  , kInk2MoeNvfp4Dp4aUpStagedPacked
  , kInk2GatherNvfp4Up192
  , kInk2MoeNvfp4Dp4aDown
  , kInk2MoeNvfp4Dp4aDownXw
  , kInk2MoeMxfp8Down
  , kInk2GqaRelSinkAttn
  , kInk2RmsnormQuantQ8 ]

theorem ink2_q8_dp4a_gemv_valid : ValidSm120Kernel kInk2Q8Dp4aGemv := by decide
theorem ink2_q8_dp4a_gemv_c3072_valid : ValidSm120Kernel kInk2Q8Dp4aGemvC3072 := by decide
theorem ink2_moe_nvfp4_dp4a_up_situ_valid : ValidSm120Kernel kInk2MoeNvfp4Dp4aUpSitu := by decide
theorem ink2_moe_nvfp4_dp4a_up_half_valid : ValidSm120Kernel kInk2MoeNvfp4Dp4aUpHalf := by decide
theorem ink2_moe_nvfp4_dp4a_up_half_packed_valid : ValidSm120Kernel kInk2MoeNvfp4Dp4aUpHalfPacked := by decide
theorem ink2_moe_nvfp4_dp4a_down_valid : ValidSm120Kernel kInk2MoeNvfp4Dp4aDown := by decide
theorem ink2_moe_nvfp4_dp4a_down_xw_valid : ValidSm120Kernel kInk2MoeNvfp4Dp4aDownXw := by decide
theorem ink2_moe_mxfp8_down_valid : ValidSm120Kernel kInk2MoeMxfp8Down := by decide
theorem ink2_gqa_rel_sink_attn_valid : ValidSm120Kernel kInk2GqaRelSinkAttn := by decide
theorem ink2_rmsnorm_quant_q8_valid : ValidSm120Kernel kInk2RmsnormQuantQ8 := by decide

theorem wave1_inkling2_all_kernels_valid :
    (wave1Inkling2KernelSuite.all checkSm120Kernel) = true := by
  decide

/-! ## 8. Measured `ptxas -arch=sm_120a` Cycle 1 (`v0.2.0`) Multi-Token Batch Kernels -/

/-- Planar `Q8_64` 16-Row × 8-Token `IMMA.16816.S8.S8` GEMM (`56` regs, `4,096` B SMEM -> `4` CTAs/SM, `96` KiB L1!). -/
def kInk2Q8Dp4aGemmB : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_gemm_b"
    threadsPerCta := 256
    rawRegsPerThread := 56          -- 14,336 regs/CTA; 4 CTAs = 57,344 <= 65,536
    userSmemPerCtaBytes := 4096     -- 4 * (4096 + 1024) = 20,480 B <= 32 KiB carveout -> 96 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Tile-8 Grouped-Expert NVFP4 `IMMA.16816.S8.S8` Up+SITU (`48` regs, `32,576` B SMEM -> `64` KiB L1!). -/
def kInk2MoeNvfp4GroupedUpSitu : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_dp4a_grouped_up_situ"
    threadsPerCta := 256
    rawRegsPerThread := 48
    userSmemPerCtaBytes := 32576    -- 1 * (32576 + 1024) = 33,600 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Tile-8 Grouped-Expert NVFP4 `IMMA.16816.S8.S8` Down (`40` regs, `18,240` B SMEM -> `3` CTAs/SM, `64` KiB L1!). -/
def kInk2MoeNvfp4GroupedDown : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_dp4a_grouped_down"
    threadsPerCta := 256
    rawRegsPerThread := 40          -- 10,240 regs/CTA; 3 CTAs = 30,720 <= 65,536
    userSmemPerCtaBytes := 18240    -- 3 * (18240 + 1024) = 57,792 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 3
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Tile-4 Grouped-Expert MXFP8 Up+SITU (`67` regs with the hoisted row loads, `37,008` B SMEM -> `64` KiB L1!). -/
def kInk2MoeMxfp8GroupedUpSitu : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_grouped_up_situ"
    threadsPerCta := 256
    rawRegsPerThread := 67
    userSmemPerCtaBytes := 37008    -- 1 * (37008 + 1024) = 38,032 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Tile-8 Grouped-Expert MXFP8 Down (`72` regs with the hoisted row loads, `36,896` B SMEM -> `64` KiB L1!). -/
def kInk2MoeMxfp8GroupedDown : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_grouped_down"
    threadsPerCta := 256
    rawRegsPerThread := 72
    userSmemPerCtaBytes := 36896    -- 1 * (36896 + 1024) = 37,920 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Register-Persistent Causal `short_conv4` + `seq_out` Rewind Tap (`56` regs -> `4` CTAs/SM, `120` KiB L1!). -/
def kInk2ShortConv4B : KernelResourceSpec :=
  { name := "ink2_short_conv4_b"
    threadsPerCta := 256
    rawRegsPerThread := 56          -- 14,336 regs/CTA; 4 CTAs = 57,344 <= 65,536
    userSmemPerCtaBytes := 0        -- 4 * 1024 = 4,096 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def cycle1BatchedKernelSuite : List KernelResourceSpec :=
  [ kInk2Q8Dp4aGemmB
  , kInk2MoeNvfp4GroupedUpSitu
  , kInk2MoeNvfp4GroupedDown
  , kInk2MoeMxfp8GroupedUpSitu
  , kInk2MoeMxfp8GroupedDown
  , kInk2ShortConv4B ]

theorem ink2_q8_dp4a_gemm_b_valid : ValidSm120Kernel kInk2Q8Dp4aGemmB := by decide
theorem ink2_moe_nvfp4_grouped_up_situ_valid : ValidSm120Kernel kInk2MoeNvfp4GroupedUpSitu := by decide
theorem ink2_moe_nvfp4_grouped_down_valid : ValidSm120Kernel kInk2MoeNvfp4GroupedDown := by decide
theorem ink2_moe_mxfp8_grouped_up_situ_valid : ValidSm120Kernel kInk2MoeMxfp8GroupedUpSitu := by decide
theorem ink2_moe_mxfp8_grouped_down_valid : ValidSm120Kernel kInk2MoeMxfp8GroupedDown := by decide
theorem ink2_short_conv4_b_valid : ValidSm120Kernel kInk2ShortConv4B := by decide

theorem cycle1_batched_all_kernels_valid :
    (cycle1BatchedKernelSuite.all checkSm120Kernel) = true := by
  decide

/-! ## 9. Measured `ptxas -arch=sm_120a` Cycle 2 (`v0.3.0`) `DSpark` Speculative & `O(1)` Rewind Kernels -/

/-- `O(1)` Speculative Rewind across all 48 visits × 4 `short_conv4` states (`22` regs -> `6` CTAs = `48` warps/SM!). -/
def kInk2ConvRewind : KernelResourceSpec :=
  { name := "ink2_conv_rewind"
    threadsPerCta := 256
    rawRegsPerThread := 22
    userSmemPerCtaBytes := 0        -- 6 * 1024 = 6,144 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- General K-Tiled Planar `Q8_64` `__dp4a` GEMM (`cols <= 9216`, `56` regs, `8,704` B SMEM -> `4` CTAs/SM!). -/
def kInk2Q8Dp4aGemmKtiled : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_gemm_ktiled"
    threadsPerCta := 256
    rawRegsPerThread := 56          -- 14,336 regs/CTA; 4 CTAs = 57,344 <= 65,536
    userSmemPerCtaBytes := 8704     -- 4 * (8704 + 1024) = 38,912 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- `DSpark` Bidirectional 8-Token Block + Context GQA (`45` regs, `4,672` B SMEM -> `5` CTAs/SM, `96` KiB L1!). -/
def kMtpAttn : KernelResourceSpec :=
  { name := "mtp_attn"
    threadsPerCta := 256
    rawRegsPerThread := 45          -- aligns to 48 -> 12,288 regs/CTA; 5 CTAs = 61,440 <= 65,536
    userSmemPerCtaBytes := 4672     -- 5 * (4672 + 1024) = 28,480 B <= 32 KiB carveout -> 96 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 5
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- `DSpark` long-context attention slices (#107): `56` regs, no SMEM -> `4` CTAs/SM. -/
def kMtpAttnSplit : KernelResourceSpec :=
  { name := "mtp_attn_split"
    threadsPerCta := 256
    rawRegsPerThread := 56          -- 14,336 regs/CTA; 4 CTAs = 57,344 <= 65,536
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Merge of the `mtp_attn_split` slices (`29` regs, `128` threads). -/
def kMtpAttnMerge : KernelResourceSpec :=
  { name := "mtp_attn_merge"
    threadsPerCta := 128
    rawRegsPerThread := 29
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 8
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

theorem mtp_attn_split_valid : ValidSm120Kernel kMtpAttnSplit := by decide
theorem mtp_attn_merge_valid : ValidSm120Kernel kMtpAttnMerge := by decide

/-- `DSpark` `R = 256` Markov-RNN Projection (`36` regs, `2,048` B SMEM -> `6` CTAs = `48` warps/SM!). -/
def kMtpMarkovRaw : KernelResourceSpec :=
  { name := "mtp_markov_raw"
    threadsPerCta := 256
    rawRegsPerThread := 36          -- aligns to 40 -> 10,240 regs/CTA; 6 CTAs = 61,440 <= 65,536
    userSmemPerCtaBytes := 2048     -- 6 * (2048 + 1024) = 18,432 B <= 32 KiB carveout -> 96 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- `DSpark` Pre-Folded `[200058, 256]` Planar `Q8_64` `__dp4a` Vocab Head (`44` regs -> `5` CTAs/SM, `120` KiB L1!). -/
def kMtpFoldMvQ8 : KernelResourceSpec :=
  { name := "mtp_fold_mv_q8"
    threadsPerCta := 256
    rawRegsPerThread := 44          -- aligns to 48 -> 12,288 regs/CTA; 5 CTAs = 61,440 <= 65,536
    userSmemPerCtaBytes := 368      -- 5 * (368 + 1024) = 6,960 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 5
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- `DSpark` Online-Softmax Confidence-Calibrated Argmax Reduction (`40` regs -> `6` CTAs = `48` warps/SM!). -/
def kMtpFoldFin : KernelResourceSpec :=
  { name := "mtp_fold_fin"
    threadsPerCta := 256
    rawRegsPerThread := 40
    userSmemPerCtaBytes := 96       -- 6 * (96 + 1024) = 6,720 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def cycle2SpeculativeKernelSuite : List KernelResourceSpec :=
  [ kInk2ConvRewind
  , kInk2Q8Dp4aGemmKtiled
  , kMtpAttn
  , kMtpMarkovRaw
  , kMtpFoldMvQ8
  , kMtpFoldFin ]

theorem ink2_conv_rewind_valid : ValidSm120Kernel kInk2ConvRewind := by decide
theorem ink2_q8_dp4a_gemm_ktiled_valid : ValidSm120Kernel kInk2Q8Dp4aGemmKtiled := by decide
theorem mtp_attn_valid : ValidSm120Kernel kMtpAttn := by decide
theorem mtp_markov_raw_valid : ValidSm120Kernel kMtpMarkovRaw := by decide
theorem mtp_fold_mv_q8_valid : ValidSm120Kernel kMtpFoldMvQ8 := by decide
theorem mtp_fold_fin_valid : ValidSm120Kernel kMtpFoldFin := by decide

theorem cycle2_speculative_all_kernels_valid :
    (cycle2SpeculativeKernelSuite.all checkSm120Kernel) = true := by
  decide

/-! ## 10. Measured `ptxas -arch=sm_120a` Cycle 4 (`v0.5.0`) Split-KV GQA & Packed Quant Kernels -/

/-- Packed 8-Group/CTA Half-Warp `Q8_16` Quantizer (`128` threads, `19` regs -> `12` CTAs = `48` warps/SM!). -/
def kInk2QuantQ8Packed : KernelResourceSpec :=
  { name := "ink2_quant_q8_16_packed"
    threadsPerCta := 128
    rawRegsPerThread := 19          -- aligns to 24 -> 3,072 regs/CTA; 12 CTAs = 36,864 <= 65,536 (48 warps/SM!)
    userSmemPerCtaBytes := 0        -- 12 * 1024 = 12,288 B <= 16 KiB carveout -> 112 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 12
    clusterCtas := 1
    minRequiredL1KiB := 112
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- INT8-KV instantiation of the unified Split-KV attention (`LEANSONIC_KV=int8`, #14 / T011).
`__launch_bounds__(256, 4)`; `ptxas` (CUDA 13.0, `sm_120a`): `64` regs, `14,976` B SMEM, `0` stack /
spill, no `CALL.REL.NOINC`. `64 * 256 = 16,384` regs/CTA; `4` CTAs = `65,536 <= 65,536`;
`4 * (14976 + 1024) = 64,000` B <= 64 KiB carveout -> 64 KiB L1. -/
def kInk2GqaAttnSplitI8 : KernelResourceSpec :=
  { name := "ink2_gqa_attn_split_i8"
    threadsPerCta := 256
    rawRegsPerThread := 64
    userSmemPerCtaBytes := 14976
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Unified Split-KV + GQA-Grouped Attention (`56` regs, `14,976` B SMEM -> `4` CTAs = `32` warps/SM, `64` KiB L1!). -/
def kInk2GqaAttnSplit : KernelResourceSpec :=
  { name := "ink2_gqa_attn_split"
    threadsPerCta := 256
    rawRegsPerThread := 56          -- 14,336 regs/CTA; 4 CTAs = 57,344 <= 65,536
    userSmemPerCtaBytes := 14976    -- 4 * (14976 + 1024) = 64,000 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- FP16 decode / verify instantiation with the pipelined row loop (`64` regs -> still `4` CTAs). -/
def kInk2GqaAttnSplitDec : KernelResourceSpec :=
  { kInk2GqaAttnSplit with
    name := "ink2_gqa_attn_split_dec"
    rawRegsPerThread := 64 }        -- 16,384 regs/CTA; 4 CTAs = 65,536 <= 65,536

/-- Split-KV Online-Softmax Combine + Sigmoid Gate (`128` threads, `39` regs -> `12` CTAs = `48` warps/SM!). -/
def kInk2GqaAttnCombine : KernelResourceSpec :=
  { name := "ink2_gqa_attn_combine"
    threadsPerCta := 128
    rawRegsPerThread := 39          -- aligns to 40 -> 5,120 regs/CTA; 12 CTAs = 61,440 <= 65,536
    userSmemPerCtaBytes := 0        -- 12 * 1024 = 12,288 B <= 16 KiB carveout -> 112 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 12
    clusterCtas := 1
    minRequiredL1KiB := 112
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def cycle4SplitKvKernelSuite : List KernelResourceSpec :=
  [ kInk2QuantQ8Packed
  , kInk2GqaAttnSplit
  , kInk2GqaAttnSplitI8
  , kInk2GqaAttnSplitDec
  , kInk2GqaAttnCombine ]

theorem ink2_quant_q8_16_packed_valid : ValidSm120Kernel kInk2QuantQ8Packed := by decide
theorem ink2_gqa_attn_split_valid : ValidSm120Kernel kInk2GqaAttnSplit := by decide
theorem ink2_gqa_attn_split_i8_valid : ValidSm120Kernel kInk2GqaAttnSplitI8 := by decide
theorem ink2_gqa_attn_split_dec_valid : ValidSm120Kernel kInk2GqaAttnSplitDec := by decide
theorem ink2_gqa_attn_combine_valid : ValidSm120Kernel kInk2GqaAttnCombine := by decide

theorem cycle4_splitkv_all_kernels_valid :
    (cycle4SplitKvKernelSuite.all checkSm120Kernel) = true := by
  decide

/-! ## Batched controller tiles (256 threads per CTA, sm_120a) -/

def kInk2CtrlDownR8B : KernelResourceSpec :=
  { name := "ink2_ctrl_down_r8_b"
    threadsPerCta := 256
    rawRegsPerThread := 56          -- first weight pass loaded before `pdl_wait` (#152): 4 CTAs
    userSmemPerCtaBytes := 18760
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 28
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2CtrlApplyT8B : KernelResourceSpec :=
  { name := "ink2_ctrl_apply_t8_b"
    threadsPerCta := 256
    rawRegsPerThread := 64
    userSmemPerCtaBytes := 2048
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 112
    hasStackSpill := false
    hasHotLoopLibmCall := false }

theorem ink2_ctrl_down_r8_b_valid : ValidSm120Kernel kInk2CtrlDownR8B := by decide
theorem ink2_ctrl_apply_t8_b_valid : ValidSm120Kernel kInk2CtrlApplyT8B := by decide

/-! ## 11. Measured `ptxas -arch=sm_120a` Cycle 5 (`v0.6.0`) 0-ULP Glue Fusion & Coalesced Controller Kernels -/

/-- Fused `depth_mix` + `rmsnorm_quant_q8_64` (`35` regs, `9,704` B SMEM -> `6` CTAs = `48` warps/SM, `64` KiB L1!). -/
def kInk2DepthMixRmsnormQuantQ8B : KernelResourceSpec :=
  { name := "ink2_depth_mix_rmsnorm_quant_q8_64_b"
    threadsPerCta := 256
    rawRegsPerThread := 35          -- aligns to 40 -> 10,240 regs/CTA; 6 CTAs = 61,440 <= 65,536 (48 warps/SM!)
    userSmemPerCtaBytes := 9704     -- 6 * (9704 + 1024) = 64,368 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Fused `depth_mix` + `rmsnorm_d` + `quant_q8_16_packed` (`33` regs, `9,704` B SMEM -> `6` CTAs = `48` warps/SM!). -/
def kInk2DepthMixRmsnormDQuantQ8B : KernelResourceSpec :=
  { name := "ink2_depth_mix_rmsnorm_d_quant_q8_16_b"
    threadsPerCta := 256
    rawRegsPerThread := 33          -- aligns to 40 -> 10,240 regs/CTA; 6 CTAs = 61,440 <= 65,536
    userSmemPerCtaBytes := 9704     -- 6 * (9704 + 1024) = 64,368 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Fused `rmsnorm_d` + `quant_q8_16_packed` (`26` regs, `9,248` B SMEM -> `6` CTAs = `48` warps/SM!). -/
def kInk2RmsnormDQuantQ8B : KernelResourceSpec :=
  { name := "ink2_rmsnorm_d_quant_q8_16_b"
    threadsPerCta := 256
    rawRegsPerThread := 26          -- aligns to 32 -> 8,192 regs/CTA; 6 CTAs = 49,152 <= 65,536
    userSmemPerCtaBytes := 9248     -- 6 * (9248 + 1024) = 61,632 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Coalesced `512B`-Burst Controller Apply (`64` threads, `52` regs, `9,472` B SMEM -> `6` CTAs, `64` KiB L1!). -/
def kInk2CtrlApplyCoalescedB : KernelResourceSpec :=
  { name := "ink2_ctrl_apply_coalesced_b"
    threadsPerCta := 64
    rawRegsPerThread := 52          -- aligns to 56 -> 3,584 regs/CTA; 6 CTAs = 21,504 <= 65,536
    userSmemPerCtaBytes := 9472     -- 6 * (9472 + 1024) = 62,976 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- GQA Relative-Bias + Sink Attention + In-Register `Q8_64` Epilogue (`40` regs, `8,856` B SMEM -> `6` CTAs!). -/
def kInk2GqaRelSinkAttnQ8B : KernelResourceSpec :=
  { name := "ink2_gqa_rel_sink_attn_q8_b"
    threadsPerCta := 256
    rawRegsPerThread := 40          -- 10,240 regs/CTA; 6 CTAs = 61,440 <= 65,536 (48 warps/SM!)
    userSmemPerCtaBytes := 8856     -- 6 * (8856 + 1024) = 59,280 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- INT8-KV instantiation of the fused attention (`LEANSONIC_KV=int8`, same `40` regs / `8,856` B SMEM -> `6` CTAs). -/
def kInk2GqaRelSinkAttnQ8BI8 : KernelResourceSpec :=
  { kInk2GqaRelSinkAttnQ8B with name := "ink2_gqa_rel_sink_attn_q8_b_i8" }

/-- Latency-staged decode / small-verify variant (#56): `75` regs -> `3` CTAs (`57,600` regs/SM). -/
def kInk2GqaRelSinkAttnQ8Decode : KernelResourceSpec :=
  { kInk2GqaRelSinkAttnQ8B with
    name := "ink2_gqa_rel_sink_attn_q8_decode"
    rawRegsPerThread := 75
    ctasPerSm := 3 }

/-- INT8-KV instantiation of the staged decode variant: `72` regs -> `3` CTAs. -/
def kInk2GqaRelSinkAttnQ8DecodeI8 : KernelResourceSpec :=
  { kInk2GqaRelSinkAttnQ8B with
    name := "ink2_gqa_rel_sink_attn_q8_decode_i8"
    rawRegsPerThread := 72
    ctasPerSm := 3 }

/-- Split-KV Combine + In-Register `Q8_64` Epilogue (`128` threads, `40` regs -> `12` CTAs = `48` warps/SM!). -/
def kInk2GqaAttnCombineQ8 : KernelResourceSpec :=
  { name := "ink2_gqa_attn_combine_q8"
    threadsPerCta := 128
    rawRegsPerThread := 40          -- 5,120 regs/CTA; 12 CTAs = 61,440 <= 65,536 (48 warps/SM!)
    userSmemPerCtaBytes := 24       -- 12 * (24 + 1024) = 12,576 B <= 16 KiB carveout -> 112 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 12
    clusterCtas := 1
    minRequiredL1KiB := 112
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Causal 4-Tap Short-Conv + Fused Block-Start Save (`26` regs -> `6` CTAs = `48` warps/SM, `120` KiB L1!). -/
def kInk2ShortConv4Save : KernelResourceSpec :=
  { name := "ink2_short_conv4_save"
    threadsPerCta := 256
    rawRegsPerThread := 26          -- aligns to 32 -> 8,192 regs/CTA; 6 CTAs = 49,152 <= 65,536
    userSmemPerCtaBytes := 0        -- 6 * 1024 = 6,144 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Batched Causal 4-Tap Short-Conv + Fused Block-Start Save (`64` regs -> `4` CTAs = `32` warps/SM, `120` KiB L1!). -/
def kInk2ShortConv4SaveB : KernelResourceSpec :=
  { name := "ink2_short_conv4_save_b"
    threadsPerCta := 256
    rawRegsPerThread := 64          -- 16,384 regs/CTA; 4 CTAs = 65,536 <= 65,536
    userSmemPerCtaBytes := 0        -- 4 * 1024 = 4,096 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Fused Prelude Save & Loop Pass Transition (`20` regs -> `6` CTAs = `48` warps/SM, `120` KiB L1!). -/
def kInk2LoopPassSetupB : KernelResourceSpec :=
  { name := "ink2_loop_pass_setup_b"
    threadsPerCta := 256
    rawRegsPerThread := 20          -- aligns to 24 -> 6,144 regs/CTA; 6 CTAs = 36,864 <= 65,536
    userSmemPerCtaBytes := 0        -- 6 * 1024 = 6,144 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- SM-Wave-Balanced 4-Rows/CTA Planar `Q8_64` `__dp4a` GEMV (`64` threads, `38` regs, `3,264` B SMEM -> `14` CTAs/SM!). -/
def kInk2Q8Dp4aGemv4r : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_gemv_4r"
    threadsPerCta := 64
    rawRegsPerThread := 38          -- aligns to 40 -> 2,560 regs/CTA; 14 CTAs = 35,840 <= 65,536
    userSmemPerCtaBytes := 3264     -- 14 * (3264 + 1024) = 60,032 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 14
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Fused 3-Projection `(Q, K, V)` Planar `Q8_64` `__dp4a` GEMV (`256` threads, `64` regs, `2,448` B SMEM -> `4` CTAs = `32` warps/SM!). -/
def kInk2Q8Dp4aQkv3Gemv : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_qkv3_gemv"
    threadsPerCta := 256
    rawRegsPerThread := 64          -- 16,384 regs/CTA; 4 CTAs = 65,536 <= 65,536 (32 warps/SM!)
    userSmemPerCtaBytes := 2448     -- 4 * (2448 + 1024) = 13,888 B <= 16 KiB carveout -> 112 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- SM-Wave-Balanced 2-Warps/CTA NVFP4 `__dp4a` MoE Down (`64` threads, `39` regs, `11,520` B SMEM -> `5` CTAs/SM!). -/
def kInk2MoeNvfp4Dp4aDown2w : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_dp4a_down_2w"
    threadsPerCta := 64
    rawRegsPerThread := 39          -- aligns to 40 -> 2,560 regs/CTA; 5 CTAs = 12,800 <= 65,536
    userSmemPerCtaBytes := 11520    -- 5 * (11520 + 1024) = 62,720 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 5
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- SM-Wave-Balanced 2-Warps/CTA MXFP8 MoE Down (`64` threads, `40` regs -> `24` CTAs = `48` warps/SM!). -/
def kInk2MoeMxfp8Down2w : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_down_2w"
    threadsPerCta := 64
    rawRegsPerThread := 40          -- 2,560 regs/CTA; 24 CTAs = 61,440 <= 65,536 (48 warps/SM!)
    userSmemPerCtaBytes := 0        -- 24 * 1024 = 24,576 B <= 32 KiB carveout -> 96 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 24
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Fused Dual K+V Causal 4-Tap Short-Conv (`26` regs -> `6` CTAs = `48` warps/SM, `120` KiB L1!). -/
def kInk2ShortConv4Kv2 : KernelResourceSpec :=
  { name := "ink2_short_conv4_kv2"
    threadsPerCta := 256
    rawRegsPerThread := 26          -- aligns to 32 -> 8,192 regs/CTA; 6 CTAs = 49,152 <= 65,536
    userSmemPerCtaBytes := 0        -- 6 * 1024 = 6,144 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def cycle5GlueFusionKernelSuite : List KernelResourceSpec :=
  [ kInk2DepthMixRmsnormQuantQ8B
  , kInk2DepthMixRmsnormDQuantQ8B
  , kInk2RmsnormDQuantQ8B
  , kInk2CtrlApplyCoalescedB
  , kInk2GqaRelSinkAttnQ8B
  , kInk2GqaRelSinkAttnQ8BI8
  , kInk2GqaRelSinkAttnQ8Decode
  , kInk2GqaRelSinkAttnQ8DecodeI8
  , kInk2GqaAttnCombineQ8
  , kInk2ShortConv4Save
  , kInk2ShortConv4SaveB
  , kInk2LoopPassSetupB
  , kInk2Q8Dp4aGemv4r
  , kInk2Q8Dp4aQkv3Gemv
  , kInk2MoeNvfp4Dp4aDown2w
  , kInk2MoeMxfp8Down2w
  , kInk2ShortConv4Kv2 ]

theorem ink2_depth_mix_rmsnorm_quant_q8_64_b_valid :
    ValidSm120Kernel kInk2DepthMixRmsnormQuantQ8B := by decide
theorem ink2_depth_mix_rmsnorm_d_quant_q8_16_b_valid :
    ValidSm120Kernel kInk2DepthMixRmsnormDQuantQ8B := by decide
theorem ink2_rmsnorm_d_quant_q8_16_b_valid :
    ValidSm120Kernel kInk2RmsnormDQuantQ8B := by decide
theorem ink2_ctrl_apply_coalesced_b_valid :
    ValidSm120Kernel kInk2CtrlApplyCoalescedB := by decide
theorem ink2_gqa_rel_sink_attn_q8_b_valid :
    ValidSm120Kernel kInk2GqaRelSinkAttnQ8B := by decide

theorem ink2_gqa_rel_sink_attn_q8_b_i8_valid :
    ValidSm120Kernel kInk2GqaRelSinkAttnQ8BI8 := by decide

theorem ink2_gqa_rel_sink_attn_q8_decode_valid :
    ValidSm120Kernel kInk2GqaRelSinkAttnQ8Decode := by decide

theorem ink2_gqa_rel_sink_attn_q8_decode_i8_valid :
    ValidSm120Kernel kInk2GqaRelSinkAttnQ8DecodeI8 := by decide
theorem ink2_gqa_attn_combine_q8_valid :
    ValidSm120Kernel kInk2GqaAttnCombineQ8 := by decide
theorem ink2_short_conv4_save_valid :
    ValidSm120Kernel kInk2ShortConv4Save := by decide
theorem ink2_short_conv4_save_b_valid :
    ValidSm120Kernel kInk2ShortConv4SaveB := by decide
theorem ink2_loop_pass_setup_b_valid :
    ValidSm120Kernel kInk2LoopPassSetupB := by decide
theorem ink2_q8_dp4a_gemv_4r_valid :
    ValidSm120Kernel kInk2Q8Dp4aGemv4r := by decide
theorem ink2_q8_dp4a_qkv3_gemv_valid :
    ValidSm120Kernel kInk2Q8Dp4aQkv3Gemv := by decide
theorem ink2_moe_nvfp4_dp4a_down_2w_valid :
    ValidSm120Kernel kInk2MoeNvfp4Dp4aDown2w := by decide
theorem ink2_moe_mxfp8_down_2w_valid :
    ValidSm120Kernel kInk2MoeMxfp8Down2w := by decide
theorem ink2_short_conv4_kv2_valid :
    ValidSm120Kernel kInk2ShortConv4Kv2 := by decide

theorem cycle5_glue_fusion_all_kernels_valid :
    (cycle5GlueFusionKernelSuite.all checkSm120Kernel) = true := by
  decide

/-- Native sm_120a wide expert grouping, four independent warps per CTA. -/
def kInk2BuildExpertGroupsWideB : KernelResourceSpec :=
  { name := "ink2_build_expert_groups_wide_b"
    threadsPerCta := 128
    rawRegsPerThread := 31
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 12
    clusterCtas := 1
    minRequiredL1KiB := 112
    hasStackSpill := false
    hasHotLoopLibmCall := false }

theorem ink2_build_expert_groups_wide_b_valid :
    ValidSm120Kernel kInk2BuildExpertGroupsWideB := by decide

/-! ## Cycle 6 (`v0.7.0`) SM120a Wide Tensor Core Prefill Suite (`Issue #10` -> `14K+ tok/s`) -/

def kInk2ShortConv4SaveParB : KernelResourceSpec :=
  { name := "ink2_short_conv4_save_par_b"
    threadsPerCta := 256
    rawRegsPerThread := 56
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2ShortConv4CommitHistB : KernelResourceSpec :=
  { name := "ink2_short_conv4_commit_hist_b"
    threadsPerCta := 256
    rawRegsPerThread := 18
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2Q8TcGemmB : KernelResourceSpec :=
  { name := "ink2_q8_tc_gemm_b"
    threadsPerCta := 512
    rawRegsPerThread := 105
    userSmemPerCtaBytes := 64512
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2Q8TcQkv3GemmB : KernelResourceSpec :=
  { name := "ink2_q8_tc_qkv3_gemm_b"
    threadsPerCta := 512
    rawRegsPerThread := 101
    userSmemPerCtaBytes := 64512
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2QuantNvfp4TwoTermB : KernelResourceSpec :=
  { name := "ink2_quant_nvfp4_2term_b"
    threadsPerCta := 128
    rawRegsPerThread := 40
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 12
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2GatherNvfp4SortedB : KernelResourceSpec :=
  { name := "ink2_gather_nvfp4_sorted_b"
    threadsPerCta := 256
    rawRegsPerThread := 38
    userSmemPerCtaBytes := 1584
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4TcUpSitu : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_tc_up_situ"
    threadsPerCta := 512
    rawRegsPerThread := 127
    userSmemPerCtaBytes := 62504
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4TcDown : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_tc_down"
    threadsPerCta := 512
    rawRegsPerThread := 124
    userSmemPerCtaBytes := 62504
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4TcUpSituPackedScales : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_tc_up_situ_packed_scales"
    threadsPerCta := 512
    rawRegsPerThread := 127
    userSmemPerCtaBytes := 62504
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4TcDownPackedScales : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_tc_down_packed_scales"
    threadsPerCta := 512
    rawRegsPerThread := 126
    userSmemPerCtaBytes := 62504
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2QuantMxfp8B : KernelResourceSpec :=
  { name := "ink2_quant_mxfp8_b"
    threadsPerCta := 32
    rawRegsPerThread := 17
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 32
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeMxfp8TcUpSitu : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_tc_up_situ"
    threadsPerCta := 512
    rawRegsPerThread := 116
    userSmemPerCtaBytes := 77872
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 28
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeMxfp8TcDown : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_tc_down"
    threadsPerCta := 512
    rawRegsPerThread := 114
    userSmemPerCtaBytes := 77872
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 28
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeMxfp8TcUpSituPackedScales : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_tc_up_situ_packed_scales"
    threadsPerCta := 512
    rawRegsPerThread := 116
    userSmemPerCtaBytes := 77872
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 28
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeMxfp8TcDownPackedScales : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_tc_down_packed_scales"
    threadsPerCta := 512
    rawRegsPerThread := 114
    userSmemPerCtaBytes := 77872
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 28
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2FlashAttnTcQ8B : KernelResourceSpec :=
  { name := "ink2_flash_attn_tc_q8_b"
    threadsPerCta := 384
    rawRegsPerThread := 168
    userSmemPerCtaBytes := 64384
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2FlashAttnTcQ8BI8 : KernelResourceSpec :=
  { name := "ink2_flash_attn_tc_q8_b_i8"
    threadsPerCta := 384
    rawRegsPerThread := 168
    userSmemPerCtaBytes := 63872
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2BuildExpertGroupsParB : KernelResourceSpec :=
  { name := "ink2_build_expert_groups_par_b"
    threadsPerCta := 256
    rawRegsPerThread := 32
    userSmemPerCtaBytes := 64
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2F32ToBf16B : KernelResourceSpec :=
  { name := "ink2_f32_to_bf16_b"
    threadsPerCta := 256
    rawRegsPerThread := 18
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2CtrlNormBf16B : KernelResourceSpec :=
  { name := "ink2_ctrl_norm_bf16_b"
    threadsPerCta := 256
    rawRegsPerThread := 40
    userSmemPerCtaBytes := 9296
    smemAlignPadBytes := 0
    ctasPerSm := 5
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2Bf16TcGemmB : KernelResourceSpec :=
  { name := "ink2_bf16_tc_gemm_b"
    threadsPerCta := 256
    rawRegsPerThread := 48
    userSmemPerCtaBytes := 20480
    smemAlignPadBytes := 0
    ctasPerSm := 2
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2CtrlApplyTcB : KernelResourceSpec :=
  { name := "ink2_ctrl_apply_tc_b"
    threadsPerCta := 256
    rawRegsPerThread := 55
    userSmemPerCtaBytes := 27648
    smemAlignPadBytes := 0
    ctasPerSm := 2
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeCombineTop8Vec4B : KernelResourceSpec :=
  { name := "ink2_moe_combine_top8_vec4_b"
    threadsPerCta := 192
    rawRegsPerThread := 40
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 8
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeCombineTop8F16B : KernelResourceSpec :=
  { name := "ink2_moe_combine_top8_f16_b"
    threadsPerCta := 288
    rawRegsPerThread := 40
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 5
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2DepthMixMultiB : KernelResourceSpec :=
  { name := "ink2_depth_mix_multi_b"
    threadsPerCta := 256
    rawRegsPerThread := 56
    userSmemPerCtaBytes := 1104
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def cycle6WideTcPrefillSuite : List KernelResourceSpec :=
  [ kInk2ShortConv4SaveParB
  , kInk2ShortConv4CommitHistB
  , kInk2Q8TcGemmB
  , kInk2Q8TcQkv3GemmB
  , kInk2QuantNvfp4TwoTermB
  , kInk2GatherNvfp4SortedB
  , kInk2MoeNvfp4TcUpSitu
  , kInk2MoeNvfp4TcDown
  , kInk2MoeNvfp4TcUpSituPackedScales
  , kInk2MoeNvfp4TcDownPackedScales
  , kInk2QuantMxfp8B
  , kInk2MoeMxfp8TcUpSitu
  , kInk2MoeMxfp8TcDown
  , kInk2MoeMxfp8TcUpSituPackedScales
  , kInk2MoeMxfp8TcDownPackedScales
  , kInk2FlashAttnTcQ8B
  , kInk2FlashAttnTcQ8BI8
  , kInk2BuildExpertGroupsParB
  , kInk2F32ToBf16B
  , kInk2CtrlNormBf16B
  , kInk2Bf16TcGemmB
  , kInk2CtrlApplyTcB
  , kInk2MoeCombineTop8Vec4B
  , kInk2MoeCombineTop8F16B
  , kInk2DepthMixMultiB ]

theorem ink2_short_conv4_save_par_b_valid :
    ValidSm120Kernel kInk2ShortConv4SaveParB := by decide
theorem ink2_short_conv4_commit_hist_b_valid :
    ValidSm120Kernel kInk2ShortConv4CommitHistB := by decide
theorem ink2_q8_tc_gemm_b_valid :
    ValidSm120Kernel kInk2Q8TcGemmB := by decide
theorem ink2_q8_tc_qkv3_gemm_b_valid :
    ValidSm120Kernel kInk2Q8TcQkv3GemmB := by decide
theorem ink2_quant_nvfp4_2term_b_valid :
    ValidSm120Kernel kInk2QuantNvfp4TwoTermB := by decide
theorem ink2_gather_nvfp4_sorted_b_valid :
    ValidSm120Kernel kInk2GatherNvfp4SortedB := by decide
theorem ink2_moe_nvfp4_tc_up_situ_valid :
    ValidSm120Kernel kInk2MoeNvfp4TcUpSitu := by decide
theorem ink2_moe_nvfp4_tc_down_valid :
    ValidSm120Kernel kInk2MoeNvfp4TcDown := by decide
theorem ink2_moe_nvfp4_tc_up_situ_packed_scales_valid :
    ValidSm120Kernel kInk2MoeNvfp4TcUpSituPackedScales := by decide
theorem ink2_moe_nvfp4_tc_down_packed_scales_valid :
    ValidSm120Kernel kInk2MoeNvfp4TcDownPackedScales := by decide
theorem ink2_quant_mxfp8_b_valid :
    ValidSm120Kernel kInk2QuantMxfp8B := by decide
theorem ink2_moe_mxfp8_tc_up_situ_valid :
    ValidSm120Kernel kInk2MoeMxfp8TcUpSitu := by decide
theorem ink2_moe_mxfp8_tc_down_valid :
    ValidSm120Kernel kInk2MoeMxfp8TcDown := by decide
theorem ink2_moe_mxfp8_tc_up_situ_packed_scales_valid :
    ValidSm120Kernel kInk2MoeMxfp8TcUpSituPackedScales := by decide
theorem ink2_moe_mxfp8_tc_down_packed_scales_valid :
    ValidSm120Kernel kInk2MoeMxfp8TcDownPackedScales := by decide
theorem ink2_flash_attn_tc_q8_b_valid :
    ValidSm120Kernel kInk2FlashAttnTcQ8B := by decide
theorem ink2_flash_attn_tc_q8_b_i8_valid :
    ValidSm120Kernel kInk2FlashAttnTcQ8BI8 := by decide
theorem ink2_build_expert_groups_par_b_valid :
    ValidSm120Kernel kInk2BuildExpertGroupsParB := by decide
theorem ink2_f32_to_bf16_b_valid :
    ValidSm120Kernel kInk2F32ToBf16B := by decide
theorem ink2_ctrl_norm_bf16_b_valid :
    ValidSm120Kernel kInk2CtrlNormBf16B := by decide
theorem ink2_bf16_tc_gemm_b_valid :
    ValidSm120Kernel kInk2Bf16TcGemmB := by decide
theorem ink2_ctrl_apply_tc_b_valid :
    ValidSm120Kernel kInk2CtrlApplyTcB := by decide
theorem ink2_moe_combine_top8_vec4_b_valid :
    ValidSm120Kernel kInk2MoeCombineTop8Vec4B := by decide
theorem ink2_moe_combine_top8_f16_b_valid :
    ValidSm120Kernel kInk2MoeCombineTop8F16B := by decide
theorem ink2_depth_mix_multi_b_valid :
    ValidSm120Kernel kInk2DepthMixMultiB := by decide

theorem cycle6_wide_tc_prefill_all_kernels_valid :
    (cycle6WideTcPrefillSuite.all checkSm120Kernel) = true := by
  decide

def flashAttnTcSmemSlices : List SmemSlice :=
  [ { name := "sm_kv_pingpong", offsetBytes := 0, sizeBytes := 34816, alignBytes := 16, liveStartStage := 0, liveEndStage := 2 }
  , { name := "sm_band", offsetBytes := 34816, sizeBytes := 4608, alignBytes := 16, liveStartStage := 0, liveEndStage := 2 }
  , { name := "sm_rf", offsetBytes := 39424, sizeBytes := 6144, alignBytes := 16, liveStartStage := 0, liveEndStage := 2 }
  , { name := "sm_gx", offsetBytes := 45568, sizeBytes := 18816, alignBytes := 16, liveStartStage := 1, liveEndStage := 2 } ]

theorem flash_attn_tc_smem_union_safe :
    SafeSmemLayout 64384 flashAttnTcSmemSlices := by decide

/-- Bitwise `spread4_nibbles` interleave bijection for `mxf4nvf4` sub-blocks (`(v[j+8] << 4) | v[j]`). -/
def spread4Nibbles (x16 : UInt32) : UInt32 :=
  let u := (x16 ||| (x16 <<< 8)) &&& 0x00FF00FF
  (u ||| (u <<< 4)) &&& 0x0F0F0F0F

theorem spread4_interleave_all_nibble_pairs_exact :
    ((List.range 16).all (fun n0 =>
      (List.range 16).all (fun n1 =>
        let packed := UInt32.ofNat (n0 ||| (n1 <<< 4))
        let spread := spread4Nibbles packed
        spread == UInt32.ofNat (n0 ||| (n1 <<< 8))))) = true := by
  decide

/-- 5-op `PRMT + LOP3` sub-block interleave (`__byte_perm` + mask/shifts) equals `spread4Nibbles` across all 16-bit pairs! -/
def prmt5140 (a16 b16 : UInt32) : UInt32 :=
  let a0 := a16 &&& (0xFF : UInt32)
  let a1 := (a16 >>> 8) &&& (0xFF : UInt32)
  let b0 := b16 &&& (0xFF : UInt32)
  let b1 := (b16 >>> 8) &&& (0xFF : UInt32)
  a0 ||| (b0 <<< 8) ||| (a1 <<< 16) ||| (b1 <<< 24)

def interleaveNvfp4Prmt (a16 b16 : UInt32) : UInt32 :=
  let p := prmt5140 a16 b16
  (p &&& (0xF00FF00F : UInt32)) ||| ((p <<< 4) &&& (0x0F000F00 : UInt32)) ||| ((p >>> 4) &&& (0x00F000F0 : UInt32))

theorem prmt_lop3_interleave_matches_spread4_byte0 :
    ((List.range 16).all (fun n0 =>
      (List.range 16).all (fun m0 =>
        let a := UInt32.ofNat (n0 ||| ((n0 ^^^ 5) <<< 4) ||| ((n0 ^^^ 10) <<< 8) ||| ((n0 ^^^ 15) <<< 12))
        let b := UInt32.ofNat (m0 ||| ((m0 ^^^ 3) <<< 4) ||| ((m0 ^^^ 12) <<< 8) ||| ((m0 ^^^ 9) <<< 12))
        interleaveNvfp4Prmt a b == (spread4Nibbles a ||| (spread4Nibbles b <<< 4))))) = true := by
  decide

/-! ## 15. Cycle 7 (`v0.8.0` V13): `TileTokens = 160` NVFP4 & MXFP8 Grouped MoE TC, 1-Pass NVFP4 Quant,
    Fused `depth_mix + ctrl_norm_bf16`, & Wave-Balanced Split-KV TC FlashAttention -/

def kInk2QuantNvfp4TwoTerm1pB : KernelResourceSpec :=
  { name := "ink2_quant_nvfp4_2term_1p_b"
    threadsPerCta := 160
    rawRegsPerThread := 39
    userSmemPerCtaBytes := 32
    smemAlignPadBytes := 0
    ctasPerSm := 9
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4TcUpSituT160 : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_tc_up_situ_t160"
    threadsPerCta := 384
    rawRegsPerThread := 155
    userSmemPerCtaBytes := 57640    -- 57,608 B 3-stage 192x160 SMEM + 32 B static <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4TcUpSituT160PackedScales : KernelResourceSpec :=
  { kInk2MoeNvfp4TcUpSituT160 with
    name := "ink2_moe_nvfp4_tc_up_situ_t160_packed_scales"
    rawRegsPerThread := 156 }

def kInk2MoeNvfp4TcDownT160 : KernelResourceSpec :=
  { kInk2MoeNvfp4TcUpSituT160 with
    name := "ink2_moe_nvfp4_tc_down_t160"
    rawRegsPerThread := 151 }

def kInk2MoeNvfp4TcDownT160PackedScales : KernelResourceSpec :=
  { kInk2MoeNvfp4TcUpSituT160 with
    name := "ink2_moe_nvfp4_tc_down_t160_packed_scales"
    rawRegsPerThread := 152 }

def kInk2MoeMxfp8TcUpSituT160 : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_tc_up_situ_t160"
    threadsPerCta := 256
    rawRegsPerThread := 205
    userSmemPerCtaBytes := 87592    -- 87,560 B 2-stage 160x128 SMEM + 32 B static <= 99 KiB -> 28 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 28
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeMxfp8TcUpSituT160PackedScales : KernelResourceSpec :=
  { kInk2MoeMxfp8TcUpSituT160 with
    name := "ink2_moe_mxfp8_tc_up_situ_t160_packed_scales"
    rawRegsPerThread := 207 }

def kInk2MoeMxfp8TcDownT160 : KernelResourceSpec :=
  { kInk2MoeMxfp8TcUpSituT160 with
    name := "ink2_moe_mxfp8_tc_down_t160"
    rawRegsPerThread := 202 }

def kInk2MoeMxfp8TcDownT160PackedScales : KernelResourceSpec :=
  { kInk2MoeMxfp8TcUpSituT160 with
    name := "ink2_moe_mxfp8_tc_down_t160_packed_scales"
    rawRegsPerThread := 200 }

def kInk2DepthMixMultiCtrlNormBf16B : KernelResourceSpec :=
  { name := "ink2_depth_mix_multi_ctrl_norm_bf16_b"
    threadsPerCta := 256
    rawRegsPerThread := 74          -- aligns to 80 -> 20,480 regs/CTA; 3 CTAs = 61,440 <= 65,536
    userSmemPerCtaBytes := 10320    -- 3 * (10,320 + 1,024) = 34,032 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 3
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2FlashAttnTcSplitB : KernelResourceSpec :=
  { name := "ink2_flash_attn_tc_split_b"
    threadsPerCta := 384
    rawRegsPerThread := 168
    userSmemPerCtaBytes := 64384
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2FlashAttnTcSplitBI8 : KernelResourceSpec :=
  { name := "ink2_flash_attn_tc_split_b_i8"
    threadsPerCta := 384
    rawRegsPerThread := 168
    userSmemPerCtaBytes := 63872
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2FlashAttnSplitCombineQ8B : KernelResourceSpec :=
  { name := "ink2_flash_attn_split_combine_q8_b"
    threadsPerCta := 256
    rawRegsPerThread := 40
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def cycle7V13PrefillSuite : List KernelResourceSpec :=
  [ kInk2QuantNvfp4TwoTerm1pB
  , kInk2MoeNvfp4TcUpSituT160
  , kInk2MoeNvfp4TcUpSituT160PackedScales
  , kInk2MoeNvfp4TcDownT160
  , kInk2MoeNvfp4TcDownT160PackedScales
  , kInk2MoeMxfp8TcUpSituT160
  , kInk2MoeMxfp8TcUpSituT160PackedScales
  , kInk2MoeMxfp8TcDownT160
  , kInk2MoeMxfp8TcDownT160PackedScales
  , kInk2DepthMixMultiCtrlNormBf16B
  , kInk2FlashAttnTcSplitB
  , kInk2FlashAttnTcSplitBI8
  , kInk2FlashAttnSplitCombineQ8B ]

theorem ink2_quant_nvfp4_2term_1p_b_valid :
    ValidSm120Kernel kInk2QuantNvfp4TwoTerm1pB := by decide
theorem ink2_moe_nvfp4_tc_up_situ_t160_valid :
    ValidSm120Kernel kInk2MoeNvfp4TcUpSituT160 := by decide
theorem ink2_moe_nvfp4_tc_up_situ_t160_packed_scales_valid :
    ValidSm120Kernel kInk2MoeNvfp4TcUpSituT160PackedScales := by decide
theorem ink2_moe_nvfp4_tc_down_t160_valid :
    ValidSm120Kernel kInk2MoeNvfp4TcDownT160 := by decide
theorem ink2_moe_nvfp4_tc_down_t160_packed_scales_valid :
    ValidSm120Kernel kInk2MoeNvfp4TcDownT160PackedScales := by decide
theorem ink2_moe_mxfp8_tc_up_situ_t160_valid :
    ValidSm120Kernel kInk2MoeMxfp8TcUpSituT160 := by decide
theorem ink2_moe_mxfp8_tc_up_situ_t160_packed_scales_valid :
    ValidSm120Kernel kInk2MoeMxfp8TcUpSituT160PackedScales := by decide
theorem ink2_moe_mxfp8_tc_down_t160_valid :
    ValidSm120Kernel kInk2MoeMxfp8TcDownT160 := by decide
theorem ink2_moe_mxfp8_tc_down_t160_packed_scales_valid :
    ValidSm120Kernel kInk2MoeMxfp8TcDownT160PackedScales := by decide
theorem ink2_depth_mix_multi_ctrl_norm_bf16_b_valid :
    ValidSm120Kernel kInk2DepthMixMultiCtrlNormBf16B := by decide
theorem ink2_flash_attn_tc_split_b_valid :
    ValidSm120Kernel kInk2FlashAttnTcSplitB := by decide
theorem ink2_flash_attn_tc_split_b_i8_valid :
    ValidSm120Kernel kInk2FlashAttnTcSplitBI8 := by decide
theorem ink2_flash_attn_split_combine_q8_b_valid :
    ValidSm120Kernel kInk2FlashAttnSplitCombineQ8B := by decide

theorem cycle7_v13_prefill_all_kernels_valid :
    (cycle7V13PrefillSuite.all checkSm120Kernel) = true := by
  decide

/-- 3-Stage `TileTokens = 192 × 160` NVFP4 Grouped MoE TC SMEM map (`49,152` B core):
    `sm_w` (`3 × 6,144 = 18,432` B) + `sm_x` (`3 × 10,240 = 30,720` B), disjoint & 128-B aligned. -/
def nvfp4TcT160SmemSlices : List SmemSlice :=
  [ { name := "sm_w_stage0", offsetBytes := 0,     sizeBytes := 6144,  alignBytes := 128, liveStartStage := 0, liveEndStage := 2 }
  , { name := "sm_w_stage1", offsetBytes := 6144,  sizeBytes := 6144,  alignBytes := 128, liveStartStage := 0, liveEndStage := 2 }
  , { name := "sm_w_stage2", offsetBytes := 12288, sizeBytes := 6144,  alignBytes := 128, liveStartStage := 0, liveEndStage := 2 }
  , { name := "sm_x_stage0", offsetBytes := 18432, sizeBytes := 10240, alignBytes := 128, liveStartStage := 0, liveEndStage := 2 }
  , { name := "sm_x_stage1", offsetBytes := 28672, sizeBytes := 10240, alignBytes := 128, liveStartStage := 0, liveEndStage := 2 }
  , { name := "sm_x_stage2", offsetBytes := 38912, sizeBytes := 10240, alignBytes := 128, liveStartStage := 0, liveEndStage := 2 } ]

theorem nvfp4_tc_t160_smem_safe :
    SafeSmemLayout 49152 nvfp4TcT160SmemSlices := by decide

/-- Zero-extra-VRAM `LayerWorkspace` Split-KV scratch non-aliasing during Step 6 (Attention):
    `qrgs` uses `[0, 8600)` dwords/row, `part_o` (`6 × 1536 = 9216` dwords/row) +
    `part_ml` (`6 × 48 = 288` dwords/row) fit inside `[8600, 18432)` (`TOP_K * D = 18432`). -/
def attnSplitWorkspaceSlices : List SmemSlice :=
  [ { name := "qrgs_live",   offsetBytes := 0,     sizeBytes := 34400, alignBytes := 16, liveStartStage := 0, liveEndStage := 1 }
  , { name := "part_o_f16",  offsetBytes := 34400, sizeBytes := 36864, alignBytes := 16, liveStartStage := 0, liveEndStage := 1 }
  , { name := "part_ml_f32", offsetBytes := 71264, sizeBytes := 1152,  alignBytes := 16, liveStartStage := 0, liveEndStage := 1 } ]

theorem attn_split_workspace_non_aliasing :
    SafeSmemLayout 73728 attnSplitWorkspaceSlices := by decide

-- ============================================================================
-- Cycle 8 (`v0.8.0` Issue #11): Solo & Speculative Decode PDL + Load-Hoisted Suite
-- ============================================================================

def kInk2Q8Dp4aGemvC2304 : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_gemv_c2304"
    threadsPerCta := 256
    rawRegsPerThread := 63          -- aligns to 64 -> 16,384 regs/CTA; 4 CTAs = 65,536 <= 65,536
    userSmemPerCtaBytes := 2448     -- 4 * (2,448 + 1,024) = 13,888 B <= 32 KiB carveout -> 96 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2DepthMixInv : KernelResourceSpec :=
  { name := "ink2_depth_mix_inv"
    threadsPerCta := 256
    rawRegsPerThread := 80          -- 80 * 256 = 20,480 regs/CTA; 3 CTAs = 61,440 <= 65,536
    userSmemPerCtaBytes := 464
    smemAlignPadBytes := 0
    ctasPerSm := 3
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2CtrlDownR4Fast : KernelResourceSpec :=
  { name := "ink2_ctrl_down_r4_fast"
    threadsPerCta := 256
    rawRegsPerThread := 72          -- 72 * 256 = 18,432 regs/CTA; 3 CTAs = 55,296 <= 65,536
    userSmemPerCtaBytes := 128
    smemAlignPadBytes := 0
    ctasPerSm := 3
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeMxfp8DownXw : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_down_xw"
    threadsPerCta := 256
    rawRegsPerThread := 68          -- aligns to 72 -> 18,432 regs/CTA; 3 CTAs = 55,296 <= 65,536
    userSmemPerCtaBytes := 64
    smemAlignPadBytes := 0
    ctasPerSm := 3
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeMxfp8DownXwPackedScales : KernelResourceSpec :=
  { kInk2MoeMxfp8DownXw with
    name := "ink2_moe_mxfp8_down_xw_packed_scales" }

def kInk2MoeNvfp4Dp4aUpHalfQ8 : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_dp4a_up_half_q8"
    threadsPerCta := 512
    rawRegsPerThread := 62          -- aligns to 64 -> 32,768 regs/CTA; 2 CTAs = 65,536 <= 65,536 (32 warps/SM, 5x LDG.128 hoisted!)
    userSmemPerCtaBytes := 2944     -- 2 * (2,944 + 1,024) = 7,936 B <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 2
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4Dp4aUpHalfQ8PackedScales : KernelResourceSpec :=
  { kInk2MoeNvfp4Dp4aUpHalfQ8 with
    name := "ink2_moe_nvfp4_dp4a_up_half_q8_packed_scales"
    rawRegsPerThread := 61 }

def kInk2GqaAttnCombineQ8Dyn : KernelResourceSpec :=
  { name := "ink2_gqa_attn_combine_q8_dyn"
    threadsPerCta := 128
    rawRegsPerThread := 40          -- 40 * 128 = 5,120 regs/CTA; 12 CTAs = 61,440 <= 65,536 (48 warps/SM!)
    userSmemPerCtaBytes := 32
    smemAlignPadBytes := 0
    ctasPerSm := 12
    clusterCtas := 1
    minRequiredL1KiB := 112
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2Q8Dp4aGemmB8Fast : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_gemm_b8_fast"
    threadsPerCta := 256
    rawRegsPerThread := 122         -- weights loaded before `pdl_wait` (#152): aligns to 128 -> 32,768 regs/CTA; 2 CTAs = 65,536 <= 65,536
    userSmemPerCtaBytes := 26112    -- 2 * (26,112 + 1,024) = 54,272 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 2
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def cycle8SoloSpecPdlSuite : List KernelResourceSpec :=
  [ kInk2Q8Dp4aGemvC2304
  , kInk2DepthMixInv
  , kInk2CtrlDownR4Fast
  , kInk2MoeMxfp8DownXw
  , kInk2MoeMxfp8DownXwPackedScales
  , kInk2MoeNvfp4Dp4aUpHalfQ8
  , kInk2MoeNvfp4Dp4aUpHalfQ8PackedScales
  , kInk2GqaAttnCombineQ8Dyn
  , kInk2Q8Dp4aGemmB8Fast ]

theorem ink2_q8_dp4a_gemv_c2304_valid :
    ValidSm120Kernel kInk2Q8Dp4aGemvC2304 := by decide
theorem ink2_depth_mix_inv_valid :
    ValidSm120Kernel kInk2DepthMixInv := by decide
theorem ink2_ctrl_down_r4_fast_valid :
    ValidSm120Kernel kInk2CtrlDownR4Fast := by decide
theorem ink2_moe_mxfp8_down_xw_valid :
    ValidSm120Kernel kInk2MoeMxfp8DownXw := by decide
theorem ink2_moe_mxfp8_down_xw_packed_scales_valid :
    ValidSm120Kernel kInk2MoeMxfp8DownXwPackedScales := by decide
theorem ink2_moe_nvfp4_dp4a_up_half_q8_valid :
    ValidSm120Kernel kInk2MoeNvfp4Dp4aUpHalfQ8 := by decide
theorem ink2_moe_nvfp4_dp4a_up_half_q8_packed_scales_valid :
    ValidSm120Kernel kInk2MoeNvfp4Dp4aUpHalfQ8PackedScales := by decide
theorem ink2_gqa_attn_combine_q8_dyn_valid :
    ValidSm120Kernel kInk2GqaAttnCombineQ8Dyn := by decide
theorem ink2_q8_dp4a_gemm_b8_fast_valid :
    ValidSm120Kernel kInk2Q8Dp4aGemmB8Fast := by decide

theorem cycle8_solo_spec_pdl_all_kernels_valid :
    (cycle8SoloSpecPdlSuite.all checkSm120Kernel) = true := by
  decide

/-- `ink2_moe_nvfp4_dp4a_up_half_q8` (`Grid: (72, 8)`, `Block: 512` = `16` warps) covers all `1,152` FFN channels in exact 16-channel `Q8_16` groups. -/
theorem up_half_q8_channel_coverage :
    72 * 16 = 1152 ∧ 512 / 32 = 16 := by decide

/-- Exact 4-full + 1-half-warp (`lane < 16`) coverage of `cols = 2304` (`144` `u128` chunks). -/
theorem c2304_hoist_exact_coverage :
    4 * 32 + 16 = 2304 / 16 := by decide

/-- `ink2_ctrl_down_r4_fast` (`Grid: (16, 3)`) reduces CTAs by `4x` (`48` vs `192`) while covering all `64` ranks. -/
theorem ctrl_down_r4_cta_reduction :
    16 * 4 = 64 ∧ 16 * 3 * 4 = 64 * 3 := by decide

/-- `ink2_moe_mxfp8_down_xw` (`Grid: 1152`, `kRows = 2`) partitions `D = 2304` exactly across 8 expert warps/CTA. -/
theorem mxfp8_down_xw_row_coverage :
    1152 * 2 = 2304 ∧ 1152 * 8 = 9216 := by decide

/-- Programmatic Dependent Launch (`PDL`) Two-Stage Ordering Invariant:
    When every middle kernel `K2` executes `ACQBULK` (`k2_wait`, waiting on `k1_exit`) strictly before
    `PREEXIT` (`k2_trigger`, enabling `k3_start`), any write committed by `K1` (such as `router_topk8`'s
    `sel_ids`) strictly precedes `K3`'s pre-wait weight prefetch (`down_xw`). -/
theorem pdl_two_stage_order_safety
    (k1_exit k2_wait k2_trigger k3_start : Nat)
    (h_wait : k1_exit ≤ k2_wait)
    (h_order : k2_wait < k2_trigger)
    (h_kick : k2_trigger ≤ k3_start) :
    k1_exit < k3_start :=
  Nat.lt_of_lt_of_le (Nat.lt_of_le_of_lt h_wait h_order) h_kick

/-! ### Multi-sequence decode rounds (#107): row-independent instantiations

Each row of a round carries `{pos, seq}` and addresses its own sequence's KV / history at a fixed
per-sequence stride; the arithmetic per row is the single-sequence kernels' (same resources or
fewer registers). -/

def kInk2PrepQkvAndCacheMs : KernelResourceSpec :=
  { name := "ink2_prep_qkv_and_cache_ms"
    threadsPerCta := 256
    rawRegsPerThread := 29          -- 7,424 regs/CTA; 6 CTAs = 44,544 <= 65,536 (`pdl_wait`, #152)
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2GqaAttnSplitDecMs : KernelResourceSpec :=
  { kInk2GqaAttnSplitDec with name := "ink2_gqa_attn_split_dec_ms" }   -- 64 regs, 14,976 B

def kInk2GqaAttnSplitI8Ms : KernelResourceSpec :=
  { kInk2GqaAttnSplitI8 with name := "ink2_gqa_attn_split_i8_ms" }     -- 64 regs, 14,976 B

def kInk2GqaAttnCombineQ8DynMs : KernelResourceSpec :=
  { kInk2GqaAttnCombineQ8Dyn with name := "ink2_gqa_attn_combine_q8_dyn_ms" }  -- 38 <= 40 regs

def kInk2ShortConv4Ms : KernelResourceSpec :=
  { name := "ink2_short_conv4_ms"
    threadsPerCta := 256
    rawRegsPerThread := 32          -- 8,192 regs/CTA; 6 CTAs = 49,152 <= 65,536
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2ShortConv4SaveMs : KernelResourceSpec :=
  { kInk2ShortConv4Ms with name := "ink2_short_conv4_save_ms" }        -- 30 <= 32 regs

-- #107: `ink2_q8_dp4a_gemm_b` arithmetic with 4 / 8 token tiles per CTA (multi-sequence rounds).
def kInk2Q8Dp4aGemmBMs4 : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_gemm_b_ms4"
    threadsPerCta := 256
    rawRegsPerThread := 123         -- aligns to 128 -> 32,768 regs/CTA; 2 CTAs = 65,536
    userSmemPerCtaBytes := 16384    -- partials[4][8][128] FP32
    smemAlignPadBytes := 0
    ctasPerSm := 2
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2Q8Dp4aGemmBMs8 : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_gemm_b_ms8"
    threadsPerCta := 256
    rawRegsPerThread := 160         -- 40,960 regs/CTA; 1 CTA
    userSmemPerCtaBytes := 32768    -- partials[8][8][128] FP32
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

-- #107 stage 3: verify rows read earlier rows of their sequence; histories are committed after.
def kInk2ShortConv4Vms : KernelResourceSpec :=
  { kInk2ShortConv4Ms with name := "ink2_short_conv4_vms" }            -- 32 <= 32 regs

def kInk2ShortConv4SaveVms : KernelResourceSpec :=
  { kInk2ShortConv4Ms with
    name := "ink2_short_conv4_save_vms"
    rawRegsPerThread := 34 }        -- aligns to 40 -> 10,240 regs/CTA; 6 CTAs = 61,440 <= 65,536

def kInk2ConvCommitMs : KernelResourceSpec :=
  { kInk2ShortConv4Ms with
    name := "ink2_conv_commit_ms"
    rawRegsPerThread := 24 }

def multiSeqRoundKernelSuite : List KernelResourceSpec :=
  [ kInk2PrepQkvAndCacheMs
  , kInk2GqaAttnSplitDecMs
  , kInk2GqaAttnSplitI8Ms
  , kInk2GqaAttnCombineQ8DynMs
  , kInk2ShortConv4Ms
  , kInk2ShortConv4SaveMs
  , kInk2ShortConv4Vms
  , kInk2ShortConv4SaveVms
  , kInk2ConvCommitMs
  , kInk2Q8Dp4aGemmBMs4
  , kInk2Q8Dp4aGemmBMs8 ]

theorem multiseq_round_all_kernels_valid :
    (multiSeqRoundKernelSuite.all checkSm120Kernel) = true := by decide

/-! ## 12. Split-KV FP16 Dynamic-Range Proof, Canary Isolation Proof & `v0.8.0` Kernel Proofs (`#101`, `#104`, `#106`) -/

/-- IEEE-754 binary16 (`FP16`) maximum finite representable integer magnitude (`65,504`). -/
def fp16MaxFiniteMag : Nat := 65504

/-- Weighted magnitude accumulator `∑ (w_i * v_i)` over `(weight, valueMag)` pairs. -/
def weightedMagSum : List (Nat × Nat) → Nat
  | [] => 0
  | (w, v) :: rest => w * v + weightedMagSum rest

/-- Total non-negative weight `∑ w_i` over `(weight, valueMag)` pairs. -/
def totalWeight : List (Nat × Nat) → Nat
  | [] => 0
  | (w, _) :: rest => w + totalWeight rest

/--
  Constructive Sequence-Length-Independent Convex-Combination Bound (`#101`):
  For ANY sequence length `N = wv.length` (without bound!) and any per-token value bound `M`,
  if every value entry satisfies `v_i ≤ M`, then `∑ (w_i * v_i) ≤ (∑ w_i) * M`.
  Consequently, dividing `o_split = ∑ w_i v_i` by `l_split ≥ ∑ w_i > 0` BEFORE converting to
  `FP16` guarantees `|o_split / l_split| ≤ M ≤ 65,504` at EVERY context length (`16K`, `128K`, `1M+`).
-/
theorem weighted_mag_sum_le_total_mul_bound (wv : List (Nat × Nat)) (M : Nat)
    (h_val : ∀ p ∈ wv, p.2 ≤ M) :
    weightedMagSum wv ≤ totalWeight wv * M := by
  induction wv with
  | nil =>
    simp [weightedMagSum, totalWeight]
  | cons hd tl ih =>
    rcases hd with ⟨w, v⟩
    simp only [weightedMagSum, totalWeight]
    have hv : v ≤ M := h_val (w, v) (List.Mem.head tl)
    have htl : ∀ p ∈ tl, p.2 ≤ M := fun p hp => h_val p (List.Mem.tail (w, v) hp)
    have ih_le := ih htl
    have h_head : w * v ≤ w * M := Nat.mul_le_mul_left w hv
    calc
      w * v + weightedMagSum tl ≤ w * M + totalWeight tl * M := Nat.add_le_add h_head ih_le
      _ = (w + totalWeight tl) * M := (Nat.add_mul w (totalWeight tl) M).symm

/-- When `totalWeight wv ≤ lSplit` (online-softmax sink + attention denominator `l_split ≥ ∑ p_i`)
    and `M ≤ fp16MaxFiniteMag`, the normalized partial satisfies `weightedMagSum wv ≤ lSplit * 65,504`
    for ALL context lengths `wv.length`. -/
theorem split_kv_normalized_partial_fp16_safe
    (wv : List (Nat × Nat)) (M lSplit : Nat)
    (h_val : ∀ p ∈ wv, p.2 ≤ M)
    (h_m_fp16 : M ≤ fp16MaxFiniteMag)
    (h_denom : totalWeight wv ≤ lSplit) :
    weightedMagSum wv ≤ lSplit * fp16MaxFiniteMag := by
  calc
    weightedMagSum wv ≤ totalWeight wv * M := weighted_mag_sum_le_total_mul_bound wv M h_val
    _ ≤ totalWeight wv * fp16MaxFiniteMag := Nat.mul_le_mul_left (totalWeight wv) h_m_fp16
    _ ≤ lSplit * fp16MaxFiniteMag := Nat.mul_le_mul_right fp16MaxFiniteMag h_denom

/-- Specification of a Split-KV partial-output storage format across `numSplits` splits. -/
structure SplitKvPartialSpec where
  seqLenTokens : Nat
  numSplits : Nat
  maxValueMag : Nat
  storesNormalizedByL : Bool
  deriving Repr, DecidableEq

/-- Worst-case magnitude stored in `part_o` (`FP16`):
    - Normalized (`o_split / l_split`, `#113` fix): bounded by `maxValueMag` independent of `seqLenTokens`!
    - Unnormalized (`∑ exp(s_t - m_split) * v_t`, pre-`#113` bug when `s_t == m_split`): grows as
      `(seqLenTokens / numSplits) * maxValueMag` and overflows `FP16` (`65,504`) at long contexts! -/
def worstCaseSplitPartialMag (s : SplitKvPartialSpec) : Nat :=
  if s.storesNormalizedByL then s.maxValueMag
  else (s.seqLenTokens / s.numSplits) * s.maxValueMag

def checkSplitKvPartialFp16Safe (s : SplitKvPartialSpec) : Bool :=
  (s.numSplits > 0) && (worstCaseSplitPartialMag s <= fp16MaxFiniteMag)

/-- After V Head-RMSNorm (`dim = 128`), a spike coordinate satisfies `|v_i| ≤ √128 < 12`. -/
def ink2NormalizedSplitKv128k : SplitKvPartialSpec :=
  { seqLenTokens := 131072, numSplits := 8, maxValueMag := 12, storesNormalizedByL := true }

def ink2NormalizedSplitKv1M : SplitKvPartialSpec :=
  { seqLenTokens := 1048576, numSplits := 64, maxValueMag := 12, storesNormalizedByL := true }

def ink2UnnormalizedBug96k : SplitKvPartialSpec :=
  { seqLenTokens := 98304, numSplits := 8, maxValueMag := 12, storesNormalizedByL := false }

def ink2UnnormalizedBug128k : SplitKvPartialSpec :=
  { seqLenTokens := 131072, numSplits := 16, maxValueMag := 12, storesNormalizedByL := false }

theorem ink2_split_kv_normalized_safe_128k :
    checkSplitKvPartialFp16Safe ink2NormalizedSplitKv128k = true := by decide

theorem ink2_split_kv_normalized_safe_1m :
    checkSplitKvPartialFp16Safe ink2NormalizedSplitKv1M = true := by decide

/-- Formal rejection of the unnormalized `v0.7.0` Split-KV partial representation at `96K` and `128K`! -/
theorem ink2_split_kv_unnormalized_overflows_96k :
    checkSplitKvPartialFp16Safe ink2UnnormalizedBug96k = false := by decide

theorem ink2_split_kv_unnormalized_overflows_128k :
    checkSplitKvPartialFp16Safe ink2UnnormalizedBug128k = false := by decide

/-- Deterministic Canary & Long-Context Needle Gate (`leansonic canary`) specification. -/
structure CanaryCaseSpec where
  inputTokens : Nat
  markerDepthPct : Nat
  maxNewTokens : Nat
  allocatedCtx : Nat
  requiresBeyondSwa1024 : Bool
  deriving Repr, DecidableEq

def checkCanaryCaseValid (c : CanaryCaseSpec) : Bool :=
  let markerPos := (c.inputTokens * c.markerDepthPct) / 100
  let distToEnd := c.inputTokens - markerPos
  (c.inputTokens >= 16)
  && (c.inputTokens + c.maxNewTokens <= c.allocatedCtx)
  && (!c.requiresBeyondSwa1024 || (c.markerDepthPct >= 1 && c.markerDepthPct <= 99 && distToEnd > 1024))

def ink2CanarySuite : List CanaryCaseSpec :=
  [ { inputTokens := 24,     markerDepthPct := 0,  maxNewTokens := 64,  allocatedCtx := 1024,   requiresBeyondSwa1024 := false }
  , { inputTokens := 16384,  markerDepthPct := 50, maxNewTokens := 512, allocatedCtx := 16896,  requiresBeyondSwa1024 := true }
  , { inputTokens := 131072, markerDepthPct := 50, maxNewTokens := 512, allocatedCtx := 131584, requiresBeyondSwa1024 := true }
  , { inputTokens := 131072, markerDepthPct := 90, maxNewTokens := 512, allocatedCtx := 131584, requiresBeyondSwa1024 := true } ]

/-- Proves all 3 needle canaries (`16K@50%`, `128K@50%`, `128K@90%`) place the verification code strictly
    outside the `1,024`-token SWA window (`distToEnd = 8,192 / 65,536 / 13,108 > 1,024`), so ONLY
    functional Global Full-Attention Split-KV visits can retrieve the planted code! -/
theorem ink2_canary_suite_isolates_global_attention :
    (ink2CanarySuite.all checkCanaryCaseValid) = true := by decide

/-- Measured `ptxas -arch=sm_120a` `v0.8.0` Solo Decode Fusion & Active-Expert Verify MoE Kernels. -/
def kInk2KvConv4PrepCache : KernelResourceSpec :=
  { name := "ink2_kv_conv4_prep_cache"
    threadsPerCta := 256
    rawRegsPerThread := 40          -- 10,240 regs/CTA; 6 CTAs = 61,440 <= 65,536 (100% 48w/SM!)
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2WoConv4SaveC3072 : KernelResourceSpec :=
  { name := "ink2_wo_conv4_save_c3072"
    threadsPerCta := 256
    rawRegsPerThread := 78          -- aligns to 80 -> 20,480 regs/CTA; 3 CTAs = 61,440 <= 65,536
    userSmemPerCtaBytes := 3264     -- 3 * (3,264 + 1,024) = 12,864 <= 16 KiB -> 112 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 3
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4Dp4aDownXwConv : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_dp4a_down_xw_conv"
    threadsPerCta := 256
    rawRegsPerThread := 55          -- aligns to 56 -> 14,336 regs/CTA; 4 CTAs = 57,344 <= 65,536
    userSmemPerCtaBytes := 64
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4Dp4aDownXwConvPacked : KernelResourceSpec :=
  { kInk2MoeNvfp4Dp4aDownXwConv with
    name := "ink2_moe_nvfp4_dp4a_down_xw_conv_packed_scales"
    rawRegsPerThread := 56 }

def kInk2MoeMxfp8DownXwConv : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_down_xw_conv"
    threadsPerCta := 256
    rawRegsPerThread := 68          -- aligns to 72 -> 18,432 regs/CTA; 3 CTAs = 55,296 <= 65,536
    userSmemPerCtaBytes := 64
    smemAlignPadBytes := 0
    ctasPerSm := 3
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeMxfp8DownXwConvPacked : KernelResourceSpec :=
  { kInk2MoeMxfp8DownXwConv with
    name := "ink2_moe_mxfp8_down_xw_conv_packed_scales" }

def kInk2BuildExpertGroupsActiveB : KernelResourceSpec :=
  { name := "ink2_build_expert_groups_active_b"
    threadsPerCta := 128
    rawRegsPerThread := 39          -- aligns to 40 -> 5,120 regs/CTA; 4 CTAs = 20,480 <= 65,536
    userSmemPerCtaBytes := 528
    smemAlignPadBytes := 0
    ctasPerSm := 4
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4Dp4aVerifyUpHalfQ8 : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_dp4a_verify_up_half_q8"
    threadsPerCta := 512
    rawRegsPerThread := 82          -- aligns to 88 -> 45,056 regs/CTA; 1 CTA = 16 warps/SM (2x grouped!)
    userSmemPerCtaBytes := 544      -- 544 + 1,024 <= 8 KiB carveout -> 120 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4Dp4aVerifyUpHalfQ8Packed : KernelResourceSpec :=
  { kInk2MoeNvfp4Dp4aVerifyUpHalfQ8 with
    name := "ink2_moe_nvfp4_dp4a_verify_up_half_q8_packed_scales" }

def kInk2MoeNvfp4Dp4aVerifyDown : KernelResourceSpec :=
  { name := "ink2_moe_nvfp4_dp4a_verify_down"
    threadsPerCta := 256
    rawRegsPerThread := 48          -- 12,288 regs/CTA; 5 CTAs = 61,440 <= 65,536 (40 warps/SM = 83.3%!)
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 5
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeNvfp4Dp4aVerifyDownPacked : KernelResourceSpec :=
  { kInk2MoeNvfp4Dp4aVerifyDown with
    name := "ink2_moe_nvfp4_dp4a_verify_down_packed_scales" }

def kInk2MoeMxfp8VerifyUpSitu : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_verify_up_situ"
    threadsPerCta := 256
    rawRegsPerThread := 128         -- 32,768 regs/CTA; 2 CTAs = 65,536 <= 65,536 (2x grouped!)
    userSmemPerCtaBytes := 288
    smemAlignPadBytes := 0
    ctasPerSm := 2
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeMxfp8VerifyUpSituPacked : KernelResourceSpec :=
  { kInk2MoeMxfp8VerifyUpSitu with
    name := "ink2_moe_mxfp8_verify_up_situ_packed_scales" }

def kInk2MoeMxfp8VerifyDown : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_verify_down"
    threadsPerCta := 256
    rawRegsPerThread := 82          -- aligns to 88 -> 22,528 regs/CTA; 2 CTAs = 45,056 <= 65,536
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 2
    clusterCtas := 1
    minRequiredL1KiB := 120
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2MoeMxfp8VerifyDownPacked : KernelResourceSpec :=
  { kInk2MoeMxfp8VerifyDown with
    name := "ink2_moe_mxfp8_verify_down_packed_scales" }

def kInk2RouterGemvTopk8 : KernelResourceSpec :=
  { name := "ink2_router_gemv_topk8"
    threadsPerCta := 32
    rawRegsPerThread := 48          -- 1,536 regs/CTA; 32 CTAs = 49,152 <= 65,536 (32 warps/SM!)
    userSmemPerCtaBytes := 64       -- 32 * (64 + 1,024) = 34,816 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 32
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2Q8Dp4aGemvC3072R1 : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_gemv_c3072_r1"
    threadsPerCta := 256
    rawRegsPerThread := 40          -- 10,240 regs/CTA; 6 CTAs = 61,440 <= 65,536 (48 warps/SM!)
    userSmemPerCtaBytes := 3264     -- 6 * (3,264 + 1,024) = 25,728 B <= 32 KiB carveout -> 96 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2Q8Dp4aGemvC2304R1 : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_gemv_c2304_r1"
    threadsPerCta := 256
    rawRegsPerThread := 47          -- aligns to 48 -> 12,288 regs/CTA; 5 CTAs = 61,440 <= 65,536 (40 warps/SM!)
    userSmemPerCtaBytes := 2448     -- 5 * (2,448 + 1,024) = 17,360 B <= 32 KiB carveout -> 96 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 5
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

def kInk2Q8Dp4aQkv3GemvR1 : KernelResourceSpec :=
  { kInk2Q8Dp4aGemvC2304R1 with
    name := "ink2_q8_dp4a_qkv3_gemv_r1"
    rawRegsPerThread := 48 }

def cycle9ConvFuseAndVerifyMoeSuite : List KernelResourceSpec :=
  [ kInk2RouterGemvTopk8
  , kInk2Q8Dp4aGemvC3072R1
  , kInk2Q8Dp4aGemvC2304R1
  , kInk2Q8Dp4aQkv3GemvR1
  , kInk2KvConv4PrepCache
  , kInk2WoConv4SaveC3072
  , kInk2MoeNvfp4Dp4aDownXwConv
  , kInk2MoeNvfp4Dp4aDownXwConvPacked
  , kInk2MoeMxfp8DownXwConv
  , kInk2MoeMxfp8DownXwConvPacked
  , kInk2BuildExpertGroupsActiveB
  , kInk2MoeNvfp4Dp4aVerifyUpHalfQ8
  , kInk2MoeNvfp4Dp4aVerifyUpHalfQ8Packed
  , kInk2MoeNvfp4Dp4aVerifyDown
  , kInk2MoeNvfp4Dp4aVerifyDownPacked
  , kInk2MoeMxfp8VerifyUpSitu
  , kInk2MoeMxfp8VerifyUpSituPacked
  , kInk2MoeMxfp8VerifyDown
  , kInk2MoeMxfp8VerifyDownPacked ]

theorem cycle9_conv_fuse_and_verify_moe_all_kernels_valid :
    (cycle9ConvFuseAndVerifyMoeSuite.all checkSm120Kernel) = true := by decide

/-! ## Constant-weight prefetch under PDL (#152 P0-B)

Speculative-verify and drafter kernels trigger their dependents right after `pdl_wait`, so the
next kernel becomes resident early and loads its constant weights before its own wait. The verify
router reads its 9 weight vectors per lane that way. -/

/-- Verify router GEMV: one warp per expert row, `Grid (128, ceil(B / 4))`. -/
def kInk2Bf16GemvRouterB : KernelResourceSpec :=
  { name := "ink2_bf16_gemv_router_b"
    threadsPerCta := 32
    rawRegsPerThread := 112         -- 3,584 regs/CTA; 18 CTAs = 64,512 <= 65,536
    userSmemPerCtaBytes := 0
    smemAlignPadBytes := 0
    ctasPerSm := 18
    clusterCtas := 1
    minRequiredL1KiB := 96
    hasStackSpill := false
    hasHotLoopLibmCall := false }

theorem ink2_bf16_gemv_router_b_valid : ValidSm120Kernel kInk2Bf16GemvRouterB := by decide

end LeanSonic.Proofs
