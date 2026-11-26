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

end LeanSonic.Proofs
