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
  , kInk2MoeNvfp4Dp4aUpSitu
  , kInk2MoeNvfp4Dp4aDown
  , kInk2MoeMxfp8Down
  , kInk2GqaRelSinkAttn
  , kInk2RmsnormQuantQ8 ]

theorem ink2_q8_dp4a_gemv_valid : ValidSm120Kernel kInk2Q8Dp4aGemv := by decide
theorem ink2_moe_nvfp4_dp4a_up_situ_valid : ValidSm120Kernel kInk2MoeNvfp4Dp4aUpSitu := by decide
theorem ink2_moe_nvfp4_dp4a_down_valid : ValidSm120Kernel kInk2MoeNvfp4Dp4aDown := by decide
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

/-- Tile-4 Grouped-Expert MXFP8 Up+SITU (`48` regs, `37,008` B SMEM -> `64` KiB L1!). -/
def kInk2MoeMxfp8GroupedUpSitu : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_grouped_up_situ"
    threadsPerCta := 256
    rawRegsPerThread := 48
    userSmemPerCtaBytes := 37008    -- 1 * (37008 + 1024) = 38,032 B <= 64 KiB carveout -> 64 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

/-- Tile-8 Grouped-Expert MXFP8 Down (`64` regs, `36,896` B SMEM -> `64` KiB L1!). -/
def kInk2MoeMxfp8GroupedDown : KernelResourceSpec :=
  { name := "ink2_moe_mxfp8_grouped_down"
    threadsPerCta := 256
    rawRegsPerThread := 64
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
  , kInk2GqaAttnCombine ]

theorem ink2_quant_q8_16_packed_valid : ValidSm120Kernel kInk2QuantQ8Packed := by decide
theorem ink2_gqa_attn_split_valid : ValidSm120Kernel kInk2GqaAttnSplit := by decide
theorem ink2_gqa_attn_split_i8_valid : ValidSm120Kernel kInk2GqaAttnSplitI8 := by decide
theorem ink2_gqa_attn_combine_valid : ValidSm120Kernel kInk2GqaAttnCombine := by decide

theorem cycle4_splitkv_all_kernels_valid :
    (cycle4SplitKvKernelSuite.all checkSm120Kernel) = true := by
  decide

/-! ## Batched controller tiles (256 threads per CTA, sm_120a) -/

def kInk2CtrlDownR8B : KernelResourceSpec :=
  { name := "ink2_ctrl_down_r8_b"
    threadsPerCta := 256
    rawRegsPerThread := 48
    userSmemPerCtaBytes := 18760
    smemAlignPadBytes := 0
    ctasPerSm := 5
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

/-- Fused 3-Projection `(Q, K, V)` Planar `Q8_64` `__dp4a` GEMV (`256` threads, `34` regs, `2,448` B SMEM -> `6` CTAs = `48` warps/SM!). -/
def kInk2Q8Dp4aQkv3Gemv : KernelResourceSpec :=
  { name := "ink2_q8_dp4a_qkv3_gemv"
    threadsPerCta := 256
    rawRegsPerThread := 34          -- aligns to 40 -> 10,240 regs/CTA; 6 CTAs = 61,440 <= 65,536 (48 warps/SM!)
    userSmemPerCtaBytes := 2448     -- 6 * (2448 + 1024) = 20,832 B <= 32 KiB carveout -> 96 KiB L1!
    smemAlignPadBytes := 0
    ctasPerSm := 6
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

end LeanSonic.Proofs
