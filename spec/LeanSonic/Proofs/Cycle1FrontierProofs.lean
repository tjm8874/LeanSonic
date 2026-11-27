/-
  LeanSonic — Cycle 1 Frontier Research & Formal Verification Upgrades
  ============================================================================
  Machine-checked Lean 4 theorems (zero `sorry`, zero `axiom`) covering:
    1. Tightened `sm_120` Hardware Launch Laws (`checkSm120KernelStrict`) &
       Planted Red-Control Counterexamples exposing boundary holes in `checkSm120Kernel`.
    2. Universal Bounds, Closed-Form Decoding & Injectivity (Zero-Collision) of
       `expertMapOffset` across all 128 experts and arbitrary `64`-aligned prefill chunks.
    4. Sharp `span <= 3` Integer-Exact Microscaling Theorem (`arXiv:2611.00486`)
       for `E2M1 -> INT8` Single-Scale GEMM without inner-loop rescaling.
    5. 4-Chain `(c0 + c1) + (c2 + c3)` `0-ULP` Batch Invariance (`dp4a` == `mma.m16n8k32`)
       over uninterpreted IEEE-754 `fma` / `fadd`.
    6. Sink-Corrected Key-Mean Smoothing Shift Invariance (`SageAttention` ×
       `Inkling-2` Attention Sink exact logit-difference preservation).
    7. `CPTC` (`arXiv:2611.22019`) 64-Byte Mirror-Fold
       Table Index Safety.

  The `.range_v1` SWAR (`BitVec 64`) and `CPTC` scalar-embedding theorems of #197 are not
  included: their `bv_decide` proofs depend on `_native.bv_decide` axioms (compiler-trusted
  evaluation, the `Lean.ofReduceBool` class that Gate 1 forbids).
-/

import LeanSonic.Axioms
import LeanSonic.Proofs.Inkling2Mini
import LeanSonic.Proofs.SpilledAndEngram

namespace LeanSonic.Proofs.Cycle1Frontier

open LeanSonic
open LeanSonic.Proofs

-- ============================================================================
-- 1. Tightened `sm_120` Hardware Launch Laws & Planted Red-Control Proofs
-- ============================================================================

/-- Hardware per-CTA thread ceiling on NVIDIA Blackwell `sm_120` (`maxThreadsPerBlock = 1024`). -/
def sm120MaxThreadsPerCta : Nat := 1024

/-- Hardware resident CTA ceiling per SM on `sm_120` (`maxBlocksPerMultiProcessor = 32`). -/
def sm120MaxCtasPerSm : Nat := 32

/-- Hardware architectural per-thread register ceiling (`R0..R254`, since `R255 = RZ`). -/
def sm120MaxRegsPerThread : Nat := 255

/-- Per-CTA dynamic + static user SMEM opt-in ceiling (`99 KiB = 101,376 B`, reserving `1 KiB` sys). -/
def sm120MaxOptInSmemPerCtaBytes : Nat := 99 * 1024

/--
  Strict `sm_120` kernel validity predicate closing four launch-time holes in `checkSm120Kernel`:
  1. `k.threadsPerCta <= 1024`
  2. `k.ctasPerSm <= 32`
  3. `k.rawRegsPerThread <= 255`
  4. `k.userSmemPerCtaBytes + k.smemAlignPadBytes <= 99 * 1024`
-/
def checkSm120KernelStrict (k : KernelResourceSpec) : Bool :=
  checkSm120Kernel k &&
  (k.threadsPerCta <= sm120MaxThreadsPerCta) &&
  (k.ctasPerSm <= sm120MaxCtasPerSm) &&
  (k.rawRegsPerThread <= sm120MaxRegsPerThread) &&
  (k.userSmemPerCtaBytes + k.smemAlignPadBytes <= sm120MaxOptInSmemPerCtaBytes)

theorem strict_implies_legacy_valid (k : KernelResourceSpec)
    (h : checkSm120KernelStrict k = true) :
    checkSm120Kernel k = true := by
  unfold checkSm120KernelStrict at h
  revert h
  cases checkSm120Kernel k <;> simp

/-- Planted Red-Control #1: `1536` threads × `1` CTA (`48` warps) falsely passes `checkSm120Kernel`
    yet fails `cuLaunchKernel` (`maxThreadsPerBlock = 1024`) and is caught by `checkSm120KernelStrict`. -/
def plantedIllegal1536Threads : KernelResourceSpec :=
  { name := "planted_illegal_1536_threads"
    threadsPerCta := 1536
    rawRegsPerThread := 32
    userSmemPerCtaBytes := 4096
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

theorem planted_1536_threads_red_control :
    checkSm120Kernel plantedIllegal1536Threads = true ∧
    checkSm120KernelStrict plantedIllegal1536Threads = false := by decide

/-- Planted Red-Control #2: `32` threads × `40` CTAs (`40` warps <= `48`) falsely passes
    `checkSm120Kernel` yet exceeds `sm_120`'s `32 CTAs/SM` hardware limit. -/
def plantedIllegal40Ctas : KernelResourceSpec :=
  { name := "planted_illegal_40_ctas"
    threadsPerCta := 32
    rawRegsPerThread := 32
    userSmemPerCtaBytes := 512
    smemAlignPadBytes := 0
    ctasPerSm := 40
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

theorem planted_40_ctas_red_control :
    checkSm120Kernel plantedIllegal40Ctas = true ∧
    checkSm120KernelStrict plantedIllegal40Ctas = false := by decide

/-- Planted Red-Control #3: `256` threads × `256` regs (`65,536 <= 65,536`) falsely passes
    `checkSm120Kernel` yet violates PTX/SASS `R255 = RZ` (`max 255 regs/thread`). -/
def plantedIllegal256Regs : KernelResourceSpec :=
  { name := "planted_illegal_256_regs"
    threadsPerCta := 256
    rawRegsPerThread := 256
    userSmemPerCtaBytes := 4096
    smemAlignPadBytes := 0
    ctasPerSm := 1
    clusterCtas := 1
    minRequiredL1KiB := 64
    hasStackSpill := false
    hasHotLoopLibmCall := false }

theorem planted_256_regs_red_control :
    checkSm120Kernel plantedIllegal256Regs = true ∧
    checkSm120KernelStrict plantedIllegal256Regs = false := by decide

/-- Every kernel suite in `Proofs/` (all fifteen `List KernelResourceSpec` suites, Inkling-2
    and Qwen3.8) passes `checkSm120KernelStrict`, not only the Cycle 10 / 11 suites. -/
theorem production_leansonic_kernels_pass_strict :
    (([kAttnBlocklistTc, kMoeSimtLowBatch, kMoeGroupedMmaInt8] ++
      phase1KernelSuite ++ wave1Inkling2KernelSuite ++ cycle1BatchedKernelSuite ++
      cycle2SpeculativeKernelSuite ++ cycle4SplitKvKernelSuite ++
      cycle5GlueFusionKernelSuite ++ cycle6WideTcPrefillSuite ++ cycle7V13PrefillSuite ++
      cycle8SoloSpecPdlSuite ++ multiSeqRoundKernelSuite ++
      cycle9ConvFuseAndVerifyMoeSuite ++ verifyLaunchFusionSuite ++
      cycle10DecodeBreakthroughSuite ++ cycle11PrefillKernelSuite ++
      cycle3Qwen38KernelSuite).all checkSm120KernelStrict) = true := by decide

-- ============================================================================
-- 2. Universal Bounds & Collision-Free Injectivity of `expertMapOffset`
-- ============================================================================

def decodeExpertFromOffset (off : Nat) : Nat :=
  (off / 64) % 128

def decodeOccFromOffset (off : Nat) : Nat :=
  ((off / 64) / 128) * 64 + (off % 64)

theorem expertMapOffset_universal_bound (cap e occ : Nat)
    (hAlign : cap % 64 = 0) (hExp : e < 128) (hOcc : occ < cap) :
    expertMapOffset e occ < 128 * cap := by
  unfold expertMapOffset
  omega

theorem expertMapOffset_left_inverse (e occ : Nat) (hExp : e < 128) :
    decodeExpertFromOffset (expertMapOffset e occ) = e ∧
    decodeOccFromOffset (expertMapOffset e occ) = occ := by
  unfold decodeExpertFromOffset decodeOccFromOffset expertMapOffset
  omega

theorem expertMapOffset_injective (e1 o1 e2 o2 : Nat)
    (hE1 : e1 < 128) (hE2 : e2 < 128)
    (hEq : expertMapOffset e1 o1 = expertMapOffset e2 o2) :
    e1 = e2 ∧ o1 = o2 := by
  have h1 := expertMapOffset_left_inverse e1 o1 hE1
  have h2 := expertMapOffset_left_inverse e2 o2 hE2
  rw [hEq] at h1
  exact ⟨h1.1.symm.trans h2.1, h1.2.symm.trans h2.2⟩

-- ============================================================================
-- 4. Sharp `span <= 3` Integer-Exact Microscaling (`arXiv:2611.00486`)
-- ============================================================================

def shiftedE2M1Int (code : Fin 16) (expDelta : Nat) : Int :=
  e2m1DoubledSpec code * (Int.ofNat (2 ^ expDelta))

def checkAllSpan3FitInt8 : Bool :=
  (List.finRange 16).all fun c =>
    (List.range 4).all fun d =>
      let v := shiftedE2M1Int c d
      (-96 <= v) && (v <= 96) && (-128 <= v) && (v <= 127)

/-- Every `E2M1` code shifted by `d ∈ {0, 1, 2, 3}` fits in `[-96, 96] ⊂ [-128, 127]` (`INT8`). -/
theorem mxfp4_span3_fits_int8 : checkAllSpan3FitInt8 = true := by decide

/-- Sharpness: at `d = 4` (`span = 4`), max `E2M1` (`+6.0`, doubled `12`) reaches `192 > 127`,
    overflowing signed `INT8`! Thus `span <= 3` is both necessary and sufficient. -/
theorem mxfp4_span4_overflows_int8 :
    shiftedE2M1Int ⟨7, by omega⟩ 4 = 192 ∧ ¬(shiftedE2M1Int ⟨7, by omega⟩ 4 <= 127) := by decide

/-- Row-level algebraic identity: absorbing `2^d` into the `INT8` weight before multiplying
    by activation `x` equals multiplying the per-block integer dot by `2^d`. -/
theorem mxfp4_single_scale_dot_identity (code : Fin 16) (d : Nat) (x : Int) :
    shiftedE2M1Int code d * x = (e2m1DoubledSpec code * x) * Int.ofNat (2 ^ d) := by
  unfold shiftedE2M1Int
  rw [Int.mul_assoc, Int.mul_comm (Int.ofNat (2 ^ d)) x, ← Int.mul_assoc]

-- ============================================================================
-- 5. 4-Chain `(c0 + c1) + (c2 + c3)` Batch-Invariant `0-ULP` Reduction
-- ============================================================================

/-- SIMT `dp4a` + butterfly warp-shuffle reduction order over 8 `dp4a` groups (`k = 32`). -/
def simtDp4aBlock32 (p : Fin 8 → Int) : Int :=
  let s0 := (p 0 + p 4) + (p 2 + p 6)
  let s1 := (p 1 + p 5) + (p 3 + p 7)
  s0 + s1

/-- Tensor Core `mma.m16n8k32` sequential `k`-chunk accumulation order over the same 8 groups. -/
def tensorCoreMmaBlock32 (p : Fin 8 → Int) : Int :=
  (((p 0 + p 1) + (p 2 + p 3)) + ((p 4 + p 5) + (p 6 + p 7)))

/-- Exact `Int` associativity/commutativity guarantees SIMT `dp4a` and Tensor Core `mma.m16n8k32`
    produce the identical integer sum on every `k = 32` sub-block! -/
theorem dp4a_eq_mma_block32_int (p : Fin 8 → Int) :
    simtDp4aBlock32 p = tensorCoreMmaBlock32 p := by
  dsimp [simtDp4aBlock32, tensorCoreMmaBlock32]
  omega

/-- Over ANY uninterpreted IEEE-754 `fma` and `fadd`, folding 4 block chains into
    `(c0 + c1) + (c2 + c3)` produces `0-ULP` identical results between `dp4a` and `mma.m16n8k32`! -/
theorem four_chain_batch_invariant_0ulp {F : Type}
    (z : F) (fma : Int → F → F → F) (fadd : F → F → F)
    (b0 b1 b2 b3 : Fin 8 → Int) (s0 s1 s2 s3 : F) :
    let simt :=
      fadd
        (fadd (fma (simtDp4aBlock32 b0) s0 z) (fma (simtDp4aBlock32 b1) s1 z))
        (fadd (fma (simtDp4aBlock32 b2) s2 z) (fma (simtDp4aBlock32 b3) s3 z))
    let tc :=
      fadd
        (fadd (fma (tensorCoreMmaBlock32 b0) s0 z) (fma (tensorCoreMmaBlock32 b1) s1 z))
        (fadd (fma (tensorCoreMmaBlock32 b2) s2 z) (fma (tensorCoreMmaBlock32 b3) s3 z))
    simt = tc := by
  simp [dp4a_eq_mma_block32_int]

-- ============================================================================
-- 6. Sink-Corrected Key-Mean Smoothing Shift Invariance (`SageAttention` × Sink)
-- ============================================================================

def dot2 (q0 q1 k0 k1 : Int) : Int :=
  q0 * k0 + q1 * k1

/--
  In an attention head with scalar `sink` logit, centering keys `k' = k - μ` shifts `q · k`
  by `- (q · μ)`. Subtracting `q · μ` from `sink` (`sink' = sink - q · μ`) preserves EVERY
  key-to-sink and key-to-key logit difference identically, making softmax outputs algebraically
  unchanged while shrinking key outlier dynamic range for `INT8` / `FP8` KV quantization!
-/
theorem sink_corrected_key_mean_smoothing_invariant
    (q0 q1 k0 k1 j0 j1 mu0 mu1 sink : Int) :
    let k0' := k0 - mu0
    let k1' := k1 - mu1
    let j0' := j0 - mu0
    let j1' := j1 - mu1
    let sink' := sink - dot2 q0 q1 mu0 mu1
    (dot2 q0 q1 k0' k1' - sink' = dot2 q0 q1 k0 k1 - sink) ∧
    (dot2 q0 q1 k0' k1' - dot2 q0 q1 j0' j1' = dot2 q0 q1 k0 k1 - dot2 q0 q1 j0 j1) := by
  unfold dot2
  constructor
  · rw [Int.mul_sub, Int.mul_sub]
    omega
  · rw [Int.mul_sub, Int.mul_sub, Int.mul_sub, Int.mul_sub]
    omega

-- ============================================================================
-- 7. `CPTC` (`arXiv:2611.22019`) Mirror-Fold `64B` Table Safety
-- ============================================================================

/-- `CPTC` Mirror-Fold (`Q = 6`, `127`-entry symmetric table stored in `64` bytes):
    Any index `raw < 127` maps strictly inside `0..63` (`64`-byte register/SMEM table). -/
def cptcMirrorFoldIdx (raw : Nat) : Nat :=
  if raw < 64 then raw else 126 - raw

theorem cptc_mirror_fold_fits_64_bytes (raw : Nat) (h : raw < 127) :
    cptcMirrorFoldIdx raw < 64 := by
  unfold cptcMirrorFoldIdx
  split <;> omega

end LeanSonic.Proofs.Cycle1Frontier
