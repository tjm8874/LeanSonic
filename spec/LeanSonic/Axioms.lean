/-
  LeanSonic — Layer 1 Immutable Trust Core (`Axioms.lean`)
  ============================================================================
  SECURITY & GOVERNANCE NOTICE:
  This file defines the formal specification, hardware resource laws for NVIDIA
  Blackwell (`sm_120`), SMEM non-aliasing predicates, 0-ULP floating-point
  reduction tree equivalence, exact NVFP4 (`E2M1`) integer lifting, and
  hybrid Looped / GDN / QSA state-pool safety invariants.

  All definitions in this file are constructive (`def`, `structure`, `inductive`).
  NO `axiom` declarations and NO `sorry` tokens are permitted anywhere in `spec/`.
  Autonomous PRs are blocked from modifying this file by Gate 0 (`CODEOWNERS` +
  `ci/check_immutable_core.py`).
-/

namespace LeanSonic

/-! ## 1. NVIDIA Blackwell (`sm_120`) Discrete Resource Model -/

/-- Total combined L1 Data Cache + Shared Memory capacity per SM on `sm_120` (128 KiB). -/
def sm120CombinedL1SmemKiB : Nat := 128

/-- Maximum dynamic + static Shared Memory allocatable per SM on `sm_120` (100 KiB). -/
def sm120MaxSmemPerSmBytes : Nat := 100 * 1024

/-- Hardware/driver reserved Shared Memory per CTA on `sm_120` (1 KiB). -/
def sm120SystemReservedSmemPerCtaBytes : Nat := 1024

/-- 32-bit registers per SM on `sm_120` (65,536 dwords = 256 KiB). -/
def sm120RegsPerSm : Nat := 65536

/-- Maximum resident warps per SM on `sm_120`. -/
def sm120MaxWarpsPerSm : Nat := 48

/-- Warp lane count. -/
def sm120WarpSize : Nat := 32

/-- Per-thread register allocation granularity (8 registers). -/
def sm120RegAllocGranularity : Nat := 8

/-- Required byte alignment for TMA 2D/3D descriptors and Planar SoA planes. -/
def sm120TmaAlignmentBytes : Nat := 128

/--
  Discrete SMEM carveout steps (in KiB) on NVIDIA Blackwell `sm_120`:
  `{0, 8, 16, 32, 64, 100}` KiB.
  Crossing `64 KiB` jumps straight to `100 KiB`, shrinking L1 from `64 KiB` to `28 KiB`.
-/
def sm120CarveoutKiB (totalSmemPerSmBytes : Nat) : Option Nat :=
  if totalSmemPerSmBytes == 0 then some 0
  else if totalSmemPerSmBytes <= 8 * 1024 then some 8
  else if totalSmemPerSmBytes <= 16 * 1024 then some 16
  else if totalSmemPerSmBytes <= 32 * 1024 then some 32
  else if totalSmemPerSmBytes <= 64 * 1024 then some 64
  else if totalSmemPerSmBytes <= 100 * 1024 then some 100
  else none

/-- Remaining L1 Data Cache (in KiB) for a given per-SM SMEM footprint. -/
def sm120RemainingL1KiB (totalSmemPerSmBytes : Nat) : Option Nat :=
  match sm120CarveoutKiB totalSmemPerSmBytes with
  | some carveout => some (sm120CombinedL1SmemKiB - carveout)
  | none => none

/-- Round `n` up to a positive `granularity`. -/
def roundUp (n granularity : Nat) : Nat :=
  if granularity == 0 then n
  else ((n + granularity - 1) / granularity) * granularity

/-- Hardware-allocated registers per thread (rounded up to granularity 8). -/
def alignedRegsPerThread (rawRegs : Nat) : Nat :=
  roundUp rawRegs sm120RegAllocGranularity

/-- Total registers consumed per CTA. -/
def regsPerCta (threadsPerCta rawRegsPerThread : Nat) : Nat :=
  threadsPerCta * alignedRegsPerThread rawRegsPerThread

/-- Total per-SM SMEM bytes including the 1 KiB per-CTA system reservation and alignment pad. -/
def totalSmemPerSmBytes (userSmemPerCtaBytes alignPadBytes ctasPerSm : Nat) : Nat :=
  ctasPerSm * (userSmemPerCtaBytes + alignPadBytes + sm120SystemReservedSmemPerCtaBytes)

/-- Static resource specification of an `sm_120` CUDA/PTX kernel. -/
structure KernelResourceSpec where
  name : String
  threadsPerCta : Nat
  rawRegsPerThread : Nat
  userSmemPerCtaBytes : Nat
  smemAlignPadBytes : Nat
  ctasPerSm : Nat
  clusterCtas : Nat
  minRequiredL1KiB : Nat
  hasStackSpill : Bool
  hasHotLoopLibmCall : Bool
  deriving Repr, DecidableEq

/-- Decidable check for whether a kernel obeys all `sm_120` static resource cliffs. -/
def checkSm120Kernel (k : KernelResourceSpec) : Bool :=
  let warpsPerCta := k.threadsPerCta / sm120WarpSize
  let totalWarps := warpsPerCta * k.ctasPerSm
  let totalRegs := regsPerCta k.threadsPerCta k.rawRegsPerThread * k.ctasPerSm
  let smemBytes := totalSmemPerSmBytes k.userSmemPerCtaBytes k.smemAlignPadBytes k.ctasPerSm
  let l1Ok := match sm120RemainingL1KiB smemBytes with
    | some remL1 => remL1 >= k.minRequiredL1KiB
    | none => false
  (k.threadsPerCta > 0) &&
  (k.threadsPerCta % sm120WarpSize == 0) &&
  (k.ctasPerSm > 0) &&
  (k.clusterCtas >= 1) &&
  (k.clusterCtas <= 2) &&
  (totalWarps <= sm120MaxWarpsPerSm) &&
  (totalRegs <= sm120RegsPerSm) &&
  (!k.hasStackSpill) &&
  (!k.hasHotLoopLibmCall) &&
  l1Ok

/-- Propositional validity predicate lifted from `checkSm120Kernel`. -/
abbrev ValidSm120Kernel (k : KernelResourceSpec) : Prop :=
  checkSm120Kernel k = true

/-! ## 2. Shared Memory Layout & Lifetime Non-Aliasing (`union Smem` Safety) -/

/-- A named Shared Memory slice with byte range and inclusive pipeline stage liveness. -/
structure SmemSlice where
  name : String
  offsetBytes : Nat
  sizeBytes : Nat
  alignBytes : Nat
  liveStartStage : Nat
  liveEndStage : Nat
  deriving Repr, DecidableEq

/-- True iff two stage intervals `[s1, e1]` and `[s2, e2]` overlap in time. -/
def stagesOverlap (a b : SmemSlice) : Bool :=
  !(a.liveEndStage < b.liveStartStage || b.liveEndStage < a.liveStartStage)

/-- True iff two byte ranges `[o1, o1+sz1)` and `[o2, o2+sz2)` are disjoint in SMEM. -/
def bytesDisjoint (a b : SmemSlice) : Bool :=
  (a.offsetBytes + a.sizeBytes <= b.offsetBytes) ||
  (b.offsetBytes + b.sizeBytes <= a.offsetBytes)

/-- Check alignment, bounds within `smemCapBytes`, and pairwise non-aliasing for overlapping lifetimes. -/
def checkSmemLayout (smemCapBytes : Nat) (slices : List SmemSlice) : Bool :=
  let wellFormed := slices.all fun s =>
    (s.sizeBytes > 0) &&
    (s.alignBytes > 0) &&
    (s.offsetBytes % s.alignBytes == 0) &&
    (s.liveStartStage <= s.liveEndStage) &&
    (s.offsetBytes + s.sizeBytes <= smemCapBytes)
  let rec pairwiseOk : List SmemSlice → Bool
    | [] => true
    | x :: xs =>
      (xs.all fun y => if stagesOverlap x y then bytesDisjoint x y else true) &&
      pairwiseOk xs
  wellFormed && pairwiseOk slices

abbrev SafeSmemLayout (smemCapBytes : Nat) (slices : List SmemSlice) : Prop :=
  checkSmemLayout smemCapBytes slices = true

/-! ## 3. Exact NVFP4 (`E2M1`) → `int8` `2×` Lifting Algebra -/

/--
  Signed 4-bit `E2M1` code (`nibble ∈ 0..15`), where bit 3 is sign (`0` = +, `1` = -)
  and bits `2..0` encode `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`.
  We define the exact doubled integer value (`2 × value`) directly in `Int`.
-/
def e2m1DoubledSpec (code : Fin 16) : Int :=
  let magIdx := code.val % 8
  let mag2x : Int :=
    match magIdx with
    | 0 => 0   -- 2 * 0.0
    | 1 => 1   -- 2 * 0.5
    | 2 => 2   -- 2 * 1.0
    | 3 => 3   -- 2 * 1.5
    | 4 => 4   -- 2 * 2.0
    | 5 => 6   -- 2 * 3.0
    | 6 => 8   -- 2 * 4.0
    | _ => 12  -- 2 * 6.0
  if code.val >= 8 then -mag2x else mag2x

/-- Hardware 16-byte `__byte_perm` LUT table used in `moe_dual_regime.cu` and Rust runtime. -/
def e2m1BytePermLut : Array Int :=
  #[0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]

def e2m1LutLookup (code : Fin 16) : Int :=
  e2m1BytePermLut.getD code.val 0

/-- Every LUT entry matches the doubled E2M1 specification and fits strictly inside signed `int8` `[-12, 12]`. -/
def checkE2M1Code (code : Fin 16) : Bool :=
  let v := e2m1LutLookup code
  (v == e2m1DoubledSpec code) && (-12 <= v) && (v <= 12) && (-128 <= v) && (v <= 127)

/-! ## 4. Non-Associative IEEE-754 `0-ULP` Butterfly Tree Equivalence -/

/--
  IEEE-754 round-to-nearest-even addition (`__fadd_rn`) is NOT associative.
  Therefore, to guarantee `0-ULP` `BATCH IDENTICAL` output between low-batch SIMT (`T <= 3`)
  and high-batch Tensor Core `mma.m16n8k32` (`T >= 4`), both paths MUST evaluate the
  exact same parenthesization tree over an uninterpreted binary operator `fadd`.
-/
def simtSubBlockReduce4 {α : Type} (fadd : α → α → α) (p0 p1 p2 p3 : α) : α :=
  fadd (fadd p0 p1) (fadd p2 p3)

/--
  `16B × 2` packed MMA reduces two 16-byte halves `(h0, h1)` where `h0 = fadd p0 p1`
  and `h1 = fadd p2 p3`, then combines `fadd h0 h1`.
-/
def mmaPacked16Bx2Reduce4 {α : Type} (fadd : α → α → α) (p0 p1 p2 p3 : α) : α :=
  let halfLo := fadd p0 p1
  let halfHi := fadd p2 p3
  fadd halfLo halfHi

/-! ## 5. Unified `VisitBytecode` & Hybrid Slot/Ring No-Clobber Model -/

/-- Unified sequence-mixing / attention operator kind across Inkling-2 and Qwen3.8/4. -/
inductive SequenceMixKind where
  | fullAttention
  | slidingWindow (windowTokens : Nat)
  | qwenSparse (topBlocks : Nat) (blockTokens : Nat)
  | gatedDeltaNet (convKernel : Nat) (stateHeads : Nat)
  deriving Repr, DecidableEq

/-- Residual stream topology (`1` for standard / Looped Blend, `4` for Qwen Gated Residual). -/
inductive ResidualTopology where
  | singleStream (gatedLoopBlend : Bool)
  | quadGatedStream
  deriving Repr, DecidableEq

/-- A single scheduled layer execution ("visit") in `VisitBytecode`. -/
structure VisitOp where
  visitIndex : Nat
  physicalLayerId : Nat
  mix : SequenceMixKind
  residual : ResidualTopology
  numExperts : Nat
  topKExperts : Nat
  hasSharedExpert : Bool
  deriving Repr, DecidableEq

/-- Architecture-level execution schedule. -/
structure ModelScheduleSpec where
  modelName : String
  physicalLayers : Nat
  visits : List VisitOp
  deriving Repr, DecidableEq

/-- Check that visit indices are contiguous `0..N-1` and every `physicalLayerId < physicalLayers`. -/
def checkScheduleValid (m : ModelScheduleSpec) : Bool :=
  let indexedOk := (m.visits.zipIdx).all fun (op, idx) =>
    (op.visitIndex == idx) &&
    (op.physicalLayerId < m.physicalLayers) &&
    (op.topKExperts > 0) &&
    (op.topKExperts <= op.numExperts)
  (!m.visits.isEmpty) && indexedOk

abbrev ValidModelSchedule (m : ModelScheduleSpec) : Prop :=
  checkScheduleValid m = true

/-- Hybrid KV + SWA Ring + GDN Recurrent State Pool sizing specification. -/
structure HybridStatePoolSpec where
  maxConcurrentSlots : Nat
  specVerifyWindow : Nat
  maxBatchRows : Nat
  kvBlockAlignTokens : Nat
  swaWindowTokens : Nat
  swaRingCapacityTokens : Nat
  recurrentStateSlots : Nat
  suffixMarginPercent : Nat
  deriving Repr, DecidableEq

/--
  No-Clobber & Speculative Co-Design Invariant:
  1. KV split chunks align to `64` tokens (`kvBlockAlignTokens == 64`).
  2. Batch row capacity covers all concurrent slots (`maxBatchRows >= maxConcurrentSlots`).
  3. SWA ring capacity holds `swaWindowTokens + maxBatchRows` aligned to `64` tokens so
     multi-row verify never overwrites active sliding-window history.
  4. Recurrent state pool allocates `maxConcurrentSlots * (1 + specVerifyWindow)` slots
     for `O(1)` rollback of unaccepted draft tokens on GDN/conv layers.
  5. Speculative verify window `W` strictly exceeds the asymmetric `(1 + margin)` cap trap:
     e.g., at `margin = 5%` (`105%`), `W * 100 >= 16 * 100` (`W >= 16`) so zero-cost suffix hits
     can beat `7.8`-token neural drafts (`8 * 100 < 8 * 105` would trap `W = 8`).
-/
def checkHybridStatePool (p : HybridStatePoolSpec) : Bool :=
  (p.maxConcurrentSlots >= 1) &&
  (p.specVerifyWindow >= 16) &&
  (p.maxBatchRows >= p.maxConcurrentSlots) &&
  (p.kvBlockAlignTokens == 64) &&
  (p.swaRingCapacityTokens % p.kvBlockAlignTokens == 0) &&
  (p.swaRingCapacityTokens >= p.swaWindowTokens + p.maxBatchRows) &&
  (p.recurrentStateSlots >= p.maxConcurrentSlots * (1 + p.specVerifyWindow)) &&
  (p.specVerifyWindow * 100 > 8 * (100 + p.suffixMarginPercent))

abbrev SafeHybridStatePool (p : HybridStatePoolSpec) : Prop :=
  checkHybridStatePool p = true

end LeanSonic
