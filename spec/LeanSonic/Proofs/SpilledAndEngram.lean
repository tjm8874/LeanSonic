/-
  LeanSonic — Spilled Mode (VRAM Offload) & Qwen3.8/4 `Engram` Formal Proofs
  ============================================================================
  Machine-checked formal proofs for:
  1. `spilledInkling2On5070Ti_valid`: 16 GiB consumer Blackwell SKU (`RTX 5070 Ti`)
     running Spilled `Inkling-2-mini` fits strictly inside VRAM (`<= 16,384 MiB`)
     with 2-buffer Ping-Pong staging and 4-op Late-Wait compute overlap.
  2. `residentInkling2OnPro5000_zero_spill_valid`: 72 GiB workstation SKU (`RTX PRO 5000`)
     with `totalSpilledExperts = 0` enforces `idlePrefetchKernelLaunches = 0`.
  3. `inkling2_48visit_ping_pong_safe`: 48-visit asynchronous `h2d_stream` <-> `compute_stream`
     Ping-Pong event timeline is strictly topological (deadlock-free) and free of
     WAR/RAW buffer overwrite hazards across all 48 visits.
  4. `resident_eq_spilled_moe_0ulp`: Universally quantified proof that for ANY expert
     set and reduction operator, lossless Host->VRAM staging produces syntactically
     identical (`0-ULP`) outputs to an all-VRAM resident execution.
  5. `qwen38_engram_51b_offload_valid`: Qwen3.8-Flash-Next / Qwen4 51B `Engram` table
     satisfies pre-Visit-0 token-only lookahead prefetch and `2`-token slot tail sizing.
  6. `scanTrigrams_append_exact`: Universal inductive proof over ALL token lists `xs, ys`
     that splitting `xs ++ ys` across ANY Chunked-Prefill boundary, single-token Decode
     step, or `W = 16` Speculative Verify window produces the EXACT same sequence of
     `Engram` trigram lookups and final `(prev2, prev1)` state tail as unsplit execution!
-/

import LeanSonic.Axioms
import LeanSonic.Proofs.Inkling2Mini

namespace LeanSonic.Proofs

/-! ## 1. Spilled vs All-Resident VRAM Budget & Zero-Spill Gate Proofs -/

/-- RTX 5070 Ti 16 GiB (`16,384` MiB) spilling 50% of cold experts to pinned DDR. -/
def spilledInkling2On5070TiSpec : SpilledOffloadSpec :=
  { skuName := "NVIDIA GeForce RTX 5070 Ti 16GB (Spilled Mode)"
    vramBudgetMiB := 16384
    residentWeightsMiB := 9600      -- Dense + Prelude/Coda + hot promoted experts
    kvAndStatePoolMiB := 3072       -- 4-stage INT8/FP8 KV + SWA ring + recurrent slots
    runtimeScratchMiB := 1024
    pingPongBuffers := 2            -- Double-buffered `staging[v % 2]` (`0 B` dynamic alloc)
    stagingBufferMiB := 256         -- `2 × 256 = 512` MiB dedicated staging -> 14,208 <= 16,384 MiB
    maxSingleVisitSpillMiB := 192
    totalSpilledExperts := 1536
    idlePrefetchKernelLaunches := 0
    lateWaitOverlapOps := 5 }       -- `router -> remap -> moe_sort -> plan -> q8_quantize`

theorem spilledInkling2On5070Ti_valid : ValidSpilledOffload spilledInkling2On5070TiSpec := by
  decide

/-- RTX PRO 5000 72 GiB (`73,728` MiB) with all experts resident (`totalSpilledExperts = 0`). -/
def residentInkling2OnPro5000Spec : SpilledOffloadSpec :=
  { skuName := "NVIDIA RTX PRO 5000 72GB (All-Resident Mode)"
    vramBudgetMiB := 73728
    residentWeightsMiB := 20480
    kvAndStatePoolMiB := 8192
    runtimeScratchMiB := 2048
    pingPongBuffers := 0
    stagingBufferMiB := 0
    maxSingleVisitSpillMiB := 0
    totalSpilledExperts := 0
    idlePrefetchKernelLaunches := 0 -- Hard-gated to 0 (`-49.8 ms` idle scan trap impossible!)
    lateWaitOverlapOps := 0 }

theorem residentInkling2OnPro5000_zero_spill_valid :
    ValidSpilledOffload residentInkling2OnPro5000Spec := by
  decide

/-! ## 2. 48-Visit Ping-Pong `HtoD` ↔ `Compute` Non-Race & Deadlock-Freedom Proof -/

/--
  Canonical 48-visit Late-Wait timeline where each visit `v` occupies monotonic
  global timeline ticks `[v * 10 .. v * 10 + 8]`:
  - `v * 10 + 1`: `h2d_stream` waits `compute_done[v - 2]` (strictly `> (v - 2)*10 + 8`!)
  - `v * 10 + 2`: `h2d_stream` issues coalesced `HtoD` + promoted `DtoD` into `buf[v % 2]`
  - `v * 10 + 4`: `h2d_stream` records `staged_event[v]`
  - `v * 10 + 5`: `compute_stream` (after running `router..quantize` concurrently) executes Late-Wait
  - `v * 10 + 6`: `compute_stream` performs first global read of `buf[v % 2]`
  - `v * 10 + 8`: `compute_stream` completes expert GEMM & records `compute_done[v]`
-/
def canonical48VisitPingPong : List PingPongVisitSync :=
  (List.range 48).map fun v =>
    { visit := v
      waitPrevComputeStep := v * 10 + 1
      htodIssueStep := v * 10 + 2
      htodRecordStep := v * 10 + 4
      computeWaitStep := v * 10 + 5
      firstExpertReadStep := v * 10 + 6
      computeDoneStep := v * 10 + 8 }

theorem inkling2_48visit_ping_pong_safe : SafePingPongSchedule canonical48VisitPingPong := by
  decide

/-! ## 3. Resident ↔ Spilled `0-ULP` Bitwise Tier-Invariance Theorem -/

/--
  For ANY weight type `W`, expert index `e`, and residency partition `isResident`,
  if VRAM-resident weights and Host-pinned weights hold lossless copies of `goldenWeight`,
  `tieredExpertFetch` returns the exact `goldenWeight e` (`0-ULP` identical).
-/
theorem tiered_fetch_eq_golden
    {W : Type} (isResident : Nat → Bool) (vramStore hostStore goldenWeight : Nat → W)
    (hVram : ∀ e, isResident e = true → vramStore e = goldenWeight e)
    (hHost : ∀ e, isResident e = false → hostStore e = goldenWeight e)
    (e : Nat) :
    tieredExpertFetch isResident vramStore hostStore e = goldenWeight e := by
  unfold tieredExpertFetch
  cases h : isResident e
  · exact hHost e h
  · exact hVram e h

/--
  Lift `tiered_fetch_eq_golden` across an arbitrary routed `(expertId)` list and
  arbitrary downstream MoE compute kernel `evalMoe`: 16 GB Spilled execution and
  72 GB All-Resident execution are syntactically (`0-ULP`) identical!
-/
theorem resident_eq_spilled_moe_0ulp
    {W Out : Type}
    (isResident : Nat → Bool)
    (vramStore hostStore goldenWeight : Nat → W)
    (hVram : ∀ e, isResident e = true → vramStore e = goldenWeight e)
    (hHost : ∀ e, isResident e = false → hostStore e = goldenWeight e)
    (evalMoe : List W → Out)
    (routedExperts : List Nat) :
    evalMoe (routedExperts.map (tieredExpertFetch isResident vramStore hostStore)) =
      evalMoe (routedExperts.map goldenWeight) := by
  have hMap :
      routedExperts.map (tieredExpertFetch isResident vramStore hostStore) =
        routedExperts.map goldenWeight := by
    induction routedExperts with
    | nil => rfl
    | cons hd tl ih =>
      simp [tiered_fetch_eq_golden isResident vramStore hostStore goldenWeight hVram hHost hd, ih]
  rw [hMap]

/-! ## 4. Qwen3.8-Flash-Next / Qwen4 `Engram` (51B N-gram Host Offload) Proofs -/

def qwen38Engram51BSpec : EngramOffloadSpec :=
  { tableParamsBillions := 51
    ngramOrder := 3                 -- Trigram (+ Bigram) local context lookup
    dependsOnHiddenState := false   -- Pure function of input token IDs!
    prefetchIssueVisit := 0         -- Issued at token ingest before Visit 0 starts
    residualInjectVisit := 2        -- Hides PCIe DMA behind Visits 0 & 1 (GDN layers)
    maxBatchRows := 64
    hostStagingRows := 64
    slotTailTokens := 2 }           -- `(prev2, prev1)` per slot in `HybridSlotPool`

theorem qwen38_engram_51b_offload_valid : ValidEngramOffload qwen38Engram51BSpec := by
  decide

/--
  Universal Inductive `Engram` Boundary Exactness Theorem:
  Splitting ANY token sequence `xs ++ ys` across ANY Chunked-Prefill boundary,
  single-token Decode step, or `W = 16` Speculative Verify window produces the
  100% exact same list of trigram table keys AND final `(prev2, prev1)` tail state
  as scanning `xs ++ ys` monolithically!
-/
theorem scanTrigrams_append_exact (p2 p1 : Nat) (xs ys : List Nat) :
    scanTrigramsWithTail p2 p1 (xs ++ ys) =
      let resX := scanTrigramsWithTail p2 p1 xs
      let resY := scanTrigramsWithTail resX.2.1 resX.2.2 ys
      (resX.1 ++ resY.1, resY.2) := by
  induction xs generalizing p2 p1 with
  | nil =>
    simp [scanTrigramsWithTail]
  | cons x rest ih =>
    simp only [List.cons_append, scanTrigramsWithTail]
    rw [ih p1 x]

end LeanSonic.Proofs
