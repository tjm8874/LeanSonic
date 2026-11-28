/-
  LeanSonic — Qwen3.8-Flash-Next built-in MTP drafter (`kernels/qwen38_mtp_sm120.cu`, #248).

  The MTP decoder layer reuses the `qwen38_sm120.cu` kernels (already covered by
  `cycle3Qwen38KernelSuite`); these are the two input-fusion kernels it adds, with the
  `ptxas -v` resources of the `sm_120a` build (0 stack, 0 spills).
-/
import LeanSonic.Axioms

namespace LeanSonic.Proofs

/-- Gemma RMSNorm over 2,560 or 10,240 values (`Grid: (1, 1)`, `Block: 256`). -/
def kQ38MtpRmsnorm : KernelResourceSpec :=
  { name := "q38_mtp_rmsnorm", threadsPerCta := 256, rawRegsPerThread := 40,
    userSmemPerCtaBytes := 36, smemAlignPadBytes := 0, ctasPerSm := 6,
    clusterCtas := 1, minRequiredL1KiB := 120, hasStackSpill := false, hasHotLoopLibmCall := false }

/-- `hc[s, d] += e[d]` over the four hyper-connection streams (`Grid: (40, 1)`, `Block: 256`). -/
def kQ38MtpAddBcast4 : KernelResourceSpec :=
  { name := "q38_mtp_add_bcast4", threadsPerCta := 256, rawRegsPerThread := 12,
    userSmemPerCtaBytes := 0, smemAlignPadBytes := 0, ctasPerSm := 6,
    clusterCtas := 1, minRequiredL1KiB := 120, hasStackSpill := false, hasHotLoopLibmCall := false }

def qwen38MtpKernelSuite : List KernelResourceSpec := [kQ38MtpRmsnorm, kQ38MtpAddBcast4]

theorem qwen38_mtp_kernels_valid :
    (qwen38MtpKernelSuite.all checkSm120Kernel) = true := by
  decide

end LeanSonic.Proofs
