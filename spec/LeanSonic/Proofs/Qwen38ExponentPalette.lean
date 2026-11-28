import LeanSonic.Proofs.SpilledAndEngram

namespace LeanSonic.Qwen38ExponentPalette

/-- Two consecutive dictionary indices occupy the low and high nibbles of one byte. -/
def encodePair (lo hi : Fin 16) : Nat := lo.val + 16 * hi.val

theorem encodePair_fits_byte (lo hi : Fin 16) : encodePair lo hi < 256 := by
  unfold encodePair
  have := lo.isLt
  have := hi.isLt
  omega

theorem decode_pair_indices (lo hi : Fin 16) :
    encodePair lo hi % 16 = lo.val ∧ encodePair lo hi / 16 = hi.val := by
  unfold encodePair
  have := lo.isLt
  omega

/-- Applies within each expert after the encoder has assigned each source byte
    to a dictionary slot. The encoder rejects a seventeenth distinct value. -/
theorem dictionary_round_trip (palette : Nat → Fin 256) (lo hi : Fin 16) :
    (palette (encodePair lo hi % 16), palette (encodePair lo hi / 16)) =
      (palette lo.val, palette hi.val) := by
  have h := decode_pair_indices lo hi
  rw [h.1, h.2]

/-- A record has 32 header words and 6,400 words of 4-bit indices. The
    decoder reads a four-byte word within the expert's 25,728-byte record. -/
theorem index_word_in_record (i : Fin 51200) :
    32 ≤ 32 + i.val / 8 ∧ 32 + i.val / 8 < 6432 := by
  have := i.isLt
  omega

/-- All 512 records for a projection fit in its allocated upper plane. -/
theorem index_word_in_plane (expert : Fin 512) (i : Fin 51200) :
    expert.val * 6432 + (32 + i.val / 8) < 512 * 6432 := by
  have := expert.isLt
  have := (index_word_in_record i).2
  omega

theorem palette_word_in_header (entry : Fin 16) : entry.val / 4 < 32 := by
  have := entry.isLt
  omega

open LeanSonic.Proofs

/-- The optional kernels use the same CTA and shared-memory layout as their stock
    counterparts. Register counts come from ptxas for the sm_120a palette build. -/
def paletteKernels : List KernelResourceSpec :=
  [ { kQ38MoeUpSwiglu11Z24 with name := "q38_moe_up_swiglu11_z24_exp4", rawRegsPerThread := 40 }
  , { kQ38MoeDownCombine11Z24 with name := "q38_moe_down_combine11_z24_exp4", rawRegsPerThread := 56 }
  , { kQ38MoeUpGroupedZ24 with name := "q38_moe_up_grouped_z24_exp4", rawRegsPerThread := 62 }
  , { kQ38MoeDownGroupedZ24 with name := "q38_moe_down_grouped_z24_exp4", rawRegsPerThread := 61 }
  ]

theorem palette_kernels_valid : paletteKernels.all checkSm120Kernel = true := by
  decide

end LeanSonic.Qwen38ExponentPalette
