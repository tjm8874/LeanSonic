import LeanSonic.Proofs.SpilledAndEngram

namespace LeanSonic.Proofs.Qwen38MixedExperts

/-- Expert IDs select a device address in a projection-specific table of 512 addresses. -/
theorem table_entry_in_bounds (proj : Fin 3) (expert : Fin 512) :
    (proj.val * 512 + expert.val + 1) * 8 ≤ 3 * 512 * 8 := by
  have := proj.isLt
  have := expert.isLt
  omega

/-- Each tier packs the same 972,800-byte Z24 record for every selected expert. -/
theorem tier_record_in_bounds (proj : Fin 3) (count slot : Nat) (selected : slot < count) :
    (proj.val * count + slot + 1) * 972800 ≤ 3 * count * 972800 := by
  have hp := proj.isLt
  have h : proj.val = 0 ∨ proj.val = 1 ∨ proj.val = 2 := by omega
  rcases h with h | h | h <;> simp [h] <;> omega

/-- Maximum byte touched by a Q4 uint4 load and each Z24 uint32 metadata load. -/
theorem plane_loads_in_bounds (group : Fin 51200) :
    (group.val + 1) * 16 ≤ 819200 ∧
    819200 + (group.val / 2 + 1) * 4 ≤ 921600 ∧
    921600 + (group.val / 4 + 1) * 4 ≤ 972800 := by
  have := group.isLt
  omega

/-- Fixed capacity on the 32 GB card, including the remainder distributed over 32 layers. -/
theorem quota_matches_19_full_layers : 32 * 203 + 16 * 202 = 19 * 512 := by decide

/-- On `72 GB` cards with `--spec K` (`2 GiB` VRAM reserved for full-VRAM MTP MoE),
    `45` full layers (`23,040` experts) + `3` mixed layers (`352` hot experts each = `1,056`)
    fit strictly within `24,096` experts (`66.85 GiB <= 68.4 GiB`). -/
theorem spec_vram_mixed_72g_fits : 45 * 512 + 3 * 352 = 24096 ∧ 24096 < 48 * 512 := by decide

end LeanSonic.Proofs.Qwen38MixedExperts
