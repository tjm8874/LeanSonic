use std::env;
use std::process::ExitCode;

use leansonic_core::bytecode::VisitProgram;
use leansonic_core::engine::{verify_inkling2_batch_invariance, MinimalEngine};
use leansonic_core::generated_spec::{
    MAX_CONCURRENT_SLOTS, MOE_GROUPED_MMA_MIN_PAIRS, RECURRENT_STATE_SLOTS, SPEC_VERIFY_WINDOW,
    SWA_RING_CAPACITY_TOKENS,
};
use leansonic_tuner::{
    compute_decode_ceiling_tok_s, rank_valid_candidates, BlackwellSkuProfile, CandidateKnob,
    ModelRooflineInput,
};

fn print_help() {
    println!("LeanSonic v0.1.0 — Lean 4-Verified Blackwell Inference Engine (Phase 0 & 1)");
    println!();
    println!("USAGE:");
    println!("  leansonic info            Print Lean-verified SM120 & Hybrid State Pool constants");
    println!("  leansonic schedules       Compile & validate Inkling-2-mini and Qwen3.8-Flash-Next schedules");
    println!(
        "  leansonic verify-parity   Run Gate 3 48-visit C=1 vs C=64 0-ULP BATCH IDENTICAL check"
    );
    println!("  leansonic tune            Run Layer 2 SM120 cliff pruning & whole-model roofline ceilings");
}

fn cmd_info() {
    println!("=== LeanSonic Verified Specification Summary ===");
    println!("  Max Concurrent Slots (C)     : {MAX_CONCURRENT_SLOTS}");
    println!("  Speculative Verify Cap (W)   : {SPEC_VERIFY_WINDOW}");
    println!("  SWA Ring Capacity (64-align) : {SWA_RING_CAPACITY_TOKENS} tokens");
    println!("  Recurrent/GDN State Pool     : {RECURRENT_STATE_SLOTS} slots (O(1) rollback)");
    println!("  MoE Grouped MMA Switch       : >= {MOE_GROUPED_MMA_MIN_PAIRS} routed pairs");
}

fn cmd_schedules() -> Result<(), String> {
    let inkling = VisitProgram::inkling2_mini();
    inkling.validate().map_err(|e| format!("{e:?}"))?;
    println!(
        "[OK] {} : {} physical layers -> {} visits ({} Full + {} SWA=1024)",
        inkling.model_name,
        inkling.physical_layers,
        inkling.visits.len(),
        inkling.count_full_attention_visits(),
        inkling.count_swa_visits()
    );

    let qwen = VisitProgram::qwen38_flash_next();
    qwen.validate().map_err(|e| format!("{e:?}"))?;
    let mut eng = MinimalEngine::new_synthetic(qwen.clone(), 64, 42);
    let step = eng.forward_step(&[(0, vec![0.1; 64])]);
    println!(
        "[OK] {} : {} layers ({} GDN + {} QSA, 4-branch GR, Engram Async HtoD: {:?})",
        qwen.model_name, qwen.physical_layers, step.gdn_visits, step.qsa_visits, step.htod_plan
    );
    Ok(())
}

fn cmd_tune() {
    let sku = BlackwellSkuProfile::rtx_pro_5000_300w();
    println!("=== Layer 2 Tuner: {} ===", sku.name);

    let candidates = vec![
        CandidateKnob {
            label: "q8_planar_128x128_tma_3stage".into(),
            tile_m: 128,
            tile_n: 128,
            tile_k: 64,
            stages: 3,
            threads_per_cta: 512,
            est_regs_per_thread: 128,
            user_smem_bytes: 61_440,
            ctas_per_sm: 1,
            cluster_ctas: 1,
            min_required_l1_kib: 64,
        },
        CandidateKnob {
            label: "moe_fp4_256x128_2stage".into(),
            tile_m: 256,
            tile_n: 128,
            tile_k: 64,
            stages: 2,
            threads_per_cta: 256,
            est_regs_per_thread: 120,
            user_smem_bytes: 77_824,
            ctas_per_sm: 1,
            cluster_ctas: 1,
            min_required_l1_kib: 28,
        },
        CandidateKnob {
            label: "rejected_75kib_l1_starved".into(),
            tile_m: 128,
            tile_n: 128,
            tile_k: 64,
            stages: 2,
            threads_per_cta: 256,
            est_regs_per_thread: 96,
            user_smem_bytes: 75_980,
            ctas_per_sm: 1,
            cluster_ctas: 1,
            min_required_l1_kib: 64,
        },
    ];

    let ranked = rank_valid_candidates(&sku, candidates);
    for (idx, r) in ranked.iter().enumerate() {
        println!(
            "  #{idx} {:<30} | score={:>6.2} | carveout={}KiB | L1={}KiB | warps/SM={}",
            r.knob.label,
            r.composite_score,
            r.carveout_kib,
            r.remaining_l1_kib,
            r.active_warps_per_sm
        );
    }

    let inkling_128k_spec8 = ModelRooflineInput {
        dense_and_head_bytes: 0.65e9,
        total_expert_bytes: 16.0e9,
        num_experts: 128,
        top_k: 8,
        kv_bytes_per_slot: 6.43e9, // 12 Full-Attention visits @ 128K FP16 KV
        recurrent_state_bytes_per_slot: 0.0,
        draft_bytes_per_step: 0.06e9,
        draft_steps_k: 8,
        mean_accepted_plus_one: 5.8,
        concurrency_slots: 1,
    };
    let ceil = compute_decode_ceiling_tok_s(&sku, &inkling_128k_spec8);
    println!("  Inkling-2-Mini 128K (C=1, W=8, E[acc+1]=5.8) Streaming Ceiling: {ceil:.1} tok/s");
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("info") => {
            cmd_info();
            ExitCode::SUCCESS
        }
        Some("schedules") => match cmd_schedules() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("[ERROR] {e}");
                ExitCode::FAILURE
            }
        },
        Some("verify-parity") => match verify_inkling2_batch_invariance(64) {
            Ok(()) => {
                println!(
                    "[PASS] Gate 3: Inkling-2-Mini 48-visit Looped MoE is 0-ULP BATCH IDENTICAL (C=1 SimtReg64 == C=64 GroupedMmaInt8)"
                );
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("[FAIL] {e}");
                ExitCode::FAILURE
            }
        },
        Some("tune") => {
            cmd_tune();
            ExitCode::SUCCESS
        }
        _ => {
            print_help();
            ExitCode::SUCCESS
        }
    }
}
