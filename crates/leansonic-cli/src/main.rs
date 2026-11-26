use std::env;
use std::io::{self, Write};
use std::process::ExitCode;

use leansonic_core::bytecode::VisitProgram;
use leansonic_core::engine::{
    verify_inkling2_batch_invariance, verify_inkling2_resident_vs_spilled_0ulp,
    verify_qwen38_engram_boundary_0ulp, MinimalEngine,
};
use leansonic_core::generated_spec::{
    ENGRAM_NGRAM_ORDER, ENGRAM_RESIDUAL_INJECT_VISIT, ENGRAM_SLOT_TAIL_TOKENS,
    MAX_CONCURRENT_SLOTS, MOE_GROUPED_MMA_MIN_PAIRS, PREFILL_CHUNK_TOKENS, RECURRENT_STATE_SLOTS,
    SERVE_DEFAULT_MAX_NEW_TOKENS, SERVE_MAX_HTTP_STREAMS, SPEC_VERIFY_WINDOW,
    SPILLED_LATE_WAIT_OVERLAP_OPS, SPILLED_PING_PONG_BUFFERS, SWA_RING_CAPACITY_TOKENS,
};
use leansonic_core::serving::{
    execute_generation_on_slot, json_escape, run_single_prompt, serve_openai_http,
    GenerationRequest,
};
use leansonic_tuner::{
    compute_decode_ceiling_tok_s, compute_spilled_engram_ceiling_tok_s, rank_valid_candidates,
    BlackwellSkuProfile, CandidateKnob, ModelRooflineInput, SpilledEngramRooflineInput,
};

fn print_help() {
    println!("LeanSonic v0.1.0 — Lean 4-Verified Blackwell Inference Engine & OpenAI Server");
    println!();
    println!("USAGE:");
    println!("  leansonic -p \"<prompt>\" [--model inkling-2-mini|qwen3.8-flash-next] [--max-tokens N] [--json]");
    println!("  leansonic serve [--host 127.0.0.1] [--port 8080] [--model inkling-2-mini]");
    println!(
        "  leansonic info            Print Lean-verified SM120, Hybrid Pool & Serving constants"
    );
    println!("  leansonic schedules       Compile & validate Inkling-2-mini and Qwen3.8-Flash-Next schedules");
    println!(
        "  leansonic verify-parity   Run Gate 3 48-visit C=1==C=64 & `-p`==`serve` 0-ULP checks"
    );
    println!("  leansonic tune            Run Layer 2 SM120 cliff pruning & whole-model roofline ceilings");
}

fn cmd_info() {
    println!("=== LeanSonic Verified Specification Summary ===");
    println!("  Max Concurrent Slots (C)     : {MAX_CONCURRENT_SLOTS}");
    println!("  Max OpenAI HTTP/SSE Streams  : {SERVE_MAX_HTTP_STREAMS}");
    println!("  Chunked Prefill Slice        : {PREFILL_CHUNK_TOKENS} tokens (64-aligned)");
    println!("  Speculative Verify Cap (W)   : {SPEC_VERIFY_WINDOW}");
    println!("  SWA Ring Capacity (64-align) : {SWA_RING_CAPACITY_TOKENS} tokens");
    println!("  Recurrent/GDN State Pool     : {RECURRENT_STATE_SLOTS} slots (O(1) rollback)");
    println!("  MoE Grouped MMA Switch       : >= {MOE_GROUPED_MMA_MIN_PAIRS} routed pairs");
    println!(
        "  Spilled Ping-Pong Staging    : {SPILLED_PING_PONG_BUFFERS} buffers, {SPILLED_LATE_WAIT_OVERLAP_OPS}-op Late-Wait"
    );
    println!(
        "  Qwen3.8 Engram Offload       : {ENGRAM_NGRAM_ORDER}-gram (tail={ENGRAM_SLOT_TAIL_TOKENS}), issue@v0 -> inject@v{ENGRAM_RESIDUAL_INJECT_VISIT}"
    );
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

fn cmd_prompt(args: &[String]) -> Result<(), String> {
    let mut prompt: Option<String> = None;
    let mut model = "inkling-2-mini".to_string();
    let mut max_tokens = 16usize;
    let mut emit_json = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-p" | "--prompt" => {
                i += 1;
                prompt = args.get(i).cloned();
            }
            "-m" | "--model" => {
                i += 1;
                model = args.get(i).cloned().ok_or("missing value after --model")?;
            }
            "-n" | "--max-tokens" => {
                i += 1;
                let raw = args.get(i).ok_or("missing value after --max-tokens")?;
                max_tokens = raw
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --max-tokens: {raw}"))?;
            }
            "--json" => {
                emit_json = true;
            }
            other if prompt.is_none() && !other.starts_with('-') => {
                prompt = Some(other.to_string());
            }
            other => return Err(format!("unrecognized flag `{other}`")),
        }
        i += 1;
    }

    let prompt_text = prompt.ok_or("missing prompt string after `-p` / `--prompt`")?;
    let req = GenerationRequest {
        model,
        prompt: prompt_text,
        max_tokens: if max_tokens == 0 {
            SERVE_DEFAULT_MAX_NEW_TOKENS
        } else {
            max_tokens
        },
        stream: !emit_json,
    };

    if emit_json {
        let out = run_single_prompt(&req, |_, _| {});
        let tok_s = if out.elapsed_ms > 0.0 {
            (out.usage.completion_tokens as f64) / (out.elapsed_ms / 1000.0)
        } else {
            0.0
        };
        println!(
            "{{\"model\":\"{}\",\"text\":\"{}\",\"finish_reason\":\"{}\",\"prefill_chunks\":{},\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}},\"elapsed_ms\":{:.2},\"tok_s\":{:.1}}}",
            out.model,
            json_escape(&out.text),
            out.finish_reason.as_openai_str(),
            out.prefill_chunks_executed,
            out.usage.prompt_tokens,
            out.usage.completion_tokens,
            out.usage.total_tokens,
            out.elapsed_ms,
            tok_s
        );
    } else {
        let out = run_single_prompt(&req, |_, piece| {
            print!("{piece}");
            let _ = io::stdout().flush();
        });
        println!();
        let tok_s = if out.elapsed_ms > 0.0 {
            (out.usage.completion_tokens as f64) / (out.elapsed_ms / 1000.0)
        } else {
            0.0
        };
        eprintln!(
            "[leansonic -p] model={} | prompt_toks={} | gen_toks={} | prefill_chunks={} | {:.2} ms ({:.1} tok/s)",
            out.model,
            out.usage.prompt_tokens,
            out.usage.completion_tokens,
            out.prefill_chunks_executed,
            out.elapsed_ms,
            tok_s
        );
    }
    Ok(())
}

fn cmd_serve(args: &[String]) -> Result<(), String> {
    let mut host = "127.0.0.1".to_string();
    let mut port: u16 = 8080;
    let mut model = "inkling-2-mini".to_string();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--host" => {
                i += 1;
                host = args.get(i).cloned().ok_or("missing value after --host")?;
            }
            "--port" => {
                i += 1;
                let raw = args.get(i).ok_or("missing value after --port")?;
                port = raw
                    .parse::<u16>()
                    .map_err(|_| format!("invalid --port: {raw}"))?;
            }
            "-m" | "--model" => {
                i += 1;
                model = args.get(i).cloned().ok_or("missing value after --model")?;
            }
            other => return Err(format!("unrecognized `serve` option `{other}`")),
        }
        i += 1;
    }

    serve_openai_http(&host, port, &model).map_err(|e| format!("server I/O error: {e}"))
}

fn verify_prompt_vs_serve_parity() -> Result<(), String> {
    let req = GenerationRequest {
        model: "inkling-2-mini".into(),
        prompt: "Verify -p vs OpenAI serve 0-ULP invariance".into(),
        max_tokens: 8,
        stream: false,
    };
    let solo = run_single_prompt(&req, |_, _| {});
    let slot_served = execute_generation_on_slot(&req, 53, |_, _| {});
    if solo.step_logit_bits != slot_served.step_logit_bits || solo.text != slot_served.text {
        return Err("`-p` solo output diverged from `serve` multi-slot execution".into());
    }
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
        kv_bytes_per_slot: 6.43e9,
        recurrent_state_bytes_per_slot: 0.0,
        draft_bytes_per_step: 0.06e9,
        draft_steps_k: 8,
        mean_accepted_plus_one: 5.8,
        concurrency_slots: 1,
    };
    let ceil = compute_decode_ceiling_tok_s(&sku, &inkling_128k_spec8);
    println!("  Inkling-2-Mini 128K (C=1, W=8, E[acc+1]=5.8) Streaming Ceiling: {ceil:.1} tok/s");

    // Qwen3.8-Flash-Next (51B Engram Host DMA hidden behind Visits 0..1 on RTX PRO 5000)
    let (qwen_engram_tok_s, qwen_exposed_pct) = compute_spilled_engram_ceiling_tok_s(
        &sku,
        &SpilledEngramRooflineInput {
            vram_compute_bytes_per_step: 3.4e9,
            overlap_hideable_vram_fraction: 0.15,
            spilled_pcie_bytes_per_step: 0.0,
            dtod_promoted_bytes_per_step: 0.0,
            engram_pcie_bytes_per_step: 262_144.0, // 64 rows × 4 KiB N-gram gather
            emitted_tokens_per_step: 5.2,
        },
    );
    println!(
        "  Qwen3.8-Flash-Next + 51B Host Engram Ceiling : {qwen_engram_tok_s:.1} tok/s (exposed PCIe stall: {qwen_exposed_pct:.2}%)"
    );

    // Spilled Inkling-2-Mini on RTX 5070 Ti 16GB (PCIe Gen4 x16 + DtoD promotion)
    let sku_spilled = BlackwellSkuProfile::rtx_5070_ti_300w();
    let (spilled_tok_s, spilled_exposed_pct) = compute_spilled_engram_ceiling_tok_s(
        &sku_spilled,
        &SpilledEngramRooflineInput {
            vram_compute_bytes_per_step: 4.8e9,
            overlap_hideable_vram_fraction: 0.72,
            spilled_pcie_bytes_per_step: 0.14e9,
            dtod_promoted_bytes_per_step: 0.35e9,
            engram_pcie_bytes_per_step: 0.0,
            emitted_tokens_per_step: 5.8,
        },
    );
    println!(
        "  RTX 5070 Ti 16GB Spilled Late-Wait Ceiling   : {spilled_tok_s:.1} tok/s (exposed PCIe stall: {spilled_exposed_pct:.2}%)"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.iter().skip(1).any(|a| a == "-p" || a == "--prompt") {
        return match cmd_prompt(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("[ERROR] {e}");
                ExitCode::FAILURE
            }
        };
    }

    match args.get(1).map(|s| s.as_str()) {
        Some("prompt") => match cmd_prompt(&args[2..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("[ERROR] {e}");
                ExitCode::FAILURE
            }
        },
        Some("serve") => match cmd_serve(&args[2..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("[ERROR] {e}");
                ExitCode::FAILURE
            }
        },
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
        Some("verify-parity") => {
            if let Err(e) = verify_inkling2_batch_invariance(64) {
                eprintln!("[FAIL] {e}");
                return ExitCode::FAILURE;
            }
            if let Err(e) = verify_inkling2_resident_vs_spilled_0ulp(64) {
                eprintln!("[FAIL] {e}");
                return ExitCode::FAILURE;
            }
            if let Err(e) = verify_qwen38_engram_boundary_0ulp(64) {
                eprintln!("[FAIL] {e}");
                return ExitCode::FAILURE;
            }
            if let Err(e) = verify_prompt_vs_serve_parity() {
                eprintln!("[FAIL] {e}");
                return ExitCode::FAILURE;
            }
            println!(
                "[PASS] Gate 3: All 4 0-ULP invariants passed (C=1==C=64 | Resident==Spilled | Qwen3.8 Engram Split/Rollback | -p==serve)"
            );
            ExitCode::SUCCESS
        }
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
