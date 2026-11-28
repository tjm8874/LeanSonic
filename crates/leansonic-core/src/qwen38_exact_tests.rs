//! Exact-size `compute-sanitizer --tool memcheck --padding 65536` harness for the
//! `kernels/qwen38_sm120.cu` kernels (#179, #248). Kept apart from `qwen38_gpu.rs` so engine
//! refactors and this test do not touch the same file.

/// Exact-size `compute-sanitizer --tool memcheck --padding 65536` harness for every
/// `kernels/qwen38_sm120.cu` kernel (#179, #248): each launch gets buffers of exactly the
/// documented shape, with indices at the last row / expert / position, so any over-read or
/// over-write lands in the sanitizer's padding. Only the sanitizer verdict matters (random
/// data); the launches must also succeed. `short = true` drops the last row of each case's
/// key buffer (the planted fault); `LEANSONIC_EXACT_ONLY=<kernel>` runs one kernel, since the
/// first fault stops later launches.
mod exact_size {
    use crate::cuda_ffi::{
        check_cu, compile_cuda_to_ptx_nvrtc, CUfunction, CUmodule, CudaContextHandle, DeviceBuf,
    };
    use crate::qwen38_gpu::{Q38_D, Q38_EXP, Q38_HC_D, Q38_MOE_FF, Q38_TOP_K, Q38_VOCAB};
    use core::ffi::c_void;
    use std::ffi::CString;
    use std::sync::Arc;

    const CUDA_HEADER_SRC: &str = include_str!("../../../kernels/include/leansonic_verified.cuh");
    const CUDA_QWEN38_SRC: &str = include_str!("../../../kernels/qwen38_sm120.cu");

    macro_rules! kargs {
            ($($arg:ident),+ $(,)?) => {
                [ $(core::ptr::from_mut(&mut $arg).cast::<c_void>()),+ ]
            };
        }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    fn only() -> Option<String> {
        std::env::var("LEANSONIC_EXACT_ONLY").ok()
    }

    fn module(ctx: &Arc<CudaContextHandle>) -> Result<CUmodule, String> {
        let image: Vec<u8> = match std::env::var("LEANSONIC_QWEN38_CUBIN_PATH") {
            Ok(p) => std::fs::read(&p).map_err(|e| format!("read {p}: {e}"))?,
            Err(_) => compile_cuda_to_ptx_nvrtc(
                &CUDA_QWEN38_SRC.replace(
                    "#include \"include/leansonic_verified.cuh\"",
                    CUDA_HEADER_SRC,
                ),
                "compute_120a",
            )?,
        };
        let mut m: CUmodule = core::ptr::null_mut();
        unsafe {
            check_cu(
                (ctx.api.cuModuleLoadData)(&mut m, image.as_ptr().cast()),
                "load qwen38",
            )?
        };
        Ok(m)
    }

    fn run(
        ctx: &Arc<CudaContextHandle>,
        m: CUmodule,
        name: &str,
        grid: (u32, u32),
        block: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), String> {
        if only().is_some_and(|k| k != name) {
            return Ok(());
        }
        let c = CString::new(name).unwrap();
        let mut f: CUfunction = core::ptr::null_mut();
        unsafe {
            check_cu((ctx.api.cuModuleGetFunction)(&mut f, m, c.as_ptr()), name)?;
            check_cu(
                (ctx.api.cuLaunchKernel)(
                    f,
                    grid.0,
                    grid.1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    ctx.compute_stream,
                    args.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                name,
            )?;
        }
        ctx.sync_compute().map_err(|e| format!("{name}: {e}"))
    }

    /// `n` random bytes (quantized codes, BF16 scale / bias pairs).
    fn noise(ctx: &Arc<CudaContextHandle>, rng: &mut Lcg, n: usize) -> Result<DeviceBuf, String> {
        let bytes: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
        DeviceBuf::from_host_bytes(ctx, &bytes)
    }

    /// `n` small finite floats in `[-0.5, 0.5)`.
    fn floats(ctx: &Arc<CudaContextHandle>, rng: &mut Lcg, n: usize) -> Result<DeviceBuf, String> {
        let bytes: Vec<u8> = (0..n)
            .flat_map(|_| ((rng.next() % 10_000) as f32 / 10_000.0 - 0.5).to_le_bytes())
            .collect();
        DeviceBuf::from_host_bytes(ctx, &bytes)
    }

    fn ints(ctx: &Arc<CudaContextHandle>, v: &[i32]) -> Result<DeviceBuf, String> {
        DeviceBuf::from_host_bytes(
            ctx,
            &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        )
    }

    fn zeros(ctx: &Arc<CudaContextHandle>, n: usize) -> Result<DeviceBuf, String> {
        DeviceBuf::alloc_zeroed(ctx, n)
    }

    /// One `q38_q4_gemv` / `q38_q4_gemv_10k` case: `w [rows, cols / 32]` u128,
    /// `sc / bi [rows, cols / 64]` u32, `x [cols]`, `y [rows]`.
    #[allow(clippy::too_many_arguments)]
    fn q4_case(
        ctx: &Arc<CudaContextHandle>,
        m: CUmodule,
        rng: &mut Lcg,
        rows: usize,
        cols: usize,
        mode: i32,
        short: usize,
    ) -> Result<(), String> {
        let w = noise(ctx, rng, (rows - short) * (cols / 32) * 16)?;
        let sc = noise(ctx, rng, rows * (cols / 64) * 4)?;
        let bi = noise(ctx, rng, rows * (cols / 64) * 4)?;
        let x = floats(ctx, rng, cols)?;
        let y = zeros(ctx, rows * 4)?;
        let (mut pw, mut ps, mut pb, mut px, mut py) = (w.ptr, sc.ptr, bi.ptr, x.ptr, y.ptr);
        let (mut r, mut c, mut md) = (rows as i32, cols as i32, mode);
        let grid = ((rows as u32).div_ceil(8), 1);
        if cols == Q38_HC_D {
            run(
                ctx,
                m,
                "q38_q4_gemv_10k",
                grid,
                256,
                &mut kargs![pw, ps, pb, px, py, r, md],
            )
        } else {
            run(
                ctx,
                m,
                "q38_q4_gemv",
                grid,
                256,
                &mut kargs![pw, ps, pb, px, py, r, c, md],
            )
        }
        .map_err(|e| format!("q4 [{rows}, {cols}] mode {mode}: {e}"))
    }

    fn cases(short: bool) -> Result<(), String> {
        let ctx = CudaContextHandle::init(0)?;
        let ctx = &ctx;
        let m = module(ctx)?;
        let mut rng = Lcg(41);
        let rng = &mut rng;
        let s = usize::from(short);
        let (d, hcd) = (Q38_D, Q38_HC_D);

        // Embedding: `[vocab, 80]` u128 codes + `[vocab, 40]` scale / bias words, last token.
        {
            let w = noise(ctx, rng, (Q38_VOCAB - s) * 80 * 16)?;
            let sc = noise(ctx, rng, Q38_VOCAB * 40 * 4)?;
            let bi = noise(ctx, rng, Q38_VOCAB * 40 * 4)?;
            let tok = ints(ctx, &[Q38_VOCAB as i32 - 1])?;
            let out = zeros(ctx, hcd * 4)?;
            let (mut pw, mut ps, mut pb, mut pt, mut po) =
                (w.ptr, sc.ptr, bi.ptr, tok.ptr, out.ptr);
            run(
                ctx,
                m,
                "q38_embed_init_hc4",
                (10, 1),
                256,
                &mut kargs![pw, ps, pb, pt, po],
            )?;
        }

        // Q4 GEMVs at every engine shape (HC down / up / inject, GDN in / out, QSA in /
        // out, PLE kv, LM head) plus a 17-row tail; all three post modes.
        for (rows, cols, mode) in [
            (320, hcd, 1),
            (4, hcd, 0),
            (hcd, 320, 2),
            (16_480, d, 0),
            (d, 6144, 0),
            (13_952, d, 0),
            (5 * d, d, 0),
            (17, 6144, 1),
            (Q38_VOCAB, d, 0),
        ] {
            q4_case(ctx, m, rng, rows, cols, mode, s)?;
        }

        // HC pre / post and PLE inject: `[4, 2560]` streams.
        {
            let hc = floats(ctx, rng, 4 * d - s)?;
            let gamma = floats(ctx, rng, 4 * d)?;
            let normed = zeros(ctx, 4 * d * 4)?;
            let (mut ph, mut pg, mut pn) = (hc.ptr, gamma.ptr, normed.ptr);
            run(ctx, m, "q38_hc_norm4", (4, 1), 256, &mut kargs![ph, pg, pn])?;
            let up = floats(ctx, rng, 4 * d - s)?;
            let mixed = zeros(ctx, d * 4)?;
            let (mut pu, mut pm) = (up.ptr, mixed.ptr);
            run(ctx, m, "q38_hc_mix4", (10, 1), 256, &mut kargs![pn, pu, pm])?;
            let branch = floats(ctx, rng, d - s)?;
            let inj = floats(ctx, rng, 4)?;
            let (mut pbr, mut pi) = (branch.ptr, inj.ptr);
            run(
                ctx,
                m,
                "q38_hc_inject4",
                (40, 1),
                256,
                &mut kargs![ph, pbr, pi],
            )?;

            let kv = floats(ctx, rng, 5 * d)?; // key [4, 2560] then value [2560]
            let (nk, nq, nc) = (
                floats(ctx, rng, 4 * d)?,
                floats(ctx, rng, 4 * d)?,
                floats(ctx, rng, 4 * d)?,
            );
            let cw = floats(ctx, rng, 4 * 4 * d)?;
            let hist = floats(ctx, rng, 9 * hcd - s)?;
            let pos = ints(ctx, &[1_000_003])?;
            let (mut pk, mut pv) = (kv.ptr, kv.ptr + (4 * d * 4) as u64);
            let (mut pnk, mut pnq, mut pnc, mut pcw, mut phi, mut pp) =
                (nk.ptr, nq.ptr, nc.ptr, cw.ptr, hist.ptr, pos.ptr);
            run(
                ctx,
                m,
                "q38_ple_inject",
                (4, 1),
                256,
                &mut kargs![ph, pk, pv, pnk, pnq, pnc, pcw, phi, pp],
            )?;
        }

        // GDN: conv over `[10240]` channels, recurrent step over 48 V-heads.
        {
            let raw = floats(ctx, rng, hcd)?;
            let cw = floats(ctx, rng, 4 * hcd)?;
            let hist = floats(ctx, rng, 3 * hcd - s)?;
            let conv = zeros(ctx, hcd * 4)?;
            let (mut pr, mut pc, mut ph, mut po) = (raw.ptr, cw.ptr, hist.ptr, conv.ptr);
            run(
                ctx,
                m,
                "q38_gdn_conv_silu",
                (40, 1),
                256,
                &mut kargs![pr, pc, ph, po],
            )?;
            let zba = floats(ctx, rng, 6240)?;
            let (al, dt, nw) = (
                floats(ctx, rng, 48)?,
                floats(ctx, rng, 48)?,
                floats(ctx, rng, 128)?,
            );
            let state = floats(ctx, rng, 48 * 128 * 128 - s)?;
            let out = zeros(ctx, 6144 * 4)?;
            let (mut pz, mut pa, mut pd, mut pn, mut ps, mut pout) =
                (zba.ptr, al.ptr, dt.ptr, nw.ptr, state.ptr, out.ptr);
            for mut gate in [0i32, 1] {
                run(
                    ctx,
                    m,
                    "q38_gdn_recurrent_step",
                    (48, 1),
                    128,
                    &mut kargs![po, pz, pa, pd, pn, gate, ps, pout],
                )?;
            }
        }

        // QSA at the last cache row: `max_ctx = 4100` (1025 key blocks, > 512 so the
        // indexer mask runs), caches `[max_ctx, 512]` FP16, `[max_ctx, 128]` and
        // `[max_ctx / 4, 128]` indexer keys.
        {
            let max_ctx = 4100usize;
            let n_blocks = max_ctx / 4;
            let proj = floats(ctx, rng, 13_952)?;
            let (qn, kn) = (floats(ctx, rng, 256)?, floats(ctx, rng, 256)?);
            let (iqn, ikn) = (floats(ctx, rng, 128)?, floats(ctx, rng, 128)?);
            let invf = floats(ctx, rng, 32)?;
            let pos = ints(ctx, &[max_ctx as i32 - 1])?;
            let (qrot, qgate) = (zeros(ctx, 24 * 256 * 4)?, zeros(ctx, 24 * 256 * 4)?);
            let kc = noise(ctx, rng, (max_ctx - s) * 512 * 2)?;
            let vc = noise(ctx, rng, max_ctx * 512 * 2)?;
            let iq = zeros(ctx, 4 * 128 * 4)?;
            let irk = zeros(ctx, max_ctx * 128 * 4)?;
            let ibk = floats(ctx, rng, n_blocks * 128)?;
            let (mut pp, mut pqn, mut pkn, mut piq, mut pik, mut pif, mut ppos) = (
                proj.ptr, qn.ptr, kn.ptr, iqn.ptr, ikn.ptr, invf.ptr, pos.ptr,
            );
            let (mut pqr, mut pqg, mut pkc, mut pvc, mut pio, mut pirk, mut pibk) = (
                qrot.ptr, qgate.ptr, kc.ptr, vc.ptr, iq.ptr, irk.ptr, ibk.ptr,
            );
            run(
                ctx,
                m,
                "q38_qsa_prep_and_cache",
                (31, 1),
                256,
                &mut kargs![pp, pqn, pkn, piq, pik, pif, ppos, pqr, pqg, pkc, pvc, pio, pirk, pibk],
            )?;
            let scores = zeros(ctx, (n_blocks - s) * 4)?;
            let mask = zeros(ctx, n_blocks * 4)?;
            let (mut psc, mut pmask, mut nb) = (scores.ptr, mask.ptr, n_blocks as i32);
            run(
                ctx,
                m,
                "q38_qsa_score_blocks",
                ((n_blocks as u32).div_ceil(8), 1),
                256,
                &mut kargs![pio, pibk, nb, psc],
            )?;
            run(
                ctx,
                m,
                "q38_qsa_select_top512",
                (1, 1),
                256,
                &mut kargs![psc, nb, pmask],
            )?;
            let out = zeros(ctx, 24 * 256 * 4)?;
            let mut pout = out.ptr;
            for mut use_mask in [0i32, 1] {
                run(
                    ctx,
                    m,
                    "q38_qsa_attn",
                    (24, 1),
                    256,
                    &mut kargs![pqr, pqg, pkc, pvc, pmask, ppos, use_mask, pout],
                )?;
            }
        }

        // Router + Top-10 + MoE: all 512 experts in VRAM with expert 511 selected, then the
        // compact `[10, ...]` staged slab.
        {
            let rw = noise(ctx, rng, (Q38_EXP - s) * 320 * 16)?;
            let x = floats(ctx, rng, d)?;
            let logits = zeros(ctx, Q38_EXP * 4)?;
            let (mut prw, mut px, mut pl) = (rw.ptr, x.ptr, logits.ptr);
            run(
                ctx,
                m,
                "q38_bf16_router_512",
                (64, 1),
                256,
                &mut kargs![prw, px, pl],
            )?;
            let lg = floats(ctx, rng, Q38_EXP - s)?;
            let ids_out = zeros(ctx, 10 * 4)?;
            let wts = floats(ctx, rng, 11)?;
            let (mut plg, mut pid, mut pwt) = (lg.ptr, ids_out.ptr, wts.ptr);
            run(
                ctx,
                m,
                "q38_softmax_topk10",
                (1, 1),
                256,
                &mut kargs![plg, pid, pwt],
            )?;

            let sel = ints(ctx, &[511, 0, 7, 300, 511, 64, 2, 499, 128, 510])?;
            let sh_gu = noise(ctx, rng, 1281 * 80 * 16)?;
            let (sh_sc, sh_bi) = (
                noise(ctx, rng, 1281 * 40 * 4)?,
                noise(ctx, rng, 1281 * 40 * 4)?,
            );
            let sh_d = noise(ctx, rng, d * 20 * 16)?;
            let (sh_dsc, sh_dbi) = (noise(ctx, rng, d * 10 * 4)?, noise(ctx, rng, d * 10 * 4)?);
            let act = zeros(ctx, 11 * Q38_MOE_FF * 4)?;
            let out = zeros(ctx, d * 4)?;
            for (experts, compact) in [(Q38_EXP, 0i32), (Q38_TOP_K, 1)] {
                // Gate / up `[E, 640, 80]` + `[E, 640, 40]`, down `[E, 2560, 20]` + `[E, 2560, 10]`.
                let gw = noise(ctx, rng, (experts * 640 - s) * 80 * 16)?;
                let uw = noise(ctx, rng, experts * 640 * 80 * 16)?;
                let gsc = noise(ctx, rng, experts * 640 * 40 * 4)?;
                let gbi = noise(ctx, rng, experts * 640 * 40 * 4)?;
                let usc = noise(ctx, rng, experts * 640 * 40 * 4)?;
                let ubi = noise(ctx, rng, experts * 640 * 40 * 4)?;
                let (mut pgw, mut pgs, mut pgb, mut puw, mut pus, mut pub_) =
                    (gw.ptr, gsc.ptr, gbi.ptr, uw.ptr, usc.ptr, ubi.ptr);
                let (mut psw, mut pss, mut psb, mut px2, mut psel, mut pw2, mut pa) = (
                    sh_gu.ptr, sh_sc.ptr, sh_bi.ptr, x.ptr, sel.ptr, wts.ptr, act.ptr,
                );
                let mut cmp = compact;
                run(
                    ctx,
                    m,
                    "q38_moe_up_swiglu11",
                    (160, 11),
                    256,
                    &mut kargs![
                        pgw, pgs, pgb, puw, pus, pub_, psw, pss, psb, px2, psel, cmp, pw2, pa
                    ],
                )?;
                drop((gw, uw, gsc, gbi, usc, ubi));
                let dw = noise(ctx, rng, (experts * d - s) * 20 * 16)?;
                let dsc = noise(ctx, rng, experts * d * 10 * 4)?;
                let dbi = noise(ctx, rng, experts * d * 10 * 4)?;
                let (mut pdw, mut pds, mut pdb, mut phw, mut phs, mut phb, mut po) = (
                    dw.ptr, dsc.ptr, dbi.ptr, sh_d.ptr, sh_dsc.ptr, sh_dbi.ptr, out.ptr,
                );
                run(
                    ctx,
                    m,
                    "q38_moe_down_combine11",
                    (320, 1),
                    256,
                    &mut kargs![pdw, pds, pdb, phw, phs, phb, pa, psel, cmp, pw2, po],
                )?;
            }
        }

        // Greedy head: `[vocab]` logits.
        {
            let logits = floats(ctx, rng, Q38_VOCAB - s)?;
            let tok = zeros(ctx, 4)?;
            let pos = ints(ctx, &[7])?;
            let (mut pl, mut pt, mut pp) = (logits.ptr, tok.ptr, pos.ptr);
            run(
                ctx,
                m,
                "q38_argmax_and_inc_pos",
                (1, 1),
                256,
                &mut kargs![pl, pt, pp],
            )?;
            run(ctx, m, "q38_inc_pos", (1, 1), 1, &mut kargs![pp])?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device; run under compute-sanitizer --tool memcheck --padding 65536"]
    fn exact_size_qwen38_kernels() -> Result<(), String> {
        cases(false)
    }

    #[test]
    #[ignore = "planted fault: memcheck --padding 65536 must report an out-of-bounds access"]
    fn exact_size_qwen38_red_control() -> Result<(), String> {
        let _ = cases(true);
        Ok(())
    }
}
