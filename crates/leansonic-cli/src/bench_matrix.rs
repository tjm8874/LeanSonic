//! `leansonic bench-matrix`: the goal oracle for `docs/goals/leansonic-fastest-engine`.
//!
//! Runs `leansonic bench` in fresh child processes (one per sample) on each requested GPU,
//! computes the median and a deterministic 95% bootstrap interval of the median, and scores
//! every cell of the target table (`docs/benchmarks/targets.json`) as PASS / FAIL /
//! RECORDED / UNMEASURED. Zero third-party dependencies: JSON, SHA-256 and `flock` are local.
use leansonic_core::sha256::sha256_hex;
use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Minimal JSON value, parser and writer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
}

pub fn parse_json(text: &str) -> Result<Json, String> {
    let bytes = text.as_bytes();
    let mut pos = 0usize;
    let v = parse_value(bytes, &mut pos)?;
    skip_ws(bytes, &mut pos);
    if pos != bytes.len() {
        return Err(format!("trailing data at byte {pos}"));
    }
    Ok(v)
}

fn skip_ws(b: &[u8], p: &mut usize) {
    while *p < b.len() && matches!(b[*p], b' ' | b'\n' | b'\r' | b'\t') {
        *p += 1;
    }
}

fn parse_value(b: &[u8], p: &mut usize) -> Result<Json, String> {
    skip_ws(b, p);
    match b.get(*p) {
        None => Err("unexpected end of JSON".into()),
        Some(b'{') => {
            *p += 1;
            let mut m = BTreeMap::new();
            skip_ws(b, p);
            if b.get(*p) == Some(&b'}') {
                *p += 1;
                return Ok(Json::Obj(m));
            }
            loop {
                skip_ws(b, p);
                let k = match parse_value(b, p)? {
                    Json::Str(s) => s,
                    _ => return Err(format!("object key must be a string at byte {p}")),
                };
                skip_ws(b, p);
                if b.get(*p) != Some(&b':') {
                    return Err(format!("expected ':' at byte {p}"));
                }
                *p += 1;
                let v = parse_value(b, p)?;
                m.insert(k, v);
                skip_ws(b, p);
                match b.get(*p) {
                    Some(b',') => *p += 1,
                    Some(b'}') => {
                        *p += 1;
                        return Ok(Json::Obj(m));
                    }
                    _ => return Err(format!("expected ',' or '}}' at byte {p}")),
                }
            }
        }
        Some(b'[') => {
            *p += 1;
            let mut a = Vec::new();
            skip_ws(b, p);
            if b.get(*p) == Some(&b']') {
                *p += 1;
                return Ok(Json::Arr(a));
            }
            loop {
                a.push(parse_value(b, p)?);
                skip_ws(b, p);
                match b.get(*p) {
                    Some(b',') => *p += 1,
                    Some(b']') => {
                        *p += 1;
                        return Ok(Json::Arr(a));
                    }
                    _ => return Err(format!("expected ',' or ']' at byte {p}")),
                }
            }
        }
        Some(b'"') => {
            *p += 1;
            let mut s = String::new();
            loop {
                let c = *b.get(*p).ok_or("unterminated string")?;
                *p += 1;
                match c {
                    b'"' => return Ok(Json::Str(s)),
                    b'\\' => {
                        let e = *b.get(*p).ok_or("bad escape")?;
                        *p += 1;
                        match e {
                            b'"' => s.push('"'),
                            b'\\' => s.push('\\'),
                            b'/' => s.push('/'),
                            b'n' => s.push('\n'),
                            b'r' => s.push('\r'),
                            b't' => s.push('\t'),
                            b'b' => s.push('\u{8}'),
                            b'f' => s.push('\u{c}'),
                            b'u' => {
                                let hex = std::str::from_utf8(b.get(*p..*p + 4).ok_or("bad \\u")?)
                                    .map_err(|_| "bad \\u")?;
                                let cp = u32::from_str_radix(hex, 16).map_err(|_| "bad \\u")?;
                                *p += 4;
                                s.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                            }
                            _ => return Err("bad escape".into()),
                        }
                    }
                    _ => {
                        // Re-decode UTF-8 runs byte-wise.
                        let start = *p - 1;
                        let mut end = *p;
                        while end < b.len() && b[end] != b'"' && b[end] != b'\\' {
                            end += 1;
                        }
                        s.push_str(std::str::from_utf8(&b[start..end]).map_err(|_| "bad utf-8")?);
                        *p = end;
                    }
                }
            }
        }
        Some(b't') if b[*p..].starts_with(b"true") => {
            *p += 4;
            Ok(Json::Bool(true))
        }
        Some(b'f') if b[*p..].starts_with(b"false") => {
            *p += 5;
            Ok(Json::Bool(false))
        }
        Some(b'n') if b[*p..].starts_with(b"null") => {
            *p += 4;
            Ok(Json::Null)
        }
        Some(_) => {
            let start = *p;
            while *p < b.len() && matches!(b[*p], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
                *p += 1;
            }
            let t = std::str::from_utf8(&b[start..*p]).map_err(|_| "bad number")?;
            t.parse::<f64>()
                .map(Json::Num)
                .map_err(|_| format!("bad JSON token at byte {start}"))
        }
    }
}

pub fn write_json(v: &Json, out: &mut String, indent: usize) {
    let pad = |n: usize| " ".repeat(n);
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(x) => {
            if x.is_finite() {
                if x.fract() == 0.0 && x.abs() < 1e15 {
                    let _ = write!(out, "{}", *x as i64);
                } else {
                    let _ = write!(out, "{x}");
                }
            } else {
                out.push_str("null");
            }
        }
        Json::Str(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => {
                        let _ = write!(out, "\\u{:04x}", c as u32);
                    }
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Json::Arr(a) => {
            if a.iter().all(|x| matches!(x, Json::Num(_))) {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_json(x, out, indent);
                }
                out.push(']');
                return;
            }
            out.push_str("[\n");
            for (i, x) in a.iter().enumerate() {
                out.push_str(&pad(indent + 2));
                write_json(x, out, indent + 2);
                if i + 1 < a.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&pad(indent));
            out.push(']');
        }
        Json::Obj(m) => {
            out.push_str("{\n");
            let n = m.len();
            for (i, (k, x)) in m.iter().enumerate() {
                out.push_str(&pad(indent + 2));
                write_json(&Json::Str(k.clone()), out, 0);
                out.push_str(": ");
                write_json(x, out, indent + 2);
                if i + 1 < n {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&pad(indent));
            out.push('}');
        }
    }
}

fn obj(pairs: Vec<(&str, Json)>) -> Json {
    Json::Obj(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn num_or_null(v: Option<f64>) -> Json {
    v.map(Json::Num).unwrap_or(Json::Null)
}

// ---------------------------------------------------------------------------
// FNV-1a for input token provenance
// ---------------------------------------------------------------------------

pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Statistics: median and deterministic bootstrap CI of the median
// ---------------------------------------------------------------------------

pub fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    Some(if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    })
}

/// 95% percentile-bootstrap interval of the median (2,000 resamples, fixed seed).
pub fn bootstrap_ci(v: &[f64]) -> Option<(f64, f64)> {
    if v.is_empty() {
        return None;
    }
    if v.len() == 1 {
        return Some((v[0], v[0]));
    }
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15 ^ (v.len() as u64);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let reps = 2000;
    let mut meds = Vec::with_capacity(reps);
    let mut buf = vec![0.0; v.len()];
    for _ in 0..reps {
        for x in buf.iter_mut() {
            *x = v[(next() % v.len() as u64) as usize];
        }
        meds.push(median(&buf).unwrap());
    }
    meds.sort_by(f64::total_cmp);
    let lo = meds[(reps as f64 * 0.025) as usize];
    let hi = meds[((reps as f64 * 0.975) as usize).min(reps - 1)];
    Some((lo, hi))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    Recorded,
    Unmeasured,
}

impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Recorded => "RECORDED",
            Verdict::Unmeasured => "UNMEASURED",
        }
    }
}

/// `higher_is_better`: PASS iff CI lower bound >= target; otherwise PASS iff CI upper bound <= target.
/// No target (`None`) with samples = RECORDED (the goal's "計測して記録" cells).
pub fn verdict(samples: &[f64], target: Option<f64>, higher_is_better: bool) -> Verdict {
    let Some((lo, hi)) = bootstrap_ci(samples) else {
        return Verdict::Unmeasured;
    };
    match target {
        None => Verdict::Recorded,
        Some(t) if higher_is_better && lo >= t => Verdict::Pass,
        Some(t) if !higher_is_better && hi <= t => Verdict::Pass,
        Some(_) => Verdict::Fail,
    }
}

// ---------------------------------------------------------------------------
// `leansonic bench` output parsing
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BenchRun {
    pub device: Option<String>,
    pub vram_gib: Option<f64>,
    pub resident: Option<String>,
    pub prefill_tok_s: Option<f64>,
    pub gen_tok_s: Option<f64>,
    pub gen_ms_tok: Option<f64>,
    pub spec_tok_step: Option<f64>,
    pub spec_acc_pct: Option<f64>,
}

fn num_before(s: &str, suffix: &str) -> Option<f64> {
    let end = s.find(suffix)?;
    let head = s[..end].trim_end();
    let start = head
        .rfind(|c: char| !(c.is_ascii_digit() || c == '.'))
        .map(|i| i + 1)
        .unwrap_or(0);
    head[start..].parse().ok()
}

/// Parses the last `[run N]` prefill / generated lines (warm run) and the engine header.
pub fn parse_bench_output(text: &str) -> BenchRun {
    let mut r = BenchRun::default();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("[LeanSonic GPU] ") {
            let mut parts = rest.split(" | ");
            r.device = parts.next().map(str::to_string);
            for p in parts {
                if let Some(v) = p.strip_prefix("VRAM used: ") {
                    r.vram_gib = v.split(' ').next().and_then(|x| x.parse().ok());
                } else if let Some(v) = p.strip_prefix("Resident expert layers: ") {
                    r.resident = v.split(' ').next().map(str::to_string);
                }
            }
        } else if line.starts_with("[run ") && line.contains("] prefill ") {
            r.prefill_tok_s = num_before(line, " tok/s");
        } else if line.starts_with("[run ") && line.contains("] generated ") {
            r.gen_ms_tok = num_before(line, " ms/tok");
            let after = &line[line.find("ms/tok").map(|i| i + 6).unwrap_or(0)..];
            r.gen_tok_s = num_before(after, " tok/s)");
            r.spec_tok_step = num_before(line, " tok/step]");
            r.spec_acc_pct = num_before(line, "%)");
        }
    }
    r
}

// ---------------------------------------------------------------------------
// GPU discovery, locking and busy check
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct GpuInfo {
    pub index: usize,
    pub name: String,
    pub uuid: String,
    pub driver: String,
    pub power_limit_w: String,
    pub pci_bus_id: String,
    pub pcie: String,
}

fn nvidia_smi(args: &[&str]) -> Option<String> {
    let out = Command::new("nvidia-smi").args(args).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// nvidia-smi indices are PCI-bus ordered, matching `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
pub fn query_gpus() -> Vec<GpuInfo> {
    let Some(text) = nvidia_smi(&[
        "--query-gpu=index,name,uuid,driver_version,power.limit,pci.bus_id,pcie.link.gen.max,pcie.link.width.max",
        "--format=csv,noheader,nounits",
    ]) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').map(str::trim).collect();
            (f.len() >= 8).then(|| GpuInfo {
                index: f[0].parse().unwrap_or(0),
                name: f[1].to_string(),
                uuid: f[2].to_string(),
                driver: f[3].to_string(),
                power_limit_w: f[4].to_string(),
                pci_bus_id: f[5].to_string(),
                pcie: format!("Gen{} x{}", f[6], f[7]),
            })
        })
        .collect()
}

/// Foreign compute processes on the given PCI bus id (other than this process tree).
fn foreign_processes(bus_id: &str) -> Vec<String> {
    let Some(text) = nvidia_smi(&[
        "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]) else {
        return Vec::new();
    };
    let norm = |s: &str| {
        s.trim()
            .to_ascii_lowercase()
            .trim_start_matches("0000")
            .to_string()
    };
    text.lines()
        .filter(|l| l.split(',').next().map(norm) == Some(norm(bus_id)))
        .map(|l| {
            l.split(',')
                .skip(1)
                .map(str::trim)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

fn hostname() -> String {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .or_else(|_| fs::read_to_string("/etc/hostname"))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown-host".into())
}

#[cfg(unix)]
mod lock {
    use std::fs::File;
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn flock(fd: i32, operation: i32) -> i32;
    }
    const LOCK_EX: i32 = 2;
    /// Shared cross-agent GPU lock (#6): `/tmp/leansonic-gpu-<host>-<pci index>.lock`.
    pub struct GpuLock(#[allow(dead_code)] File);
    pub fn acquire(path: &str) -> Result<GpuLock, String> {
        let f = File::create(path).map_err(|e| format!("open {path}: {e}"))?;
        // SAFETY: valid fd owned by `f` for the lifetime of the returned guard.
        let rc = unsafe { flock(f.as_raw_fd(), LOCK_EX) };
        if rc != 0 {
            return Err(format!("flock {path} failed"));
        }
        Ok(GpuLock(f))
    }
}

// ---------------------------------------------------------------------------
// Cells
// ---------------------------------------------------------------------------

/// One measured configuration: (cell id, token spec, spec K, gen tokens).
struct Probe {
    id: &'static str,
    tokens: String,
    spec: u32,
    gen: usize,
    /// `LEANSONIC_CTX` override (long-context probes need `N + headroom`).
    ctx: Option<usize>,
}

#[derive(Default, Clone)]
struct ProbeSamples {
    runs: Vec<BenchRun>,
    input_fnv: String,
    errors: Vec<String>,
}

fn sku_key(name: &str, targets: &Json) -> Option<String> {
    let Json::Obj(skus) = targets.get("skus")? else {
        return None;
    };
    skus.iter()
        .find(|(_, v)| {
            v.get("match")
                .and_then(Json::as_str)
                .is_some_and(|m| name.contains(m))
        })
        .map(|(k, _)| k.clone())
}

struct Opts {
    weights: String,
    gpus: Vec<usize>,
    n: usize,
    ctx: usize,
    quick: bool,
    long: bool,
    skip_canary: bool,
    /// `--c64`: 64 independent sequences decoded in shared rounds (`multiseq`, #107).
    c64: bool,
    /// `--c4-32k <ids>`: 4 sequences of a 32K-token prompt, DSpark verify rounds (#107).
    c4_32k: Option<PathBuf>,
    /// `--kld <ref.kld>`: run `eval-kld` once per GPU for the `kld_mean` cell.
    kld_ref: Option<PathBuf>,
    spec_prompts: Option<PathBuf>,
    /// `--wide-prompts <dir>`: every `*.txt` (via `bench --tokens chat:@file`) and `*.ids` file
    /// becomes a DSpark probe for the RECORDED `spec_wide` / `spec_accept_wide` cells.
    wide_prompts: Option<PathBuf>,
    targets: PathBuf,
    out: Option<PathBuf>,
    bin: PathBuf,
    require_idle: bool,
    commit: String,
}

fn parse_opts(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        weights: String::new(),
        gpus: Vec::new(),
        n: 5,
        ctx: 20480,
        quick: false,
        long: false,
        skip_canary: false,
        c64: false,
        c4_32k: None,
        kld_ref: None,
        spec_prompts: None,
        wide_prompts: None,
        targets: PathBuf::from("docs/benchmarks/targets.json"),
        out: None,
        bin: env::current_exe().map_err(|e| e.to_string())?,
        require_idle: false,
        commit: env::var("LEANSONIC_COMMIT").unwrap_or_else(|_| "unknown".into()),
    };
    let mut gpus_arg = "all".to_string();
    let mut i = 0;
    let val = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        args.get(*i)
            .cloned()
            .ok_or(format!("missing value after {flag}"))
    };
    while i < args.len() {
        match args[i].as_str() {
            "-w" | "--weights" => o.weights = val(&mut i, "--weights")?,
            "--gpus" => gpus_arg = val(&mut i, "--gpus")?,
            "--n" => o.n = val(&mut i, "--n")?.parse().map_err(|_| "bad --n")?,
            "--ctx" => o.ctx = val(&mut i, "--ctx")?.parse().map_err(|_| "bad --ctx")?,
            "--quick" => o.quick = true,
            "--long" => o.long = true,
            "--no-canary" => o.skip_canary = true,
            "--c64" => o.c64 = true,
            "--c4-32k" => o.c4_32k = Some(PathBuf::from(val(&mut i, "--c4-32k")?)),
            "--kld" => o.kld_ref = Some(PathBuf::from(val(&mut i, "--kld")?)),
            "--require-idle" => o.require_idle = true,
            "--wide-prompts" => {
                o.wide_prompts = Some(PathBuf::from(val(&mut i, "--wide-prompts")?))
            }
            "--spec-prompts" => {
                o.spec_prompts = Some(PathBuf::from(val(&mut i, "--spec-prompts")?))
            }
            "--targets" => o.targets = PathBuf::from(val(&mut i, "--targets")?),
            "--out" => o.out = Some(PathBuf::from(val(&mut i, "--out")?)),
            "--bin" => o.bin = PathBuf::from(val(&mut i, "--bin")?),
            "--commit" => o.commit = val(&mut i, "--commit")?,
            other => return Err(format!("unknown bench-matrix option `{other}`")),
        }
        i += 1;
    }
    if o.weights.is_empty() {
        return Err("bench-matrix requires --weights <dir>".into());
    }
    if o.quick {
        o.n = 1;
    }
    o.n = o.n.max(1);
    o.gpus = if gpus_arg == "all" {
        query_gpus().iter().map(|g| g.index).collect()
    } else {
        gpus_arg
            .split(',')
            .map(|s| s.trim().parse().map_err(|_| format!("bad GPU index `{s}`")))
            .collect::<Result<_, _>>()?
    };
    if o.gpus.is_empty() {
        return Err("no GPUs found (nvidia-smi unavailable?) — pass --gpus 0,1".into());
    }
    Ok(o)
}

fn probes(o: &Opts) -> Vec<Probe> {
    let mut v = vec![
        Probe {
            id: "short",
            tokens: "synth:32".into(),
            spec: 0,
            gen: 128,
            ctx: Some(8192),
        },
        Probe {
            id: "p2k",
            tokens: "synth:2048".into(),
            spec: 0,
            gen: 32,
            ctx: Some(8192),
        },
    ];
    if !o.quick {
        v.push(Probe {
            id: "p16k",
            tokens: "synth:16384".into(),
            spec: 0,
            gen: 128,
            ctx: None,
        });
    }
    if o.c64 {
        v.push(Probe {
            id: "c64",
            tokens: "synth:400".into(),
            spec: 0,
            gen: 64,
            ctx: Some(4096),
        });
    }
    if let Some(ids) = &o.c4_32k {
        v.push(Probe {
            id: "c4_32k",
            tokens: format!("@{}", ids.display()),
            spec: 7,
            gen: 256,
            ctx: Some(32_768 + 512),
        });
    }
    if o.long && !o.quick {
        v.push(Probe {
            id: "p128k",
            tokens: "synth:131072".into(),
            spec: 0,
            gen: 128,
            ctx: Some(131_072 + 512),
        });
    }
    if let Some(dir) = &o.spec_prompts {
        for k in ["conv", "struct", "prose"] {
            let f = dir.join(format!("{k}.ids"));
            if f.exists() {
                v.push(Probe {
                    id: match k {
                        "conv" => "spec_conv",
                        "struct" => "spec_struct",
                        _ => "spec_prose",
                    },
                    tokens: format!("@{}", f.display()),
                    spec: 7,
                    gen: 256,
                    ctx: None,
                });
            }
        }
    }
    if let Some(dir) = &o.wide_prompts {
        let mut files: Vec<PathBuf> = fs::read_dir(dir)
            .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        files.retain(|f| matches!(f.extension().and_then(|e| e.to_str()), Some("txt" | "ids")));
        files.sort();
        for f in files {
            let stem = f.file_stem().and_then(|s| s.to_str()).unwrap_or("p");
            let tokens = if f.extension().and_then(|e| e.to_str()) == Some("txt") {
                format!("chat:@{}", f.display())
            } else {
                format!("@{}", f.display())
            };
            v.push(Probe {
                // Probe ids are `&'static str`; a CLI run builds this list once.
                id: Box::leak(format!("{WIDE_PREFIX}{stem}").into_boxed_str()),
                tokens,
                spec: 7,
                gen: 256,
                ctx: None,
            });
        }
    }
    v
}

const WIDE_PREFIX: &str = "wide_";

/// Per sample index, the mean over all `--wide-prompts` probes of `f` (samples with any
/// missing prompt are dropped, so every value averages the same prompt set).
fn wide_mean(
    m: &BTreeMap<&'static str, ProbeSamples>,
    f: fn(&BenchRun) -> Option<f64>,
) -> Vec<f64> {
    let cols: Vec<Vec<f64>> = m
        .iter()
        .filter(|(k, _)| k.starts_with(WIDE_PREFIX))
        .map(|(_, s)| s.runs.iter().filter_map(f).collect())
        .collect();
    if cols.is_empty() {
        return Vec::new();
    }
    let n = cols.iter().map(Vec::len).min().unwrap_or(0);
    (0..n)
        .map(|i| cols.iter().map(|c| c[i]).sum::<f64>() / cols.len() as f64)
        .collect()
}

fn input_fingerprint(tokens: &str) -> String {
    let tokens = tokens.strip_prefix("chat:").unwrap_or(tokens);
    let bytes = match tokens.strip_prefix('@') {
        Some(path) => fs::read(path).unwrap_or_default(),
        None => tokens.as_bytes().to_vec(),
    };
    format!("{:016x}", fnv1a64(&bytes))
}

fn run_probe(o: &Opts, gpu: usize, p: &Probe) -> ProbeSamples {
    let mut s = ProbeSamples {
        input_fnv: input_fingerprint(&p.tokens),
        ..Default::default()
    };
    for _ in 0..o.n {
        if p.id == "c64" || p.id == "c4_32k" {
            // `c64`: 64 sequences, one batched decode round per step. `c4_32k`: 4 sequences,
            // per-sequence DSpark drafts verified together. The total decode rate is the
            // sample (`gen_tok_s`). LEANSONIC_SEQS is set by `multiseq` itself.
            let gen = p.gen.to_string();
            let spec = p.spec.to_string();
            let mut args = vec![
                "multiseq",
                "--weights",
                &o.weights,
                "--tokens",
                &p.tokens,
                "--round",
                "64",
                "--gen",
                &gen,
                "--no-check",
            ];
            if p.id == "c64" {
                args.extend(["--seqs", "64"]);
            } else {
                args.extend(["--seqs", "4", "--spec", &spec, "--dspark"]);
            }
            let out = Command::new(&o.bin)
                .args(&args)
                .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
                .env("CUDA_VISIBLE_DEVICES", gpu.to_string())
                .env("LEANSONIC_CTX", p.ctx.unwrap_or(o.ctx).to_string())
                .output();
            match out {
                Ok(out) => {
                    let text = format!(
                        "{}{}",
                        String::from_utf8_lossy(&out.stdout),
                        String::from_utf8_lossy(&out.stderr)
                    );
                    let total = text
                        .lines()
                        .find(|l| l.starts_with("batched rounds:"))
                        .and_then(|l| num_before(l, "tok/s total"));
                    match (out.status.success(), total) {
                        (true, Some(t)) => s.runs.push(BenchRun {
                            gen_tok_s: Some(t),
                            ..Default::default()
                        }),
                        _ => s.errors.push(
                            text.lines()
                                .find(|l| l.contains("ERROR"))
                                .unwrap_or("multiseq failed")
                                .to_string(),
                        ),
                    }
                }
                Err(e) => s.errors.push(format!("spawn {}: {e}", o.bin.display())),
            }
            eprint!(".");
            continue;
        }
        let out = Command::new(&o.bin)
            .args([
                "bench",
                "--weights",
                &o.weights,
                "--tokens",
                &p.tokens,
                "--spec",
                &p.spec.to_string(),
                "--gen",
                &p.gen.to_string(),
                "--repeat",
                "2",
            ])
            .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            .env("CUDA_VISIBLE_DEVICES", gpu.to_string())
            .env("LEANSONIC_CTX", p.ctx.unwrap_or(o.ctx).to_string())
            .output();
        match out {
            Ok(out) => {
                let text = format!(
                    "{}{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                let r = parse_bench_output(&text);
                if !out.status.success() || r.prefill_tok_s.is_none() {
                    let err = text
                        .lines()
                        .find(|l| l.contains("ERROR"))
                        .unwrap_or("bench failed")
                        .to_string();
                    s.errors.push(err);
                } else {
                    s.runs.push(r);
                }
            }
            Err(e) => s.errors.push(format!("spawn {}: {e}", o.bin.display())),
        }
        eprint!(".");
    }
    s
}

/// 100-chunk teacher-forced KLD for the `kld_mean` cell. `eval-kld` is deterministic for a
/// fixed binary, weights and GPU, so one run is the sample.
#[derive(Default)]
struct KldRun {
    mean: Option<f64>,
    max: Option<f64>,
    top1: Option<f64>,
    ppl: Option<f64>,
    error: Option<String>,
}

fn json_num(line: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn parse_kld_output(text: &str) -> KldRun {
    match text.lines().rev().find(|l| l.contains("\"mean_kld\"")) {
        Some(line) => KldRun {
            mean: json_num(line, "mean_kld"),
            max: json_num(line, "max_kld"),
            top1: json_num(line, "top1_agreement"),
            ppl: json_num(line, "ppl"),
            error: None,
        },
        None => KldRun {
            error: Some(
                text.lines()
                    .find(|l| l.contains("ERROR"))
                    .unwrap_or("eval-kld produced no result")
                    .to_string(),
            ),
            ..Default::default()
        },
    }
}

fn run_kld(o: &Opts, gpu: usize, reference: &Path) -> KldRun {
    let out = Command::new(&o.bin)
        .args(["eval-kld", "--weights", &o.weights, "--ref"])
        .arg(reference)
        .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        .env("CUDA_VISIBLE_DEVICES", gpu.to_string())
        .output();
    match out {
        Ok(out) => parse_kld_output(&format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )),
        Err(e) => KldRun {
            error: Some(format!("spawn {}: {e}", o.bin.display())),
            ..Default::default()
        },
    }
}

fn run_canary(o: &Opts, gpu: usize) -> (Option<Json>, Option<String>) {
    let mut cmd = Command::new(&o.bin);
    cmd.args(["canary", "--weights", &o.weights]);
    if o.quick {
        cmd.arg("--short-only");
    } else if o.long {
        cmd.arg("--long");
    }
    cmd.env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        .env("CUDA_VISIBLE_DEVICES", gpu.to_string());
    match cmd.output() {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            let parsed = stdout
                .lines()
                .rev()
                .find(|l| l.contains("\"canary_pass\""))
                .and_then(|l| parse_json(l).ok());
            let pass = parsed
                .as_ref()
                .and_then(|j| j.get("canary_pass"))
                .is_some_and(|v| *v == Json::Bool(true));
            if out.status.success() && pass {
                (parsed, None)
            } else {
                let msg = stderr
                    .lines()
                    .find(|l| l.contains("ERROR") || l.contains("canary failed"))
                    .unwrap_or("canary check failed")
                    .to_string();
                (parsed, Some(msg))
            }
        }
        Err(e) => (None, Some(format!("spawn {}: {e}", o.bin.display()))),
    }
}

/// (cell id, label, higher_is_better, extractor over probes)
type Extract = fn(&BTreeMap<&'static str, ProbeSamples>) -> Vec<f64>;

fn col(
    m: &BTreeMap<&'static str, ProbeSamples>,
    id: &str,
    f: fn(&BenchRun) -> Option<f64>,
) -> Vec<f64> {
    m.get(id)
        .map(|s| s.runs.iter().filter_map(f).collect())
        .unwrap_or_default()
}

fn spec_avg(m: &BTreeMap<&'static str, ProbeSamples>) -> Vec<f64> {
    let a = col(m, "spec_conv", |r| r.gen_tok_s);
    let b = col(m, "spec_struct", |r| r.gen_tok_s);
    let c = col(m, "spec_prose", |r| r.gen_tok_s);
    let n = a.len().min(b.len()).min(c.len());
    (0..n).map(|i| (a[i] + b[i] + c[i]) / 3.0).collect()
}

const CELLS: &[(&str, &str, bool, Extract)] = &[
    ("prefill_2k", "Prefill 2K (tok/s)", true, |m| {
        col(m, "p2k", |r| r.prefill_tok_s)
    }),
    ("prefill_16k", "Prefill 16K (tok/s)", true, |m| {
        col(m, "p16k", |r| r.prefill_tok_s)
    }),
    (
        "decode_short",
        "Decode short, spec off (tok/s)",
        true,
        |m| col(m, "short", |r| r.gen_tok_s),
    ),
    ("decode_16k", "Decode @16K, spec off (tok/s)", true, |m| {
        col(m, "p16k", |r| r.gen_tok_s)
    }),
    (
        "spec_avg",
        "Speculative avg conv/struct/prose (tok/s)",
        true,
        spec_avg,
    ),
    (
        "vram_gib",
        "VRAM spec off (GiB, lower=better)",
        false,
        |m| col(m, "short", |r| r.vram_gib),
    ),
    ("prefill_128k", "Prefill 128K (tok/s, --long)", true, |m| {
        col(m, "p128k", |r| r.prefill_tok_s)
    }),
    (
        "long_128k_decode",
        "Decode @128K, spec off (tok/s, --long)",
        true,
        |m| col(m, "p128k", |r| r.gen_tok_s),
    ),
    (
        "spec_wide",
        "Speculative avg, --wide-prompts (tok/s)",
        true,
        |m| wide_mean(m, |r| r.gen_tok_s),
    ),
    (
        "spec_accept_wide",
        "DSpark acceptance, --wide-prompts (%)",
        true,
        |m| wide_mean(m, |r| r.spec_acc_pct),
    ),
    (
        "c64_total",
        "C=64 aggregate decode (tok/s, --c64)",
        true,
        |m| col(m, "c64", |r| r.gen_tok_s),
    ),
    (
        "c4_32k_code",
        "32K C=4 aggregate, DSpark (tok/s, --c4-32k)",
        true,
        |m| col(m, "c4_32k", |r| r.gen_tok_s),
    ),
    (
        "kld_mean",
        "100-chunk KLD mean (lower=better)",
        false,
        |_| Vec::new(),
    ),
];

fn cell_json(samples: &[f64], target: Option<f64>, hib: bool) -> (Json, Verdict) {
    let v = verdict(samples, target, hib);
    let ci = bootstrap_ci(samples);
    let med = median(samples);
    let ratio = match (med, target) {
        (Some(m), Some(t)) if t != 0.0 => Some(if hib { m / t } else { t / m }),
        _ => None,
    };
    (
        obj(vec![
            ("verdict", Json::Str(v.label().into())),
            ("median", num_or_null(med)),
            ("ci95_lo", num_or_null(ci.map(|c| c.0))),
            ("ci95_hi", num_or_null(ci.map(|c| c.1))),
            ("target", num_or_null(target)),
            ("ratio_to_target", num_or_null(ratio)),
            ("n", Json::Num(samples.len() as f64)),
            (
                "samples",
                Json::Arr(samples.iter().map(|x| Json::Num(*x)).collect()),
            ),
        ]),
        v,
    )
}

fn fmt_opt(v: Option<f64>) -> String {
    v.map(|x| {
        if x >= 100.0 {
            format!("{x:.0}")
        } else if x.abs() < 0.1 {
            // KLD-scale values (e.g. 0.003151) would print as `0.00` with two decimals.
            format!("{x:.5}")
        } else {
            format!("{x:.2}")
        }
    })
    .unwrap_or_else(|| "-".into())
}

/// Prints the cell table for one or more GPU result objects (UUIDs are never printed).
pub fn print_table(gpus: &[Json]) {
    let mut header = format!("{:<44}", "cell");
    for g in gpus {
        let name = g.get("name").and_then(Json::as_str).unwrap_or("?");
        let short = name
            .replace("NVIDIA ", "")
            .replace(" Blackwell", "")
            .replace("GeForce ", "");
        let _ = write!(header, " | {short:>24}");
    }
    println!("{header}");
    println!("{}", "-".repeat(header.chars().count()));
    for (id, label, _, _) in CELLS {
        let mut line = format!("{label:<44}");
        for g in gpus {
            let c = g.get("cells").and_then(|c| c.get(id));
            let med = c.and_then(|c| c.get("median")).and_then(Json::as_f64);
            let tgt = c.and_then(|c| c.get("target")).and_then(Json::as_f64);
            let v = c
                .and_then(|c| c.get("verdict"))
                .and_then(Json::as_str)
                .unwrap_or("UNMEASURED");
            let cell = match (med, tgt) {
                (Some(m), Some(t)) => format!("{v} {}/{}", fmt_opt(Some(m)), fmt_opt(Some(t))),
                (Some(m), None) => format!("{v} {}", fmt_opt(Some(m))),
                _ => v.to_string(),
            };
            let _ = write!(line, " | {cell:>24}");
        }
        println!("{line}");
    }
}

pub fn run(args: &[String]) -> Result<(), String> {
    if args.first().map(String::as_str) == Some("--merge") {
        let mut gpus = Vec::new();
        for path in &args[1..] {
            let text = fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
            let j = parse_json(&text)?;
            if let Some(Json::Arr(a)) = j.get("gpus") {
                gpus.extend(a.iter().cloned());
            }
        }
        print_table(&gpus);
        let fails = gpus
            .iter()
            .flat_map(|g| match g.get("cells") {
                Some(Json::Obj(m)) => m.values().cloned().collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .filter(|c| {
                c.get("verdict").and_then(Json::as_str) != Some("PASS")
                    && c.get("verdict").and_then(Json::as_str) != Some("RECORDED")
            })
            .count();
        println!("\nnon-PASS cells: {fails}");
        return Ok(());
    }

    let o = parse_opts(args)?;
    let targets_text = fs::read_to_string(&o.targets)
        .map_err(|e| format!("read targets {}: {e}", o.targets.display()))?;
    let targets = parse_json(&targets_text)?;
    let bin_sha = fs::read(&o.bin).map(|b| sha256_hex(&b)).unwrap_or_default();
    let inventory = query_gpus();
    let host = hostname();

    let mut gpu_results = Vec::new();
    for &idx in &o.gpus {
        let info = inventory
            .iter()
            .find(|g| g.index == idx)
            .cloned()
            .unwrap_or(GpuInfo {
                index: idx,
                name: format!("gpu{idx}"),
                ..Default::default()
            });
        let lock_path = format!("/tmp/leansonic-gpu-{host}-{idx}.lock");
        eprintln!(
            "[bench-matrix] GPU {idx} {} — waiting for {lock_path}",
            info.name
        );
        #[cfg(unix)]
        let _guard = lock::acquire(&lock_path)?;
        let foreign = foreign_processes(&info.pci_bus_id);
        if !foreign.is_empty() {
            eprintln!("[bench-matrix] GPU {idx} has other compute processes: {foreign:?}");
            if o.require_idle {
                return Err(format!("GPU {idx} is busy (--require-idle)"));
            }
        }
        let sku = sku_key(&info.name, &targets);
        let sku_t = sku
            .as_deref()
            .and_then(|k| targets.get("skus").and_then(|s| s.get(k)));

        let mut probe_samples: BTreeMap<&'static str, ProbeSamples> = BTreeMap::new();
        let mut inputs = BTreeMap::new();
        let mut errors = Vec::new();
        for p in probes(&o) {
            eprint!("[bench-matrix] GPU {idx} {} x{} ", p.id, o.n);
            let s = run_probe(&o, idx, &p);
            eprintln!(" ok={} err={}", s.runs.len(), s.errors.len());
            inputs.insert(
                p.id.to_string(),
                Json::Str(format!(
                    "{} fnv1a64={}",
                    p.tokens.rsplit('/').next().unwrap_or(""),
                    s.input_fnv
                )),
            );
            errors.extend(s.errors.iter().map(|e| Json::Str(format!("{}: {e}", p.id))));
            probe_samples.insert(p.id, s);
        }

        let kld = o.kld_ref.as_ref().map(|r| {
            eprint!("[bench-matrix] GPU {idx} eval-kld x1 ");
            let k = run_kld(&o, idx, r);
            eprintln!(
                " {}",
                k.mean
                    .map(|m| format!("mean={m:.6}"))
                    .unwrap_or_else(|| "failed".into())
            );
            if let Some(e) = &k.error {
                errors.push(Json::Str(format!("kld: {e}")));
            }
            k
        });

        let canary = if o.skip_canary {
            None
        } else {
            eprint!("[bench-matrix] GPU {idx} canary x1 ");
            let (cj, cerr) = run_canary(&o, idx);
            if let Some(ref e) = cerr {
                eprintln!(" FAIL ({e})");
                errors.push(Json::Str(format!("canary: {e}")));
            } else {
                eprintln!(" PASS");
            }
            Some((cj, cerr.is_some()))
        };
        let canary_failed = canary.as_ref().is_some_and(|(_, failed)| *failed);

        let mut cells = BTreeMap::new();
        for (id, _, hib, f) in CELLS {
            let mut samples = f(&probe_samples);
            if *id == "kld_mean" {
                if let Some(m) = kld.as_ref().and_then(|k| k.mean) {
                    samples = vec![m];
                }
            }
            let target = sku_t.and_then(|t| t.get(id)).and_then(Json::as_f64);
            let (mut j, _) = cell_json(&samples, target, *hib);
            if canary_failed {
                if let Json::Obj(ref mut m) = j {
                    if m.get("verdict") == Some(&Json::Str("PASS".into())) {
                        m.insert("verdict".into(), Json::Str("FAIL".into()));
                    }
                }
            }
            cells.insert(id.to_string(), j);
        }
        let resident = probe_samples
            .get("short")
            .and_then(|s| s.runs.first())
            .and_then(|r| r.resident.clone());
        let device = probe_samples
            .values()
            .find_map(|s| s.runs.first().and_then(|r| r.device.clone()));
        if let (Some(d), false) = (&device, info.name.is_empty()) {
            if !d.contains(info.name.trim_start_matches("NVIDIA ")) {
                errors.push(Json::Str(format!(
                    "device mismatch: nvidia-smi index {idx} is `{}` but engine reported `{d}`",
                    info.name
                )));
            }
        }
        gpu_results.push(obj(vec![
            ("index", Json::Num(idx as f64)),
            ("name", Json::Str(info.name.clone())),
            ("engine_device", device.map(Json::Str).unwrap_or(Json::Null)),
            (
                "uuid",
                Json::Str(if info.uuid.starts_with("GPU-") {
                    "GPU-REDACTED".into()
                } else {
                    info.uuid.clone()
                }),
            ),
            ("driver", Json::Str(info.driver.clone())),
            ("power_limit_w", Json::Str(info.power_limit_w.clone())),
            ("pcie", Json::Str(info.pcie.clone())),
            ("sku", sku.map(Json::Str).unwrap_or(Json::Null)),
            (
                "resident_expert_layers",
                resident.map(Json::Str).unwrap_or(Json::Null),
            ),
            (
                "foreign_processes",
                Json::Arr(foreign.into_iter().map(Json::Str).collect()),
            ),
            (
                "kld",
                kld.as_ref()
                    .map(|k| {
                        obj(vec![
                            ("mean", num_or_null(k.mean)),
                            ("max", num_or_null(k.max)),
                            ("top1_agreement", num_or_null(k.top1)),
                            ("ppl", num_or_null(k.ppl)),
                        ])
                    })
                    .unwrap_or(Json::Null),
            ),
            (
                "canary",
                canary.and_then(|(cj, _)| cj).unwrap_or(Json::Null),
            ),
            ("inputs", Json::Obj(inputs)),
            ("errors", Json::Arr(errors)),
            ("cells", Json::Obj(cells)),
        ]));
    }

    let report = obj(vec![
        ("schema", Json::Str("leansonic.bench-matrix.v1".into())),
        ("commit", Json::Str(o.commit.clone())),
        ("binary_sha256", Json::Str(bin_sha)),
        ("host", Json::Str(host)),
        ("weights", Json::Str(o.weights.clone())),
        ("ctx", Json::Num(o.ctx as f64)),
        ("n_per_cell", Json::Num(o.n as f64)),
        ("engine_env", Json::Obj(engine_env())),
        ("method", Json::Str("fresh process per sample; warm 2nd request per process; median + 95% percentile bootstrap (2000, fixed seed) of the median; PASS iff CI bound clears target".into())),
        ("gpus", Json::Arr(gpu_results.clone())),
    ]);
    let env_line: Vec<String> = engine_env()
        .into_iter()
        .map(|(k, v)| match v {
            Json::Str(v) => format!("{k}={v}"),
            _ => k,
        })
        .collect();
    println!(
        "engine env: {}",
        if env_line.is_empty() {
            "(defaults)".to_string()
        } else {
            env_line.join(" ")
        }
    );
    print_table(&gpu_results);
    if let Some(out) = &o.out {
        let mut s = String::new();
        write_json(&report, &mut s, 0);
        s.push('\n');
        if let Some(parent) = Path::new(out).parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(out, s).map_err(|e| format!("write {}: {e}", out.display()))?;
        eprintln!(
            "[bench-matrix] wrote {} (contains GPU UUIDs — do not publish verbatim)",
            out.display()
        );
    }
    Ok(())
}

/// `LEANSONIC_*` variables inherited by every sample process (KV format, split counts, ...),
/// (sorted by the map), so a report records the engine configuration it measured.
/// `LEANSONIC_CTX` is set per probe and reported as `ctx`.
fn engine_env() -> BTreeMap<String, Json> {
    env::vars()
        .filter(|(k, _)| k.starts_with("LEANSONIC_") && k != "LEANSONIC_CTX")
        .map(|(k, val)| (k, Json::Str(val)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip() {
        let text = r#"{"a": [1, 2.5, -3e2], "b": {"c": "x\"yé", "d": null, "e": true}}"#;
        let v = parse_json(text).unwrap();
        let mut s = String::new();
        write_json(&v, &mut s, 0);
        assert_eq!(parse_json(&s).unwrap(), v);
        assert_eq!(
            v.get("b").unwrap().get("c").unwrap().as_str(),
            Some("x\"yé")
        );
        assert!(parse_json("{\"a\": }").is_err());
    }

    #[test]
    fn median_and_bootstrap() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
        let v = [10.0, 10.1, 9.9, 10.05, 9.95];
        let (lo, hi) = bootstrap_ci(&v).unwrap();
        assert!(lo <= 10.0 && hi >= 10.0 && lo >= 9.9 && hi <= 10.1);
        assert_eq!(bootstrap_ci(&v), bootstrap_ci(&v), "deterministic");
    }

    #[test]
    fn wide_mean_averages_prompt_sets() {
        let run = |t: f64, a: f64| BenchRun {
            gen_tok_s: Some(t),
            spec_acc_pct: Some(a),
            ..Default::default()
        };
        let mut m: BTreeMap<&'static str, ProbeSamples> = BTreeMap::new();
        m.insert(
            "wide_a",
            ProbeSamples {
                runs: vec![run(100.0, 50.0), run(110.0, 52.0)],
                ..Default::default()
            },
        );
        m.insert(
            "wide_b",
            ProbeSamples {
                runs: vec![run(200.0, 60.0)],
                ..Default::default()
            },
        );
        m.insert(
            "spec_conv",
            ProbeSamples {
                runs: vec![run(999.0, 99.0)],
                ..Default::default()
            },
        );
        assert_eq!(wide_mean(&m, |r| r.gen_tok_s), vec![150.0]);
        assert_eq!(wide_mean(&m, |r| r.spec_acc_pct), vec![55.0]);
        assert!(wide_mean(&BTreeMap::new(), |r| r.gen_tok_s).is_empty());
        assert_eq!(
            input_fingerprint("chat:@/nonexistent"),
            input_fingerprint("@/nonexistent")
        );
    }

    #[test]
    fn kld_output_parse() {
        let text = "[LeanSonic GPU] x\n{\"chunks\":100,\"ppl\":7.065,\"mean_kld\":0.003151,\"max_kld\":0.608,\"top1_agreement\":0.977}\n";
        let k = parse_kld_output(text);
        assert_eq!(k.mean, Some(0.003151));
        assert_eq!(k.max, Some(0.608));
        assert_eq!(k.top1, Some(0.977));
        assert_eq!(k.ppl, Some(7.065));
        assert!(k.error.is_none());
        let bad = parse_kld_output("[ERROR] no ref\n");
        assert!(bad.mean.is_none());
        assert_eq!(bad.error.as_deref(), Some("[ERROR] no ref"));
    }

    #[test]
    fn verdict_rules() {
        assert_eq!(verdict(&[150.0; 5], Some(149.0), true), Verdict::Pass);
        assert_eq!(verdict(&[148.0; 5], Some(149.0), true), Verdict::Fail);
        assert_eq!(verdict(&[16.5; 5], Some(16.6), false), Verdict::Pass);
        assert_eq!(verdict(&[17.2; 5], Some(16.6), false), Verdict::Fail);
        assert_eq!(verdict(&[50.0], None, true), Verdict::Recorded);
        assert_eq!(verdict(&[], Some(1.0), true), Verdict::Unmeasured);
    }

    #[test]
    fn parses_bench_output() {
        let text = "[LeanSonic GPU] NVIDIA RTX PRO 5000 72GB Blackwell | VRAM used: 17.20 GiB / 71.12 GiB | Resident expert layers: 24/24 (Spilled: 0) | Spec: off | Loaded in 3.68s\n\
[run 0] prefill 32 tokens: 33.46 ms, 956.4 tok/s\n\
[run 0] generated 128 tokens: 859.57 ms (6.72 ms/tok, 148.9 tok/s)\n\
[run 1] prefill 38 tokens: 40.85 ms, 930.3 tok/s\n\
[run 1] generated 256 tokens: 1408.25 ms (5.50 ms/tok, 181.8 tok/s) [spec acc=141/243 (58.0%), 2.53 tok/step]\n";
        let r = parse_bench_output(text);
        assert_eq!(
            r.device.as_deref(),
            Some("NVIDIA RTX PRO 5000 72GB Blackwell")
        );
        assert_eq!(r.vram_gib, Some(17.2));
        assert_eq!(r.resident.as_deref(), Some("24/24"));
        assert_eq!(r.prefill_tok_s, Some(930.3));
        assert_eq!(r.gen_ms_tok, Some(5.5));
        assert_eq!(r.gen_tok_s, Some(181.8));
        assert_eq!(r.spec_tok_step, Some(2.53));
        assert_eq!(r.spec_acc_pct, Some(58.0));
    }

    #[test]
    fn targets_file_parses_and_matches_skus() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/benchmarks/targets.json"
        );
        let t = parse_json(&fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(
            sku_key("NVIDIA RTX PRO 5000 72GB Blackwell", &t).as_deref(),
            Some("rtx_pro_5000")
        );
        assert_eq!(
            sku_key("NVIDIA RTX PRO 4500 Blackwell", &t).as_deref(),
            Some("rtx_pro_4500")
        );
        assert_eq!(
            sku_key("NVIDIA GeForce RTX 5070 Ti", &t).as_deref(),
            Some("rtx_5070_ti")
        );
        assert_eq!(
            sku_key("NVIDIA GeForce RTX 5060 Ti", &t).as_deref(),
            Some("rtx_5060_ti")
        );
        for k in ["rtx_pro_5000", "rtx_pro_4500", "rtx_5070_ti", "rtx_5060_ti"] {
            let s = t.get("skus").unwrap().get(k).unwrap();
            assert!(s.get("prefill_2k").unwrap().as_f64().unwrap() > 0.0, "{k}");
        }
    }
}
