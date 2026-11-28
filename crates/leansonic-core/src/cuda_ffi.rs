//! Zero-Dependency Dynamic CUDA Driver (`libcuda.so.1`) & NVRTC (`libnvrtc.so`) FFI.
//!
//! Binds NVIDIA Driver API symbols at runtime via POSIX `dlopen`/`dlsym` so `LeanSonic`
//! compiles with zero third-party crates on macOS/CPU CI and drives `sm_120` Blackwell GPUs
//! natively on Linux (`cuda01` RTX PRO 4500 & `cuda02` RTX 5070 Ti), including CUDA Graphs
//! and Pinned Device-Mapped Host RAM (`cuMemHostAlloc`) for Spilled Mode.

#![allow(non_camel_case_types, non_snake_case)]

use core::ffi::{c_char, c_int, c_uint, c_void};
use std::ffi::CString;
use std::sync::Arc;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const CU_MEMHOSTALLOC_PORTABLE: c_uint = 0x01;
pub const CU_MEMHOSTALLOC_DEVICEMAP: c_uint = 0x02;
pub const CU_STREAM_NON_BLOCKING: c_uint = 0x01;
pub const CU_EVENT_DISABLE_TIMING: c_uint = 0x02;
pub const CU_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;
pub const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: c_int = 6;

#[repr(C)]
#[derive(Clone, Copy)]
pub union CUlaunchAttributeValue {
    pub programmatic_stream_serialization_allowed: c_int,
    pub raw: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUlaunchAttribute {
    pub id: c_int,
    pub pad: [u8; 4],
    pub value: CUlaunchAttributeValue,
}

impl CUlaunchAttribute {
    #[inline]
    pub fn programmatic_stream_serialization() -> Self {
        let mut val = CUlaunchAttributeValue { raw: [0; 8] };
        val.programmatic_stream_serialization_allowed = 1;
        Self {
            id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: [0; 4],
            value: val,
        }
    }
}

#[repr(C)]
pub struct CUlaunchConfig {
    pub gridDimX: c_uint,
    pub gridDimY: c_uint,
    pub gridDimZ: c_uint,
    pub blockDimX: c_uint,
    pub blockDimY: c_uint,
    pub blockDimZ: c_uint,
    pub sharedMemBytes: c_uint,
    pub _pad0: c_uint,
    pub hStream: CUstream,
    pub attrs: *mut CUlaunchAttribute,
    pub numAttrs: c_uint,
    pub _pad1: c_uint,
}

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;
#[cfg(target_os = "linux")]
const RTLD_GLOBAL: c_int = 0x100;
#[cfg(all(unix, not(target_os = "linux")))]
const RTLD_GLOBAL: c_int = 0x8;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CUuuid {
    pub bytes: [u8; 16],
}

fn format_cuda_uuid(u: &CUuuid) -> String {
    let b = &u.bytes;
    format!(
        "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
    )
}

#[cfg(unix)]
unsafe fn open_first_lib_with_path<'a>(
    candidates: &[&'a str],
) -> Result<(*mut c_void, &'a str), String> {
    for &path in candidates {
        let Ok(cstr) = CString::new(path) else {
            continue;
        };
        let h = unsafe { dlopen(cstr.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            return Ok((h, path));
        }
    }
    Err(format!("unable to dlopen any of {candidates:?}"))
}

#[cfg(not(unix))]
unsafe fn open_first_lib_with_path<'a>(
    _candidates: &[&'a str],
) -> Result<(*mut c_void, &'a str), String> {
    Err("dynamic CUDA FFI is only supported on Unix/Linux hosts".into())
}

unsafe fn open_first_lib(candidates: &[&str]) -> Result<*mut c_void, String> {
    unsafe { open_first_lib_with_path(candidates).map(|(h, _)| h) }
}

#[cfg(unix)]
unsafe fn load_sym<T: Copy>(handle: *mut c_void, name: &str) -> Result<T, String> {
    let cstr = CString::new(name).unwrap();
    let ptr = unsafe { dlsym(handle, cstr.as_ptr()) };
    if ptr.is_null() {
        return Err(format!("missing symbol `{name}`"));
    }
    Ok(unsafe { core::mem::transmute_copy(&ptr) })
}

#[cfg(not(unix))]
unsafe fn load_sym<T: Copy>(_handle: *mut c_void, name: &str) -> Result<T, String> {
    Err(format!("missing symbol `{name}` on non-Unix"))
}

pub struct CudaApi {
    pub cuInit: unsafe extern "C" fn(c_uint) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    pub cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    pub cuDeviceGetUuid: Option<unsafe extern "C" fn(*mut CUuuid, CUdevice) -> CUresult>,
    pub cuCtxCreate_v2: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
    pub cuCtxDestroy_v2: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuMemGetInfo_v2: unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult,
    pub cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    pub cuMemsetD8_v2: unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult,
    pub cuMemHostAlloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CUresult,
    pub cuMemFreeHost: unsafe extern "C" fn(*mut c_void) -> CUresult,
    pub cuMemHostGetDevicePointer_v2:
        unsafe extern "C" fn(*mut CUdeviceptr, *mut c_void, c_uint) -> CUresult,
    pub cuMemcpyHtoD_v2: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    pub cuMemcpyDtoH_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    pub cuMemcpyHtoDAsync_v2:
        unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    pub cuMemcpyDtoDAsync_v2:
        unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult,
    pub cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    pub cuModuleGetFunction:
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    pub cuFuncSetAttribute: unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult,
    pub cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
    pub cuLaunchKernelEx: Option<
        unsafe extern "C" fn(
            *const CUlaunchConfig,
            CUfunction,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> CUresult,
    >,
    pub cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    pub cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    pub cuEventDestroy_v2: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuStreamWaitEvent: unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUresult,
    pub cuStreamBeginCapture_v2: unsafe extern "C" fn(CUstream, c_int) -> CUresult,
    pub cuStreamEndCapture: unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult,
    pub cuGraphInstantiateWithFlags:
        unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult,
    pub cuGraphLaunch: unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult,
    pub cuGraphExecDestroy: unsafe extern "C" fn(CUgraphExec) -> CUresult,
    pub cuGraphDestroy: unsafe extern "C" fn(CUgraph) -> CUresult,
}

unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

pub fn check_cu(code: CUresult, op: &str) -> Result<(), String> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("CUDA error {code} in `{op}`"))
    }
}

pub struct CudaContextHandle {
    pub api: Arc<CudaApi>,
    pub device: CUdevice,
    pub ctx: CUcontext,
    pub device_name: String,
    pub device_uuid: String,
    pub compute_stream: CUstream,
    pub h2d_stream: CUstream,
}

unsafe impl Send for CudaContextHandle {}
unsafe impl Sync for CudaContextHandle {}

impl CudaContextHandle {
    pub fn init(device_ordinal: i32) -> Result<Arc<Self>, String> {
        let lib = unsafe {
            open_first_lib(&[
                "libcuda.so.1",
                "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                "/usr/lib/wsl/lib/libcuda.so.1",
            ])?
        };
        let api = unsafe {
            let get_uuid = load_sym(lib, "cuDeviceGetUuid_v2")
                .or_else(|_| load_sym(lib, "cuDeviceGetUuid"))
                .ok();
            Arc::new(CudaApi {
                cuInit: load_sym(lib, "cuInit")?,
                cuDeviceGet: load_sym(lib, "cuDeviceGet")?,
                cuDeviceGetName: load_sym(lib, "cuDeviceGetName")?,
                cuDeviceGetUuid: get_uuid,
                cuCtxCreate_v2: load_sym(lib, "cuCtxCreate_v2")?,
                cuCtxDestroy_v2: load_sym(lib, "cuCtxDestroy_v2")?,
                cuCtxSetCurrent: load_sym(lib, "cuCtxSetCurrent")?,
                cuMemGetInfo_v2: load_sym(lib, "cuMemGetInfo_v2")?,
                cuMemAlloc_v2: load_sym(lib, "cuMemAlloc_v2")?,
                cuMemFree_v2: load_sym(lib, "cuMemFree_v2")?,
                cuMemsetD8_v2: load_sym(lib, "cuMemsetD8_v2")?,
                cuMemHostAlloc: load_sym(lib, "cuMemHostAlloc")?,
                cuMemFreeHost: load_sym(lib, "cuMemFreeHost")?,
                cuMemHostGetDevicePointer_v2: load_sym(lib, "cuMemHostGetDevicePointer_v2")?,
                cuMemcpyHtoD_v2: load_sym(lib, "cuMemcpyHtoD_v2")?,
                cuMemcpyDtoH_v2: load_sym(lib, "cuMemcpyDtoH_v2")?,
                cuMemcpyHtoDAsync_v2: load_sym(lib, "cuMemcpyHtoDAsync_v2")?,
                cuMemcpyDtoDAsync_v2: load_sym(lib, "cuMemcpyDtoDAsync_v2")?,
                cuModuleLoadData: load_sym(lib, "cuModuleLoadData")?,
                cuModuleGetFunction: load_sym(lib, "cuModuleGetFunction")?,
                cuFuncSetAttribute: load_sym(lib, "cuFuncSetAttribute")?,
                cuLaunchKernel: load_sym(lib, "cuLaunchKernel")?,
                cuLaunchKernelEx: load_sym(lib, "cuLaunchKernelEx").ok(),
                cuStreamCreate: load_sym(lib, "cuStreamCreate")?,
                cuStreamSynchronize: load_sym(lib, "cuStreamSynchronize")?,
                cuEventCreate: load_sym(lib, "cuEventCreate")?,
                cuEventRecord: load_sym(lib, "cuEventRecord")?,
                cuEventDestroy_v2: load_sym(lib, "cuEventDestroy_v2")?,
                cuStreamWaitEvent: load_sym(lib, "cuStreamWaitEvent")?,
                cuStreamBeginCapture_v2: load_sym(lib, "cuStreamBeginCapture_v2")?,
                cuStreamEndCapture: load_sym(lib, "cuStreamEndCapture")?,
                cuGraphInstantiateWithFlags: load_sym(lib, "cuGraphInstantiateWithFlags")?,
                cuGraphLaunch: load_sym(lib, "cuGraphLaunch")?,
                cuGraphExecDestroy: load_sym(lib, "cuGraphExecDestroy")?,
                cuGraphDestroy: load_sym(lib, "cuGraphDestroy")?,
            })
        };

        unsafe {
            check_cu((api.cuInit)(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            check_cu((api.cuDeviceGet)(&mut dev, device_ordinal), "cuDeviceGet")?;
            let mut name_buf = [0u8; 128];
            check_cu(
                (api.cuDeviceGetName)(name_buf.as_mut_ptr().cast(), 128, dev),
                "cuDeviceGetName",
            )?;
            let nul = name_buf.iter().position(|&b| b == 0).unwrap_or(128);
            let device_name = String::from_utf8_lossy(&name_buf[..nul]).into_owned();
            let device_uuid = if let Some(uuid_fn) = api.cuDeviceGetUuid {
                let mut raw_uuid = CUuuid::default();
                if uuid_fn(&mut raw_uuid, dev) == CUDA_SUCCESS {
                    format_cuda_uuid(&raw_uuid)
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            let mut ctx: CUcontext = core::ptr::null_mut();
            check_cu((api.cuCtxCreate_v2)(&mut ctx, 0, dev), "cuCtxCreate_v2")?;
            let mut handle = Self {
                api,
                device: dev,
                ctx,
                device_name,
                device_uuid,
                compute_stream: core::ptr::null_mut(),
                h2d_stream: core::ptr::null_mut(),
            };
            check_cu(
                (handle.api.cuStreamCreate)(&mut handle.compute_stream, CU_STREAM_NON_BLOCKING),
                "cuStreamCreate(compute)",
            )?;
            check_cu(
                (handle.api.cuStreamCreate)(&mut handle.h2d_stream, CU_STREAM_NON_BLOCKING),
                "cuStreamCreate(h2d)",
            )?;
            Ok(Arc::new(handle))
        }
    }

    pub fn bind_current(&self) -> Result<(), String> {
        unsafe { check_cu((self.api.cuCtxSetCurrent)(self.ctx), "cuCtxSetCurrent") }
    }

    pub fn mem_info(&self) -> Result<(usize, usize), String> {
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe {
            check_cu(
                (self.api.cuMemGetInfo_v2)(&mut free, &mut total),
                "cuMemGetInfo_v2",
            )?;
        }
        Ok((free, total))
    }

    pub fn sync_compute(&self) -> Result<(), String> {
        unsafe {
            check_cu(
                (self.api.cuStreamSynchronize)(self.compute_stream),
                "cuStreamSynchronize(compute)",
            )
        }
    }

    pub(crate) fn sync_stream(&self, stream: CUstream) -> Result<(), String> {
        unsafe {
            check_cu(
                (self.api.cuStreamSynchronize)(stream),
                "cuStreamSynchronize",
            )
        }
    }

    pub fn sync_h2d(&self) -> Result<(), String> {
        unsafe {
            check_cu(
                (self.api.cuStreamSynchronize)(self.h2d_stream),
                "cuStreamSynchronize(h2d)",
            )
        }
    }
}

impl Drop for CudaContextHandle {
    fn drop(&mut self) {
        // The driver also releases the streams, modules and events owned by this context.
        unsafe { (self.api.cuCtxDestroy_v2)(self.ctx) };
    }
}

pub struct DeviceBuf {
    ctx: Arc<CudaContextHandle>,
    pub ptr: CUdeviceptr,
    pub byte_len: usize,
}

impl DeviceBuf {
    /// Allocate device memory without zero-filling (`cuMemAlloc_v2` only).
    /// Callers MUST overwrite the entire `[0..byte_len)` range before reading it on GPU.
    pub fn alloc_uninit(ctx: &Arc<CudaContextHandle>, byte_len: usize) -> Result<Self, String> {
        let alloc_len = byte_len.max(16);
        let mut ptr: CUdeviceptr = 0;
        unsafe {
            check_cu(
                (ctx.api.cuMemAlloc_v2)(&mut ptr, alloc_len),
                "cuMemAlloc_v2",
            )?;
        }
        Ok(Self {
            ctx: Arc::clone(ctx),
            ptr,
            byte_len,
        })
    }

    pub fn alloc_zeroed(ctx: &Arc<CudaContextHandle>, byte_len: usize) -> Result<Self, String> {
        let alloc_len = byte_len.max(16);
        let buf = Self::alloc_uninit(ctx, byte_len)?;
        // `cuMemsetD8` on device memory is asynchronous and runs on the legacy stream, which the
        // engine's non-blocking streams do not wait for: a kernel launched right after the
        // allocation could otherwise run before the clear and have its output zeroed.
        unsafe {
            check_cu(
                (ctx.api.cuMemsetD8_v2)(buf.ptr, 0, alloc_len),
                "cuMemsetD8_v2",
            )?;
            check_cu(
                (ctx.api.cuStreamSynchronize)(core::ptr::null_mut()),
                "cuStreamSynchronize(legacy)",
            )?;
        }
        Ok(buf)
    }

    pub fn from_host_bytes(ctx: &Arc<CudaContextHandle>, src: &[u8]) -> Result<Self, String> {
        if src.is_empty() {
            return Self::alloc_zeroed(ctx, 0);
        }
        let buf = Self::alloc_uninit(ctx, src.len())?;
        buf.copy_from_host_bytes(src)?;
        Ok(buf)
    }

    pub fn copy_from_host_bytes(&self, src: &[u8]) -> Result<(), String> {
        self.copy_from_host_at(0, src)
    }

    pub fn copy_from_host_at(&self, offset: usize, src: &[u8]) -> Result<(), String> {
        assert!(offset.saturating_add(src.len()) <= self.byte_len);
        if src.is_empty() {
            return Ok(());
        }
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoD_v2)(
                    self.ptr + (offset as u64),
                    src.as_ptr().cast(),
                    src.len(),
                ),
                "cuMemcpyHtoD_v2",
            )
        }
    }

    pub fn copy_to_host_f32(&self, dst: &mut [f32]) -> Result<(), String> {
        let bytes = core::mem::size_of_val(dst);
        assert!(bytes <= self.byte_len);
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyDtoH_v2)(dst.as_mut_ptr().cast(), self.ptr, bytes),
                "cuMemcpyDtoH_v2",
            )
        }
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                let _ = (self.ctx.api.cuMemFree_v2)(self.ptr);
            }
        }
    }
}

/// Pinned Host DDR buffer mapped into GPU address space (`CU_MEMHOSTALLOC_DEVICEMAP`).
pub struct PinnedHostBuf {
    ctx: Arc<CudaContextHandle>,
    pub host_ptr: *mut u8,
    pub dev_ptr: CUdeviceptr,
    pub byte_len: usize,
}

unsafe impl Send for PinnedHostBuf {}
unsafe impl Sync for PinnedHostBuf {}

impl PinnedHostBuf {
    pub fn alloc_uninit(ctx: &Arc<CudaContextHandle>, byte_len: usize) -> Result<Self, String> {
        let alloc_len = byte_len.max(16);
        let mut raw: *mut c_void = core::ptr::null_mut();
        let mut dptr: CUdeviceptr = 0;
        unsafe {
            check_cu(
                (ctx.api.cuMemHostAlloc)(
                    &mut raw,
                    alloc_len,
                    CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP,
                ),
                "cuMemHostAlloc",
            )?;
        }
        let mut buf = Self {
            ctx: Arc::clone(ctx),
            host_ptr: raw.cast::<u8>(),
            dev_ptr: 0,
            byte_len,
        };
        unsafe {
            check_cu(
                (ctx.api.cuMemHostGetDevicePointer_v2)(&mut dptr, raw, 0),
                "cuMemHostGetDevicePointer_v2",
            )?;
        }
        buf.dev_ptr = dptr;
        Ok(buf)
    }

    pub fn alloc_zeroed(ctx: &Arc<CudaContextHandle>, byte_len: usize) -> Result<Self, String> {
        let buf = Self::alloc_uninit(ctx, byte_len)?;
        unsafe {
            core::ptr::write_bytes(buf.host_ptr, 0, byte_len.max(16));
        }
        Ok(buf)
    }

    pub fn from_bytes(ctx: &Arc<CudaContextHandle>, src: &[u8]) -> Result<Self, String> {
        let buf = Self::alloc_uninit(ctx, src.len())?;
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), buf.host_ptr, src.len());
        }
        Ok(buf)
    }
}

impl Drop for PinnedHostBuf {
    fn drop(&mut self) {
        if !self.host_ptr.is_null() {
            unsafe {
                let _ = (self.ctx.api.cuMemFreeHost)(self.host_ptr.cast());
            }
        }
    }
}

fn collect_python_nvidia_lib_dirs() -> Vec<std::path::PathBuf> {
    let mut dirs = Vec::new();
    for env_key in ["VIRTUAL_ENV", "CONDA_PREFIX"] {
        if let Ok(prefix) = std::env::var(env_key) {
            let lib_root = std::path::Path::new(&prefix).join("lib");
            if let Ok(entries) = std::fs::read_dir(&lib_root) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    if name.to_string_lossy().starts_with("python") {
                        let sp = entry.path().join("site-packages").join("nvidia");
                        for sub in ["cu13", "cuda_nvrtc"] {
                            let d = sp.join(sub).join("lib");
                            if d.is_dir() {
                                dirs.push(d);
                            }
                        }
                    }
                }
            }
        }
    }
    dirs
}

#[cfg(unix)]
unsafe fn preload_nvrtc_builtins_from_dir(dir: &std::path::Path) {
    // Preload `libnvrtc-builtins.so*` with `RTLD_NOW | RTLD_GLOBAL` so `libnvrtc.so.13`'s
    // internal `dlopen("libnvrtc-builtins.so.13.0", ...)` resolves from the process link map
    // even on hosts without a system CUDA Toolkit or `LD_LIBRARY_PATH` (Issue #15).
    for exact in [
        "libnvrtc-builtins.so.13.0",
        "libnvrtc-builtins.so.13",
        "libnvrtc-builtins.so",
    ] {
        let p = dir.join(exact);
        if p.exists() {
            if let Ok(cstr) = CString::new(p.to_string_lossy().as_bytes()) {
                let _ = unsafe { dlopen(cstr.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name();
            if fname.to_string_lossy().starts_with("libnvrtc-builtins.so") {
                if let Ok(cstr) = CString::new(entry.path().to_string_lossy().as_bytes()) {
                    let _ = unsafe { dlopen(cstr.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
                }
            }
        }
    }
}

#[cfg(not(unix))]
unsafe fn preload_nvrtc_builtins_from_dir(_dir: &std::path::Path) {}

fn fnv1a64_pair(a: &[u8], b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in a.iter().chain(b.iter()) {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Compile CUDA C++ source via `libnvrtc.so` at runtime, preferring native `sm_120a` CUBIN
/// (`nvrtcGetCUBIN` + `/tmp` content-addressed cache) when `arch` targets SM120, with seamless
/// fallback to `compute_120` PTX if `nvrtcGetCUBIN` is unavailable.
pub fn compile_cuda_to_ptx_nvrtc(cuda_src: &str, arch: &str) -> Result<Vec<u8>, String> {
    let want_sm120_cubin = (arch == "compute_120" || arch == "sm_120a")
        && std::env::var("LEANSONIC_FORCE_PTX_JIT")
            .map(|v| v != "1")
            .unwrap_or(true);
    let cubin_hash = fnv1a64_pair(cuda_src.as_bytes(), b"::sm_120a::v2");
    let cache_path = std::env::temp_dir().join(format!("leansonic_sm120a_{cubin_hash:016x}.cubin"));
    if want_sm120_cubin {
        if let Ok(bytes) = std::fs::read(&cache_path) {
            // ELF magic `\x7fELF` sanity check
            if bytes.len() > 1024 && bytes.starts_with(b"\x7fELF") {
                return Ok(bytes);
            }
        }
    }

    type NvrtcCreateProgram = unsafe extern "C" fn(
        *mut *mut c_void,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int;
    type NvrtcCompileProgram =
        unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int;
    type NvrtcGetSize = unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int;
    type NvrtcGetStr = unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int;
    type NvrtcDestroy = unsafe extern "C" fn(*mut *mut c_void) -> c_int;

    const HINT: &str = "Hint: set LEANSONIC_CUBIN_PATH=/path/to/sm120a.cubin to skip runtime JIT, \
        or set LEANSONIC_NVRTC_PATH=/path/to/libnvrtc.so.13 (with libnvrtc-builtins.so.13.0 alongside it).";

    let py_dirs = collect_python_nvidia_lib_dirs();
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("LEANSONIC_NVRTC_PATH") {
        candidates.push(p);
    }
    candidates.extend([
        "libnvrtc.so.13".to_string(),
        "libnvrtc.so".to_string(),
        "/usr/local/cuda/lib64/libnvrtc.so.13".to_string(),
        "/usr/local/cuda-13.0/lib64/libnvrtc.so.13".to_string(),
    ]);
    for d in &py_dirs {
        candidates.push(d.join("libnvrtc.so.13").to_string_lossy().into_owned());
        candidates.push(d.join("libnvrtc.so").to_string_lossy().into_owned());
    }
    candidates.push("/models/mtp-off-three-engines-5070ti-20261105/libs13/libnvrtc.so.13".into());

    let refs: Vec<&str> = candidates.iter().map(String::as_str).collect();
    let (lib, matched_path) =
        unsafe { open_first_lib_with_path(&refs).map_err(|e| format!("{e}\n{HINT}"))? };

    unsafe {
        if let Some(parent) = std::path::Path::new(matched_path).parent() {
            if !parent.as_os_str().is_empty() {
                preload_nvrtc_builtins_from_dir(parent);
            }
        }
        for d in &py_dirs {
            preload_nvrtc_builtins_from_dir(d);
        }

        let create_prog: NvrtcCreateProgram = load_sym(lib, "nvrtcCreateProgram")?;
        let compile_prog: NvrtcCompileProgram = load_sym(lib, "nvrtcCompileProgram")?;
        let get_ptx_size: NvrtcGetSize = load_sym(lib, "nvrtcGetPTXSize")?;
        let get_ptx: NvrtcGetStr = load_sym(lib, "nvrtcGetPTX")?;
        let get_cubin_size: Option<NvrtcGetSize> = load_sym(lib, "nvrtcGetCUBINSize").ok();
        let get_cubin: Option<NvrtcGetStr> = load_sym(lib, "nvrtcGetCUBIN").ok();
        let get_log_size: NvrtcGetSize = load_sym(lib, "nvrtcGetProgramLogSize")?;
        let get_log: NvrtcGetStr = load_sym(lib, "nvrtcGetProgramLog")?;
        let destroy_prog: NvrtcDestroy = load_sym(lib, "nvrtcDestroyProgram")?;

        let c_src = CString::new(cuda_src).map_err(|e| e.to_string())?;
        let c_name = CString::new("inkling2_sm120.cu").unwrap();

        // Try native `sm_120a` CUBIN compilation first when supported by `libnvrtc`.
        if want_sm120_cubin {
            if let (Some(fn_csz), Some(fn_cb)) = (get_cubin_size, get_cubin) {
                let mut prog: *mut c_void = core::ptr::null_mut();
                if create_prog(
                    &mut prog,
                    c_src.as_ptr(),
                    c_name.as_ptr(),
                    0,
                    core::ptr::null(),
                    core::ptr::null(),
                ) == 0
                {
                    let opt_arch = CString::new("--gpu-architecture=sm_120a").unwrap();
                    let opt_std = CString::new("--std=c++17").unwrap();
                    let opt_dev = CString::new("-default-device").unwrap();
                    let opt_fm = CString::new("--use_fast_math").unwrap();
                    let opts = [
                        opt_arch.as_ptr(),
                        opt_std.as_ptr(),
                        opt_dev.as_ptr(),
                        opt_fm.as_ptr(),
                    ];
                    if compile_prog(prog, opts.len() as c_int, opts.as_ptr()) == 0 {
                        let mut csz = 0usize;
                        if fn_csz(prog, &mut csz) == 0 && csz > 1024 {
                            let mut cbuf = vec![0u8; csz];
                            if fn_cb(prog, cbuf.as_mut_ptr().cast()) == 0
                                && cbuf.starts_with(b"\x7fELF")
                            {
                                let _ = destroy_prog(&mut prog);
                                let tmp_path = std::env::temp_dir().join(format!(
                                    "leansonic_sm120a_{cubin_hash:016x}.{}.tmp",
                                    std::process::id()
                                ));
                                if std::fs::write(&tmp_path, &cbuf).is_ok() {
                                    let _ = std::fs::rename(&tmp_path, &cache_path);
                                }
                                return Ok(cbuf);
                            }
                        }
                    }
                    let _ = destroy_prog(&mut prog);
                }
            }
        }

        let mut prog: *mut c_void = core::ptr::null_mut();
        let rc = create_prog(
            &mut prog,
            c_src.as_ptr(),
            c_name.as_ptr(),
            0,
            core::ptr::null(),
            core::ptr::null(),
        );
        if rc != 0 {
            return Err(format!("nvrtcCreateProgram failed ({rc})\n{HINT}"));
        }

        let fallback_arch = if arch == "sm_120a" {
            "compute_120"
        } else {
            arch
        };
        let opt_arch = CString::new(format!("--gpu-architecture={fallback_arch}")).unwrap();
        let opt_std = CString::new("--std=c++17").unwrap();
        let opt_dev = CString::new("-default-device").unwrap();
        let opt_fm = CString::new("--use_fast_math").unwrap();
        let opts = [
            opt_arch.as_ptr(),
            opt_std.as_ptr(),
            opt_dev.as_ptr(),
            opt_fm.as_ptr(),
        ];
        let comp_rc = compile_prog(prog, opts.len() as c_int, opts.as_ptr());
        if comp_rc != 0 {
            let mut log_sz = 0usize;
            let _ = get_log_size(prog, &mut log_sz);
            let mut log_buf = vec![0u8; log_sz.max(1)];
            let _ = get_log(prog, log_buf.as_mut_ptr().cast());
            let _ = destroy_prog(&mut prog);
            return Err(format!(
                "NVRTC compile failed ({comp_rc}):\n{}\n{HINT}",
                String::from_utf8_lossy(&log_buf).trim_end_matches('\0')
            ));
        }

        let mut ptx_sz = 0usize;
        let _ = get_ptx_size(prog, &mut ptx_sz);
        let mut ptx_buf = vec![0u8; ptx_sz];
        let _ = get_ptx(prog, ptx_buf.as_mut_ptr().cast());
        let _ = destroy_prog(&mut prog);
        Ok(ptx_buf)
    }
}

pub struct CapturedGraphExec {
    ctx: Arc<CudaContextHandle>,
    graph: CUgraph,
    exec: CUgraphExec,
}

unsafe impl Send for CapturedGraphExec {}
unsafe impl Sync for CapturedGraphExec {}

impl CapturedGraphExec {
    pub fn capture<F>(ctx: &Arc<CudaContextHandle>, record_fn: F) -> Result<Self, String>
    where
        F: FnOnce(CUstream) -> Result<(), String>,
    {
        unsafe {
            check_cu(
                (ctx.api.cuStreamBeginCapture_v2)(
                    ctx.compute_stream,
                    CU_STREAM_CAPTURE_MODE_GLOBAL,
                ),
                "cuStreamBeginCapture_v2",
            )?;
        }
        let recorded = record_fn(ctx.compute_stream);
        let mut graph: CUgraph = core::ptr::null_mut();
        let ended = unsafe {
            check_cu(
                (ctx.api.cuStreamEndCapture)(ctx.compute_stream, &mut graph),
                "cuStreamEndCapture",
            )
        };
        if let Err(error) = ended {
            recorded?;
            return Err(error);
        }
        let mut captured = Self {
            ctx: Arc::clone(ctx),
            graph,
            exec: core::ptr::null_mut(),
        };
        recorded?;
        let mut exec: CUgraphExec = core::ptr::null_mut();
        unsafe {
            check_cu(
                (ctx.api.cuGraphInstantiateWithFlags)(&mut exec, graph, 0),
                "cuGraphInstantiateWithFlags",
            )?;
        }
        captured.exec = exec;
        Ok(captured)
    }

    pub fn replay(&self) -> Result<(), String> {
        unsafe {
            check_cu(
                (self.ctx.api.cuGraphLaunch)(self.exec, self.ctx.compute_stream),
                "cuGraphLaunch",
            )
        }
    }
}

impl Drop for CapturedGraphExec {
    fn drop(&mut self) {
        unsafe {
            if !self.exec.is_null() {
                let _ = (self.ctx.api.cuGraphExecDestroy)(self.exec);
            }
            if !self.graph.is_null() {
                let _ = (self.ctx.api.cuGraphDestroy)(self.graph);
            }
        }
    }
}

#[cfg(all(test, unix))]
mod context_tests {
    use super::*;

    #[test]
    #[ignore = "requires a CUDA device"]
    fn last_buffer_releases_its_context_and_restores_the_previous_one() -> Result<(), String> {
        let outer = CudaContextHandle::init(0)?;
        let get_current: unsafe extern "C" fn(*mut CUcontext) -> CUresult =
            unsafe { load_sym(open_first_lib(&["libcuda.so.1"])?, "cuCtxGetCurrent")? };
        let current = || -> Result<CUcontext, String> {
            let mut ctx = core::ptr::null_mut();
            unsafe { check_cu(get_current(&mut ctx), "cuCtxGetCurrent")? };
            Ok(ctx)
        };

        let inner = CudaContextHandle::init(0)?;
        let inner_ctx = inner.ctx;
        let buffer = DeviceBuf::alloc_zeroed(&inner, 16)?;
        drop(inner);
        assert_eq!(current()?, inner_ctx);
        drop(buffer);
        assert_eq!(current()?, outer.ctx);
        Ok(())
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn pdl_launch_abi_matches_cuda_header_layout() {
        assert_eq!(core::mem::size_of::<CUlaunchAttribute>(), 72);
        assert_eq!(core::mem::align_of::<CUlaunchAttribute>(), 8);
        assert_eq!(core::mem::offset_of!(CUlaunchAttribute, id), 0);
        assert_eq!(core::mem::offset_of!(CUlaunchAttribute, value), 8);
        assert_eq!(core::mem::size_of::<CUlaunchConfig>(), 56);
        assert_eq!(core::mem::align_of::<CUlaunchConfig>(), 8);
        assert_eq!(core::mem::offset_of!(CUlaunchConfig, gridDimX), 0);
        assert_eq!(core::mem::offset_of!(CUlaunchConfig, blockDimX), 12);
        assert_eq!(core::mem::offset_of!(CUlaunchConfig, sharedMemBytes), 24);
        assert_eq!(core::mem::offset_of!(CUlaunchConfig, hStream), 32);
        assert_eq!(core::mem::offset_of!(CUlaunchConfig, attrs), 40);
        assert_eq!(core::mem::offset_of!(CUlaunchConfig, numAttrs), 48);
    }
}
