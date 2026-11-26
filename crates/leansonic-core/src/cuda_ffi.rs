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

#[cfg(unix)]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: c_int = 2;

#[cfg(unix)]
unsafe fn open_first_lib(candidates: &[&str]) -> Result<*mut c_void, String> {
    for &path in candidates {
        let cstr = CString::new(path).unwrap();
        let h = unsafe { dlopen(cstr.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            return Ok(h);
        }
    }
    Err(format!("unable to dlopen any of {candidates:?}"))
}

#[cfg(not(unix))]
unsafe fn open_first_lib(_candidates: &[&str]) -> Result<*mut c_void, String> {
    Err("dynamic CUDA FFI is only supported on Unix/Linux hosts".into())
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
    pub cuCtxCreate_v2: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
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
    pub cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    pub cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
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
            Arc::new(CudaApi {
                cuInit: load_sym(lib, "cuInit")?,
                cuDeviceGet: load_sym(lib, "cuDeviceGet")?,
                cuDeviceGetName: load_sym(lib, "cuDeviceGetName")?,
                cuCtxCreate_v2: load_sym(lib, "cuCtxCreate_v2")?,
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
                cuLaunchKernel: load_sym(lib, "cuLaunchKernel")?,
                cuStreamCreate: load_sym(lib, "cuStreamCreate")?,
                cuStreamSynchronize: load_sym(lib, "cuStreamSynchronize")?,
                cuEventCreate: load_sym(lib, "cuEventCreate")?,
                cuEventRecord: load_sym(lib, "cuEventRecord")?,
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

            let mut ctx: CUcontext = core::ptr::null_mut();
            check_cu((api.cuCtxCreate_v2)(&mut ctx, 0, dev), "cuCtxCreate_v2")?;
            let mut compute_stream: CUstream = core::ptr::null_mut();
            let mut h2d_stream: CUstream = core::ptr::null_mut();
            check_cu(
                (api.cuStreamCreate)(&mut compute_stream, CU_STREAM_NON_BLOCKING),
                "cuStreamCreate(compute)",
            )?;
            check_cu(
                (api.cuStreamCreate)(&mut h2d_stream, CU_STREAM_NON_BLOCKING),
                "cuStreamCreate(h2d)",
            )?;

            Ok(Arc::new(Self {
                api,
                device: dev,
                ctx,
                device_name,
                compute_stream,
                h2d_stream,
            }))
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

    pub fn sync_h2d(&self) -> Result<(), String> {
        unsafe {
            check_cu(
                (self.api.cuStreamSynchronize)(self.h2d_stream),
                "cuStreamSynchronize(h2d)",
            )
        }
    }
}

pub struct DeviceBuf {
    ctx: Arc<CudaContextHandle>,
    pub ptr: CUdeviceptr,
    pub byte_len: usize,
}

impl DeviceBuf {
    pub fn alloc_zeroed(ctx: &Arc<CudaContextHandle>, byte_len: usize) -> Result<Self, String> {
        let alloc_len = byte_len.max(16);
        let mut ptr: CUdeviceptr = 0;
        unsafe {
            check_cu(
                (ctx.api.cuMemAlloc_v2)(&mut ptr, alloc_len),
                "cuMemAlloc_v2",
            )?;
            check_cu((ctx.api.cuMemsetD8_v2)(ptr, 0, alloc_len), "cuMemsetD8_v2")?;
        }
        Ok(Self {
            ctx: Arc::clone(ctx),
            ptr,
            byte_len,
        })
    }

    pub fn from_host_bytes(ctx: &Arc<CudaContextHandle>, src: &[u8]) -> Result<Self, String> {
        let buf = Self::alloc_zeroed(ctx, src.len())?;
        if !src.is_empty() {
            buf.copy_from_host_bytes(src)?;
        }
        Ok(buf)
    }

    pub fn copy_from_host_bytes(&self, src: &[u8]) -> Result<(), String> {
        assert!(src.len() <= self.byte_len);
        if src.is_empty() {
            return Ok(());
        }
        unsafe {
            check_cu(
                (self.ctx.api.cuMemcpyHtoD_v2)(self.ptr, src.as_ptr().cast(), src.len()),
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
    pub fn from_bytes(ctx: &Arc<CudaContextHandle>, src: &[u8]) -> Result<Self, String> {
        let alloc_len = src.len().max(16);
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
            if !src.is_empty() {
                core::ptr::copy_nonoverlapping(src.as_ptr(), raw.cast::<u8>(), src.len());
            }
            check_cu(
                (ctx.api.cuMemHostGetDevicePointer_v2)(&mut dptr, raw, 0),
                "cuMemHostGetDevicePointer_v2",
            )?;
        }
        Ok(Self {
            ctx: Arc::clone(ctx),
            host_ptr: raw.cast::<u8>(),
            dev_ptr: dptr,
            byte_len: src.len(),
        })
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

/// Compile CUDA C++ source to `sm_120` PTX via `libnvrtc.so` at runtime!
pub fn compile_cuda_to_ptx_nvrtc(cuda_src: &str, arch: &str) -> Result<Vec<u8>, String> {
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

    let mut candidates: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("LEANSONIC_NVRTC_PATH") {
        candidates.push(p);
    }
    candidates.extend([
        "libnvrtc.so.13".to_string(),
        "libnvrtc.so".to_string(),
        "/usr/local/cuda/lib64/libnvrtc.so.13".to_string(),
        "/models/mtp-off-three-engines-5070ti-20261105/libs13/libnvrtc.so.13".to_string(),
    ]);
    let refs: Vec<&str> = candidates.iter().map(String::as_str).collect();
    let lib = unsafe { open_first_lib(&refs)? };

    unsafe {
        let create_prog: NvrtcCreateProgram = load_sym(lib, "nvrtcCreateProgram")?;
        let compile_prog: NvrtcCompileProgram = load_sym(lib, "nvrtcCompileProgram")?;
        let get_ptx_size: NvrtcGetSize = load_sym(lib, "nvrtcGetPTXSize")?;
        let get_ptx: NvrtcGetStr = load_sym(lib, "nvrtcGetPTX")?;
        let get_log_size: NvrtcGetSize = load_sym(lib, "nvrtcGetProgramLogSize")?;
        let get_log: NvrtcGetStr = load_sym(lib, "nvrtcGetProgramLog")?;
        let destroy_prog: NvrtcDestroy = load_sym(lib, "nvrtcDestroyProgram")?;

        let c_src = CString::new(cuda_src).map_err(|e| e.to_string())?;
        let c_name = CString::new("inkling2_sm120.cu").unwrap();
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
            return Err(format!("nvrtcCreateProgram failed ({rc})"));
        }

        let opt_arch = CString::new(format!("--gpu-architecture={arch}")).unwrap();
        let opt_std = CString::new("--std=c++17").unwrap();
        let opts = [opt_arch.as_ptr(), opt_std.as_ptr()];
        let comp_rc = compile_prog(prog, opts.len() as c_int, opts.as_ptr());
        if comp_rc != 0 {
            let mut log_sz = 0usize;
            let _ = get_log_size(prog, &mut log_sz);
            let mut log_buf = vec![0u8; log_sz.max(1)];
            let _ = get_log(prog, log_buf.as_mut_ptr().cast());
            let _ = destroy_prog(&mut prog);
            return Err(format!(
                "NVRTC compile failed ({comp_rc}):\n{}",
                String::from_utf8_lossy(&log_buf)
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
        if let Err(e) = record_fn(ctx.compute_stream) {
            let mut dummy: CUgraph = core::ptr::null_mut();
            unsafe {
                let _ = (ctx.api.cuStreamEndCapture)(ctx.compute_stream, &mut dummy);
            }
            return Err(e);
        }
        let mut graph: CUgraph = core::ptr::null_mut();
        let mut exec: CUgraphExec = core::ptr::null_mut();
        unsafe {
            check_cu(
                (ctx.api.cuStreamEndCapture)(ctx.compute_stream, &mut graph),
                "cuStreamEndCapture",
            )?;
            check_cu(
                (ctx.api.cuGraphInstantiateWithFlags)(&mut exec, graph, 0),
                "cuGraphInstantiateWithFlags",
            )?;
        }
        Ok(Self {
            ctx: Arc::clone(ctx),
            graph,
            exec,
        })
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
