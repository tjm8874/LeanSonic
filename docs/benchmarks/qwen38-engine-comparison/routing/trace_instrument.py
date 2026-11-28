"""Apply only to a disposable checkout of 9b3e4f2. The trace synchronizes each layer."""

from pathlib import Path

p = Path("crates/leansonic-core/src/qwen38_gpu.rs")
s = p.read_text()
mark = """                        "q38_softmax_topk10",
                    )?;
"""
if s.count(mark) != 1:
    raise ValueError("unexpected router source")
s = s.replace(
    mark,
    mark
    + """                    let mut selected = [0i32; Q38_TOP_K];
                    check_cu((self.ctx.api.cuStreamSynchronize)(stream), "trace sync")?;
                    check_cu((self.ctx.api.cuMemcpyDtoH_v2)(selected.as_mut_ptr().cast(),
                             self.d_sel_ids10.ptr, core::mem::size_of_val(&selected)), "trace read")?;
                    eprintln!("ROUTE {} {} {:?}", self.host_pos, l_idx, selected);
""",
)
p.write_text(s)
p = Path("crates/leansonic-core/examples/q38_stream.rs")
s = p.read_text()
s = s.replace(
    "Qwen38GpuEngine::load_from_dir(model, context)?",
    "Qwen38GpuEngine::load_from_dir_with_opts(model, context, false)?",
)
if "load_from_dir_with_opts" not in s:
    raise ValueError("missing trace mode")
p.write_text(s)
