"""Cross BF16 and native head inputs with official and dequantized MLX projection weights."""

import json
from pathlib import Path

import numpy as np
import torch
from bf16_teacher import Weights, load


@torch.inference_mode()
def main():
    model = load(Path("official-bf16"), "cuda")
    seqs = [
        list(map(int, l.split())) for l in Path("fixtures.ids").read_text().splitlines()
    ]
    states = []

    def save_input(module, args):
        states.append(args[0][0, :-1].detach().clone())

    hook = model.lm_head.register_forward_pre_hook(save_input)
    outputs = []
    for seq in seqs:
        out = model(
            input_ids=torch.tensor([seq], device="cuda"),
            use_cache=True,
            return_dict=True,
        )
        outputs.append(out.logits[0, :-1].float().cpu().numpy().copy())
    hook.remove()
    h = torch.cat(states).float()
    direct = torch.from_numpy(
        np.fromfile("native-head.f32", dtype="<f4").reshape(h.shape)
    ).cuda()
    w = model.lm_head.weight.detach()
    del out, model
    source = Weights(Path("restored"))
    prefix = "language_model.lm_head"
    packed = source.dense(prefix + ".weight").to("cuda")
    sc = source.dense(prefix + ".scales").float().cuda()
    bias = source.dense(prefix + ".biases").float().cuda()
    n, dim = w.shape
    assert (
        h.shape == (49, dim)
        and packed.shape == (n, dim // 8)
        and sc.shape == bias.shape == (n, dim // 32)
    )
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device="cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {
        k: []
        for k in [
            "bf_hidden_official_fp32",
            "bf_hidden_mlx_fp32",
            "native_hidden_official_fp32",
            "native_hidden_mlx_fp32",
        ]
    }
    for start in range(0, n, 2048):
        ss = slice(start, start + 2048)
        u = (
            ((packed[ss].to(torch.int32).unsqueeze(-1) >> shifts) & 15)
            .reshape(-1, dim // 32, 32)
            .float()
        )
        deq = (u * sc[ss, :, None] + bias[ss, :, None]).reshape(-1, dim)
        weights = w[ss].float()
        for key, state, plane in [
            ("bf_hidden_official_fp32", h, weights),
            ("bf_hidden_mlx_fp32", h, deq),
            ("native_hidden_official_fp32", direct, weights),
            ("native_hidden_mlx_fp32", direct, deq),
        ]:
            results[key].append((state @ plane.T).cpu().numpy())
    results = {k: np.concatenate(v, axis=1) for k, v in results.items()}
    results["bf_hidden_official_upstream"] = np.concatenate(outputs, axis=0)
    rows = np.loadtxt("score-bf16/rows.tsv", dtype=np.int64)
    ref = np.memmap("score-bf16/logits.f32", mode="r", dtype="<f4").reshape(
        len(rows), n
    )
    native = np.memmap(
        "wave1-score-direct-engram/logits.f32", mode="r", dtype="<f4"
    ).reshape(len(rows), n)

    def measure(a, b):
        kl = []
        hits = 0
        nll = 0
        for i, (p, q) in enumerate(zip(a, b)):
            p, q = p.astype(np.float64), q.astype(np.float64)
            for v in (p, q):
                v -= v.max()
                v -= np.log(np.exp(v).sum())
            hits += int(p.argmax() == q.argmax())
            nll -= q[rows[i, 3]]
            kl.append(float(np.dot(np.exp(p), p - q)))
        worst = int(np.argmax(kl))
        jp = np.flatnonzero((rows[:, 0] == 1) & (rows[:, 1] == 8)).item()
        return {
            "mean_kl": float(np.mean(kl)),
            "max_kl": kl[worst],
            "worst_case_position": rows[worst, :2].tolist(),
            "top1_matches": hits,
            "candidate_ppl": float(np.exp(nll / len(rows))),
            "japanese_position8_kl": kl[jp],
        }

    print(
        json.dumps(
            {
                "rows": len(rows),
                "vocab": n,
                "against_official_bf16": {
                    k: measure(ref, v)
                    for k, v in {**results, "native_production": native}.items()
                },
                "native_vs_mlx_fp32_same_hidden": measure(
                    native, results["native_hidden_mlx_fp32"]
                ),
                "official_repeat_byte_equal": bool(
                    np.array_equal(ref, results["bf_hidden_official_upstream"])
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
