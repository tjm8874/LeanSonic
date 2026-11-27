"""Exercise an already-owned loopback server with raw and ASCII-escaped requests."""

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def port_of(child):
    deadline = time.monotonic() + 180
    while child.poll() is None and time.monotonic() < deadline:
        fds = Path(f"/proc/{child.pid}/fd")
        sockets = set()
        for fd in fds.iterdir():
            try:
                sockets.add(os.readlink(fd))
            except FileNotFoundError:
                continue
        for line in Path("/proc/net/tcp").read_text().splitlines()[1:]:
            fields = line.split()
            if (
                fields[3] == "0A"
                and fields[1].startswith("0100007F:")
                and ("socket:[" + fields[9] + "]") in sockets
            ):
                return int(fields[1].split(":")[1], 16)
        time.sleep(0.15)
    raise RuntimeError(
        f"no listening socket for child {child.pid}, exit={child.poll()}"
    )


def equivalent(got, want):
    if type(got) is not type(want):
        return False
    if isinstance(want, list):
        return len(got) == len(want) and all(
            equivalent(a, b) for a, b in zip(got, want)
        )
    if isinstance(want, dict):
        return got.keys() == want.keys() and all(
            equivalent(got[k], v) for k, v in want.items()
        )
    return got == want


def passes(r, text):
    if r["kind"] == "text":
        return text.strip() == r["expected"]
    try:

        def unique(pairs):
            if len(dict(pairs)) != len(pairs):
                raise ValueError("duplicate JSON key")
            return dict(pairs)

        return equivalent(
            json.loads(text.strip(), object_pairs_hook=unique), r["expected"]
        )
    except ValueError:
        return False


def run(cli, model, max_tokens, compare_encodings):
    fixtures = json.loads(Path(__file__).with_name("fixtures.json").read_text())
    output = []
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with open("server.log", "xb") as log:
        child = subprocess.Popen(
            [
                cli,
                "serve",
                "--weights",
                model,
                "--spec",
                "0",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--effort",
                "none",
            ],
            stdout=log,
            stderr=log,
        )
        try:
            address = f"http://127.0.0.1:{port_of(child)}"
            with opener.open(address + "/health", timeout=10) as resp:
                backend = json.load(resp)["backend"]
                if "blackwell-gpu:" not in backend:
                    raise ValueError(backend)
            # Change request order between encodings. Repeat the first case at the end.
            order = [(f, False, "raw") for f in fixtures]
            if compare_encodings:
                order.extend((f, True, "escaped") for f in reversed(fixtures))
            order.append((fixtures[0], False, "repeat"))
            for r, esc, kind in order:
                data = {
                    "model": "inkling-2-mini",
                    "messages": [{"role": "user", "content": r["prompt"]}],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "reasoning_effort": "none",
                    "stream": False,
                }
                request = urllib.request.Request(
                    address + "/v1/chat/completions",
                    data=json.dumps(data, ensure_ascii=esc).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with opener.open(request, timeout=180) as resp:
                    obj = json.load(resp)
                (choice,) = obj["choices"]
                text = choice["message"]["content"]
                output.append(
                    {
                        "case": r["id"],
                        "encoding": kind,
                        "text": text,
                        "finish": choice["finish_reason"],
                        "usage": obj["usage"],
                        "expected_prompt_tokens": len(
                            Path("ids", r["id"] + ".ids").read_text().split()
                        ),
                        "passed": passes(r, text) and choice["finish_reason"] == "stop",
                    }
                )
            return {"backend": backend, "requests": output}
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cli")
    p.add_argument("model")
    p.add_argument("--cap", choices=[128, 384], type=int, default=128)
    p.add_argument("--compare-encodings", action="store_true")
    args = p.parse_args()
    result = run(args.cli, args.model, args.cap, args.compare_encodings)
    Path("results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
