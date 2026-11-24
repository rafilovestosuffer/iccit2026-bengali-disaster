"""Add finished G-run member arrays to the n01 HF inputs and push a new dataset version.

python jobs/n01_meta/add_new_members.py g2_s0 [g2_s1 ...]
For each member: newest runs/*/outputs/<m>_oof.npy (+ _test.npy) from a run whose result.json has
exit_code 0, checked for shape (6323, 8) / (1580, 8), finite values and rows summing to 1; copied to
hf/iccit2845-meta-inputs/new/ with its source run, kernel and sha256 recorded in new/SOURCES.txt; then
`gpu.cmd data push hf/iccit2845-meta-inputs iccit2845-meta-inputs`.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
NEW = ROOT / "hf/iccit2845-meta-inputs/new"
NEW.mkdir(parents=True, exist_ok=True)
src_f = NEW / "SOURCES.txt"
lines = [l for l in (src_f.read_text(encoding="utf-8").splitlines() if src_f.exists() else []) if l and l != "none"]
for m in sys.argv[1:]:
    hits = []
    for p in RUNS.glob(f"*/outputs/{m}_oof.npy"):
        res = p.parent.parent / "result.json"
        if res.exists() and json.loads(res.read_text()).get("exit_code") == 0:
            hits.append(p)
    if not hits:
        sys.exit(f"{m}: no finished run with {m}_oof.npy")
    p = max(hits, key=lambda q: q.stat().st_mtime)
    o, t = np.load(p), np.load(p.parent / f"{m}_test.npy")
    assert o.shape == (6323, 8) and t.shape == (1580, 8), (o.shape, t.shape)
    for a in (o, t):
        assert np.isfinite(a).all() and np.allclose(a.sum(1), 1, atol=1e-3), "bad probabilities"
    np.save(NEW / f"{m}_oof.npy", o.astype(np.float32))
    np.save(NEW / f"{m}_test.npy", t.astype(np.float32))
    run = json.loads((p.parent.parent / "run.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256((NEW / f"{m}_oof.npy").read_bytes()).hexdigest()[:16]
    lines = [l for l in lines if not l.startswith(f"{m}:")]
    lines.append(f"{m}: run {p.parent.parent.name}, kernel {run.get('kernel')}, oof sha256 {sha}")
    print("added", lines[-1])
src_f.write_text("\n".join(lines) + "\n", encoding="utf-8")
r = subprocess.run(["gpu.cmd", "data", "push", str(ROOT / "hf/iccit2845-meta-inputs"),
                    "iccit2845-meta-inputs"], capture_output=True, text=True)
print((r.stdout or "")[-400:], (r.stderr or "")[-400:])
