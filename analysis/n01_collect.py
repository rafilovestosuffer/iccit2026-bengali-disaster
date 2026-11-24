"""Copy finished n01 Kaggle outputs (runs/n01h-*/outputs) into analysis/out/n01/ with provenance.
Only runs whose result.json has exit_code 0 are taken; QUICK outputs are never copied. If two runs produced the same
config/meta-seed, the newest wins and the choice is recorded in analysis/out/n01/provenance.json.
"""
import hashlib
import json
import shutil
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = Path(__file__).resolve().parents[1] / "analysis/out/n01"
OUT.mkdir(parents=True, exist_ok=True)
prov_f = OUT / "provenance.json"
prov = json.loads(prov_f.read_text()) if prov_f.exists() else {}
for rd in sorted(RUNS.glob("n01h-*"), key=lambda p: p.stat().st_mtime):
    res = rd / "result.json"
    if not res.exists() or json.loads(res.read_text()).get("exit_code") != 0:
        continue
    run = json.loads((rd / "run.json").read_text(encoding="utf-8"))
    for f in sorted((rd / "outputs").glob("*")):
        if "QUICK" in f.name or f.suffix not in (".npz", ".csv", ".json"):
            continue
        shutil.copy2(f, OUT / f.name)
        prov[f.name] = {"run": rd.name, "kernel": run.get("kernel"), "account": run.get("account"),
                        "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
prov_f.write_text(json.dumps(prov, indent=1), encoding="utf-8")
print(len(prov), "files recorded in", prov_f)
