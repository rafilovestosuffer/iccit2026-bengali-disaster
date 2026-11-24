"""Code-only Kaggle CPU job for n01_nested_meta.py; inputs come from the private HF dataset
iccit2845-meta-inputs, not released (built from jobs/n01_meta/pack.py published -> hf/iccit2845-meta-inputs).
python jobs/n01_meta/pack_hf.py <config> [extra hf dataset ...]  ->  jobs/n01_meta/<config>_hf/
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cfg = sys.argv[1]
d = ROOT / "jobs/n01_meta" / f"{cfg}_hf"
if d.exists():
    shutil.rmtree(d)
d.mkdir(parents=True)
for f in ("n01_nested_meta.py", "stackfeat_defs.py"):
    shutil.copy(ROOT / "analysis" / f, d / f)
(d / "job.yaml").write_text(f"""project: iccit2845
name: n01h-{cfg.lower()}
entry: n01_nested_meta.py
args: "--config {cfg} --meta-seed {{seed}}"
code: .
gpu: false
hours: 3
internet: true
hf_datasets: [iccit2845-meta-inputs]
""", encoding="utf-8")
print("packed", d)
