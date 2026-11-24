"""Pack a self-contained Kaggle CPU job for n01_nested_meta.py (one code dir per config).

python jobs/n01_meta/pack.py <config>  ->  jobs/n01_meta/<config>/ with the script, its helpers, data/ and job.yaml.
inputs/ (not data/: the gpu tool excludes data dirs): folds.csv (image_id, fold, label), test_ids.csv, textfeat.npz (stacker text features, train/test),
imgstats.csv (16 image statistics per image from K1, if available), oof/ (original member arrays),
new/ (G-run member arrays + SOURCES.txt). Test labels are never packed.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
import n01_nested_meta as N  # noqa: E402
import stackfeat_defs as SF  # noqa: E402

cfg = sys.argv[1]
members = N.CONFIGS[cfg]
d = ROOT / "jobs/n01_meta" / cfg
if d.exists():
    shutil.rmtree(d)
(d / "inputs/oof").mkdir(parents=True)
(d / "inputs/new").mkdir(parents=True)
for f in ("n01_nested_meta.py", "stackfeat_defs.py"):
    shutil.copy(ROOT / "analysis" / f, d / f)
folds = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
test = pd.read_csv(ROOT / "contest/Disaster_test.csv", encoding="utf-8-sig")
folds[["image_id", "fold", "label"]].to_csv(d / "inputs/folds.csv", index=False)
test[["image_id"]].to_csv(d / "inputs/test_ids.csv", index=False)
np.savez_compressed(d / "inputs/textfeat.npz",
                    train=np.array([SF.text_feats(x) for x in folds["context"]], np.float32),
                    test=np.array([SF.text_feats(x) for x in test["context"]], np.float32))
st, src = N.image_stats()
if st is not None:
    cols = [f"img{j}" for j in range(16)]
    np.savez_compressed(d / "inputs/imgstats.npz", train=st.loc[folds["image_id"], cols].values.astype(np.float32),
                        test=st.loc[test["image_id"], cols].values.astype(np.float32))
    print("image stats from", src)
else:
    print("WARNING: no image stats (K1) -> job must run with --no-imgstats")
sources = []
for m in members:
    if m in N.ORIG:
        for suf in ("_oof.npy", "_test_probs.npy"):
            shutil.copy(N.OOF / f"{N.ORIG[m]}{suf}", d / "inputs/oof" / f"{N.ORIG[m]}{suf}")
    else:
        o, t, s = N.find_new_run(m)
        np.save(d / "inputs/new" / f"{m}_oof.npy", o)
        np.save(d / "inputs/new" / f"{m}_test.npy", t)
        sources.append(f"{m}: {s}")
(d / "inputs/new/SOURCES.txt").write_text("\n".join(sources) or "none", encoding="utf-8")
extra = "" if st is not None else " --no-imgstats"
(d / "job.yaml").write_text(f"""project: iccit2845
name: n01-{cfg.lower()}
entry: n01_nested_meta.py
args: "--config {cfg} --meta-seed {{seed}}{extra}"
code: .
gpu: false
hours: 2
internet: false
""", encoding="utf-8")
print("packed", d)
