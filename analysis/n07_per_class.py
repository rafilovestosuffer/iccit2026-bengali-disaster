"""Per-class recall and confusion matrices under nested evaluation (OOF on the 6,323 training rows only; no test labels).

Systems: the three original members (MuRIL-Large, EVA-02-L, BanglaBERT-base) and, for each available config
(primary pipeline P if complete, else clean3; plus clean3), late fusion and the full pipeline (bias_boot), each as the
mean over meta-seeds 0-4 of per-seed recall. Also on the dedup subset (rows with no cross-fold twin).
Writes analysis/out/n07/per_class_recall.csv and confusion_<system>.csv (meta-seed 0 for pipeline stages).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from n06_compare import PRIMARY, member_pred, meta_files, stage_preds  # noqa: E402

OUTD = ROOT / "analysis/out/n07"
OUTD.mkdir(parents=True, exist_ok=True)
folds = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
y = folds.label.values.astype(int)
names = folds.groupby("label").category.first().to_dict()
K = len(names)
keep = np.load(ROOT / "analysis/out/dedup_masks.npz")["train_keep"]

systems = {f"member:{m}": [member_pred(m)] for m in ("muril_L", "eva02_L", "bb_base")}
for cfg in dict.fromkeys((PRIMARY, "clean3")):
    if meta_files(cfg):
        for st in ("late_fusion", "bias_boot"):
            systems[f"{cfg}:{st}"] = stage_preds(cfg, st)

rows = []
for sysname, preds in systems.items():
    for subset, mask in (("all_6323", np.ones(len(y), bool)), ("dedup_5321", keep)):
        rec = np.array([recall_score(y[mask], p[mask], labels=range(K), average=None, zero_division=0) for p in preds])
        err = np.array([[int(((y[mask] == k) & (p[mask] != k)).sum()) for k in range(K)] for p in preds])
        for k in range(K):
            rows.append(dict(system=sysname, subset=subset, n_seeds=len(preds), label=k, cls=names[k],
                             support=int((y[mask] == k).sum()), recall_mean=float(rec[:, k].mean()),
                             recall_min=float(rec[:, k].min()), recall_max=float(rec[:, k].max()),
                             missed_mean=float(err[:, k].mean())))
    cm = confusion_matrix(y, preds[0], labels=range(K))
    pd.DataFrame(cm, index=[names[k] for k in range(K)], columns=[names[k] for k in range(K)]).to_csv(
        OUTD / f"confusion_{sysname.replace(':', '_')}.csv")

df = pd.DataFrame(rows)
df.to_csv(OUTD / "per_class_recall.csv", index=False)
print("primary pipeline:", PRIMARY)
piv = df[df.subset == "all_6323"].pivot(index="cls", columns="system", values="recall_mean").round(4)
print(piv.to_string())
piv = df[df.subset == "all_6323"].pivot(index="cls", columns="system", values="missed_mean")
print("\nmissed (mean over meta-seeds):\n" + piv.to_string())
