"""N09: post-hoc controls (NOT part of the pre-specified family; added 2026-11-19 after an independent check of the
revision, D37). Training OOF rows only: the 1,580-row partition is not read.

(a) LF weight fitted on rows disjoint from the evaluated rows. In n02, LF's alpha for outer fold k is fitted on the
    outer-training rows, whose text predictions T come from G1 fold models that were trained on fold k (the scope note
    of Section IV-A). Here, for each fold k, the rows of k are split into two halves by duplicate group (identical
    normalised caption, or a train-train image pair with pHash distance <= 4 from K1, joined into connected
    components; the half is a hash of the group, no labels used). alpha is fitted with n01's fit_alpha on one half and
    applied to the other half, and vice versa. On fold k both inputs exclude fold k: T is G1's fold-k model and the
    image head is n02's nested logistic regression (C chosen without fold k).
(b) LF with a fixed alpha = 0.5 (no fitting).
(c) Full pipeline P with J1 trained with seed 1 or 2 instead of seed 0 (other members unchanged): nested errors of
    the bootstrap-bias stage per meta-seed, from the existing n01 runs P_s1 and P_s2 (exploratory, D33).

Writes analysis/out/n09/{lf_controls.csv, alphas.csv, j1seed.csv, meta.json}.
"""
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from n01_nested_meta import fit_alpha  # noqa: E402

O = ROOT / "analysis/out"
OUTD = O / "n09"
OUTD.mkdir(parents=True, exist_ok=True)
SALT = "2845-control-v1/"

folds = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
y = folds.label.to_numpy().astype(int)
fold = folds.fold.to_numpy()
keep = np.load(O / "dedup_masks.npz")["train_keep"]
n2 = np.load(O / "n02/preds.npz")
meta2 = json.loads((O / "n02/meta.json").read_text(encoding="utf-8"))
img = n2["img_oof"]
k1 = Path(meta2["sources"]["g1_s0"]).parents[1] / "k1-imgindex-1118-1725-s0/outputs"
near = k1 / "near_dups.csv"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm_caption(s):
    return " ".join(unicodedata.normalize("NFC", str(s)).split())


# ---- duplicate groups (no labels)
pos = {k: i for i, k in enumerate(folds.image_id)}
e = pd.read_csv(near)
e = e[(e.kind == "train-train") & (e.phash_dist <= 4)]
src, dst = list(e.a.map(pos)), list(e.b.map(pos))
assert not (pd.isna(src).any() or pd.isna(dst).any()), "near_dups ids not all in the training folds"
for idx in folds.groupby(folds.context.map(norm_caption)).indices.values():
    src += [idx[0]] * (len(idx) - 1)
    dst += list(idx[1:])
n = len(y)
_, comp = connected_components(coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)), directed=False)
gid = folds.image_id.groupby(comp).transform("min").to_numpy()
half = np.array([hashlib.sha256((SALT + g).encode()).digest()[0] % 2 for g in gid])

seeds = sorted(int(k.split("__s")[1]) for k in n2.files if k.startswith("oof__T__s"))
rows, arows = [], []
for s in seeds:
    T = np.load(Path(meta2["sources"][f"g1_s{s}"]) / f"g1_s{s}_oof.npy")
    assert np.array_equal(T.argmax(1), n2[f"oof__T__s{s}"]), f"G1 seed {s} OOF differs from n02's T"
    ctl = np.full(n, -1)
    for k in range(5):
        for h in (0, 1):
            cal, ev = (fold == k) & (half == h), (fold == k) & (half != h)
            assert not set(comp[cal]) & set(comp[ev])
            a = fit_alpha(T[cal], img[cal], y[cal])
            ctl[ev] = (a * T[ev] + (1 - a) * img[ev]).argmax(1)
            arows.append(dict(seed=s, fold=k, fit_half=h, n_fit=int(cal.sum()), n_eval=int(ev.sum()), alpha=float(a)))
    assert (ctl >= 0).all()
    fixed = (0.5 * T + 0.5 * img).argmax(1)
    preds = {"T": n2[f"oof__T__s{s}"], "LF": n2[f"oof__LF__s{s}"], "LF_halffold": ctl, "LF_fixed05": fixed,
             "J1": n2[f"oof__J1__s{s}"]}
    if f"oof__JP__s{s}" in n2.files:
        preds["JP"] = n2[f"oof__JP__s{s}"]
    for arm, p in preds.items():
        for sub, m in (("all_6323", np.ones(n, bool)), ("dedup_5321", keep)):
            rows.append(dict(arm=arm, seed=s, subset=sub, err=int((p[m] != y[m]).sum()),
                             f1=float(f1_score(y[m], p[m], average="macro"))))
lf = pd.DataFrame(rows)
lf.to_csv(OUTD / "lf_controls.csv", index=False)
pd.DataFrame(arows).to_csv(OUTD / "alphas.csv", index=False)
print(lf[lf.subset == "all_6323"].pivot(index="arm", columns="seed", values="err"))
print(lf[lf.subset == "dedup_5321"].pivot(index="arm", columns="seed", values="err"))

# ---- (c) J1 training seed in the full pipeline
jrows = []
for cfg in ("P", "P_s1", "P_s2"):
    for m in range(5):
        f = O / f"n01/preds_{cfg}_m{m}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        for stage in ("bias_boot", "late_fusion"):
            p = z[f"nested__{stage}"].argmax(1)
            jrows.append(dict(config=cfg, meta_seed=m, stage=stage, err=int((p != y).sum()),
                              f1=float(f1_score(y, p, average="macro")),
                              dd_err=int((p[keep] != y[keep]).sum())))
js = pd.DataFrame(jrows)
js.to_csv(OUTD / "j1seed.csv", index=False)
print(js.groupby(["config", "stage"])[["err", "f1", "dd_err"]].mean())

inputs = [O / "n02/preds.npz", O / "dedup_masks.npz", near, ROOT / "drive_oof/x/meta/folds_canonical.csv"]
inputs += [Path(meta2["sources"][f"g1_s{s}"]) / f"g1_s{s}_oof.npy" for s in seeds]
inputs += sorted((O / "n01").glob("preds_P*_m[0-4].npz"))
(OUTD / "meta.json").write_text(json.dumps(dict(
    status="post hoc (D37), exploratory; not part of the pre-specified family",
    split_rule=f"duplicate-group halves within each fold: sha256('{SALT}' + group id)[0] % 2",
    half_sizes=pd.crosstab(fold, half).values.tolist(),
    inputs={str(p): sha(p) for p in inputs}), indent=1), encoding="utf-8")
