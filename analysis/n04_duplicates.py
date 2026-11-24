"""N04: duplicate / near-duplicate sensitivity (STRATEGY C5 + C7).

Inputs: canonical folds, K1 near_dups.csv (pHash Hamming <= 10 between contest images), member OOF arrays,
n01 nested predictions. Reports:
  * caption duplicates in train (groups, rows, cross-fold groups, label conflicts) and test-train caption overlap;
  * image near-duplicates at pHash distance 0 and <= 4: train-train cross-fold pairs, test-train pairs;
  * OOF macro-F1 / errors of each member and of key nested systems on all rows vs rows WITHOUT a cross-fold
    duplicate (caption or image), to show whether duplicates inflate OOF scores;
  * EVA-02-L OOF accuracy on rows that have a cross-fold near-duplicate image vs rows that do not.
Test labels are not used here.
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from n01_nested_meta import ORIG, OOF, fast_f1  # noqa: E402

RUNS = ROOT / "runs"
OUTD = ROOT / "analysis/out/n04"


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    f = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
    y, fold, ids = f.label.values, f.fold.values, f.image_id.values
    te = pd.read_csv(ROOT / "contest/Disaster_test.csv", encoding="utf-8-sig")
    pos = {i: n for n, i in enumerate(ids)}
    rep = {}

    # caption duplicates
    k = f.context.map(norm)
    grp = f.assign(k=k).groupby("k")
    dup_rows = grp.filter(lambda d: len(d) > 1)
    cross_cap = set()
    for _, d in dup_rows.groupby("k"):
        if d.fold.nunique() > 1:
            cross_cap.update(d.image_id)
    tk = te.context.map(norm)
    rep["caption"] = {"train_rows_in_dup_groups": len(dup_rows), "groups": int(dup_rows.k.nunique()),
                      "rows_with_crossfold_caption_dup": len(cross_cap),
                      "groups_with_label_conflict": int((dup_rows.groupby("k").label.nunique() > 1).sum()),
                      "test_rows_caption_in_train": int(tk.isin(set(k)).sum()),
                      "test_rows_in_test_dup_groups": int(tk.duplicated(keep=False).sum())}

    # image near-duplicates
    nd_p = sorted(RUNS.glob("k1-imgindex-*/outputs/near_dups.csv"))
    cross_img = {0: set(), 4: set()}
    if nd_p:
        nd = pd.read_csv(nd_p[-1])
        for thr in (0, 4, 10):
            sub = nd[nd.phash_dist <= thr]
            tt = sub[sub.kind == "train-train"]
            cf = tt[[fold[pos[a]] != fold[pos[b]] for a, b in zip(tt.a, tt.b)]]
            lab_conf = int(sum(y[pos[a]] != y[pos[b]] for a, b in zip(cf.a, cf.b)))
            rows = set(cf.a) | set(cf.b)
            if thr in cross_img:
                cross_img[thr] = rows
            rep[f"image_phash_le_{thr}"] = {"train_pairs": len(tt), "crossfold_pairs": len(cf),
                                             "crossfold_pairs_label_conflict": lab_conf,
                                             "rows_with_crossfold_image_dup": len(rows),
                                             "test_train_pairs": int((sub.kind == "test-train").sum()),
                                             "test_rows_with_train_image_dup": int(sub[sub.kind == "test-train"].a.nunique())}
    else:
        rep["image"] = "K1 near_dups.csv not available yet"

    # sensitivity of OOF scores
    masks = {"all": np.ones(len(y), bool),
             "no_crossfold_caption_dup": ~np.isin(ids, list(cross_cap)),
             "no_crossfold_image_dup_le4": ~np.isin(ids, list(cross_img[4])),
             "no_crossfold_dup_any_le4": ~np.isin(ids, list(cross_cap | cross_img[4]))}
    rows = []
    systems = {m: np.load(OOF / f"{v}_oof.npy").argmax(1) for m, v in ORIG.items()}
    for cfg in ("clean4", "P", "published"):
        p = ROOT / f"analysis/out/n01/preds_{cfg}_m0.npz"
        if p.exists():
            z = np.load(p)
            for s in ("late_fusion", "blend", "bias_boot"):
                if f"nested__{s}" in z.files:
                    systems[f"{cfg}:{s}(nested)"] = z[f"nested__{s}"].argmax(1)
    for name, pred in systems.items():
        r = {"system": name}
        for mk, m in masks.items():
            r[f"{mk}_n"] = int(m.sum())
            r[f"{mk}_f1"] = round(fast_f1(y[m], pred[m]), 5)
            r[f"{mk}_err"] = int((pred[m] != y[m]).sum())
        rows.append(r)
    sens = pd.DataFrame(rows)
    sens.to_csv(OUTD / "sensitivity.csv", index=False)
    if cross_img[4]:
        e = systems["eva02_L"] == y
        has = np.isin(ids, list(cross_img[4]))
        rep["eva_acc_rows_with_crossfold_image_dup_le4"] = round(float(e[has].mean()), 4) if has.any() else None
        rep["eva_acc_rows_without"] = round(float(e[~has].mean()), 4)
    import json
    json.dump(rep, open(OUTD / "report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
