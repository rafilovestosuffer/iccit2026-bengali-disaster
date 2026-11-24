"""N02: matched fusion-mechanism comparison M (answers R3.5 "restrict the fusion conclusion to the tested
architecture" and R2.3 / R3.4 "repeated seeds, paired CIs").

All arms share the canonical folds, MuRIL-L with one recipe, and the same frozen pretrained EVA-02-L features (G0):
  T   text only (G1, seed s)
  LF  late fusion: alpha * T + (1 - alpha) * I, where I is a logistic-regression head on the frozen pooled EVA
      features; alpha and the head are fitted fully inside the outer loop (fold k never used)
  J1  single-token joint head (G2, seed s)  = the submitted design, now without fold-label leakage
  JP  patch cross-attention over 64 patch tokens (G3, seed s) = what Eq. (2) of the paper describes
Seeds 0, 1, 2 (seed-matched differences). Test-set scores are saved, never evaluated here.
Run: .venv/Scripts/python.exe analysis/n02_mechanism.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from n01_nested_meta import fast_f1, fit_alpha  # noqa: E402
from n03_stats import Bootstrapper, holm  # noqa: E402

RUNS = ROOT / "runs"
OUTD = ROOT / "analysis/out/n02"
K = 8
C_GRID = [0.01, 0.1, 1.0]


def newest(pattern):
    hits = sorted(RUNS.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def lr_fit_predict(Xtr, ytr, Xs, C):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=3000)
    m.fit(sc.transform(Xtr), ytr)
    return [m.predict_proba(sc.transform(X)) for X in Xs]


def image_head(X, y, fold, Xte):
    """Fully nested LR head on frozen pooled EVA features. For outer fold k, C is chosen by inner CV on the other folds;
    the inner-CV predictions for the chosen C (img_inner[k], used to fit alpha) and the head refit on the other folds
    (img_oof rows of fold k) never see fold k. Final head for the test rows: C by 5-fold CV on all training rows.
    Cached in analysis/out/n02/image_head.npz (independent of the G runs)."""
    cache = OUTD / "image_head.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["oof"], z["test"], z["chosen"].item(), float(z["c_final"]), z["inner"].item()
    oof = np.zeros((len(y), K))
    chosen, inner_pred = {}, {}
    for k in range(5):
        tr = fold != k
        inner = fold[tr]
        best_c, best_f, best_p = None, -1, None
        for C in C_GRID:
            p = np.zeros((tr.sum(), K))
            for j in np.unique(inner):
                p[inner == j] = lr_fit_predict(X[tr][inner != j], y[tr][inner != j], [X[tr][inner == j]], C)[0]
            f = fast_f1(y[tr], p.argmax(1))
            if f > best_f:
                best_f, best_c, best_p = f, C, p
        chosen[k], inner_pred[k] = best_c, best_p
        oof[fold == k] = lr_fit_predict(X[tr], y[tr], [X[fold == k]], best_c)[0]
        print(f"  image head outer fold {k}: C={best_c} inner F1={best_f:.5f}", flush=True)
    best_c, best_f = None, -1
    for C in C_GRID:
        p = np.zeros((len(y), K))
        for j in range(5):
            p[fold == j] = lr_fit_predict(X[fold != j], y[fold != j], [X[fold == j]], C)[0]
        f = fast_f1(y, p.argmax(1))
        if f > best_f:
            best_f, best_c = f, C
    test = lr_fit_predict(X, y, [Xte], best_c)[0]
    OUTD.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, oof=oof, test=test, chosen=np.array(chosen, dtype=object), c_final=best_c,
                        inner=np.array(inner_pred, dtype=object))
    return oof, test, chosen, best_c, inner_pred


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
    y, fold = folds["label"].values.astype(int), folds["fold"].values.astype(int)
    pooled_p = newest("*/outputs/eva_pooled.npy")
    feats = np.load(pooled_p).astype(np.float32)
    X, Xte = feats[:6323], feats[6323:]
    img_oof, img_test, chosen, c_final, img_inner_k = image_head(X, y, fold, Xte)
    print("image head (frozen EVA, LR): OOF F1", round(fast_f1(y, img_oof.argmax(1)), 5),
          "err", int((img_oof.argmax(1) != y).sum()), "C per outer fold", chosen)
    if "--image-head-only" in sys.argv:
        return
    seeds, arms = [], {}
    for s in range(3):
        g = {b: newest(f"*/outputs/{b}_s{s}_oof.npy") for b in ("g1", "g2", "g3")}
        if g["g1"] and g["g2"]:  # D23: T, LF and J1 need G1 and G2; JP (G3) only if present for every seed
            seeds.append(s)
            for b, p in g.items():
                if p:
                    arms[(b, s)] = (np.load(p), np.load(p.parent / f"{b}_s{s}_test.npy"), str(p.parent))
    if not seeds:
        sys.exit("no seed with g1 and g2 outputs yet")
    ARMS = ("T", "LF", "J1") + (("JP",) if all(("g3", s) in arms for s in seeds) else ())
    print("seeds", seeds, "arms", ARMS)

    preds, tests, rows = {}, {}, []
    for s in seeds:
        T, Tte, _ = arms[("g1", s)]
        # late fusion: alpha fitted on outer-training rows only, using an image head that never saw fold k
        lf = np.zeros_like(T)
        for k in range(5):
            tr = fold != k
            a = fit_alpha(T[tr], img_inner_k[k], y[tr])
            lf[fold == k] = a * T[fold == k] + (1 - a) * img_oof[fold == k]
        a_full = fit_alpha(T, img_oof, y)
        preds[("T", s)], preds[("LF", s)] = T.argmax(1), lf.argmax(1)
        preds[("J1", s)], tests[("J1", s)] = arms[("g2", s)][0].argmax(1), arms[("g2", s)][1]
        tests[("T", s)], tests[("LF", s)] = Tte, a_full * Tte + (1 - a_full) * img_test
        if "JP" in ARMS:
            preds[("JP", s)], tests[("JP", s)] = arms[("g3", s)][0].argmax(1), arms[("g3", s)][1]
        for arm in ARMS:
            p = preds[(arm, s)]
            rows.append(dict(arm=arm, seed=s, oof_f1=fast_f1(y, p), oof_err=int((p != y).sum())))
    # D12: size of per-fold best-epoch selection on the evaluation fold, measured on the G runs
    es = []
    for (b, s), (_, _, d) in arms.items():
        f = Path(d) / f"{b}_s{s}_epochval.npy"
        if not f.exists() or (b == "g3" and "JP" not in ARMS):
            continue
        ev = np.load(f).astype(np.float32)  # (n_epochs, 6323, K), each fold's rows filled by its own fold model
        final = ev[-1].argmax(1)
        sel = final.copy()
        for k in range(5):
            m = fold == k
            best = max(range(ev.shape[0]), key=lambda e: fast_f1(y[m], ev[e][m].argmax(1)))
            sel[m] = ev[best][m].argmax(1)
        es.append(dict(branch=b, seed=s, f1_final=fast_f1(y, final), f1_selected=fast_f1(y, sel),
                       d_f1=fast_f1(y, sel) - fast_f1(y, final), d_err=int((sel != y).sum()) - int((final != y).sum())))
    if es:
        pd.DataFrame(es).to_csv(OUTD / "epoch_selection.csv", index=False)
        print(pd.DataFrame(es).to_string(index=False))
    per = pd.DataFrame(rows)
    agg = per.groupby("arm").agg(n=("seed", "size"), f1_mean=("oof_f1", "mean"), f1_sd=("oof_f1", "std"),
                                 err_mean=("oof_err", "mean"), err_sd=("oof_err", "std")).reset_index()
    print(per.to_string(index=False))
    print(agg.to_string(index=False))

    bs = Bootstrapper(y)
    comps = [c for c in [("LF", "T"), ("J1", "T"), ("JP", "T"), ("J1", "LF"), ("JP", "LF"), ("JP", "J1")]
             if set(c) <= set(ARMS)]
    out = []
    for a_, b_ in comps:
        r = bs.compare_seeds([preds[(a_, s)] for s in seeds], [preds[(b_, s)] for s in seeds])
        # pooled McNemar over seeds is not standard; report per-seed exact McNemar
        ps = [bs.compare(preds[(a_, s)], preds[(b_, s)])["mcnemar_p"] for s in seeds]
        out.append(dict(a=a_, b=b_, **r, mcnemar_p_per_seed=[round(x, 4) for x in ps]))
    cmp_df = pd.DataFrame(out)
    print(cmp_df.to_string(index=False))
    per.to_csv(OUTD / "per_seed.csv", index=False)
    agg.to_csv(OUTD / "aggregate.csv", index=False)
    cmp_df.to_csv(OUTD / "comparisons.csv", index=False)
    np.savez_compressed(OUTD / "preds.npz", img_oof=img_oof, img_test=img_test,
                        **{f"oof__{a}__s{s}": v for (a, s), v in preds.items()},
                        **{f"test__{a}__s{s}": v for (a, s), v in tests.items()})
    json.dump({"seeds": seeds, "sources": {f"{b}_s{s}": v[2] for (b, s), v in arms.items()},
               "eva_pooled": str(pooled_p), "C_outer": chosen, "C_final": c_final}, open(OUTD / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
