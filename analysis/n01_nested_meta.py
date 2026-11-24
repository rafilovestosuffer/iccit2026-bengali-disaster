"""N01: nested cross-validation of every meta-level stage (answers R2.1 / R3.1).

Outer loop = the 5 canonical folds. For outer fold k, every downstream stage is fitted on the OOF rows of the
other four folds only and then applied once to fold k:
  late fusion alpha (MuRIL-L + EVA-02-L)        grid of 201 values
  Dirichlet blend                                50 Dirichlet starts, each refined by Nelder-Mead (macro-F1)
  LightGBM stack                                 fixed parameters, 400 rounds, no early stopping, no Optuna;
                                                 stack predictions for the training rows come from an inner
                                                 4-fold CV over the remaining canonical folds
  alpha-mix of blend and stack                   grid of 201 values
  per-class log-bias, single-shot                coordinate ascent, 8 passes, offsets in [-0.5, 0.5] step 0.01
  per-class log-bias, bootstrap median           15 bootstrap resamples, 4 passes each
  blend -> bootstrap bias (no stack)             the simpler pipeline
The same procedure fitted and scored on all 6,323 rows (as in the submitted paper) gives the "in-sample" figure.
A final refit on all 6,323 rows produces 1,580-row test scores, which are SAVED BUT NOT SCORED here
(test labels are never read by this script; see n05_score_test_once.py).

Run: .venv/Scripts/python.exe analysis/n01_nested_meta.py --config clean4 --meta-seed 0
"""
import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import os

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stackfeat_defs as SF  # noqa: E402

# bundle mode (Kaggle CPU job): inputs packed next to the script in inputs/ by jobs/n01_meta/pack.py
_HF = Path(os.environ.get("DATA_DIR", "/nonexistent")) / "iccit2845-meta-inputs"
BUNDLE = _HF if (_HF / "folds.csv").exists() else (HERE / "inputs" if (HERE / "inputs" / "folds.csv").exists() else None)
OOF = BUNDLE / "oof" if BUNDLE else ROOT / "drive_oof/x/oof"
RUNS = ROOT / "runs"
OUTD = Path(os.environ["OUT_DIR"]) if BUNDLE and "OUT_DIR" in os.environ else ROOT / "analysis/out/n01"
EPS = 1e-12
K = 8
QUICK = False
ORIG = {
    "bb_base": "text_bbB_s123_20260509_0550", "bb_multi": "text_bbmulti_20260509_0837",
    "muril_L": "text_murilL_20260509_0210", "eva02_L": "img_eva02L_448_20260511_0327",
    "mm_old": "mm_eva_muril_20260512_1041", "pl_muril": "pl_muril_20260512_1725",
}
CONFIGS = {
    # the submitted system's members (transductive PL-MuRIL + fusion member with fold-label leakage)
    "published": ["bb_base", "bb_multi", "muril_L", "eva02_L", "mm_old", "pl_muril"],
    # non-transductive, no fusion member. bb_multi is dropped: it is (S1 BanglaBERT-base + 2 x bb_base) / 3,
    # i.e. a seed average of the same monolingual model, not a multilingual variant (audit 2026-11-18)
    "clean3": ["bb_base", "muril_L", "eva02_L"],
    # primary pipeline P: clean3 + leakage-free single-token joint head G2 (seed 0)
    "P": ["bb_base", "muril_L", "eva02_L", "g2_s0"],
    "P_s1": ["bb_base", "muril_L", "eva02_L", "g2_s1"],
    "P_s2": ["bb_base", "muril_L", "eva02_L", "g2_s2"],
}


# ------------------------------------------------------------------------------------- utilities
def fast_f1(y, pred):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    d = 2 * tp + fp + fn
    return float(np.mean(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0)))


def find_new_run(branch_seed):
    """g2_s0 -> newest complete run dir holding g2_s0_oof.npy (bundle mode: data/new/)."""
    if BUNDLE:
        d = BUNDLE / "new"
        return np.load(d / f"{branch_seed}_oof.npy"), np.load(d / f"{branch_seed}_test.npy"), (d / "SOURCES.txt").read_text()
    hits = sorted(RUNS.glob(f"*/outputs/{branch_seed}_oof.npy"), key=lambda p: p.stat().st_mtime)
    if not hits:
        sys.exit(f"member {branch_seed} not found under {RUNS}")
    d = hits[-1].parent
    return np.load(d / f"{branch_seed}_oof.npy"), np.load(d / f"{branch_seed}_test.npy"), str(d)


def load_members(names):
    O, T, src = {}, {}, {}
    for n in names:
        if n in ORIG:
            O[n], T[n] = np.load(OOF / f"{ORIG[n]}_oof.npy"), np.load(OOF / f"{ORIG[n]}_test_probs.npy")
            src[n] = ORIG[n]
        else:
            O[n], T[n], src[n] = find_new_run(n)
        O[n] = O[n].astype(np.float64)
        T[n] = T[n].astype(np.float64)
        assert O[n].shape == (6323, K) and T[n].shape == (1580, K), n
        assert np.isfinite(O[n]).all() and np.isfinite(T[n]).all(), n
    return O, T, src


def image_stats():
    if BUNDLE:
        if not (BUNDLE / "imgstats.npz").exists():
            return None, None
        z = np.load(BUNDLE / "imgstats.npz", allow_pickle=False)
        ids = list(pd.read_csv(BUNDLE / "folds.csv").image_id) + list(pd.read_csv(BUNDLE / "test_ids.csv").image_id)
        return pd.DataFrame(np.vstack([z["train"], z["test"]]), index=ids,
                            columns=[f"img{j}" for j in range(16)]), "bundle:imgstats.npz"
    hits = sorted(RUNS.glob("k1-imgindex-*/outputs/contest_index.csv"))
    if not hits:
        return None, None
    d = pd.read_csv(hits[-1])
    d = d.set_index("stem")
    cols = [f"img{j}" for j in range(16)]
    return d[cols], str(hits[-1])


# ---------------------------------------------------------------------------------- meta stages
def fit_alpha(pa, pb, y, grid=201):
    best_a, best_f = 0.0, -1.0
    for a in np.linspace(0, 1, grid):
        f = fast_f1(y, (a * pa + (1 - a) * pb).argmax(1))
        if f > best_f:
            best_f, best_a = f, a
    return best_a


def fit_blend(oofs, y, meta_seed, n_restarts=50):
    n_restarts = 2 if QUICK else n_restarts
    def neg(w):
        w = np.maximum(w, 0)
        s = w.sum()
        if s < 1e-9:
            return 0.0
        w = w / s
        return -fast_f1(y, sum(wi * oi for wi, oi in zip(w, oofs)).argmax(1))

    best_w, best_s = None, -1.0
    for t in range(n_restarts):
        np.random.seed(1000 * meta_seed + t)
        w0 = np.random.dirichlet(np.ones(len(oofs)))
        r = minimize(neg, w0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 8000})
        if -r.fun > best_s:
            best_s = -r.fun
            best_w = np.maximum(r.x, 0)
            best_w = best_w / best_w.sum()
    return best_w


def per_model_stats(p):
    sp = np.sort(p, axis=1)
    return np.stack([sp[:, -1], sp[:, -1] - sp[:, -2], -(p * np.log(p + EPS)).sum(1)], axis=1)


def pairwise_feats(pa, pb):
    kl_ab = (pa * (np.log(pa + EPS) - np.log(pb + EPS))).sum(1)
    kl_ba = (pb * (np.log(pb + EPS) - np.log(pa + EPS))).sum(1)
    cos = (pa * pb).sum(1) / (np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1) + EPS)
    agree = (pa.argmax(1) == pb.argmax(1)).astype(np.float32)
    return np.stack([kl_ab, kl_ba, cos, agree], axis=1)


def stack_matrix(arrs, tf, im):
    """Same construction as the competition stacker (ml-contest-2 cell 28, build_stack_matrix)."""
    names = list(arrs)
    blocks = [arrs[n] for n in names] + [per_model_stats(arrs[n]) for n in names]
    blocks += [pairwise_feats(arrs[a], arrs[b]) for a, b in combinations(names, 2)]
    blocks.append(np.mean([arrs[n] for n in names], axis=0))
    blocks.append(tf)
    if im is not None:
        blocks.append(im)
    return np.concatenate(blocks, axis=1).astype(np.float32)


def lgb_params(meta_seed):
    return dict(objective="multiclass", num_class=K, learning_rate=0.03, num_leaves=31, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=40, verbose=-1, seed=42 + meta_seed,
                num_threads=2, deterministic=True, force_row_wise=True)


def lgb_fit_predict(Xtr, ytr, Xs, meta_seed, rounds=400):
    rounds = 10 if QUICK else rounds
    m = lgb.train(lgb_params(meta_seed), lgb.Dataset(Xtr, label=ytr), num_boost_round=rounds)
    return [m.predict(X) for X in Xs]


def fit_bias_single(p, y, passes=8, lo=-0.5, hi=0.5, step=0.01):
    passes = 1 if QUICK else passes
    lp = np.log(p + EPS)
    grid = np.arange(lo, hi + step / 2, step)
    b = np.zeros(K)
    for _ in range(passes):
        for c in range(K):
            o = b[c]
            bl, bs = o, fast_f1(y, (lp + b).argmax(1))
            for d in grid:
                b[c] = o + d
                v = fast_f1(y, (lp + b).argmax(1))
                if v > bs:
                    bs, bl = v, b[c]
            b[c] = bl
    return b


def fit_bias_boot(p, y, meta_seed, n_boot=15):
    n_boot = 2 if QUICK else n_boot
    out = []
    for sd in range(n_boot):
        idx = np.random.default_rng(1000 * meta_seed + sd).choice(len(y), len(y), replace=True)
        out.append(fit_bias_single(p[idx], y[idx], passes=4))
    return np.median(np.stack(out), 0)


def run_procedure(O_tr, y_tr, f_tr, F_tr, apply_sets, members, meta_seed):
    """Fit every stage on (O_tr, y_tr) only; return stage scores for each apply set.

    O_tr: dict member -> (n_tr, K) OOF probs; f_tr: canonical fold ids of those rows (used for the inner stack CV);
    F_tr: side features (text, image stats) of those rows; apply_sets: list of (dict member -> probs, side features).
    """
    out = [dict() for _ in apply_sets]
    fitted = {}
    oofs_tr = [O_tr[m] for m in members]
    # uniform mean (no fitting)
    for o, (A, _) in zip(out, apply_sets):
        o["uniform"] = np.mean([A[m] for m in members], 0)
    # late fusion of MuRIL-L and EVA-02-L
    if "muril_L" in members and "eva02_L" in members:
        a = fit_alpha(O_tr["muril_L"], O_tr["eva02_L"], y_tr)
        fitted["late_fusion_alpha"] = float(a)
        for o, (A, _) in zip(out, apply_sets):
            o["late_fusion"] = a * A["muril_L"] + (1 - a) * A["eva02_L"]
    # blend
    w = fit_blend(oofs_tr, y_tr, meta_seed)
    fitted["blend_w"] = dict(zip(members, np.round(w, 4).tolist()))
    blend_tr = sum(wi * oi for wi, oi in zip(w, oofs_tr))
    for o, (A, _) in zip(out, apply_sets):
        o["blend"] = sum(wi * A[m] for wi, m in zip(w, members))
    # stack: inner CV for training-row predictions, full fit for apply sets
    X_tr = stack_matrix({m: O_tr[m] for m in members}, *F_tr)
    stack_tr = np.zeros((len(y_tr), K))
    for f in np.unique(f_tr):
        ti, vi = f_tr != f, f_tr == f
        stack_tr[vi] = lgb_fit_predict(X_tr[ti], y_tr[ti], [X_tr[vi]], meta_seed)[0]
    Xs = [stack_matrix({m: A[m] for m in members}, *F) for A, F in apply_sets]
    for o, sp in zip(out, lgb_fit_predict(X_tr, y_tr, Xs, meta_seed)):
        o["stack"] = sp
    # alpha-mix
    a = fit_alpha(blend_tr, stack_tr, y_tr)
    fitted["mix_alpha"] = float(a)
    mix_tr = a * blend_tr + (1 - a) * stack_tr
    for o in out:
        o["alpha_mix"] = a * o["blend"] + (1 - a) * o["stack"]
    # biases
    b1 = fit_bias_single(mix_tr, y_tr)
    b2 = fit_bias_boot(mix_tr, y_tr, meta_seed)
    b3 = fit_bias_boot(blend_tr, y_tr, meta_seed)
    fitted.update(bias_single=np.round(b1, 3).tolist(), bias_boot=np.round(b2, 3).tolist(),
                  blend_bias_boot=np.round(b3, 3).tolist())
    for o in out:
        o["bias_single"] = np.log(o["alpha_mix"] + EPS) + b1
        o["bias_boot"] = np.log(o["alpha_mix"] + EPS) + b2
        o["blend_bias_boot"] = np.log(o["blend"] + EPS) + b3
    return out, fitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--meta-seed", type=int, default=0)
    ap.add_argument("--no-imgstats", action="store_true")
    ap.add_argument("--quick", action="store_true", help="code-path test only: tiny restarts/rounds/bootstraps")
    a = ap.parse_args()
    if a.quick:
        global QUICK
        QUICK = True
    t0 = time.time()
    OUTD.mkdir(parents=True, exist_ok=True)
    members = CONFIGS[a.config]
    if BUNDLE:
        folds = pd.read_csv(BUNDLE / "folds.csv")
        test = pd.read_csv(BUNDLE / "test_ids.csv")
        z = np.load(BUNDLE / "textfeat.npz")
        tf_tr, tf_te = z["train"], z["test"]
    else:
        folds = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
        test = pd.read_csv(ROOT / "contest/Disaster_test.csv", encoding="utf-8-sig")  # captions only, no labels
        tf_tr = np.array([SF.text_feats(x) for x in folds["context"]], np.float32)
        tf_te = np.array([SF.text_feats(x) for x in test["context"]], np.float32)
    y, fold = folds["label"].values.astype(int), folds["fold"].values.astype(int)
    O, T, src = load_members(members)
    # Text side-features: the competition stacker used 21 (8 generic statistics + 5 emoji flags + 8 class-keyword
    # counts). The emoji flags were selected for 100% precision on the full labelled training set and the keyword
    # lexicons were hand-built from it, so the revised configs keep only the 8 generic statistics (DECISION_LOG D16);
    # the 'published' replica keeps all 21.
    text_set = "full" if a.config == "published" else "generic"
    if text_set == "generic":
        tf_tr, tf_te = tf_tr[:, :8], tf_te[:, :8]
    im_tr = im_te = None
    im_src = None
    # The 16 image statistics nearly fingerprint an image; with 851 train rows that have an exact (pHash-0)
    # cross-fold twin, the stacker could carry a twin's label across folds. They are used only in the
    # 'published' replica (as in the competition stacker), not in the revised configs (DECISION_LOG D19).
    if a.config != "published":
        a.no_imgstats = True
    if not a.no_imgstats:
        st, im_src = image_stats()
        if st is None:
            sys.exit("image stats (K1) not available; rerun with --no-imgstats or wait for K1")
        im_tr = st.loc[folds["image_id"]].values.astype(np.float32)
        im_te = st.loc[test["image_id"]].values.astype(np.float32)
    STAGES = None
    nested = {}
    fitted_all = {}
    for k in range(5):
        tr, ev = fold != k, fold == k
        res, fitted = run_procedure({m: O[m][tr] for m in members}, y[tr], fold[tr],
                                    (tf_tr[tr], None if im_tr is None else im_tr[tr]),
                                    [({m: O[m][ev] for m in members}, (tf_tr[ev], None if im_tr is None else im_tr[ev]))],
                                    members, a.meta_seed)
        fitted_all[f"outer{k}"] = fitted
        STAGES = list(res[0])
        for s in STAGES:
            nested.setdefault(s, np.zeros((len(y), K)))
            nested[s][ev] = res[0][s]
        print(f"  outer fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    # in-sample (as in the submitted paper): fit on all rows, score on the same rows; stack via 5-fold CV;
    # and the final refit applied to the 1,580 test rows (scores saved, not evaluated here)
    res, fitted = run_procedure(O, y, fold, (tf_tr, im_tr),
                                [(O, (tf_tr, im_tr)), (T, (tf_te, im_te))], members, a.meta_seed)
    fitted_all["full"] = fitted
    insample, test_scores = res
    # in-sample stack must be the CV stack (as originally), not the full-fit stack predicting its own rows
    X_all = stack_matrix(O, tf_tr, im_tr)
    st_cv = np.zeros((len(y), K))
    for f in range(5):
        st_cv[fold == f] = lgb_fit_predict(X_all[fold != f], y[fold != f], [X_all[fold == f]], a.meta_seed)[0]
    w = np.array([fitted["blend_w"][m] for m in members])
    blend_in = sum(wi * O[m] for wi, m in zip(w, members))
    insample["stack"] = st_cv
    insample["alpha_mix"] = fitted["mix_alpha"] * blend_in + (1 - fitted["mix_alpha"]) * st_cv
    insample["bias_single"] = np.log(insample["alpha_mix"] + EPS) + np.array(fitted["bias_single"])
    insample["bias_boot"] = np.log(insample["alpha_mix"] + EPS) + np.array(fitted["bias_boot"])

    rows = []
    for m in members:
        rows.append(dict(config=a.config, meta_seed=a.meta_seed, stage=f"member:{m}",
                         nested_f1=fast_f1(y, O[m].argmax(1)), nested_err=int((O[m].argmax(1) != y).sum()),
                         insample_f1=np.nan, insample_err=np.nan))
    for s in STAGES:
        pn, pi = nested[s].argmax(1), insample[s].argmax(1)
        rows.append(dict(config=a.config, meta_seed=a.meta_seed, stage=s, nested_f1=fast_f1(y, pn),
                         nested_err=int((pn != y).sum()), insample_f1=fast_f1(y, pi), insample_err=int((pi != y).sum())))
    df = pd.DataFrame(rows)
    tag = f"{a.config}_m{a.meta_seed}" + ("_QUICK" if a.quick else "")
    df.to_csv(OUTD / f"summary_{tag}.csv", index=False)
    np.savez_compressed(OUTD / f"preds_{tag}.npz",
                        **{f"nested__{s}": nested[s].astype(np.float32) for s in STAGES},
                        **{f"insample__{s}": insample[s].astype(np.float32) for s in STAGES},
                        **{f"test__{s}": test_scores[s].astype(np.float32) for s in STAGES})
    json.dump({"config": a.config, "members": members, "sources": src, "image_stats": im_src, "text_features": text_set,
               "meta_seed": a.meta_seed, "fitted": fitted_all, "seconds": round(time.time() - t0)},
              open(OUTD / f"fitted_{tag}.json", "w"), indent=1)
    pd.set_option("display.width", 160)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"done in {time.time() - t0:.0f}s -> {OUTD}")


if __name__ == "__main__":
    main()
