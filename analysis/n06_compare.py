"""N06: nested-OOF tables and the pre-registered comparisons (DECISION_LOG D8, D18, D20, D22).

Inputs: analysis/out/n01/preds_{config}_m{seed}.npz (meta-seeds 0-4, collected by n01_collect.py),
        analysis/out/n02/preds.npz (mechanism arms T, LF, J1, JP; seeds 0-2), analysis/out/dedup_masks.npz.
Writes analysis/out/n06/:
  stage_table.csv   config x stage: nested F1/errors mean, sd over meta-seeds (all rows and dedup subset),
                    in-sample F1/errors mean, optimism = in-sample - nested
  primary.csv       the 4 pre-registered comparisons (D20), all rows and dedup subset, seed-aware CI, Holm, resolved
  exploratory.csv   other stage-vs-stage comparisons over meta-seeds (no multiplicity adjustment; labelled exploratory)
  optimism.csv      in-sample vs nested per stage, over meta-seeds, with seed-aware CI
Test-partition predictions are never read here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from n03_stats import Bootstrapper, holm  # noqa: E402
from n01_nested_meta import CONFIGS, ORIG, OOF, fast_f1  # noqa: E402

IN = ROOT / "analysis/out/n01"
N02 = ROOT / "analysis/out/n02/preds.npz"
OUTD = ROOT / "analysis/out/n06"
META_SEEDS = range(5)


def meta_files(cfg):
    return {s: IN / f"preds_{cfg}_m{s}.npz" for s in META_SEEDS if (IN / f"preds_{cfg}_m{s}.npz").exists()}


def mech_complete():
    """D20/D23: the matched comparison needs T, LF and J1 (G1 and G2) for every training seed 0-2."""
    if not N02.exists():
        return False
    keys = set(np.load(N02).files)
    return all(f"oof__{a}__s{s}" in keys for a in ("T", "LF", "J1") for s in range(3))


# D22/D23: P is the primary pipeline only when G1 and G2 seeds 0-2 are in and all five P meta-seeds are in;
# otherwise clean3 (fallback, no fusion-mechanism claim)
MECH_OK = mech_complete()
PRIMARY = "P" if MECH_OK and len(meta_files("P")) == len(META_SEEDS) else "clean3"


def stage_names(cfg):
    return [k[len("nested__"):] for k in np.load(next(iter(meta_files(cfg).values()))).files if k.startswith("nested__")]


def stage_preds(cfg, stage, kind="nested"):
    out = []
    for s, f in meta_files(cfg).items():
        z = np.load(f)
        if f"{kind}__{stage}" in z.files:
            out.append(z[f"{kind}__{stage}"].argmax(1))
    return out


def member_pred(m):
    if m in ORIG:
        return np.load(OOF / f"{ORIG[m]}_oof.npy").argmax(1)
    from n01_nested_meta import find_new_run
    return find_new_run(m)[0].argmax(1)


def sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    y = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv").label.values.astype(int)
    keep = np.load(ROOT / "analysis/out/dedup_masks.npz")["train_keep"]
    cfgs = list(dict.fromkeys(c for c in ("published", "clean3", PRIMARY) if meta_files(c)))
    print("primary pipeline:", PRIMARY)
    if not cfgs:
        sys.exit("no n01 predictions collected yet (run n01_collect.py)")

    # ---- stage table (D22: mean over meta-seeds is the headline)
    rows = []
    for cfg in cfgs:
        for m in CONFIGS[cfg]:
            p = member_pred(m)
            rows.append(dict(config=cfg, stage=f"member:{m}", n_meta_seeds=1, nested_f1_mean=fast_f1(y, p),
                             nested_err_mean=int((p != y).sum()), dedup_f1_mean=fast_f1(y[keep], p[keep]),
                             dedup_err_mean=int((p[keep] != y[keep]).sum())))
        for st in stage_names(cfg):
            nest, ins = stage_preds(cfg, st), stage_preds(cfg, st, "insample")
            f = [fast_f1(y, p) for p in nest]
            e = [int((p != y).sum()) for p in nest]
            fd = [fast_f1(y[keep], p[keep]) for p in nest]
            ed = [int((p[keep] != y[keep]).sum()) for p in nest]
            fi = [fast_f1(y, p) for p in ins]
            ei = [int((p != y).sum()) for p in ins]
            rows.append(dict(config=cfg, stage=st, n_meta_seeds=len(nest),
                             nested_f1_mean=np.mean(f), nested_f1_sd=sd(f), nested_err_mean=np.mean(e), nested_err_sd=sd(e),
                             nested_err_min=min(e), nested_err_max=max(e),
                             dedup_f1_mean=np.mean(fd), dedup_err_mean=np.mean(ed),
                             insample_f1_mean=np.mean(fi), insample_err_mean=np.mean(ei),
                             optimism_f1=np.mean(fi) - np.mean(f), optimism_err=np.mean(e) - np.mean(ei)))
    st = pd.DataFrame(rows)
    st.to_csv(OUTD / "stage_table.csv", index=False)
    print(st.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # ---- primary family (D20)
    fam = []
    if MECH_OK:
        z = np.load(N02)
        seeds = sorted({int(k.split("__s")[1]) for k in z.files if k.startswith("oof__")})
        have = {a for a in ("T", "LF", "J1", "JP") if all(f"oof__{a}__s{s}" in z.files for s in seeds)}
        mt = []
        for a_ in ("T", "LF", "J1", "JP"):
            if a_ not in have:
                continue
            ps = [z[f"oof__{a_}__s{s}"] for s in seeds]
            f = [fast_f1(y, p) for p in ps]
            e = [int((p != y).sum()) for p in ps]
            fd = [fast_f1(y[keep], p[keep]) for p in ps]
            ed = [int((p[keep] != y[keep]).sum()) for p in ps]
            mt.append(dict(arm=a_, n_seeds=len(ps), f1_mean=np.mean(f), f1_sd=sd(f), err_mean=np.mean(e), err_sd=sd(e),
                           err_per_seed=e, dedup_f1_mean=np.mean(fd), dedup_err_mean=np.mean(ed)))
        pd.DataFrame(mt).to_csv(OUTD / "mech_table.csv", index=False)
        print(pd.DataFrame(mt).to_string(index=False))
        for a_, b_, lab in [("LF", "T", "(1) late fusion vs text only"),
                            ("J1", "LF", "(2) single-token joint head vs late fusion"),
                            ("JP", "LF", "(3) patch cross-attention vs late fusion")]:
            if {a_, b_} <= have:
                fam.append((lab, f"M:{a_}", f"M:{b_}", [z[f"oof__{a_}__s{s}"] for s in seeds],
                            [z[f"oof__{b_}__s{s}"] for s in seeds]))
    else:
        print("n02 mechanism predictions for seeds 0-2 not complete -> comparisons (1)-(3) skipped (D23)")
    if PRIMARY in cfgs:
        fam.append((f"(4) {PRIMARY} full pipeline (bootstrap bias) vs {PRIMARY} late fusion", f"{PRIMARY}:bias_boot",
                    f"{PRIMARY}:late_fusion", stage_preds(PRIMARY, "bias_boot"), stage_preds(PRIMARY, "late_fusion")))
    out = []
    for subset, mask in (("all_6323", np.ones(len(y), bool)), ("dedup_5321", keep)):
        bs = Bootstrapper(y[mask])
        rs = []
        for lab, a, b, pa, pb in fam:
            r = bs.compare_seeds([p[mask] for p in pa], [p[mask] for p in pb])
            rs.append(dict(subset=subset, comparison=lab, a=a, b=b, **r))
        if rs:
            for r, h in zip(rs, holm([r["p"] for r in rs])):  # D25: Satterthwaite-t p
                r["p_holm"] = float(h)
                r["resolved_holm"] = bool(h < 0.05 and r["same_sign"])
        out += rs
    prim = pd.DataFrame(out)
    prim.to_csv(OUTD / "primary.csv", index=False)
    if len(prim):
        print(prim[["subset", "comparison", "n_seeds", "d_f1_mean", "ci_lo", "ci_hi", "d_f1_per_seed", "d_err_mean",
                    "df_satt", "p", "p_holm", "same_sign", "resolved_holm"]].to_string(index=False))

    # ---- exploratory (no adjustment), over meta-seeds; all rows and the D18 dedup subset
    bs = Bootstrapper(y)
    bsd = Bootstrapper(y[keep])
    ex = []
    for cfg in [c for c in dict.fromkeys((PRIMARY, "clean3", "published")) if c in cfgs]:
        n = len(meta_files(cfg))
        pairs = [("member:muril_L", "member:eva02_L", "text vs image (best unimodal members)"),
                 ("late_fusion", "member:muril_L", "late fusion vs best unimodal"),
                 ("blend", "late_fusion", "learned blend vs late fusion"),
                 ("blend", "uniform", "learned blend vs uniform mean"),
                 ("stack", "blend", "stack vs blend"),
                 ("alpha_mix", "blend", "alpha-mix with stack vs blend"),
                 ("bias_boot", "alpha_mix", "bootstrap bias vs alpha-mix"),
                 ("bias_single", "bias_boot", "single-shot vs bootstrap bias"),
                 ("bias_boot", "blend_bias_boot", "full pipeline vs blend + bias"),
                 ("bias_boot", "late_fusion", "full pipeline vs late fusion")]
        if "g2_s0" in CONFIGS[cfg]:
            pairs.append(("member:g2_s0", "late_fusion", "G2 member vs late fusion"))
        for ka, kb, lab in pairs:
            pa = [member_pred(ka[7:])] * n if ka.startswith("member:") else stage_preds(cfg, ka)
            pb = [member_pred(kb[7:])] * n if kb.startswith("member:") else stage_preds(cfg, kb)
            if pa and pb and len(pa) == len(pb):
                ex.append(dict(subset="all_6323", config=cfg, comparison=lab, a=ka, b=kb, **bs.compare_seeds(pa, pb)))
                ex.append(dict(subset="dedup_5321", config=cfg, comparison=lab, a=ka, b=kb,
                               **bsd.compare_seeds([p[keep] for p in pa], [p[keep] for p in pb])))
    if "clean3" in cfgs and PRIMARY in cfgs and PRIMARY != "clean3":
        ex.append(dict(subset="all_6323", config="P vs clean3", comparison="adding the G2 member", a="P:bias_boot",
                       b="clean3:bias_boot",
                       **bs.compare_seeds(stage_preds(PRIMARY, "bias_boot"), stage_preds("clean3", "bias_boot"))))
    exd = pd.DataFrame(ex)
    if len(exd):
        exd["label"] = "exploratory (no multiplicity adjustment)"
        print(exd[["subset", "config", "comparison", "d_f1_mean", "ci_lo", "ci_hi", "d_f1_per_seed", "d_err_mean",
                   "resolved"]].to_string(index=False))
    exd.to_csv(OUTD / "exploratory.csv", index=False)

    # ---- optimism (in-sample vs nested), over meta-seeds
    opt = []
    for cfg in cfgs:
        for s_ in stage_names(cfg):
            a, b = stage_preds(cfg, s_, "insample"), stage_preds(cfg, s_)
            if a and len(a) == len(b):
                opt.append(dict(config=cfg, stage=s_, **bs.compare_seeds(a, b)))
    pd.DataFrame(opt).to_csv(OUTD / "optimism.csv", index=False)
    if opt:
        print(pd.DataFrame(opt)[["config", "stage", "d_f1_mean", "ci_lo", "ci_hi", "d_err_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
