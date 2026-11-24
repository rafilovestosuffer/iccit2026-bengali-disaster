"""Independent check of revision/generated/numbers.json: recompute each number from the raw prediction arrays with
sklearn (not the fast_f1 / CSV path used by n06 and make_paper_numbers) and compare. Exit code 1 on any mismatch.
Exceptions, stated honestly: the duplicate counts (n04/report.json), token-length counts (n08) and the bootstrap
p-values/intervals (n06/primary.csv) are TRANSCRIPTION checks against the same files the generator reads (the bootstrap
is not re-run here); their means and signs are recomputed where possible. Static audit numbers in the response are
checked against the archived audit outputs (analysis/verify_tmp/meta_audit).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "analysis/out"
num = json.loads((ROOT / "revision/generated/numbers.json").read_text(encoding="utf-8"))
y = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv").label.values.astype(int)
keep = np.load(O / "dedup_masks.npz")["train_keep"]
F = lambda yt, p: f1_score(yt, p, average="macro")  # noqa: E731
bad, checked = [], 0


def first_float(s):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s.replace("{,}", ""))
    return float(m.group()) if m else None


def check(key, value, tol):
    global checked
    if key not in num:
        return
    checked += 1
    got = first_float(num[key])
    if got is None or abs(got - value) > tol:
        bad.append(f"{key}: paper {num[key]!r} vs recomputed {value:.6f}")


ORIG = {"Bb": "text_bbB_s123_20260509_0550", "Muril": "text_murilL_20260509_0210", "Eva": "img_eva02L_448_20260511_0327"}
for k, f in ORIG.items():
    p = np.load(ROOT / f"drive_oof/x/oof/{f}_oof.npy").argmax(1)
    check(f"{k}Fone", F(y, p), 6e-6)
    check(f"{k}Err", (p != y).sum(), 0)
    check(f"{k}DdFone", F(y[keep], p[keep]), 6e-6)

cfg = num.get("PrimaryCfg", "clean3")
files = sorted((O / "n01").glob(f"preds_{cfg}_m[0-4].npz"))
for key, stage in (("Uni", "uniform"), ("LF", "late_fusion"), ("Blend", "blend"), ("Stack", "stack"), ("Mix", "alpha_mix"),
                   ("Bs", "bias_single"), ("Bb", "bias_boot")):
    if not files:
        break
    ps = [np.load(f)[f"nested__{stage}"].argmax(1) for f in files]
    pi = [np.load(f)[f"insample__{stage}"].argmax(1) for f in files]
    check(f"P{key}Fone", np.mean([F(y, p) for p in ps]), 6e-6)
    check(f"P{key}Err", np.mean([(p != y).sum() for p in ps]), 0.051)
    check(f"P{key}InFone", np.mean([F(y, p) for p in pi]), 6e-6)
    check(f"P{key}DdFone", np.mean([F(y[keep], p[keep]) for p in ps]), 6e-6)

if files:  # per-class recall of the full pipeline (mean over meta-seeds), lowest class
    from sklearn.metrics import recall_score
    ps = [np.load(f)["nested__bias_boot"].argmax(1) for f in files]
    rec = np.mean([recall_score(y, p, average=None) for p in ps], axis=0)
    check("PcMinRecall", rec.min(), 6e-5)
    recd = np.mean([recall_score(y[keep], p[keep], average=None) for p in ps], axis=0)
    check("PcMinRecallDd", recd.min(), 6e-5)
    # both are quoted as "at least": the printed value must not exceed the true minimum
    for key, true_min in (("PcMinRecall", rec.min()), ("PcMinRecallDd", recd.min())):
        if key in num and first_float(num[key]) > true_min + 1e-12:
            bad.append(f"{key}: printed lower bound {num[key]} exceeds the true minimum {true_min:.6f}")

if cfg == "P" and "JoneSzFone" in num:  # J1 (seed 0), a member of P, from its raw OOF file
    pj = np.load(ROOT / "hf/iccit2845-meta-inputs/new/g2_s0_oof.npy").argmax(1)
    check("JoneSzFone", F(y, pj), 6e-6)
    check("JoneSzErr", (pj != y).sum(), 0)
    check("JoneSzDdFone", F(y[keep], pj[keep]), 6e-6)

# ---- data-level numbers added after the 2026-11-18 review (independent recomputation)
WORDNUM = {"none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}


def check_word(key, value):
    global checked
    if key not in num:
        return
    checked += 1
    got = WORDNUM.get(num[key], first_float(num[key]))
    if got != value:
        bad.append(f"{key}: paper {num[key]!r} vs recomputed {value}")


six = {"bb_base": "text_bbB_s123_20260509_0550", "bb_multi": "text_bbmulti_20260509_0837",
       "muril_L": "text_murilL_20260509_0210", "eva02_L": "img_eva02L_448_20260511_0327",
       "mm_old": "mm_eva_muril_20260512_1041", "pl_muril": "pl_muril_20260512_1725"}
P6 = np.stack([np.load(ROOT / f"drive_oof/x/oof/{v}_oof.npy").argmax(1) for v in six.values()])
wrong_all = (P6 != y).all(0)
check_word("AllSixWrong", int(wrong_all.sum()))
check_word("AllSixSameWrong", int((wrong_all & (P6 == P6[0]).all(0)).sum()))
share = pd.Series(y).value_counts(normalize=True) * 100
check("ClassShareMin", share.min(), 0.051)
check("ClassShareMax", share.max(), 0.051)
rep = json.loads((O / "n04/report.json").read_text())
check("CapGroupsCross", rep["caption"]["rows_with_crossfold_caption_dup"] / 2, 0)  # all cross-fold groups are pairs
check("TestCapInTrain", rep["caption"]["test_rows_caption_in_train"], 0)
tl = json.loads((O / "n08/token_lengths.json").read_text()) if (O / "n08/token_lengths.json").exists() else None
if tl:
    check_word("BbTruncTrain", tl["banglabert_base_train"]["over"])
    check_word("MurilTruncTrain", tl["muril_large_train"]["over"])
dump = (ROOT / "analysis/verify_tmp/meta_audit/final_nb_dump.txt").read_text(encoding="utf-8", errors="replace")
wblock = dump[dump.find("Weights:"):]
mw = re.search(r"^.*pl_muril\s*=\s*([0-9.]+)", wblock[wblock.find("\n", wblock.find("pl_muril") - 200):], re.M)
if mw:
    check("PlBlendW", float(mw.group(1)), 0)
# (InGainErr is checked in the block added after the third review)
# static numbers in main.tex, recomputed: caption length statistics, class counts, the original stacker's Optuna trials
TEX = (ROOT / "revision/src_revised/main.tex").read_text(encoding="utf-8")
L = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv").context.astype(str).str.len()
_lmax = f"{L.max():,}".replace(",", "{,}")
_cap = f"${L.mean():.1f}$ characters (median ${L.median():.0f}$, maximum\n${_lmax}$"
if "Captions average" in TEX:  # the caption statistics were cut for space on 2026-11-19 (D35)
    checked += 1
    if _cap not in TEX:
        bad.append(f"caption statistics: expected '{_cap}' in main.tex")
_cnt = pd.Series(y).value_counts()
_wf = int(_cnt.min())
_rest = _cnt.drop(_cnt.idxmin())
checked += 1
if not (f"${_rest.min()}$--${_rest.max()}$ training instances" in TEX and f"class at ${_wf}$" in TEX):
    bad.append(f"class counts: {_rest.min()}-{_rest.max()} and {_wf} not as printed in Section III")
_nt = re.search(r"n_trials=(\d+)", (ROOT / "analysis/verify_tmp/meta_audit/mlc2_dump.txt").read_text(
    encoding="utf-8", errors="replace"))
checked += 1
if not (_nt and f"${_nt.group(1)}$ Optuna trials" in TEX):
    bad.append("Optuna trial count in IV-A does not match the archived stacker notebook")

pub = sorted((O / "n01").glob("preds_published_m[0-4].npz"))
if pub:
    ps = [np.load(f)["nested__bias_boot"].argmax(1) for f in pub]
    check("SubBbFone", np.mean([F(y, p) for p in ps]), 6e-6)

if (O / "n02/preds.npz").exists():
    z = np.load(O / "n02/preds.npz")
    for key, arm in (("MT", "T"), ("MLF", "LF"), ("MJone", "J1"), ("MJP", "JP")):
        ks = sorted(k for k in z.files if k.startswith(f"oof__{arm}__s"))
        if ks:
            check(f"{key}Fone", np.mean([F(y, z[k]) for k in ks]), 6e-6)
            check(f"{key}DdFone", np.mean([F(y[keep], z[k][keep]) for k in ks]), 6e-6)

if (O / "n05/test_preds.npz").exists():
    z = np.load(O / "n05/test_preds.npz")
    idx = json.loads((O / "n05/test_preds_index.json").read_text())
    ps = [z[f"p{i}"] for i, k in enumerate(idx) if k.startswith(f"{cfg}:bias_boot|")]
    if ps:
        check("TestPFone", np.mean([F(z["y"], p) for p in ps]), 6e-6)

# ---- added after the third review (2026-11-18): every printed count, and the facts that select generated wording
def flag(cond, msg):
    global checked
    checked += 1
    if not cond:
        bad.append(msg)


for k, f in ORIG.items():  # dedup error counts of the original branches
    p = np.load(ROOT / f"drive_oof/x/oof/{f}_oof.npy").argmax(1)
    check(f"{k}DdErr", (p[keep] != y[keep]).sum(), 0)
check("DedupN", int(keep.sum()), 0)
check("DedupTestN", int(np.load(O / "dedup_masks.npz")["test_keep"].sum()), 0)
check("DupTrainCross", rep["image_phash_le_4"]["rows_with_crossfold_image_dup"], 0)
check("DupTestTrain", rep["image_phash_le_4"]["test_rows_with_train_image_dup"], 0)
check("EvaAccTwin", 100 * rep["eva_acc_rows_with_crossfold_image_dup_le4"], 0.051)
check("EvaAccNoTwin", 100 * rep["eva_acc_rows_without"], 0.051)


def errs_of(cfg, stage):
    """Mean nested, in-sample and dedup-nested error counts and per-seed nested predictions over meta-seeds 0-4."""
    fs = sorted((O / "n01").glob(f"preds_{cfg}_m[0-4].npz"))
    if not fs:
        return None
    zs = [np.load(f) for f in fs]
    pn = [z[f"nested__{stage}"].argmax(1) for z in zs]
    pi = [z[f"insample__{stage}"].argmax(1) for z in zs]
    return dict(nested=np.mean([(p != y).sum() for p in pn]), insample=np.mean([(p != y).sum() for p in pi]),
                dedup=np.mean([(p[keep] != y[keep]).sum() for p in pn]), pn=pn, pi=pi)


E = {s: errs_of(cfg, s) for s in ("uniform", "late_fusion", "blend", "stack", "alpha_mix", "bias_single", "bias_boot")}
if all(v is not None for v in E.values()):
    for key, stage in (("Uni", "uniform"), ("LF", "late_fusion"), ("Blend", "blend"), ("Stack", "stack"),
                       ("Mix", "alpha_mix"), ("Bs", "bias_single"), ("Bb", "bias_boot")):
        check(f"P{key}InErr", E[stage]["insample"], 0.051)
        check(f"P{key}DdErr", E[stage]["dedup"], 0.051)
    check("OptimismErr", E["bias_boot"]["nested"] - E["bias_boot"]["insample"], 0.051)
    check("UniCostErr", E["uniform"]["nested"] - E["blend"]["nested"], 0.051)
    mu_e = (np.load(ROOT / f"drive_oof/x/oof/{ORIG['Muril']}_oof.npy").argmax(1) != y)
    check("LFErrRemovedPct", round(100 * (mu_e.sum() - E["late_fusion"]["nested"]) / mu_e.sum()), 0)
    check("LFErrRemovedPctDd", round(100 * (mu_e[keep].sum() - E["late_fusion"]["dedup"]) / mu_e[keep].sum()), 0)
    # 'largest' qualifier of UniCostText: same comparison set as the generator, recomputed from the raw arrays
    largest = True
    later = ("blend", "stack", "alpha_mix", "bias_single", "bias_boot")
    for col in ("nested", "insample"):  # uniform->blend against every pairwise step among the later stages
        u = E["uniform"][col] - E["blend"][col]
        other = max(abs(E[a][col] - E[b][col]) for i, a in enumerate(later) for b in later[i + 1:])
        largest &= bool(u > 0 and u > other)
    flag(num.get("UniCostLargest") == ("yes" if largest else "no"), f"UniCostLargest {num.get('UniCostLargest')} "
         f"vs recomputed {largest}")
    flag(("larger change than any step" in num.get("UniCostText", "")) <= largest,
         "UniCostText states the superlative but the recomputed gaps do not support it")
    # in-sample sentence of V-C: which stages appear to make fewer errors than LF_p in-sample
    low = [E[s]["insample"] < E["late_fusion"]["insample"] for s in ("blend", "stack", "alpha_mix", "bias_single",
                                                                     "bias_boot")]
    want = "every stage from the blend on" if all(low) else "the full pipeline," if low[-1] else "do not appear"
    flag(want in num.get("InSampleText", ""), f"InSampleText does not match the recomputed in-sample errors ({want})")
    check("InGainErr", E["late_fusion"]["insample"] - E["bias_boot"]["insample"], 0.051)
    # comparison (4), full pipeline minus LF_p: mean dF1 and mean error difference over meta-seeds, all and dedup rows
    d_f1 = np.mean([F(y, a) - F(y, b) for a, b in zip(E["bias_boot"]["pn"], E["late_fusion"]["pn"])])
    d_er = np.mean([(a != y).sum() - (b != y).sum() for a, b in zip(E["bias_boot"]["pn"], E["late_fusion"]["pn"])])
    check("CfD", d_f1, 6e-5)
    check("CfE", d_er, 0.051)
    # (the dedup clause of MetaText is checked below, in the form its verdict selects)
    # optimism direction from the raw arrays must match optimism.csv, and AbsOpt must follow optimism.csv
    ot = pd.read_csv(O / "n06/optimism.csv")
    ot = ot[(ot.config == cfg) & (ot.stage == "bias_boot")].iloc[0]
    raw_opt = np.mean([F(y, a) - F(y, b) for a, b in zip(E["bias_boot"]["pi"], E["bias_boot"]["pn"])])
    flag(np.sign(raw_opt) == np.sign(ot.d_f1_mean), "optimism sign: raw arrays vs optimism.csv")
    opt_res = bool(ot.resolved and ot.d_f1_mean > 0 and E["bias_boot"]["nested"] > E["bias_boot"]["insample"])
    flag(("understates" in num.get("AbsOpt", "")) == opt_res, "AbsOpt wording does not match optimism.csv")
# exploratory selectors: LFDdText and VAHead follow exploratory.csv; MetaText follows primary.csv
ex = pd.read_csv(O / "n06/exploratory.csv")


def ex_ok(comp, subset):
    q = ex[(ex.config == cfg) & (ex.comparison == comp) & (ex.subset == subset)]
    return bool(len(q) and q.iloc[0].resolved and q.iloc[0].d_f1_mean > 0)


if "LFDdText" in num:
    flag(("holds on the dedup rows" in num["LFDdText"]) == ex_ok("late fusion vs best unimodal", "dedup_5321"),
         "LFDdText does not match exploratory.csv")
if "VAHead" in num:
    flag(num["VAHead"].startswith("Text leads") == ex_ok("text vs image (best unimodal members)", "all_6323"),
         "VAHead (text half) does not match exploratory.csv")
pr = pd.read_csv(O / "n06/primary.csv")
q4 = pr[(pr.subset == "all_6323") & pr.comparison.str.startswith("(4)")]
if len(q4) and "MetaText" in num:
    flag(("is not distinguishable from" in num["MetaText"]) == (not q4.iloc[0].resolved_holm),
         "MetaText verdict does not match primary.csv comparison (4)")
# matched heads (MECH state): mean error counts from n02's per-seed predictions
if (O / "n02/preds.npz").exists():
    z = np.load(O / "n02/preds.npz")
    for key, arm in (("MT", "T"), ("MLF", "LF"), ("MJone", "J1"), ("MJP", "JP")):
        ks = sorted(k for k in z.files if k.startswith(f"oof__{arm}__s"))
        if ks:
            check(f"{key}Err", np.mean([(z[k] != y).sum() for k in ks]), 0.051)
# published (submitted) configuration through the nested procedure
S = errs_of("published", "bias_boot")
if S is not None:
    check("SubBbErr", S["nested"], 0.051)
    check("SubBbInErr", S["insample"], 0.051)
    check("SubOptimismErr", S["nested"] - S["insample"], 0.051)
    check("SubBbInFone", np.mean([F(y, p) for p in S["pi"]]), 6e-6)
    check("SubBbDdFone", np.mean([F(y[keep], p[keep]) for p in S["pn"]]), 6e-6)

# ---- static audit numbers in the response (response_draft.md), against the archived audit outputs
resp = (ROOT / "revision/response_draft.md").read_text(encoding="utf-8")
aud = ROOT / "analysis/verify_tmp/meta_audit"
a01 = (aud / "a01.out.txt").read_text(encoding="utf-8", errors="replace")
v01 = (aud / "v01.out.txt").read_text(encoding="utf-8", errors="replace")
n_aft = int(re.search(r"hazard\+por \(aftermath regex\) matches: (\d+)", a01).group(1))
m = re.search(r"aftermath regex hits on all 1580 test rows: (\d+) \| among kw rows: (\d+)", v01)
n_all, n_kw = int(m.group(1)), int(m.group(2))
flag(f"{n_aft} training captions; {n_kw} exclusions among keyword rows ({n_all} = matches over all 1,580 rows)"
     in resp, f"C6 vs audit: {n_aft} / {n_kw} / {n_all}")
m = re.search(r"A1_stable: gate passes=\d+ changes=(\d+) \| before rule test F1=[0-9.]+ err=\d+ \| after rule "
              r"F1=([0-9.]+) err=(\d+)", v01)
flag(f'the "{m.group(3)} errors" belong to a third run ({m.group(2)})' in resp, "C4 vs audit (A1_stable)")
n_single = int(re.search(r"A2_single: gate passes=\d+ changes=(\d+)", v01).group(1))
blk = v01[v01.find("A2_single: gate passes"):v01.find("A1_stable: gate passes")]
n_wrong, n_right = blk.count("rule change WRONG"), blk.count("rule change RIGHT")
blk1 = v01[v01.find("A1_stable: gate passes"):]
blk1 = blk1[:blk1.find("==")] if "==" in blk1 else blk1
flag((n_single, n_wrong, n_right) == (3, 1, 2) and int(m.group(1)) == 1 and blk1.count("rule change RIGHT") == 1
     and "changed three labels" in resp and "changed one label" in resp,
     f"R2.4 rule changes vs audit: single {n_single} ({n_wrong} wrong, {n_right} right), bootstrap {m.group(1)}")
five = int(((P6 != y).sum(0) >= 5).sum())
flag(f'the submitted "{five}" counted rows wrong in at least five branches' in resp, f"R3.5(b) count vs {five}")

# ---- added after the fourth review (2026-11-18): p-values, intervals, ranges, sentence gates, response tables
CLS = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv").groupby("label").category.first().to_dict()
MECH = bool(prim_any := len(pr[pr.comparison.str.startswith("(1)")]))


def prow(start, subset="all_6323"):
    q = pr[(pr.subset == subset) & pr.comparison.str.startswith(start)]
    return None if q.empty else q.iloc[0]


def pnum(p):  # the generator's printed p: three decimals, or "<0.001"
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def sg(x, fmt):
    return f"${x:{fmt}}$"


# pre-registered comparisons: printed D (mean and CI), E, P against primary.csv; means recomputed from predictions
ARMS = {"Co": ("LF", "T"), "Ct": ("J1", "LF"), "Cth": ("JP", "LF")}
z2 = np.load(O / "n02/preds.npz") if (O / "n02/preds.npz").exists() else None
for key, start in (("Co", "(1)"), ("Ct", "(2)"), ("Cth", "(3)"), ("Cf", "(4)")):
    r = prow(start)
    if r is None:
        continue
    if f"{key}D" in num:
        flag(num[f"{key}D"] == f"{sg(r.d_f1_mean, '+.4f')} [{sg(r.ci_lo, '+.4f')}, {sg(r.ci_hi, '+.4f')}]",
             f"{key}D {num[f'{key}D']!r} vs primary.csv")
    if f"{key}P" in num:
        flag(num[f"{key}P"] == (f"{r.p_holm:.3f}" if r.p_holm >= 0.001 else "$<$0.001"), f"{key}P vs primary.csv")
    if key in ARMS and z2 is not None:
        a_, b_ = ARMS[key]
        sd = sorted({k.split("__s")[1] for k in z2.files if k.startswith(f"oof__{a_}__s")})
        da = [F(y, z2[f"oof__{a_}__s{s}"]) - F(y, z2[f"oof__{b_}__s{s}"]) for s in sd]
        de = [(z2[f"oof__{a_}__s{s}"] != y).sum() - (z2[f"oof__{b_}__s{s}"] != y).sum() for s in sd]
        check(f"{key}D", np.mean(da), 6e-5)
        check(f"{key}E", np.mean(de), 0.051)
# abstract: the Holm p of comparison (4)
r4 = prow("(4)")
if r4 is not None and "AbsMeta" in num:
    _p = f"$p{{=}}{r4.p_holm:.3f}$" if r4.p_holm >= 0.001 else "$p{<}0.001$"
    if len(pr[pr.subset == "all_6323"]) > 1:
        flag(f"Holm-adjusted {_p}" in num["AbsMeta"], "AbsMeta Holm p vs primary.csv")
    else:  # a family of one: the p-value is not called Holm-adjusted
        flag(f"{_p}, the single pre-specified comparison" in num["AbsMeta"] and "Holm" not in num["AbsMeta"],
             "AbsMeta single-comparison p vs primary.csv")
    if E.get("bias_boot") is not None and "errors" in num["AbsMeta"]:  # the abstract's mean error difference of (4)
        _de = np.mean([(a != y).sum() - (b != y).sum() for a, b in zip(E["bias_boot"]["pn"], E["late_fusion"]["pn"])])
        _ab = num["AbsMeta"]
        flag(f"${_de:+.1f}$ errors" in _ab or f"{abs(_de):.1f} {'fewer' if _de < 0 else 'more'} errors" in _ab,
             f"AbsMeta error difference vs recomputed {_de:+.1f}")
# MetaText: fallback cites the interval of (4); dedup sentence form chosen by the dedup verdict of (4)
r4d = prow("(4)", "dedup_5321")
if r4 is not None and "MetaText" in num:
    if not MECH:
        flag(f"$\\Delta$F1 {num.get('CfD', '?')}" in num["MetaText"], "MetaText does not cite the interval of (4)")
    if r4d is not None and E.get("bias_boot") is not None:
        ddv = np.mean([(a[keep] != y[keep]).sum() - (b[keep] != y[keep]).sum()
                       for a, b in zip(E["bias_boot"]["pn"], E["late_fusion"]["pn"])])
        _pp = f"$p_H{{=}}{r4d.p_holm:.3f}$" if r4d.p_holm >= 0.001 else "$p_H{<}0.001$"
        cons = (r4d.d_f1_mean > 0 and r4d.d_err_mean < 0) or (r4d.d_f1_mean < 0 and r4d.d_err_mean > 0)
        want = (f"${ddv:+.1f}$ errors on average, {_pp}" if not r4d.resolved_holm else
                f"{abs(ddv):.1f} {'fewer' if r4d.d_f1_mean > 0 else 'more'} errors than" if cons else
                f"${ddv:+.1f}$ errors on average; resolved, {_pp}")
        flag(want in num["MetaText"], f"MetaText dedup clause: expected {want!r}")
# meta-seed ranges printed as 'mean (lo--hi)'
for key, stage in (("Uni", "uniform"), ("LF", "late_fusion"), ("Blend", "blend"), ("Stack", "stack"),
                   ("Mix", "alpha_mix"), ("Bs", "bias_single"), ("Bb", "bias_boot")):
    if E.get(stage) is not None and f"P{key}Err" in num:
        ev = [(p != y).sum() for p in E[stage]["pn"]]
        want = f"{np.mean(ev):g}" if min(ev) == max(ev) else f"{np.mean(ev):g} ({min(ev)}--{max(ev)})"
        flag(num[f"P{key}Err"] == want, f"P{key}Err {num[f'P{key}Err']!r} vs recomputed {want!r}")
if S is not None and "SubBbErr" in num:
    ev = [(p != y).sum() for p in S["pn"]]
    want = f"{np.mean(ev):g}" if min(ev) == max(ev) else f"{np.mean(ev):g} ({min(ev)}--{max(ev)})"
    flag(num["SubBbErr"] == want, f"SubBbErr {num['SubBbErr']!r} vs recomputed {want!r}")


# sentence gates (exactly as in the generator)
def ex_row(comp, subset="all_6323"):
    q = ex[(ex.config == cfg) & (ex.comparison == comp) & (ex.subset == subset)]
    return None if q.empty else q.iloc[0]


_lfa = ex_row("late fusion vs best unimodal")
_lfa_ok = bool(_lfa is not None and _lfa.resolved and _lfa.d_f1_mean > 0 and _lfa.d_err_mean < 0)
flag(("LFErrRemovedPct" in num) == _lfa_ok, "LFErrRemovedPct present without (or missing despite) a resolved gain")
_r1 = prow("(1)")
_img = bool(_lfa is not None and _lfa.resolved and _lfa.d_f1_mean > 0
            and (not MECH or (_r1.resolved_holm and _r1.d_f1_mean > 0)))
if "VAHead" in num:
    flag(("image is not redundant" in num["VAHead"]) == _img, "VAHead (image half) does not match its gate")
_ub = ex_row("learned blend vs uniform mean")
if _ub is not None and "UniCostText" in num and E.get("blend") is not None:
    raw = np.mean([F(y, a) - F(y, b) for a, b in zip(E["blend"]["pn"], E["uniform"]["pn"])])
    flag(np.sign(raw) == np.sign(_ub.d_f1_mean), "blend-vs-uniform sign: raw arrays vs exploratory.csv")
    t_ = num["UniCostText"]
    if not _ub.resolved:
        ok = "not distinguishable" in t_
    elif _ub.d_f1_mean > 0 and _ub.d_err_mean < 0:
        ok = "costs" in t_
    elif _ub.d_f1_mean < 0:
        ok = "uniform average scores higher" in t_
    else:
        ok = "learned blend scores higher" in t_
    flag(ok, "UniCostText variant does not match exploratory.csv")
if "the largest drop of the three" in TEX:
    drops = {}
    for k, f in ORIG.items():
        p = np.load(ROOT / f"drive_oof/x/oof/{f}_oof.npy").argmax(1)
        drops[k] = F(y, p) - F(y[keep], p[keep])
    flag(max(drops, key=drops.get) == "Eva", f"V-A 'largest drop of the three' is not EVA-02-L: {drops}")
if files and "PcMinClass" in num:
    from sklearn.metrics import recall_score
    rec = np.mean([recall_score(y, np.load(f)["nested__bias_boot"].argmax(1), average=None) for f in files], axis=0)
    flag(num["PcMinClass"] == CLS[int(rec.argmin())], f"PcMinClass {num['PcMinClass']} vs {CLS[int(rec.argmin())]}")

# response R2.3 table: one line per primary.csv row, in order; McNemar b / c recomputed for seed 0
if "RespPrimaryTable" in num:
    tl_ = num["RespPrimaryTable"].split("\n")[2:]
    flag(len(tl_) == len(pr), "RespPrimaryTable row count vs primary.csv")
    for line, (_, r) in zip(tl_, pr.iterrows()):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        hp = f"{r.p_holm:.3f}" if r.p_holm >= 0.001 else "< 0.001"
        flag(cells[2] == f"{r.d_f1_mean:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]" and cells[3] == f"{r.d_err_mean:+.1f}"
             and cells[6] == hp and cells[7] == ("yes" if r.resolved_holm else "no"),
             f"RespPrimaryTable row {r.comparison[:3]} {r.subset} vs primary.csv")
        m_ = keep if r.subset == "dedup_5321" else np.ones(len(y), bool)
        if str(r.comparison).startswith("(4)") and files:
            z0 = np.load(files[0])
            pa, pb = z0["nested__bias_boot"].argmax(1), z0["nested__late_fusion"].argmax(1)
        elif z2 is not None and str(r.comparison)[:3] in ("(1)", "(2)", "(3)"):
            a_, b_ = ARMS[{"(1)": "Co", "(2)": "Ct", "(3)": "Cth"}[str(r.comparison)[:3]]]
            pa, pb = z2[f"oof__{a_}__s0"], z2[f"oof__{b_}__s0"]
        else:
            continue
        b0 = int(((pa == y) & (pb != y) & m_).sum())
        c0 = int(((pa != y) & (pb == y) & m_).sum())
        flag(cells[8].startswith(f"{b0} / {c0},"), f"McNemar b/c for {r.comparison[:3]} {r.subset}: {cells[8]} vs {b0} / {c0}")

# response R2.4: class transitions and correctness of the rule changes, derived from the audit replay; gate threshold
_transition = r"(test_\d+): final ([A-Za-z ]+?) -> rule ([A-Za-z ]+?) \| true ([A-Za-z ]+?) \|.*?rule change (RIGHT|WRONG)"
_rows = re.findall(_transition, blk)
_rule_sentences = resp.split("**R2.4", 1)[1].split("## 2.", 1)[0].split(". ")
flag(len(_rows) == 3, "three replayed rule transitions")
# D40: the response names the public BanglaCalamityMMD records, never the contest file ids (anonymity, D29)
_pub = pd.read_csv(O / "n12/map_contest_to_bcmmd.csv").set_index("stem")
for row_id, before, after, truth, outcome in _rows:
    phrase = (f"corrected {before} predictions to the reference label {after}" if outcome == "RIGHT"
              else f"changed a correct {before} prediction to {after}")
    rec = _pub.loc[row_id]
    flag(truth == (after if outcome == "RIGHT" else before)
         and any(rec.public_id in sentence and f"public {rec.public_split} record" in sentence and phrase in sentence
                 for sentence in _rule_sentences),
         f"R2.4 row, transition and reference label for {row_id} ({rec.public_split}/{rec.public_id})")
_r1x = re.findall(_transition, blk1)
flag(len(_r1x) == 1 and _r1x[0] in _rows and _r1x[0][-1] == "RIGHT"
     and f"changed one label, correcting {_pub.loc[_r1x[0][0]].public_id}" in resp, "R2.4 bootstrap-bias change")
flag(not re.search(r"\btest_\d{4}\b", resp), "no contest file ids in the response")
_th = re.search(r"CONF_TH = ([0-9.]+)", dump)
flag(bool(_th) and f"(> {float(_th.group(1)):.2f})" in resp, "R2.4 gate threshold vs archived notebook")
# C5: the submitted '33' (rows wrong in at least five branches); bounded so '133' cannot match
flag(re.search(rf"(?<!\d){five} = rows wrong in at least five", resp) is not None, f"C5 count vs {five}")

# ---- 1,580-row partition (after n05): every printed number recomputed from the saved predictions
if (O / "n05/test_preds.npz").exists():
    zt = np.load(O / "n05/test_preds.npz")
    idx = json.loads((O / "n05/test_preds_index.json").read_text())
    yt, kt = zt["y"], zt["dedup_keep"].astype(bool)

    def tp(system):
        return [zt[f"p{i}"] for i, k in enumerate(idx) if k.startswith(f"{system}|")]
    pb = tp(f"{cfg}:bias_boot")
    if pb:
        ev = [int((p != yt).sum()) for p in pb]
        want = f"{np.mean(ev):g}" if min(ev) == max(ev) else f"{np.mean(ev):g} ({min(ev)}--{max(ev)})"
        flag(num.get("TestPErr") == want, f"TestPErr {num.get('TestPErr')!r} vs {want!r}")
        check("TestPDdFone", np.mean([F(yt[kt], p[kt]) for p in pb]), 6e-6)
    pl = tp(f"{cfg}:late_fusion")
    if pl:
        check("TestLFFone", np.mean([F(yt, p) for p in pl]), 6e-6)
        check("TestLFErr", np.mean([(p != yt).sum() for p in pl]), 0.051)
        note = num.get("TestDirNote", "")
        if pb:  # V-D: the pipeline clause is present exactly when it has fewer errors here and (4) is unresolved
            want_p = (np.mean([(p != yt).sum() for p in pb]) < np.mean([(p != yt).sum() for p in pl])
                      and not (r4 is not None and r4.resolved_holm))
            flag(("the full pipeline makes fewer errors than" in note) == want_p, "TestDirNote pipeline clause gate")
            check("TestPErrMean", np.mean([(p != yt).sum() for p in pb]), 0.051)
        # matched heads on the 1,580 rows: printed error means and the 'reverse of their out-of-fold order' clause
        arms_t = [a for a in ("T", "LF", "J1", "JP") if tp(f"M:{a}") and (a != "JP" or prow("(3)") is not None)]
        if MECH and {"T", "LF", "J1"} <= set(arms_t):
            te = {a: np.mean([(p != yt).sum() for p in tp(f"M:{a}")]) for a in arms_t}
            want_h = ", ".join(f"{te[a]:.1f}" for a in arms_t[:-1]) + f" and {te[arms_t[-1]]:.1f}"
            flag(num.get("TestHeadErr") == want_h, f"TestHeadErr {num.get('TestHeadErr')!r} vs {want_h!r}")
            joint = [a for a in ("J1", "JP") if a in arms_t]
            oof_more = all(
                np.mean([(z2[f"oof__{a}__s{s}"] != y).sum() for s in (0, 1, 2)]) >
                np.mean([(z2[f"oof__LF__s{s}"] != y).sum() for s in (0, 1, 2)]) for a in joint)
            rev_t = all(te[a] < te["LF"] for a in joint) and oof_more
            flag(("the reverse of their out-of-fold order" in note) == rev_t, "TestDirNote reversal clause gate")
    trs = pd.read_csv(O / "n05/test_results.csv")
    for line in num.get("RespTestTable", "").split("\n")[2:]:
        name = line.strip().strip("|").split("|")[0].strip()
        sysid = {"revised pipeline (full)": f"{cfg}:bias_boot", "LF_p": f"{cfg}:late_fusion",
                 "submitted configuration (transductive, leaky member)": "published:bias_boot"}.get(name, f"M:{name}")
        ps_ = tp(sysid)
        if ps_:
            flag(f"{np.mean([F(yt, p) for p in ps_]):.5f}" in line, f"RespTestTable {name} macro-F1")
    import hashlib  # noqa: E402
    h = hashlib.sha256((O / "n05/frozen_config.json").read_bytes()).hexdigest()
    flag(num.get("FrozenHash") == h[:16] and h in (ROOT / "DECISION_LOG.md").read_text(encoding="utf-8"),
         "FrozenHash vs sha256(frozen_config.json) and DECISION_LOG")

# ---- added after the reviewer simulation (2026-11-19): sentences on the late stages, LF_p and the text lead
check("DupTrainExact", rep["image_phash_le_0"]["rows_with_crossfold_image_dup"], 0)  # transcription (n04)
NAMES = {"bb_base": "BanglaBERT-base", "g2_s0": "J1 (seed 0)"}
fits = sorted((O / "n01").glob(f"fitted_{cfg}_m[0-4].json"))
if fits:
    fj = [json.loads(f.read_text())["fitted"] for f in fits]
    mem = list(fj[0]["full"]["blend_w"])
    extra = [NAMES.get(m_, m_) for m_ in mem if m_ not in ("muril_L", "eva02_L")]
    flag(num.get("ExtraMembers") == " and ".join(extra), f"ExtraMembers {num.get('ExtraMembers')!r} vs members {mem}")
    mix = [v["mix_alpha"] for j in fj for v in j.values()]
    lfw = [v["late_fusion_alpha"] for j in fj for v in j.values()]
    if E.get("stack") is not None:
        late_up = E["bias_boot"]["nested"] > E["stack"]["nested"] and E["alpha_mix"]["nested"] > E["stack"]["nested"]
        lt = num.get("LateText", "")
        flag(("do not reduce errors relative to the stacker alone" in lt) == late_up, "LateText gate vs nested errors")
        if late_up:
            flag(f"ranges from {min(mix):g} to {max(mix):g} across" in lt, f"LateText mix range vs {min(mix)}-{max(mix)}")
        stg = ("uniform", "late_fusion", "blend", "stack", "alpha_mix", "bias_single", "bias_boot")
        fewest = min(stg, key=lambda s_: E[s_]["nested"]) == "stack"
        sf = num.get("StackFewest", "")
        flag(bool(sf) == fewest, "StackFewest present without (or missing despite) the stacker having fewest errors")
        if sf:
            flag(f"two branches instead of {[k for k, v in WORDNUM.items() if v == len(mem)][0]}" in sf,
                 f"StackFewest branch count vs {len(mem)} members")
    # LF_p: a constant weight means identical nested and in-sample predictions (and the uniform mean has no weight)
    same = all(np.array_equal(a, b) for st_ in ("late_fusion", "uniform") if E.get(st_) is not None
               for a, b in zip(E[st_]["pn"], E[st_]["pi"]))
    note = num.get("LFAlphaNote", "")
    flag(bool(note) == (len(set(lfw)) == 1), "LFAlphaNote present without (or missing despite) a constant LF_p weight")
    if note:
        flag(f"$\\alpha{{=}}{lfw[0]:g}$" in note and same, f"LFAlphaNote: weight {set(lfw)}, predictions identical {same}")
# LF_p vs MuRIL-Large (V-A): mean dF1 recomputed; interval transcribed from exploratory.csv; McNemar recomputed (m0)
if _lfa is not None and "LFCI" in num and E.get("late_fusion") is not None:
    mu_p = np.load(ROOT / f"drive_oof/x/oof/{ORIG['Muril']}_oof.npy").argmax(1)
    dlf = np.mean([F(y, p) - F(y, mu_p) for p in E["late_fusion"]["pn"]])
    p0 = E["late_fusion"]["pn"][0]
    b0, c0 = int(((p0 == y) & (mu_p != y)).sum()), int(((p0 != y) & (mu_p == y)).sum())
    want = (f"$\\Delta$F1 {sg(dlf, '+.4f')} [{sg(_lfa.ci_lo, '+.4f')}, {sg(_lfa.ci_hi, '+.4f')}], "
            f"McNemar {b0}/{c0};")
    flag(want in num["LFCI"], f"LFCI {num['LFCI']!r} vs {want!r}")
# matched heads: 'in every seed' wording (abstract, conclusion, V-B) gated on per-seed error counts from n02
if z2 is not None and MECH:
    heads_em = []
    for key, a_ in (("Ct", "J1"), ("Cth", "JP")):
        r_ = prow({"Ct": "(2)", "Cth": "(3)"}[key])
        if r_ is None:
            continue
        sd_ = sorted({k.split("__s")[1] for k in z2.files if k.startswith(f"oof__{a_}__s")})
        if all((z2[f"oof__{a_}__s{s}"] != y).sum() > (z2[f"oof__LF__s{s}"] != y).sum() for s in sd_) \
                and not r_.resolved_holm:
            heads_em.append(a_)
    # D37: the abstract/conclusion state the head order as a mean error count only, and only if it holds against
    # both the pre-specified LF and the post-hoc half-fold LF control (n09)
    flag("in every seed" not in num.get("AbsMech", ""), "AbsMech must not claim a per-seed order (D37)")
    _ctl = pd.read_csv(O / "n09/lf_controls.csv") if (O / "n09/lf_controls.csv").exists() else None
    _hm = {a_: np.mean([(z2[k] != y).sum() for k in z2.files if k.startswith(f"oof__{a_}__s")]) for a_ in ("LF", "J1", "JP")
           if any(k.startswith(f"oof__{a_}__s") for k in z2.files)}
    _cm = None if _ctl is None else _ctl[(_ctl.arm == "LF_halffold") & (_ctl.subset == "all_6323")].err.mean()
    _on_avg = all(_hm[h] > _hm["LF"] and (_cm is None or _hm[h] > _cm) for h in ("J1", "JP") if h in _hm) \
        and not any(prow(c) is not None and prow(c).resolved_holm for c in ("(2)", "(3)"))
    flag(("more errors on average" in num.get("AbsMech", "")) == _on_avg, "AbsMech 'on average' gate (D37)")
    flag(("in every seed" in num.get("MechText", "")) == bool(heads_em), "MechText 'in every seed' gate")
    if heads_em:
        flag(f"LF makes fewer errors than {' and '.join(heads_em)} in every seed" in num.get("MechText", ""),
             "MechText per-seed head list")
# text lead (V-A): EVA-02-L vs BanglaBERT-base order on all rows and on the dedup rows
_pe = np.load(ROOT / f"drive_oof/x/oof/{ORIG['Eva']}_oof.npy").argmax(1)
_pbb = np.load(ROOT / f"drive_oof/x/oof/{ORIG['Bb']}_oof.npy").argmax(1)
_o_all = (_pe != y).sum() < (_pbb != y).sum()
_o_dd = F(y[keep], _pe[keep]) > F(y[keep], _pbb[keep])
flag((num.get("TextOrder", "") == "") == (_o_all == _o_dd), "TextOrder present without (or missing despite) a reversal")

# ---- D37 post-hoc controls (n09), J1-seed sensitivity, image copies (n10), Table IV protocol, response pointers
if MECH and (O / "n09/lf_controls.csv").exists():
    c9 = pd.read_csv(O / "n09/lf_controls.csv")
    t9 = c9[c9.subset == "all_6323"].pivot(index="arm", columns="seed", values="err")
    sd9 = list(t9.columns)
    # the n09 table's T/LF/J1/JP rows must equal n02's predictions; LF_fixed05 is recomputed from the raw inputs
    m2 = json.loads((O / "n02/meta.json").read_text(encoding="utf-8"))
    for s in sd9:
        for arm in ("T", "LF", "J1", "JP"):
            if f"oof__{arm}__s{s}" in z2.files:
                flag(int(t9.loc[arm, s]) == int((z2[f"oof__{arm}__s{s}"] != y).sum()), f"n09 {arm} s{s} vs n02")
        Tp = np.load(Path(m2["sources"][f"g1_s{s}"]) / f"g1_s{s}_oof.npy")
        fx = (0.5 * Tp + 0.5 * z2["img_oof"]).argmax(1)
        flag(int(t9.loc["LF_fixed05", s]) == int((fx != y).sum()), f"n09 LF_fixed05 s{s} recomputed")
    ctl = [int(t9.loc["LF_halffold", s]) for s in sd9]
    flag(num.get("LFCtlErr") == ", ".join(map(str, ctl[:-1])) + " and " + str(ctl[-1]), "LFCtlErr vs n09")
    worse = [s for s in sd9 if all(t9.loc["LF_halffold", s] > t9.loc[h, s] for h in ("J1", "JP") if h in t9.index)]
    flag((f"in seed {worse[0]}," in num.get("MechText", "")) == (len(worse) == 1), "MechText control seed clause")
    lst = lambda a: ", ".join(str(int(t9.loc[a, s])) for s in sd9)  # noqa: E731
    for a in ("LF_halffold", "LF", "J1", "JP", "T", "LF_fixed05"):
        flag(lst(a) in num.get("RespLFCtl", ""), f"RespLFCtl {a} list")
    a9 = pd.read_csv(O / "n09/alphas.csv")
    flag(f"{int(a9.n_fit.min())}-{int(a9.n_fit.max())} rows" in num.get("RespLFCtl", ""), "RespLFCtl n_fit range")
if cfg == "P":
    _je = {}
    for c_ in ("P", "P_s1", "P_s2"):
        fs = [O / f"n01/preds_{c_}_m{m}.npz" for m in range(5)]
        if all(f.exists() for f in fs):
            _je[c_] = np.mean([(np.load(f)["nested__bias_boot"].argmax(1) != y).sum() for f in fs])
    if len(_je) == 3:
        for c_ in ("P_s1", "P_s2", "P"):
            flag(f"{_je[c_]:.1f}" in num.get("RespJ1Seed", ""), f"RespJ1Seed {c_} mean errors")
_icf = O / "n10/image_copies.json"
if _icf.exists():
    ic = json.loads(_icf.read_text(encoding="utf-8"))
    flag(ic["working"]["cross_fold_le4"] == rep["image_phash_le_4"]["rows_with_crossfold_image_dup"],
         "n10 working-copy count vs n04 report")
    flag(ic["working"]["dedup_keep"] == int(keep.sum()), "n10 working-copy dedup size vs frozen mask")
    for v in (ic["public"]["cross_fold_d0"], ic["working"]["cross_fold_d0"], ic["public"]["cross_fold_le4"]):
        flag(f"{v:,}" in num.get("RespImgCopy", ""), f"RespImgCopy number {v}")
    check("DupTrainExact", ic["working"]["cross_fold_d0"], 0)
_prior_tex = re.findall(r"~\\cite\{banglacalamity\}\s*&\s*([0-9.]+)\s*&", TEX)
flag(len(_prior_tex) == 3, "Table IV prior-work rows")
_att = lambda v, n: any(round(100 * k / n, 2) == v for k in range(n + 1))  # noqa: E731
flag(bool(num.get("RespC22")) == all(not _att(float(v), 790) for v in _prior_tex), "RespC22 gate (790 rows)")
flag("& $790$ test" not in TEX and TEX.count(r"& \rev{as published}") == 3, "Table IV protocol column")
_rf = (ROOT / "revision/response_filled.md").read_text(encoding="utf-8")
flag(not re.findall(r"\b(?:Sections?|Tables?|Fig\.|Eq\.)\s*[),.;:]", _rf), "empty cross-reference in the response")
flag("\\ref" not in (ROOT / "jobs/latex/response/main.tex").read_text(encoding="utf-8"), "\\ref in response LaTeX")
flag("(Section )" not in _rf and "leakage-free matched" not in _rf, "response wording fixed by D37")

public_map = O / "n12/map_contest_to_bcmmd.csv"
if public_map.exists():
    mapping = pd.read_csv(public_map)
    order = pd.read_csv(ROOT / "jobs/g_fusion/rows.csv.gz")
    flag(mapping.stem.tolist() == order.image_id.tolist() and mapping.split.equals(order.split),
         "public mapping covers every working row in order")
    flag(mapping.bc_path.is_unique and mapping[["public_split", "public_id"]].duplicated().sum() == 0,
         "public records mapped one to one")
    evaluation = mapping[mapping.split == "test"]
    flag(evaluation.public_split.value_counts().to_dict() == {"test": 790, "validation": 790},
         "published evaluation split composition")
    with np.load(O / "dedup_masks.npz") as masks:
        for split in ("train", "test"):
            flag(np.array_equal(mapping.loc[mapping.split == split, "working_keep"], masks[split + "_keep"]),
                 f"row-aligned working-copy mask, {split}")
    flag("unique match by image MD5 and normalized caption" in _rf and
         "6,323 training images differ in bytes" in _rf, "training and evaluation provenance in response")

# Verify the post-hoc grouped refit from row-level scores, independently of the training script.
grouped_dir = O / "n11"
if grouped_dir.is_dir():
    grouped = json.loads((grouped_dir / "grouped_refit.json").read_bytes())
    with np.load(grouped_dir / "grouped_predictions.npz") as record:
        ids = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv").image_id
        flag(np.array_equal(record["row_id"], ids), "grouped refit row alignment")
        folds = record["fold"]
        group_folds = pd.DataFrame({"group": record["group"], "fold": folds}).groupby("group").fold
        flag(len(group_folds) == grouped["n_groups"] and group_folds.nunique().max() == 1,
             "no duplicate component crosses grouped partitions")
        tst, img = record["eval_caption"], record["eval_image"]
        scores = {"caption": tst, "image": img, "equal": (tst + img) / 2, "calibrated": np.zeros_like(tst)}
        for run in grouped["folds"]:
            outer = run["fold"]
            ev, cal = folds == outer, folds == (outer + 1) % 5
            flag((ev.sum(), cal.sum(), (~(cal | ev)).sum()) ==
                 (run["n_eval"], run["n_cal"], run["n_fit"]), f"grouped partition counts, fold {outer}")
            ct, ci = record["cal_caption"][cal], record["cal_image"][cal]
            grid = np.linspace(0, 1, 201)
            best = max(grid, key=lambda a: F(y[cal], (a * ct + (1 - a) * ci).argmax(1)))
            flag(best == run["alpha"], f"calibration-only fusion weight, fold {outer}")
            scores["calibrated"][ev] = best * tst[ev] + (1 - best) * img[ev]
            for arm, values in scores.items():
                flag(int((values[ev].argmax(1) != y[ev]).sum()) == run["errors"][arm],
                     f"grouped fold {outer} errors, {arm}")
        for arm, values in scores.items():
            pred = values.argmax(1)
            errors = int((pred != y).sum())
            f1 = F(y, pred)
            flag(grouped["pooled"][arm]["errors"] == errors and
                 abs(grouped["pooled"][arm]["macro_f1"] - f1) < 1e-12,
                 f"grouped pooled result, {arm}")
            flag(str(errors) in num["RespGroupedRefit"], f"grouped response error count, {arm}")
            if arm in ("calibrated", "caption"):
                flag(f"{f1:.5f}" in num["RespGroupedRefit"], f"grouped response F1, {arm}")
    flag(num["RespGroupedRefit"] in _rf, "grouped results printed in the reviewer response")

# ---- no silent skips: every macro main.tex prints must exist, except those this state legitimately lacks
used = set(re.findall(r"\\num([A-Za-z]+)", TEX))
allowed = set()
if not (O / "n05/test_results.csv").exists():
    allowed |= {k for k in used if k.startswith("Test")} | {"FrozenHash"}
if not MECH:
    allowed |= {k for k in used if re.match(r"(Co|Ct|Cth|MT|MLF|MJone|MJP|EpochSel)[A-Z]", k)}
if cfg != "P":
    allowed |= {k for k in used if k.startswith("JoneSz")}
if prow("(3)") is None:
    allowed |= {k for k in used if k.startswith("Cth") or k.startswith("MJP")}
flag(not (used - set(num) - allowed), f"macros printed by main.tex but missing: {sorted(used - set(num) - allowed)}")

print(f"checked {checked} numbers; mismatches: {len(bad)}")
for b in bad:
    print("  MISMATCH", b)
sys.exit(1 if bad else 0)
