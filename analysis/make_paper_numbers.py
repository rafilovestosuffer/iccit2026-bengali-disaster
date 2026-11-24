"""Write every number (and every result-dependent sentence) of the revised paper as LaTeX macros, from result files only.

Reads  analysis/out/n06/{stage_table,primary,mech_table}.csv, analysis/out/n04/report.json,
       analysis/out/dedup_masks.npz, analysis/out/n05/test_results.csv (if scored), analysis/out/n02/epoch_selection.csv.
Writes revision/generated/numbers.tex (\\newcommand per macro used in revision/src_revised/main.tex; missing -> \\ph{}),
       a copy in revision/src_revised/numbers.tex, and revision/generated/numbers.json.
Result-dependent sentences follow DECISION_LOG D20: a difference is stated as a difference only if resolved
(Holm-adjusted p < 0.05 and every seed agrees in sign); otherwise "not distinguishable".
The primary configuration is P when its nested results exist, else clean3 (D23).
"""
import ast
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# LAYOUT TESTS ONLY: PAPER_NUM_IN points at a synthetic copy of analysis/out and PAPER_NUM_OUT at a layout-test folder;
# then nothing under revision/ is written. Deliverables are always generated without these variables.
O = Path(os.environ.get("PAPER_NUM_IN", ROOT / "analysis/out"))
OUT_ONLY = Path(os.environ["PAPER_NUM_OUT"]) if os.environ.get("PAPER_NUM_OUT") else None
GEN = ROOT / "revision/generated"
SRC = ROOT / "revision/src_revised"
GEN.mkdir(parents=True, exist_ok=True)
M = {}


def rd(p):
    try:
        return pd.read_csv(p) if Path(p).exists() else None
    except pd.errors.EmptyDataError:
        return None


def f5(x):
    return f"{x:.5f}"


def errs(row):
    # mean (range over meta-seeds); round brackets, so a range never reads as a citation range
    if pd.isna(row.get("nested_err_min", np.nan)) or row["nested_err_min"] == row["nested_err_max"]:
        return f"{row['nested_err_mean']:g}"
    return f"{row['nested_err_mean']:g} ({int(row['nested_err_min'])}--{int(row['nested_err_max'])})"


def sgn(x, fmt):
    """Signed number in math mode, so LaTeX prints a real minus sign."""
    return f"${x:{fmt}}$"


st = rd(O / "n06/stage_table.csv")
prim = rd(O / "n06/primary.csv")
mech = rd(O / "n06/mech_table.csv")
rep = json.loads((O / "n04/report.json").read_text())
masks = np.load(O / "dedup_masks.npz")

PRIMARY = "P" if st is not None and (st.config == "P").any() else "clean3"
M["PrimaryCfg"] = PRIMARY
M["PrimaryMembers"] = ("BanglaBERT-base, MuRIL-Large, EVA-02-L and J1 (seed 0)" if PRIMARY == "P"
                       else "BanglaBERT-base, MuRIL-Large and EVA-02-L")
N_BRANCHES = 4 if PRIMARY == "P" else 3  # n01 CONFIGS: clean3 = 3 members, P = clean3 + g2_s0
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
M["NBranches"] = WORDS[N_BRANCHES]

# ---- data audit
M["DupTrainCross"] = f"{rep['image_phash_le_4']['rows_with_crossfold_image_dup']:,}".replace(",", "{,}")
M["DupTestTrain"] = str(rep["image_phash_le_4"]["test_rows_with_train_image_dup"])
M["DupTrainExact"] = f"{rep['image_phash_le_0']['rows_with_crossfold_image_dup']:,}".replace(",", "{,}")
M["EvaAccTwin"] = f"{100 * rep['eva_acc_rows_with_crossfold_image_dup_le4']:.1f}"
M["EvaAccNoTwin"] = f"{100 * rep['eva_acc_rows_without']:.1f}"
M["DedupN"] = f"{int(masks['train_keep'].sum()):,}".replace(",", "{,}")
M["DedupTestN"] = f"{int(masks['test_keep'].sum()):,}".replace(",", "{,}")
M["TestCapInTrain"] = str(rep["caption"]["test_rows_caption_in_train"])
# caption-duplicate groups spanning folds, with n04's caption normalisation (NFC + whitespace collapse)
import unicodedata  # noqa: E402
_f = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
_f["k"] = _f.context.map(lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip())
_g = _f.groupby("k").fold.nunique()
M["CapGroupsCross"] = str(int((_g > 1).sum()))
# captions truncated by the text branches (n08)
_tl = O / "n08/token_lengths.json"
if _tl.exists():
    _t = json.loads(_tl.read_text())
    _w0 = {0: "none", **WORDS}
    M["BbTruncTrain"] = _w0.get(_t["banglabert_base_train"]["over"], str(_t["banglabert_base_train"]["over"]))
    M["MurilTruncTrain"] = _w0.get(_t["muril_large_train"]["over"], str(_t["muril_large_train"]["over"]))
# rows wrong in all six submitted branches (OOF), and with the same wrong label
_ORIG = {"bb_base": "text_bbB_s123_20260509_0550", "bb_multi": "text_bbmulti_20260509_0837",
         "muril_L": "text_murilL_20260509_0210", "eva02_L": "img_eva02L_448_20260511_0327",
         "mm_old": "mm_eva_muril_20260512_1041", "pl_muril": "pl_muril_20260512_1725"}
_y = _f.label.values.astype(int)
_share = np.bincount(_y) / len(_y) * 100
M["ClassShareMin"], M["ClassShareMax"] = f"{_share.min():.1f}", f"{_share.max():.1f}"
_P6 = np.stack([np.load(ROOT / f"drive_oof/x/oof/{v}_oof.npy").argmax(1) for v in _ORIG.values()])
_aw = (_P6 != _y).all(0)
M["AllSixWrong"] = WORDS.get(int(_aw.sum()), str(int(_aw.sum())))
M["AllSixSameWrong"] = WORDS.get(int((_aw & (_P6 == _P6[0]).all(0)).sum()), str(int((_aw & (_P6 == _P6[0]).all(0)).sum())))
# leaderboard scorings of the 1,580-row partition during development (recorded in seen_test_numbers.md)
_seen = (O / "seen_test_numbers.md").read_text(encoding="utf-8")
_m = re.search(r"scores of (\d+) submissions", _seen)
if _m:
    M["LBScorings"] = _m.group(1)


# ---- stage table
def row(cfg, stage):
    if st is None:
        return None
    r = st[(st.config == cfg) & (st.stage == stage)]
    return None if r.empty else r.iloc[0]


for key, m in (("Bb", "bb_base"), ("Muril", "muril_L"), ("Eva", "eva02_L")):
    r = row("clean3", f"member:{m}")
    if r is not None:
        M[f"{key}Fone"], M[f"{key}Err"] = f5(r.nested_f1_mean), f"{int(r.nested_err_mean)}"
        M[f"{key}DdFone"], M[f"{key}DdErr"] = f5(r.dedup_f1_mean), f"{int(r.dedup_err_mean)}"
for key, stage in (("Uni", "uniform"), ("LF", "late_fusion"), ("Blend", "blend"), ("Stack", "stack"),
                   ("Mix", "alpha_mix"), ("Bs", "bias_single"), ("Bb", "bias_boot"), ("BlBb", "blend_bias_boot")):
    r = row(PRIMARY, stage)
    if r is not None:
        M[f"P{key}Fone"], M[f"P{key}Err"] = f5(r.nested_f1_mean), errs(r)
        M[f"P{key}InFone"], M[f"P{key}InErr"] = f5(r.insample_f1_mean), f"{r.insample_err_mean:g}"
        M[f"P{key}DdFone"], M[f"P{key}DdErr"] = f5(r.dedup_f1_mean), f"{r.dedup_err_mean:g}"
r = row("published", "bias_boot")
if r is not None:
    M["SubBbFone"], M["SubBbErr"] = f5(r.nested_f1_mean), errs(r)
    M["SubBbInFone"], M["SubBbDdFone"] = f5(r.insample_f1_mean), f5(r.dedup_f1_mean)
    M["SubBbInErr"] = f"{r.insample_err_mean:g}"
    M["SubOptimismErr"] = f"{r.optimism_err:g}"
r = row(PRIMARY, "member:g2_s0")  # J1 (seed 0) is a member of P
if r is not None:
    M["JoneSzFone"], M["JoneSzErr"], M["JoneSzDdFone"] = (f5(r.nested_f1_mean), f"{int(r.nested_err_mean)}",
                                                         f5(r.dedup_f1_mean))
ex = rd(O / "n06/exploratory.csv")
opt_t = rd(O / "n06/optimism.csv")


def erow(cfg, comparison, subset="all_6323"):
    if ex is None or "subset" not in ex.columns:
        return None
    q = ex[(ex.config == cfg) & (ex.comparison == comparison) & (ex.subset == subset)]
    return None if q.empty else q.iloc[0]


lf, mu = row(PRIMARY, "late_fusion"), row("clean3", "member:muril_L")
_lfa = erow(PRIMARY, "late fusion vs best unimodal")
_lfd = erow(PRIMARY, "late fusion vs best unimodal", "dedup_5321")
# The relative reduction in the total error count is reported only for a resolved exploratory gain.
if lf is not None and mu is not None and _lfa is not None and _lfa.resolved and _lfa.d_f1_mean > 0 \
        and _lfa.d_err_mean < 0:
    M["LFErrRemovedPct"] = f"{100 * (mu.nested_err_mean - lf.nested_err_mean) / mu.nested_err_mean:.0f}"
if _lfd is not None:
    if _lfd.resolved and _lfd.d_f1_mean > 0 and _lfd.d_err_mean < 0:
        M["LFDdText"] = (", and its reduction of MuRIL-Large's errors holds on the dedup rows (\\numMurilDdErr{} to "
                         "\\numPLFDdErr{} errors)")
        if mu is not None and lf is not None:  # the dedup reduction is its own percentage, not the all-rows one
            M["LFErrRemovedPctDd"] = f"{100 * (mu.dedup_err_mean - lf.dedup_err_mean) / mu.dedup_err_mean:.0f}"
            M["LFDdConcl"] = ", and \\numLFErrRemovedPctDd{}\\% on rows without a cross-fold near-duplicate"
    else:
        M["LFDdText"] = ("; on the dedup rows MuRIL-Large makes \\numMurilDdErr{} errors and LF$_\\text{p}$ "
                         "\\numPLFDdErr{}, which is not a resolved difference")
        M["LFDdConcl"] = " (on rows without a cross-fold near-duplicate the reduction is not resolved)"
bb = row(PRIMARY, "bias_boot")
if bb is not None:
    M["OptimismErr"] = f"{bb.optimism_err:g}"
# in-sample optimism of the full pipeline is stated as a direction only if positive and resolved (D8)
_o = None if opt_t is None else opt_t[(opt_t.config == PRIMARY) & (opt_t.stage == "bias_boot")]
OPT_RESOLVED = bool(bb is not None and _o is not None and len(_o) and _o.iloc[0].resolved
                    and _o.iloc[0].d_f1_mean > 0 and bb.optimism_err > 0)
M["AbsOpt"] = ("in an exploratory comparison, in-sample evaluation of the meta stages, as in our original analysis, "
               "understates the full pipeline's error count by \\numOptimismErr{} relative to nested evaluation"
               if OPT_RESOLVED else
               "in an exploratory comparison, the full pipeline's in-sample and nested error counts are not "
               "distinguishable")
if bb is not None and lf is not None:
    # apparent in-sample gain of the full pipeline over late fusion (positive = fewer errors in-sample)
    M["InGainErr"] = f"{lf.insample_err_mean - bb.insample_err_mean:g}"
    # Section V-C claim about the in-sample column, stated only as far as the data support it
    # the later stages combine more members than LF_p (BanglaBERT-base, and J1 seed 0 in P), so the comparison is
    # with LF_p, not a gain of the stages over late fusion
    _after = [row(PRIMARY, s_) for s_ in ("blend", "stack", "alpha_mix", "bias_single", "bias_boot")]
    _low = [r_ is not None and r_.insample_err_mean < lf.insample_err_mean for r_ in _after]
    _extra = "BanglaBERT-base and J1 (seed 0)" if PRIMARY == "P" else "BanglaBERT-base"
    M["ExtraMembers"] = _extra
    M["InSampleText"] = (f"every stage from the blend on, which also uses {_extra}, appears to make fewer errors than "
                         "LF$_\\text{p}$" if all(_low) else
                         f"the full pipeline, which also uses {_extra}, appears to make fewer errors than "
                         "LF$_\\text{p}$" if _low[-1] else
                         f"the stages from the blend on, which also use {_extra}, do not appear to make fewer errors "
                         "than LF$_\\text{p}$")
# exploratory interval for the headline late-fusion gain over MuRIL-Large (single training runs; V-A)
if _lfa is not None:
    M["LFCI"] = (f" (exploratory; $\\Delta$F1 {sgn(_lfa.d_f1_mean, '+.4f')} [{sgn(_lfa.ci_lo, '+.4f')}, {sgn(_lfa.ci_hi, '+.4f')}], "
                 f"McNemar {int(_lfa.mcnemar_seed0_b)}/{int(_lfa.mcnemar_seed0_c)}; conditional on single training runs)")
# the text lead is a MuRIL-Large lead: EVA-02-L vs BanglaBERT-base order on all rows and on the dedup rows (V-A)
_bbm, _evm = row("clean3", "member:bb_base"), row("clean3", "member:eva02_L")
if _bbm is not None and _evm is not None and mu is not None:
    _all = _evm.nested_err_mean < _bbm.nested_err_mean
    _dd = _evm.dedup_f1_mean > _bbm.dedup_f1_mean
    M["TextOrder"] = ("" if _all == _dd else
                      (f"; EVA-02-L makes fewer errors than BanglaBERT-base on all rows (\\numEvaErr{{}} against "
                       "\\numBbErr{}) but not on the dedup rows, so the text lead is MuRIL-Large's" if _all else
                       "; BanglaBERT-base makes fewer errors than EVA-02-L on all rows but not on the dedup rows, "
                       "so the text lead is MuRIL-Large's"))
# late meta stages under nested evaluation (V-B) and the fitted weights that explain them (fitted_*.json)
_fit = sorted((ROOT / "analysis/out/n01").glob(f"fitted_{PRIMARY}_m[0-4].json"))
_ma, _lfw = [], []
for _p in _fit:
    _j = json.loads(_p.read_text())["fitted"]
    for _k, _v in _j.items():
        _ma.append(_v["mix_alpha"])
        _lfw.append(_v["late_fusion_alpha"])
_skr, _mxr, _bbr, _lfr = (row(PRIMARY, s_) for s_ in ("stack", "alpha_mix", "bias_boot", "late_fusion"))
if all(r_ is not None for r_ in (_skr, _mxr, _bbr)) and _ma:
    if _bbr.nested_err_mean > _skr.nested_err_mean and _mxr.nested_err_mean > _skr.nested_err_mean:
        M["LateText"] = ("\\rev{Under nested evaluation the $\\alpha$-mix and bias stages, fitted on the blend's in-sample "
                         "predictions (Section~\\ref{sec:meta}), do not reduce errors relative "
                         "to the stacker alone (\\numPStackErr{}, \\numPMixErr{} and \\numPBbErr{} errors; exploratory); "
                         f"the fitted $\\alpha$-mix weight ranges from {min(_ma):g} to {max(_ma):g} across outer folds "
                         "and meta-seeds.}")
    else:
        M["LateText"] = ("\\rev{Under nested evaluation the stacker alone makes \\numPStackErr{} errors, the "
                         "$\\alpha$-mix \\numPMixErr{} and the full pipeline \\numPBbErr{} (exploratory).}")
    # the stacker alone has the fewest nested errors of the pipeline's stages: say so where LF_p is recommended
    _stages = [row(PRIMARY, s_) for s_ in ("uniform", "late_fusion", "blend", "stack", "alpha_mix", "bias_single",
                                            "bias_boot")]
    M["StackFewest"] = ""
    if all(s_ is not None for s_ in _stages) and min(_stages, key=lambda s_: s_.nested_err_mean) is _stages[3] \
            and _lfr is not None:
        M["StackFewest"] = (" \\rev{Numerically the stacker alone makes the fewest nested errors (\\numPStackErr{} "
                            "against \\numPLFErr{}; exploratory); LF$_\\text{p}$ needs two branches instead of "
                            f"{WORDS[N_BRANCHES]} and tolerates a missing image.}}")
if _lfw and len(set(_lfw)) == 1:
    M["LFAlphaNote"] = (f" LF$_\\text{{p}}$'s weight is $\\alpha{{=}}{_lfw[0]:g}$ in every outer fit and the uniform mean "
                        "has none, so their nested and in-sample scores coincide.")
else:
    M["LFAlphaNote"] = ""
un, bl = row(PRIMARY, "uniform"), row(PRIMARY, "blend")
if un is not None and bl is not None:
    M["UniCostErr"] = f"{un.nested_err_mean - bl.nested_err_mean:.1f}"
    # V-C sentence on the uniform mean: the superlative is stated only if the data support it in both columns
    _ub = erow(PRIMARY, "learned blend vs uniform mean")
    _st = {s_: row(PRIMARY, s_) for s_ in ("uniform", "blend", "stack", "alpha_mix", "bias_single", "bias_boot")}
    # the comparison set is named in the sentence: uniform -> blend against the steps between blend, stacker,
    # alpha-mix and bias (late fusion is not in the set; its effect over MuRIL-Large is reported in V-A)
    _largest = False
    if all(v is not None for v in _st.values()):
        # every pairwise gap among the blend, stacker, alpha-mix and bias stages (the set the sentence names)
        _later = ("blend", "stack", "alpha_mix", "bias_single", "bias_boot")
        for col in ("nested_err_mean", "insample_err_mean"):
            _u = _st["uniform"][col] - _st["blend"][col]
            _other = max(abs(_st[a][col] - _st[b][col]) for i, a in enumerate(_later) for b in _later[i + 1:])
            _largest = (_u > 0 and _u > _other) if col == "nested_err_mean" else (_largest and _u > 0 and _u > _other)
    M["UniCostLargest"] = "yes" if _largest else "no"  # read by verify_paper_numbers.py, not printed
    if _ub is not None and _ub.resolved and _ub.d_f1_mean > 0 and _ub.d_err_mean < 0:
        M["UniCostText"] = ("\\rev{Replacing the learned blend with a uniform average costs \\numUniCostErr{} errors "
                            "(exploratory)" + (", a larger change than any step between the blend, stacker, "
                                               "$\\alpha$-mix and bias stages under both evaluations"
                                               if _largest else "") + ".}")
    elif _ub is not None and _ub.resolved and _ub.d_f1_mean < 0:  # d_f1 = blend minus uniform
        M["UniCostText"] = ("\\rev{A uniform average scores higher than the learned blend under nested evaluation "
                            "(exploratory).}")
    elif _ub is not None and _ub.resolved:  # blend higher in macro-F1 without fewer errors
        M["UniCostText"] = ("\\rev{The learned blend scores higher than a uniform average in macro-F1 under nested "
                            "evaluation (exploratory).}")
    elif _ub is not None:
        M["UniCostText"] = ("\\rev{The learned blend and a uniform average are not distinguishable under nested "
                            "evaluation (exploratory).}")

# ---- mechanism table
if mech is not None:
    for key, arm in (("MT", "T"), ("MLF", "LF"), ("MJone", "J1"), ("MJP", "JP")):
        r = mech[mech.arm == arm]
        if not r.empty:
            r = r.iloc[0]
            M[f"{key}Fone"] = f"{r.f1_mean:.5f} $\\pm$ {r.f1_sd:.5f}"
            M[f"{key}Err"] = f"{r.err_mean:.1f} $\\pm$ {r.err_sd:.1f}"
            M[f"{key}DdFone"] = f5(r.dedup_f1_mean)

# ---- pre-specified comparisons (all rows) and the sentences that depend on them
C = {}
if prim is not None and len(prim):
    pa = prim[prim.subset == "all_6323"]
    for key, start in (("Co", "(1)"), ("Ct", "(2)"), ("Cth", "(3)"), ("Cf", "(4)")):
        r = pa[pa.comparison.str.startswith(start)]
        if r.empty:
            continue
        r = r.iloc[0]
        C[key] = r
        M[f"{key}D"] = f"{sgn(r.d_f1_mean, '+.4f')} [{sgn(r.ci_lo, '+.4f')}, {sgn(r.ci_hi, '+.4f')}]"
        M[f"{key}E"] = sgn(r.d_err_mean, "+.1f")
        M[f"{key}P"] = f"{r.p_holm:.3f}" if r.p_holm >= 0.001 else "$<$0.001"
        M[f"{key}R"] = "yes" if r.resolved_holm else "no"
        M[f"{key}Seeds"] = ", ".join(sgn(x, "+.4f") for x in ast.literal_eval(str(r.d_f1_per_seed)))
# the same family on the D18 dedup subset (its own Holm adjustment, D20)
CD = {}
if prim is not None and len(prim):
    pdd = prim[prim.subset == "dedup_5321"]
    for key, start in (("Co", "(1)"), ("Ct", "(2)"), ("Cth", "(3)"), ("Cf", "(4)")):
        r = pdd[pdd.comparison.str.startswith(start)]
        if not r.empty:
            CD[key] = r.iloc[0]

# D23: a fusion-mechanism claim needs the G1 and G2 runs (comparisons (1) and (2)); without G3, (3) is dropped
MECH = "Co" in C and "Ct" in C
M["NComparisons"] = WORDS[len(C)] if C else "four"
M["RespPrereg"] = "pre-specified comparisons" if MECH else "pre-specified comparison"  # response C14
if MECH and "Cth" not in C:
    M["FamilyNote"] = (" \\rev{JP was not completed for all three seeds and is not reported, so the family has three "
                       "comparisons.}")
elif C and not MECH:
    M["FamilyNote"] = (" \\rev{The head comparisons need three training seeds per head; these runs were not complete, so "
                       "only the last comparison is reported (a family of one, so Holm adjustment leaves $p$ unchanged) "
                       "and no fusion-mechanism claim is made.}")
else:
    M["FamilyNote"] = ""
LFP = "LF$_\\text{p}$"  # late fusion of the pipeline's MuRIL-Large and fine-tuned EVA-02-L (comparison (4))


def pstr(key, D=None):
    r = (D or C)[key]
    return f"$p_H{{=}}{r.p_holm:.3f}$" if r.p_holm >= 0.001 else "$p_H{<}0.001$"


def consistent(r):
    """Direction is taken from macro-F1 (the resolution rule uses per-seed F1); True if the mean error difference
    points the same way (A better in F1 <-> A fewer errors)."""
    return (r.d_f1_mean > 0 and r.d_err_mean < 0) or (r.d_f1_mean < 0 and r.d_err_mean > 0)


def diff_phrase(key, a, b, D=None):
    """D20 wording: a direction is stated only for a resolved difference."""
    r = (D or C)[key]
    p = pstr(key, D)
    if not r.resolved_holm:
        return f"{a} and {b} are not distinguishable ({sgn(r.d_err_mean, '+.1f')} errors on average, {p})"
    if consistent(r):
        return (f"{a} makes {abs(r.d_err_mean):.1f} {'fewer' if r.d_f1_mean > 0 else 'more'} errors than {b} "
                f"on average (resolved, {p})")
    return (f"{a} scores {'higher' if r.d_f1_mean > 0 else 'lower'} than {b} in macro-F1 "
            f"({sgn(r.d_err_mean, '+.1f')} errors on average; resolved, {p})")


def same_verdict(key):
    """True if the dedup-subset verdict equals the all-rows verdict (same resolution and, if resolved, same sign)."""
    a, d = C[key], CD.get(key)
    if d is None:
        return False
    return bool(a.resolved_holm == d.resolved_holm and
                (not a.resolved_holm or np.sign(a.d_f1_mean) == np.sign(d.d_f1_mean)))


MLF = "convex late fusion with the same text recipe and image features"  # the matched LF (not LF_p)


def every_more(key):
    """True if A (a joint head) makes more errors than B (LF) in every training seed (all rows)."""
    r = C[key]
    ea, eb = ast.literal_eval(str(r.err_a)), ast.literal_eval(str(r.err_b))
    return len(ea) == len(eb) and all(a > b for a, b in zip(ea, eb))
M["RespPriorTab"] = "Table IV" if MECH else "Table III"  # Table III (matched heads) exists only with \mechtrue


def dd_qual(key):
    """Qualifier for a summary clause whose pre-specified dedup verdict differs from the all-rows verdict (D18/D20)."""
    if same_verdict(key):
        return ""
    d = CD.get(key)
    if d is None or not d.resolved_holm:
        return " (not resolved on the rows without a cross-fold near-duplicate)"
    if C[key].resolved_holm:  # resolved on both subsets, opposite signs
        return " (reversed on the rows without a cross-fold near-duplicate)"
    return " (resolved on the rows without a cross-fold near-duplicate)"


def head_clause(key, name, ref=MLF, qual=True):
    r = C[key]
    q = dd_qual(key) if qual else ""
    if not r.resolved_holm:
        return f"{name} was not distinguishable from {ref}{q}"
    if consistent(r):
        return f"{name} made {abs(r.d_err_mean):.1f} {'fewer' if r.d_f1_mean > 0 else 'more'} errors than {ref}{q}"
    return f"{name} scored {'higher' if r.d_f1_mean > 0 else 'lower'} than {ref} in macro-F1{q}"


# ---- post-hoc controls (n09, D37; not in the pre-specified family): LF's alpha fitted on half-fold rows disjoint from
# the evaluated rows, a fixed alpha of 0.5, and the full pipeline with J1 training seeds 1 and 2
_n9 = rd(O / "n09/lf_controls.csv")
_j9 = rd(O / "n09/j1seed.csv")
_a9 = rd(O / "n09/alphas.csv")
_c9 = None
if _n9 is not None:
    _c9 = {sub: _n9[_n9.subset == sub].pivot(index="arm", columns="seed", values="err")
           for sub in ("all_6323", "dedup_5321")}


def _ctl_mean_order(head):
    """True if the half-fold LF control makes fewer errors on average than `head` (or no control exists)."""
    if _c9 is None:
        return True
    t = _c9["all_6323"]
    return head in t.index and t.loc["LF_halffold"].mean() < t.loc[head].mean()


if MECH:
    heads = [(k, n) for k, n in (("Ct", "a jointly trained single-token head"), ("Cth", "patch cross-attention"))
             if k in C]
    if len(heads) == 2 and all(not C[k].resolved_holm and same_verdict(k) and C[k].d_err_mean > 0
                               and _ctl_mean_order({"Ct": "J1", "Cth": "JP"}[k]) for k, _ in heads):
        # direction stated as a mean error count, not as a resolved difference (D20); the per-seed order depends on
        # how LF's alpha is fitted (n09, D37), so it is stated in V-B with the control, not here
        mech_sum = ("a jointly trained single-token head and patch cross-attention each made more errors on average "
                    f"than {MLF}, but neither difference was resolved")
    elif len(heads) == 2 and all(not C[k].resolved_holm and same_verdict(k) for k, _ in heads):
        mech_sum = ("neither a jointly trained single-token head nor patch cross-attention was distinguishable "
                    f"from {MLF}")
    else:
        # MLF is spelled out once; a dedup qualifier shared by both heads is stated once at the end
        quals = {dd_qual(k) for k, _ in heads}
        shared = len(heads) == 2 and len(quals) == 1
        mech_sum = "; ".join(head_clause(k, n, MLF if i == 0 else "late fusion", qual=not shared)
                             for i, (k, n) in enumerate(heads))
        q0 = next(iter(quals))
        if shared and q0:
            _rest = " on the rows without a cross-fold near-duplicate)"
            mech_sum += {" (not resolved" + _rest: " (neither difference is resolved" + _rest,
                         " (resolved" + _rest: " (both differences are resolved" + _rest,
                         " (reversed" + _rest: " (both differences are reversed" + _rest}[q0]
    M["MechSummary"] = mech_sum + "."
    M["AbsMech"] = "In a matched comparison over three seeds on frozen image features, " + mech_sum + "."
    parts = [diff_phrase("Co", "Late fusion (LF)", "the text-only head (T)") + "."]
    hp = [diff_phrase(k, n, "LF") for k, n in (("Ct", "J1"), ("Cth", "JP")) if k in C]
    parts.append(" and ".join(hp) + ".")  # both phrases start with the head's name (J1 / JP)
    _em = [n for k, n in (("Ct", "J1"), ("Cth", "JP")) if k in C and not C[k].resolved_holm and every_more(k)]
    if _em:
        parts.append(f"LF makes fewer errors than {' and '.join(_em)} in every seed.")
    M["LFCtlErr"] = ""
    if _c9 is not None:
        t9 = _c9["all_6323"]
        sd9 = [int(s) for s in t9.columns]
        ctl = [int(t9.loc["LF_halffold", s]) for s in sd9]
        M["LFCtlErr"] = ", ".join(str(v) for v in ctl[:-1]) + " and " + str(ctl[-1])
        h9 = [h for h in ("J1", "JP") if h in t9.index]
        below_t = all(t9.loc["LF_halffold", s] < t9.loc["T", s] for s in sd9)
        worse = [s for s in sd9 if all(t9.loc["LF_halffold", s] > t9.loc[h, s] for h in h9)]
        every9 = all(t9.loc["LF_halffold", s] < t9.loc[h, s] for s in sd9 for h in h9)
        if worse:
            tail = (f"more than {' and '.join(h9)} in seed{'s' if len(worse) > 1 else ''} "
                    f"{' and '.join(str(s) for s in worse)}, so the order of LF and the joint heads depends on how "
                    "$\\alpha$ is fitted")
        elif not every9:
            tail = ("no longer fewer than " + " and ".join(h9) + " in every seed, so the order of LF and the joint "
                    "heads depends on how $\\alpha$ is fitted")
        else:
            tail = "still fewer than " + " and ".join(h9) + " in every seed"
        parts.append("Post hoc, with $\\alpha$ fitted on half of each evaluation fold, LF "
                     f"makes \\numLFCtlErr{{}} errors (seeds {sd9[0]}--{sd9[-1]}), "
                     + ("still far fewer than T but " if below_t and not every9 else
                        "fewer than T and " if below_t else "not fewer than T in every seed, and ")
                     + tail + ".")
    hkeys = [k for k in ("Co", "Ct", "Cth") if k in C]
    if all(same_verdict(k) for k in hkeys):
        parts.append("Each verdict is the same on the dedup rows.")
    else:
        names = {"Co": ("LF", "T"), "Ct": ("J1", "LF"), "Cth": ("JP", "LF")}
        parts.append("On the dedup rows, " + "; ".join(diff_phrase(k, *names[k], D=CD) for k in hkeys
                                                         if k in CD and not same_verdict(k)) + ".")
    parts.append(f"These results concern the {'two heads' if 'Cth' in C else 'head'} and the frozen features we tested, "
                 "not cross-attention in general.")
    M["MechText"] = "\\rev{" + " ".join(parts) + "}"
    _held = f"this concerns the {'heads' if 'Cth' in C else 'head'} and frozen features we tested"
    M["ConclText"] = "\\rev{Under matched conditions, " + mech_sum + "; " + _held + ".}"
    n_run = len(hkeys) + ("Cf" in C)
    M["RespSum4"] = ("New fusion runs with three training seeds, five meta-seeds for every meta stage, paired "
                     "class-stratified bootstrap 95% CIs that include the seed spread, a pre-specified family of four "
                     "comparisons with Holm adjustment" + ("" if n_run == 4 else
                                                           f" ({WORDS[n_run]} were run; JP could not be completed for "
                                                           "all seeds in time)") + ", and actual error counts everywhere.")
    M["RespSum5"] = f"The fusion conclusion is restricted to the {'heads' if 'Cth' in C else 'head'} we tested; "
    # the template adds the closing full stop after {{RespMechA}}
    M["RespMechA"] = ("the conclusion now reads \"Under matched conditions, " + mech_sum + "; " + _held
                      + ".\", and Section V-B adds \"" + parts[-1].rstrip(".") + "\"")
    jp = "Cth" in C
    M["RespR23Runs"] = ("We trained new runs on the same folds with one recipe and the final epoch (no epoch selection on "
                        "the evaluation fold): text-only MuRIL-L (T), the single-token joint head of the submission "
                        "retrained without the label leak (J1)"
                        + (" and a patch cross-attention head over 64 frozen EVA-02-L patch tokens (JP), each with seeds "
                           "0, 1, 2" if jp else ", each with seeds 0, 1, 2 (a patch cross-attention head over 64 frozen "
                           "EVA-02-L patch tokens, JP, was also planned but could not be completed for all seeds)")
                        + ", and late fusion (LF) of T with a logistic-regression head on the same frozen image "
                        "features, fitted inside the outer loop.")
    M["RespSeedNote"] = f"; the {WORDS[len([k for k in ('Co', 'Ct', 'Cth') if k in C])]} head comparisons include " \
                        "training-seed variance"
    M["RespC1"] = "Section IV-C rewritten; J1 (same design, no leak) " + (
        "and JP (64 patch tokens) trained" if jp else
        "trained; JP (64 patch tokens) could not be completed for all seeds in time")
    M["RespC2"] = ("Branch removed from every revised configuration; it appears only in the labelled "
                   "submitted-configuration row; retrained without the fold-label leak on frozen pretrained features ("
                   + ("J1, JP" if jp else "J1") + ")")
    M["RespC3"] = "Matched comparison redone (R2.3)"
else:  # D23 fallback text (also shown while the G runs are pending)
    M["RespR23Runs"] = ("We set up new runs on the same folds with one recipe and the final epoch (no epoch selection on "
                        "the evaluation fold): text-only MuRIL-L (T), the single-token joint head of the submission "
                        "retrained without the label leak (J1) and a patch cross-attention head over 64 frozen EVA-02-L "
                        "patch tokens (JP), each with seeds 0, 1, 2, and late fusion (LF) of T with a "
                        "logistic-regression head on the same frozen image features, fitted inside the outer loop.")
    M["RespSeedNote"] = ""
    M["RespC1"] = "Section IV-C rewritten; the branch is withdrawn and the paper makes no fusion-mechanism claim"
    M["RespC2"] = ("Branch removed from every revised configuration; it appears only in the labelled "
                   "submitted-configuration row")
    M["RespC3"] = ("The comparison built on 0.99290 is removed; the pipeline's own late fusion LF_p is the reference "
                   "of the last pre-specified comparison (R2.3)")
    M["RespMechA"] = ("the leakage-free retrain was not complete for all three training seeds, so the paper withdraws the submitted fusion "
                      "branch and makes no fusion-mechanism claim")
    M["MechSummary"] = ("the leakage-free retrain of the submitted fusion branch could not be completed in time, so no "
                        "claim about the fusion mechanism is made.")
    M["AbsMech"] = ("The submitted fusion branch, whose image features leaked fold labels, is withdrawn; a leakage-free "
                    "comparison of fusion mechanisms is left to future work.")
    M["MechText"] = "\\rev{" + M["AbsMech"] + "}"
    M["ConclText"] = ("\\rev{The submitted fusion branch was withdrawn because its image features leaked fold labels; a "
                      "leakage-free comparison of fusion mechanisms is left to future work.}")
    M["RespSum4"] = ("Five meta-seeds for every meta stage, a paired class-stratified bootstrap 95% CI that includes the "
                     "meta-seed spread for the one pre-specified comparison that could be run (the full pipeline vs its "
                     "own late fusion), and actual error counts everywhere. The planned fusion-head runs with three "
                     "training seeds could not be completed in time, so the base branches remain single training runs.")
    M["RespSum5"] = "The submitted fusion branch is withdrawn and the paper makes no fusion-mechanism claim; "

# Section V-A run-in heading: each half only if the data support it (exploratory comparisons; in the MECH state the
# image half also needs pre-specified comparison (1), LF vs T, resolved in LF's favour)
_tv = erow(PRIMARY, "text vs image (best unimodal members)")
_text_leads = bool(_tv is not None and _tv.resolved and _tv.d_f1_mean > 0)
_img_needed = bool(_lfa is not None and _lfa.resolved and _lfa.d_f1_mean > 0
                   and (not MECH or (C["Co"].resolved_holm and C["Co"].d_f1_mean > 0)))
M["VAHead"] = ("Text leads, and the image is not redundant" if _text_leads and _img_needed else
               "Text leads" if _text_leads else "The image is not redundant" if _img_needed else
               "Branches and modalities")

if "Cf" in C:
    r = C["Cf"]
    ref ="Table~\\ref{tab:mech}, last row" if MECH else f"$\\Delta$F1 {M['CfD']}"
    M["PDeltaErr"] = sgn(r.d_err_mean, "+.1f")
    M["PDeltaVerdict"] = ("resolved, " if r.resolved_holm else "not resolved, ") + pstr("Cf")
    # abstract clause (D20 wording; Holm p written out because p_H is defined only in the body)
    _hp = f"Holm-adjusted $p{{=}}{r.p_holm:.3f}$" if r.p_holm >= 0.001 else "Holm-adjusted $p{<}0.001$"
    if len(C) == 1:  # a family of one: Holm adjustment does nothing, so do not suggest a multiplicity correction
        _hp = (f"$p{{=}}{r.p_holm:.3f}$" if r.p_holm >= 0.001 else "$p{<}0.001$") + ", the single pre-specified comparison"
    _lfn = "late fusion of MuRIL-Large and EVA-02-Large"  # the abstract does not define the short form EVA-02-L
    # a differing pre-specified dedup verdict of (4) is stated wherever the all-rows verdict is summarised
    cfq = dd_qual("Cf") if "Cf" in CD else ""
    _hpq = _hp + (f"; {cfq.strip()[1:-1]}" if cfq else "")
    if not r.resolved_holm:
        M["AbsMeta"] = (f"Under nested evaluation, the full pipeline and {_lfn} are not distinguishable "
                        f"({sgn(r.d_err_mean, '+.1f')} errors on average, {_hpq})")
    elif consistent(r):
        M["AbsMeta"] = (f"Under nested evaluation, the full pipeline makes {abs(r.d_err_mean):.1f} "
                        f"{'fewer' if r.d_f1_mean > 0 else 'more'} errors than {_lfn} ({_hpq})")
    else:
        M["AbsMeta"] = (f"Under nested evaluation, the full pipeline scores {'higher' if r.d_f1_mean > 0 else 'lower'} "
                        f"than {_lfn} in macro-F1 ({_hpq})")
    # the optimism comparison is outside the pre-specified family (D20: labelled exploratory)
    opt = ("in-sample evaluation understates its error count by \\numOptimismErr{} relative to nested evaluation "
           "(exploratory)" if OPT_RESOLVED
           else "its in-sample and nested error counts are not distinguishable (exploratory)")
    # the same comparison on the dedup rows (pre-specified, D18/D20)
    dd = ""
    if "Cf" in CD:
        dd = " On the dedup rows, " + diff_phrase("Cf", "the full pipeline", LFP, D=CD) + "."
    insample_fewer = bb is not None and lf is not None and bb.insample_err_mean < lf.insample_err_mean
    if not r.resolved_holm:
        M["MetaText"] = f"\\rev{{The full pipeline is not distinguishable from {LFP} ({ref}), and {opt}.{dd}}}"
        M["DeployText"] = (f"\\rev{{Because the full pipeline was not distinguishable from {LFP} under nested "
                           f"evaluation{cfq}, two branches combined by one fitted weight is the configuration we would "
                           "deploy.}")
        verdict = f"under nested evaluation it is not distinguishable from {LFP}{cfq}"
    elif r.d_f1_mean < 0:
        amount = (f"makes {abs(r.d_err_mean):.1f} more errors than {LFP}" if consistent(r)
                  else f"scores lower than {LFP} in macro-F1")
        M["MetaText"] = (f"\\rev{{The full pipeline {amount} (resolved, {ref})"
                         + (", although in-sample it appeared to make fewer" if insample_fewer else "") + f".{dd}}}")
        M["DeployText"] = (f"\\rev{{Because the full pipeline did worse than {LFP} under nested evaluation{cfq}, two "
                           "branches combined by one fitted weight is the configuration we would deploy.}")
        verdict = f"under nested evaluation it is worse than {LFP}{cfq}"
    else:
        amount = (f"makes {abs(r.d_err_mean):.1f} fewer errors than {LFP}" if consistent(r)
                  else f"scores higher than {LFP} in macro-F1")
        M["MetaText"] = f"\\rev{{The full pipeline {amount} (resolved, {ref}); {opt}.{dd}}}"
        M["DeployText"] = (f"\\rev{{The full pipeline's gain over {LFP}{cfq} has to be weighed against serving "
                           f"{WORDS[N_BRANCHES]} branches and four fitted stages.}}")
        verdict = f"under nested evaluation it still improves on {LFP}{cfq}"
    # a joint head that beat late fusion in the matched comparison changes the deployment advice
    _better = [k for k in ("Ct", "Cth") if MECH and k in C and C[k].resolved_holm and C[k].d_f1_mean > 0]
    if _better:
        k = _better[0]
        _how = "made fewer errors than" if consistent(C[k]) else "scored higher in macro-F1 than"
        M["DeployText"] = M["DeployText"][:-1] + (f" A joint head {_how} late fusion in the matched comparison"
                                                  f"{dd_qual(k)}; that gain has to be weighed against serving both "
                                                  "encoders in one graph.}")
    # conclusion sentence on in-sample optimism (F02): names the apparent gain and the optimism separately
    gain = (f"fitted and scored on the same rows, the full pipeline appeared to make \\numInGainErr{{}} fewer errors "
            f"than {LFP}, and " if insample_fewer else "")
    _opt_c = (opt.replace("its error count", "the full pipeline's error count") if OPT_RESOLVED
              else "the full pipeline's in-sample and nested error counts are not distinguishable (exploratory)")
    M["OptText"] = ("\\rev{Stages fitted on the predictions they are scored on should be evaluated in a nested loop: "
                    f"{gain}{_opt_c}; {verdict}.}}")
    # D25 was adopted after the clean3 result of comparison (4) had been printed (not P's): disclose it in that state
    if PRIMARY == "clean3":
        same4 = (round(r.ci_lo, 4) == round(r.ci_lo_normal, 4) and round(r.ci_hi, 4) == round(r.ci_hi_normal, 4))
        M["TRefNote"] = (" The $t$ reference replaced a normal one after the comparison with " + LFP + " had been "
                         "computed; it can only widen intervals" + (" and leaves that one unchanged to four decimals."
                                                                    if same4 else "."))
        M["RespTNote"] = ("; the comparison reported below had already been computed with the normal reference, and the "
                          "change " + ("leaves its interval unchanged to four decimals" if same4 else
                                       "widens its interval"))
    else:
        M["TRefNote"], M["RespTNote"] = "", ""

# ---- response-file tables (Markdown; plain text, filled into response_draft.md by revision/fill_response.py)
if prim is not None and len(prim):
    lines = ["| Rows | Comparison (A vs B) | Mean dF1 [95% CI] | Mean d errors | dF1 per seed | df | Holm p | Resolved "
             "| McNemar, seed 0 (b / c, exact p) |",
             "|---|---|---|---|---|---|---|---|---|"]
    CLAB = {"(1)": "(1) LF vs T", "(2)": "(2) J1 vs LF", "(3)": "(3) JP vs LF",
            "(4)": "(4) full revised pipeline vs its own late fusion LF_p"}
    for _, r in prim.iterrows():
        rows_lab = "all 6,323" if r.subset == "all_6323" else "dedup 5,321"
        r = r.copy()
        r["comparison"] = CLAB.get(str(r.comparison)[:3], r.comparison)
        seeds_ = ", ".join(f"{x:+.4f}" for x in ast.literal_eval(str(r.d_f1_per_seed)))
        hp = f"{r.p_holm:.3f}" if r.p_holm >= 0.001 else "< 0.001"
        dfv = "inf" if not np.isfinite(r.df_satt) else f"{r.df_satt:.0f}"
        mcn = (f"{int(r.mcnemar_seed0_b)} / {int(r.mcnemar_seed0_c)}, "
               + (f"{r.mcnemar_seed0_p:.3f}" if r.mcnemar_seed0_p >= 0.001 else f"{r.mcnemar_seed0_p:.0e}"))
        lines.append(f"| {rows_lab} | {r.comparison} | {r.d_f1_mean:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | "
                     f"{r.d_err_mean:+.1f} | {seeds_} | {dfv} | {hp} | {'yes' if r.resolved_holm else 'no'} | {mcn} |")
    M["RespPrimaryTable"] = "\n".join(lines)
_tr = rd(O / "n05/test_results.csv")
if _tr is not None and len(_tr):
    # only the systems the paper names, under the paper's names (n05 scores every frozen system; the rest stay in
    # analysis/out/n05/test_results.csv)
    SHOW = [(f"{PRIMARY}:bias_boot", "revised pipeline (full)"), (f"{PRIMARY}:late_fusion", "LF_p"),
            ("published:bias_boot", "submitted configuration (transductive, leaky member)")]
    if MECH:
        SHOW += [(f"M:{a}", a) for a in ("T", "LF", "J1") + (("JP",) if "Cth" in C else ())]
    lines = ["| System | Refits / seeds | Macro-F1 mean (SD) | Errors mean (min-max) | Row-bootstrap 95% CI "
             "| Dedup macro-F1 |", "|---|---|---|---|---|---|"]
    for sysid, name in SHOW:
        q = _tr[_tr.system == sysid]
        if q.empty:
            continue
        r = q.iloc[0]
        sd_ = "" if not np.isfinite(r.test_f1_sd) else f" ({r.test_f1_sd:.5f})"
        lines.append(f"| {name} | {int(r.n_seeds)} | {r.test_f1_mean:.5f}{sd_} | {r.test_err_mean:.1f} "
                     f"({r.test_err_min:g}-{r.test_err_max:g}) | [{r.row_boot_ci_lo:.5f}, {r.row_boot_ci_hi:.5f}] | "
                     f"{r.dedup_f1_mean:.5f} |")
    M["RespTestTable"] = "\n".join(lines)
# R3.4: exploratory comparisons (no multiplicity adjustment) and per-stage error counts, primary configuration
if ex is not None and len(ex):
    EXL = [("text vs image (best unimodal members)", "MuRIL-L vs EVA-02-L (single runs)"),
           ("late fusion vs best unimodal", "LF_p vs MuRIL-L"),
           ("learned blend vs uniform mean", "learned blend vs uniform mean"),
           ("stack vs blend", "stacker vs blend"),
           ("alpha-mix with stack vs blend", "alpha-mix vs blend"),
           ("bootstrap bias vs alpha-mix", "+ bias (bootstrap) vs alpha-mix")]
    lines = ["| Rows | Comparison (A vs B) | Mean dF1 [95% CI] | Mean d errors | McNemar, seed 0 (b / c, exact p) |",
             "|---|---|---|---|---|"]
    for sub_, rows_lab in (("all_6323", "all 6,323"), ("dedup_5321", "dedup 5,321")):
        for comp, lab in EXL:
            q = ex[(ex.config == PRIMARY) & (ex.comparison == comp) & (ex.subset == sub_)]
            if q.empty:
                continue
            r = q.iloc[0]
            pm = f"{r.mcnemar_seed0_p:.3f}" if r.mcnemar_seed0_p >= 0.001 else f"{r.mcnemar_seed0_p:.0e}"
            lines.append(f"| {rows_lab} | {lab} | {r.d_f1_mean:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | "
                         f"{r.d_err_mean:+.1f} | {int(r.mcnemar_seed0_b)} / {int(r.mcnemar_seed0_c)}, {pm} |")
    if _o is not None and len(_o):
        r = _o.iloc[0]
        pm = f"{r.mcnemar_seed0_p:.3f}" if r.mcnemar_seed0_p >= 0.001 else f"{r.mcnemar_seed0_p:.0e}"
        lines.append(f"| all 6,323 | full pipeline, in-sample vs nested (optimism) | {r.d_f1_mean:+.4f} "
                     f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.d_err_mean:+.1f} | {int(r.mcnemar_seed0_b)} / "
                     f"{int(r.mcnemar_seed0_c)}, {pm} |")
    M["RespExplTable"] = "\n".join(lines)
SLAB = [("uniform", "uniform mean"), ("late_fusion", "late fusion LF_p"), ("blend", "convex blend"),
        ("stack", "GBM stacker"), ("alpha_mix", "alpha-mix"), ("bias_single", "+ bias, single fit"),
        ("bias_boot", "+ bias, bootstrap (full pipeline)")]
lines = ["| Stage | Nested errors, mean (range) | In-sample errors | Dedup errors (nested) |", "|---|---|---|---|"]
for s_, lab in SLAB:
    r = row(PRIMARY, s_)
    if r is None:
        continue
    rng = (f"{r.nested_err_mean:g}" if r.nested_err_min == r.nested_err_max else
           f"{r.nested_err_mean:g} ({int(r.nested_err_min)}-{int(r.nested_err_max)})")
    lines.append(f"| {lab} | {rng} | {r.insample_err_mean:g} | {r.dedup_err_mean:g} |")
M["RespStageErrTable"] = "\n".join(lines)
# R2.3, full state with JP: why J1 and JP give near-identical per-seed differences, and JP's conditional entry
M["RespJJ"], M["RespJPCond"] = "", ""
_z2 = O / "n02/preds.npz"
if MECH and "Cth" in C and _z2.exists():
    _zz = np.load(_z2)
    _yy = rd(ROOT / "drive_oof/x/meta/folds_canonical.csv").label.values.astype(int)
    _diff = [int((_zz[f"oof__J1__s{s}"] != _zz[f"oof__JP__s{s}"]).sum()) for s in (0, 1, 2)]
    _ej = [int((_zz[f"oof__J1__s{s}"] != _yy).sum()) for s in (0, 1, 2)]
    _ep = [int((_zz[f"oof__JP__s{s}"] != _yy).sum()) for s in (0, 1, 2)]
    _same = [s for s in (0, 1, 2) if _ej[s] == _ep[s]]
    _btw = [int((_zz[f"oof__{h}__s{a}"] != _zz[f"oof__{h}__s{b}"]).sum()) for h in ("J1", "JP")
            for a, b in ((0, 1), (0, 2), (1, 2))]
    M["RespJJ"] = (f" J1 and JP share the seed and the text fine-tuning recipe and differ only in the image tokens. Their "
                   f"OOF predictions differ on {_diff[0]}, {_diff[1]} and {_diff[2]} of the 6,323 rows (seeds 0-2), "
                   f"fewer than two seeds of the same head differ ({min(_btw)}-{max(_btw)} rows)"
                   + (f"; in seed{'s' if len(_same) > 1 else ''} {' and '.join(str(s) for s in _same)} they make the "
                      f"same number of errors ({', '.join(str(_ej[s]) for s in _same)}) on different rows, which is why "
                      "their per-seed differences from LF coincide to four decimals" if _same else "")
                   + ". The two comparisons are therefore strongly correlated; Holm's procedure is valid under any "
                   "dependence.")
    _raw = prim[prim.subset == "all_6323"].copy()
    _raw = _raw[~_raw.comparison.str.startswith("(3)")]
    _ps = sorted(_raw.p.tolist())
    _m = len(_ps)
    _adj, _run = {}, 0.0
    for i_, p_ in enumerate(_ps):
        _run = max(_run, min(1.0, (_m - i_) * p_))
        _adj[p_] = _run
    _p2 = _adj[float(_raw[_raw.comparison.str.startswith("(2)")].p.iloc[0])]
    M["RespJPCond"] = (" Comparison (3) was to enter the family only if all three JP seeds finished before a cut-off "
                       "fixed in advance (otherwise Holm over three comparisons); they did. With three comparisons the "
                       f"Holm-adjusted p of (2) would have been {_p2:.3f}; no verdict changes.")
# R2.3, full state only: what the D30 code-path dry run showed before the real fusion-head results existed
M["RespDryRun"] = (" After the four comparisons were fixed, a code-path dry run of the analysis on synthetic fusion-head "
                   "outputs also printed the real comparison (4) for this pipeline and its per-class misses; the choice "
                   "of primary pipeline follows the rule fixed before (all seeds of T and J1 and five meta-seeds "
                   "present)." if MECH and PRIMARY == "P" else "")
# R2.3: post-hoc controls (n09, D37)
M["RespLFCtl"], M["RespJ1Seed"] = "", ""
if MECH and _c9 is not None and _a9 is not None:
    t9 = _c9["all_6323"]
    sd9 = [int(s) for s in t9.columns]
    _l = lambda arm: ", ".join(str(int(t9.loc[arm, s])) for s in sd9)  # noqa: E731
    _fs = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv").fold.value_counts()
    _ntr = sorted(int(_fs.sum() - v) for v in _fs.values)
    _h9 = [h for h in ("J1", "JP") if h in t9.index]
    M["RespLFCtl"] = (
        "Post-hoc control (not part of the pre-specified family): LF's weight α is fitted on the outer-training rows, "
        "whose text predictions come from fold models that saw the evaluation fold (the scope note in R2.1). Fitting α "
        "instead on one half of each evaluation fold (halves formed from duplicate groups, so identical captions and "
        "near-duplicate images stay together; both inputs exclude that fold) and applying it to the other half gives "
        f"LF {_l('LF_halffold')} errors for seeds {', '.join(str(s) for s in sd9)} (pre-specified LF {_l('LF')}; "
        + "; ".join(f"{h} {_l(h)}" for h in _h9) + f"; T {_l('T')}); with α fixed at 0.5, LF makes "
        f"{_l('LF_fixed05')}. LF makes far fewer errors than T under every choice, but its order relative to the joint "
        "heads depends on how α is fitted (Section V-B now says so). The half-fold fit uses "
        f"{int(_a9.n_fit.min())}-{int(_a9.n_fit.max())} rows instead of {_ntr[0]:,}-{_ntr[-1]:,}, so the change "
        "cannot be attributed to the indirect path alone.")
if _j9 is not None and PRIMARY == "P":
    _g = _j9[_j9.stage == "bias_boot"].groupby("config").err.mean()
    _lp = _j9[(_j9.stage == "late_fusion") & (_j9.config == "P")].err.mean()
    if all(c in _g.index for c in ("P", "P_s1", "P_s2")):
        M["RespJ1Seed"] = (
            f" Exploratory: the full pipeline uses J1 trained with seed 0; with J1 seed 1 or 2 instead (other members "
            f"unchanged) it makes {_g['P_s1']:.1f} and {_g['P_s2']:.1f} nested errors (mean over meta-seeds; seed 0: "
            f"{_g['P']:.1f}; LF_p: {_lp:.1f}). These runs vary one member only and are not independent replications.")

# ---- C14 addendum: which copy of the images the duplicate counts and the frozen mask refer to (n10, D37)
M["RespImgCopy"] = ""
_ic = O / "n10/image_copies.json"
if _ic.exists():
    _c = json.loads(_ic.read_text(encoding="utf-8"))
    _pd = _c["pair_phash_dist_counts"]
    assert _c["frozen_mask_matches_working"] and _c["md5_equal"] == 0
    M["RespImgCopy"] = (
        " These counts and the frozen mask use our working files. "
        f"All {_c['n_train']:,} training images differ in bytes from their public counterparts. "
        "Their dimensions agree and their perceptual-hash distances are "
        + ", ".join(f"{k} for {v:,}" for k, v in sorted(_pd.items(), key=lambda kv: int(kv[0]))) + ". On the public training "
        f"files the same procedure finds {_c['public']['cross_fold_d0']:,} rows with a distance-0 cross-fold twin "
        f"(ours: {_c['working']['cross_fold_d0']:,}) and {_c['public']['cross_fold_le4']:,} at distance ≤ 4 (ours: "
        f"{_c['working']['cross_fold_le4']:,}); {_c['le4_flag_diff']} rows change status, and the dedup subset has "
        f"{_c['public']['dedup_keep']:,} rows in both cases. We will release the row-level mask at camera-ready.")

# The optional grouped refit is a distinct post-hoc experiment on the training split.
M["RespGroupedRefit"] = ""
_grouped_path = O / "n11/grouped_refit.json"
if _grouped_path.exists():
    _grouped = json.loads(_grouped_path.read_bytes())
    _pooled = _grouped["pooled"]
    M["RespGroupedRefit"] = (
        f"An additional post-hoc check refitted simpler models on the {_grouped['n_rows']:,} training rows. "
        f"We kept {_grouped['n_groups']:,} connected groups of identical normalized captions or near-duplicate "
        "images (pHash distance at most four) together in five new folds. For each evaluation fold, three folds "
        "trained a character n-gram text classifier and a logistic classifier on frozen EVA features. "
        "A fourth fold chose the text weight on a grid from 0 to 1 in steps of 0.005. The fifth fold supplied "
        f"only evaluation predictions. Pooled across the five rotations, text made {_pooled['caption']['errors']} "
        f"errors (macro-F1 {_pooled['caption']['macro_f1']:.5f}), image {_pooled['image']['errors']}, "
        f"equal weighting {_pooled['equal']['errors']}, and fitted late fusion {_pooled['calibrated']['errors']} "
        f"(macro-F1 {_pooled['calibrated']['macro_f1']:.5f}). This checks the image contribution with "
        "disjoint duplicate groups. Evaluation of the finalized ensemble on unseen events remains open.")

# ---- C22: protocol of the prior-work accuracies in Table IV (D37). The values are copied from the benchmark paper;
# test whether they are attainable as exact accuracies on 790 or 1,580 rows (two-decimal percentages)
_prior = {"mBERT": 79.90, "Swin": 78.65, "DisasterMultiFusionNet": 85.25}


def _attainable(v, n):
    return any(round(100 * k / n, 2) == v for k in range(n + 1))


_bad = {n: [m for m, v in _prior.items() if not _attainable(v, n)] for n in (790, 1580)}
M["RespC22"] = ""
if len(_bad[790]) == len(_prior):
    M["RespC22"] = ("| C22 | Prior-work accuracies (Table IV) attributed to all 790 test rows | The benchmark "
                    "authors report " + ", ".join(f"{v:.2f}%" for v in _prior.values()) + ". Their public Swin+mBERT "
                    "early-fusion notebook evaluates image-matched test rows and contains no saved outputs. "
                    "We could not verify the exact denominator or run behind the published figures. | Table IV "
                    "reports their accuracies as published and identifies our metric as macro-F1. |")

# ---- submitted blend weight of PL-MuRIL, as printed by the archived competition notebook
nb = ROOT / "analysis/verify_tmp/meta_audit/final_nb_dump.txt"
if nb.exists():
    # anchored to the "Weights:" block; the first pl_muril line of the dump is its OOF F1, not its weight
    mm = re.search(r"Weights:.*?pl_muril\s*=\s*([0-9.]+)", nb.read_text(encoding="utf-8", errors="replace"), re.S)
    if mm:
        assert float(mm.group(1)) < 1, f"PlBlendW parse error: {mm.group(1)}"
        M["PlBlendW"] = mm.group(1)

# ---- epoch-selection effect (D12), measured on the G runs by n02
es = rd(O / "n02/epoch_selection.csv")
if es is not None and len(es):
    M["EpochSelErr"] = sgn(es.d_err.mean(), "+.1f")
    M["EpochSelFone"] = f"{es.d_f1.mean():+.5f}"
if MECH and "EpochSelErr" in M:
    M["RespC11"] = ("Each original branch kept the best epoch chosen on its own evaluation fold; new runs use the final "
                    "epoch; on the new runs, picking each fold's best epoch on its evaluation fold would change the OOF "
                    f"error count by {M['EpochSelErr']} on average (macro-F1 {M['EpochSelFone']})")
else:  # D23 fallback: no completed new runs to measure the effect on
    M["RespC11"] = ("Each original branch kept the best epoch chosen on its own evaluation fold, so its OOF score is "
                    "optimistic")

# ---- per-class recall under nested evaluation (n07; OOF rows only), full pipeline of the primary config
pc = rd(O / "n07/per_class_recall.csv")
if pc is not None and len(pc):
    q = pc[(pc.system == f"{PRIMARY}:bias_boot") & (pc.subset == "all_6323")]
    if len(q) and q.n_seeds.iloc[0] == 5:
        def floor4(x):  # quoted as "at least": round down (guarding against float error), never up
            return f"{math.floor(round(x * 1e4, 6)) / 1e4:.4f}"
        w = q.loc[q.recall_mean.idxmin()]
        M["PcMinRecall"], M["PcMinClass"] = floor4(w.recall_mean), str(w.cls)
        M["PcMaxMissed"] = f"{q.missed_mean.max():g}"
        qd = pc[(pc.system == f"{PRIMARY}:bias_boot") & (pc.subset == "dedup_5321")]
        if len(qd):
            M["PcMinRecallDd"] = floor4(qd.recall_mean.min())

# ---- 1,580-row partition (descriptive, D21)
tr = rd(O / "n05/test_results.csv")
if tr is not None:
    def trow(system):
        q = tr[tr.system == system]
        return None if q.empty else q.iloc[0]
    r = trow(f"{PRIMARY}:bias_boot")
    if r is not None:
        M["TestPFone"] = f5(r.test_f1_mean)
        # mean (range) over the five meta-seed refits, as for the nested error counts
        M["TestPErr"] = (f"{r.test_err_mean:g}" if r.test_err_min == r.test_err_max else
                         f"{r.test_err_mean:g} ({int(r.test_err_min)}--{int(r.test_err_max)})")
        M["TestPDdFone"] = f5(r.dedup_f1_mean)
    r_lf = trow(f"{PRIMARY}:late_fusion")
    if r_lf is not None:
        M["TestLFFone"] = f5(r_lf.test_f1_mean)
        M["TestLFErr"] = f"{r_lf.test_err_mean:g}"
        # V-D: the full pipeline vs LF_p, and the matched heads, on the same frozen single scoring (descriptive);
        # an ordering that differs from the nested / OOF results is stated, not weighed against them
        M["TestDirNote"], M["RespTestNote"] = "", ""
        tex_p, txt_p = [], []
        if r is not None:
            M["TestPErrMean"] = f"{r.test_err_mean:g}"
            if r.test_err_mean < r_lf.test_err_mean and not ("Cf" in C and C["Cf"].resolved_holm):
                tex_p.append("the full pipeline makes fewer errors than LF$_\\text{p}$ (\\numTestPErrMean{} against "
                             "\\numTestLFErr{})")
                txt_p.append(f"the full pipeline makes fewer errors than LF_p ({r.test_err_mean:g} against "
                             f"{r_lf.test_err_mean:g}), whereas under nested evaluation the two were not "
                             "distinguishable (comparison (4))")
        arms = [a for a in ("T", "LF", "J1", "JP") if trow(f"M:{a}") is not None and (a != "JP" or "Cth" in C)]
        if MECH and {"T", "LF", "J1"} <= set(arms):
            te = {a: trow(f"M:{a}").test_err_mean for a in arms}
            M["TestHeadErr"] = ", ".join(f"{te[a]:.1f}" for a in arms[:-1]) + f" and {te[arms[-1]]:.1f}"
            joint = [a for a in ("J1", "JP") if a in arms]
            rev_ = all(te[a] < te["LF"] for a in joint) and all(
                C[k].d_err_mean > 0 for k in (("Ct",) + (("Cth",) if "JP" in joint else ())))
            names = ", ".join(arms[:-1]) + f" and {arms[-1]}"
            both = "both joint heads" if len(joint) == 2 else "the joint head"
            tex_p.append(f"the matched heads {names} make \\numTestHeadErr{{}} errors" +
                         (f", so {both} make{'s' if len(joint) == 1 else ''} fewer errors than LF, the reverse of "
                          "their out-of-fold order" if rev_ else ""))
            txt_p.append(f"the matched heads {names} make {M['TestHeadErr']} errors" +
                         (f", so {both} make{'s' if len(joint) == 1 else ''} fewer errors than LF, the reverse of "
                          "their out-of-fold order" if rev_ else ""))
        if tex_p:
            M["TestDirNote"] = (" \\rev{Here " + "; ".join(tex_p) + ". These labels were used during development "
                                "(Section~\\ref{sec:data}), so all these figures are descriptive only.}")
            M["RespTestNote"] = ("On these rows " + "; ".join(txt_p) + ". Because these labels were used during "
                                 "development, the paper reports these figures descriptively (Section V-D) and does not "
                                 "set them against the pre-specified comparisons.")
        else:
            M["TestDirNote"] = (" \\rev{These labels were used during development (Section~\\ref{sec:data}), so these "
                                "figures are descriptive only.}")
cfgf = O / "n05/frozen_config.json"
if cfgf.exists():
    M["FrozenHash"] = hashlib.sha256(cfgf.read_bytes()).hexdigest()[:16]

# ---- write: every macro used by the paper is defined; unknown ones show as \ph{}
used = set()
if (SRC / "main.tex").exists():
    used = set(re.findall(r"\\num([A-Za-z]+)", (SRC / "main.tex").read_text(encoding="utf-8")))
lines = ["% GENERATED by analysis/make_paper_numbers.py -- do not edit by hand.",
         "\\providecommand{\\ph}[1][]{\\textcolor{red}{[?]}}"]
for k in sorted(used | set(M)):
    if k.startswith("Resp"):  # Markdown tables for the response file only (they contain '%' and '|')
        continue
    lines.append(f"\\newcommand{{\\num{k}}}{{{M.get(k, chr(92) + 'ph{}')}}}")
# switches used by main.tex (declared there with \newif): fusion-mechanism results present (D23); P is primary
lines.append("\\mechtrue" if MECH else "\\mechfalse")
lines.append("\\primPtrue" if PRIMARY == "P" else "\\primPfalse")
lines.append("\\jptrue" if MECH and "Cth" in C else "\\jpfalse")  # D23: JP (G3) present for every seed
txt = "\n".join(lines) + "\n"
if OUT_ONLY is not None:  # layout test: write only there
    OUT_ONLY.mkdir(parents=True, exist_ok=True)
    (OUT_ONLY / "numbers.tex").write_text(txt, encoding="utf-8")
    (OUT_ONLY / "numbers_layouttest.json").write_text(json.dumps(M, indent=1), encoding="utf-8")
else:
    (GEN / "numbers.tex").write_text(txt, encoding="utf-8")
    if SRC.exists():
        (SRC / "numbers.tex").write_text(txt, encoding="utf-8")
    (GEN / "numbers.json").write_text(json.dumps(M, indent=1), encoding="utf-8")
missing = sorted(used - set(M))
print(f"{len(M)} values; primary config = {PRIMARY}; paper macros missing ({len(missing)}): {', '.join(missing)}")
