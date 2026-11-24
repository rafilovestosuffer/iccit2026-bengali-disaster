"""N05: score the 1,580-row partition ONCE per pre-registered system (DECISION_LOG D7, D21, D22).

Step 1 (freeze):  python n05_score_test_once.py freeze
    Lists every system to score: each test__<stage> array of analysis/out/n01/preds_{config}_m{seed}.npz
    (configs published, clean3, P, P_s1, P_s2 when present; meta-seeds 0-4) and each test__<arm>__s<seed> array of
    analysis/out/n02/preds.npz. Writes analysis/out/n05/frozen_config.json with the SHA-256 of every prediction file
    and of the label file, then prints the SHA-256 of frozen_config.json. That hash must be appended to
    DECISION_LOG.md BEFORE step 2. Refuses to freeze unless clean3 and published have all 5 meta-seeds.
Step 2 (score):   python n05_score_test_once.py score <sha256-of-frozen_config.json>
    Refuses if the hash is not logged, if any file changed, or if results already exist. Scores every system once
    on all 1,580 rows and on the D18 dedup subset (test rows with no near-duplicate train image or caption), and writes
    analysis/out/n05/test_results_per_seed.csv, test_results.csv (mean, sd over seeds; descriptive only, D21) and
    test_preds.npz.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from n03_stats import Bootstrapper  # noqa: E402

OUTD = ROOT / "analysis/out/n05"
CFG = OUTD / "frozen_config.json"
RES = OUTD / "test_results.csv"
LABELS = ['Drought', 'Earthquake', 'Flood', 'Human Damage', 'Landslides', 'Non Disaster', 'Tropical Storm', 'Wildfire']
N01 = "analysis/out/n01"
MECH = "analysis/out/n02/preds.npz"
CONFIGS = ("published", "clean3", "P", "P_s1", "P_s2")
REQUIRED = ("published", "clean3")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    OUTD.mkdir(parents=True, exist_ok=True)
    if RES.exists():
        sys.exit("results already exist; the partition has been scored")
    systems = []  # (system, seed, file, key)
    for cfg in CONFIGS:
        files = {s: f"{N01}/preds_{cfg}_m{s}.npz" for s in range(5) if (ROOT / f"{N01}/preds_{cfg}_m{s}.npz").exists()}
        if cfg in REQUIRED and len(files) < 5:
            sys.exit(f"{cfg}: only meta-seeds {sorted(files)} present; all 5 required before freezing")
        for s, f in files.items():
            for k in np.load(ROOT / f).files:
                if k.startswith("test__"):
                    systems.append((f"{cfg}:{k[6:]}", s, f, k))
    if (ROOT / MECH).exists():
        for k in np.load(ROOT / MECH).files:
            if k.startswith("test__") and "__s" in k:
                arm, s = k[6:].rsplit("__s", 1)
                systems.append((f"M:{arm}", int(s), MECH, k))
    files = sorted({f for _, _, f, _ in systems})
    cfg = {"systems": [dict(system=a, seed=b, file=c, key=d) for a, b, c, d in systems],
           "file_sha256": {f: sha(ROOT / f) for f in files},
           "labels_file": "analysis/out/test_labels_bcmmd.csv", "labels_sha256": sha(ROOT / "analysis/out/test_labels_bcmmd.csv"),
           "dedup_mask": "analysis/out/dedup_masks.npz:test_keep", "dedup_sha256": sha(ROOT / "analysis/out/dedup_masks.npz")}
    CFG.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print("frozen", len(systems), "system-seed entries from", len(files), "files ->", CFG)
    print("SHA-256(frozen_config.json) =", sha(CFG))
    print("Append this hash to DECISION_LOG.md, then run: n05_score_test_once.py score <hash>")


def score(h):
    if RES.exists():
        sys.exit("results already exist; refusing to score twice")
    if sha(CFG) != h:
        sys.exit("frozen_config.json hash mismatch")
    if h not in (ROOT / "DECISION_LOG.md").read_text(encoding="utf-8"):
        sys.exit("hash not logged in DECISION_LOG.md")
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    for f, s in cfg["file_sha256"].items():
        if sha(ROOT / f) != s:
            sys.exit(f"prediction file changed after freezing: {f}")
    if sha(ROOT / cfg["labels_file"]) != cfg["labels_sha256"] or sha(ROOT / "analysis/out/dedup_masks.npz") != cfg["dedup_sha256"]:
        sys.exit("label file or dedup mask changed after freezing")
    test = pd.read_csv(ROOT / "contest/Disaster_test.csv", encoding="utf-8-sig")
    lab = pd.read_csv(ROOT / cfg["labels_file"])
    yt = test[["image_id"]].merge(lab, on="image_id", how="left")["label_bcmmd"].map({l: i for i, l in enumerate(LABELS)})
    assert yt.notna().all()
    yt = yt.values.astype(int)
    keep = np.load(ROOT / "analysis/out/dedup_masks.npz")["test_keep"]
    bs_all, bs_dd = Bootstrapper(yt), Bootstrapper(yt[keep])
    rows, preds, boots = [], {}, {}
    for e in cfg["systems"]:
        p = np.load(ROOT / e["file"])[e["key"]].argmax(1)
        f1, err = bs_all.point(p)
        f1d, errd = bs_dd.point(p[keep])
        rows.append(dict(system=e["system"], seed=e["seed"], test_f1=f1, test_err=err, dedup_f1=f1d, dedup_err=errd))
        preds[f"{e['system']}|s{e['seed']}"] = p
        boots.setdefault(e["system"], []).append(bs_all.boot_f1(p))
    per = pd.DataFrame(rows)
    per.to_csv(OUTD / "test_results_per_seed.csv", index=False)
    agg = per.groupby("system", sort=False).agg(n_seeds=("seed", "size"), test_f1_mean=("test_f1", "mean"),
                                                test_f1_sd=("test_f1", "std"), test_err_mean=("test_err", "mean"),
                                                test_err_min=("test_err", "min"), test_err_max=("test_err", "max"),
                                                dedup_f1_mean=("dedup_f1", "mean"), dedup_err_mean=("dedup_err", "mean")).reset_index()
    ci = {k: np.percentile(np.mean(v, axis=0), [2.5, 97.5]) for k, v in boots.items()}
    agg["row_boot_ci_lo"] = agg.system.map(lambda k: ci[k][0])
    agg["row_boot_ci_hi"] = agg.system.map(lambda k: ci[k][1])
    agg["note"] = "descriptive only (D21)"
    agg.to_csv(RES, index=False)
    np.savez_compressed(OUTD / "test_preds.npz", y=yt, dedup_keep=keep, **{f"p{i}": v for i, v in enumerate(preds.values())})
    (OUTD / "test_preds_index.json").write_text(json.dumps(list(preds.keys()), indent=1), encoding="utf-8")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "freeze":
        freeze()
    elif len(sys.argv) == 3 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        print(__doc__)
