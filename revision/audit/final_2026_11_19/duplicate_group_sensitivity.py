"""Post-hoc paired bootstrap of duplicate groups on the training OOF predictions."""
from pathlib import Path
import json
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import t
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/out"
SEED = 284519
REPLICATES = 10_000


def duplicate_groups(rows):
    source = json.loads((OUT / "n02/meta.json").read_bytes())["sources"]["g1_s0"]
    path = Path(source).parents[1] / "k1-imgindex-1118-1725-s0/outputs/near_dups.csv"
    edges = pd.read_csv(path)
    edges = edges[(edges.kind == "train-train") & (edges.phash_dist <= 4)]
    order = dict(zip(rows.image_id, range(len(rows))))
    left, right = edges.a.map(order).tolist(), edges.b.map(order).tolist()
    caption = rows.context.map(lambda v: " ".join(unicodedata.normalize("NFC", v).split()))
    for ids in rows.groupby(caption).indices.values():
        left.extend([ids[0]] * (len(ids) - 1))
        right.extend(ids[1:])
    graph = coo_matrix((np.ones(len(left)), (left, right)), shape=(len(rows), len(rows)))
    return connected_components(graph, directed=False)[1]


def get_predictions():
    with np.load(OUT / "n02/preds.npz") as archive:
        arms = {arm: [archive[f"oof__{arm}__s{s}"] for s in range(3)]
                for arm in ("T", "LF", "J1", "JP")}
    for stage in ("bias_boot", "late_fusion"):
        arms[stage] = []
        for s in range(5):
            with np.load(OUT / f"n01/preds_P_m{s}.npz") as archive:
                arms[stage].append(archive["nested__" + stage].argmax(axis=1))
    return arms


def bootstrap(rows, groups, arms, keep):
    # Groups with identical confusion statistics are exchangeable when sampling
    # components uniformly. Mixed-label groups stay intact.
    labels = rows.label.to_numpy()[keep]
    _, index = np.unique(groups[keep], return_inverse=True)
    sizes = np.bincount(index)
    ordered = [(arm, i) for arm, runs in arms.items() for i in range(len(runs))]
    nclasses = 8

    def totals(values, weights=None):
        return np.bincount(index * nclasses + values, weights=weights,
                           minlength=len(sizes) * nclasses).reshape(-1, nclasses)

    observed = totals(labels)
    blocks = [observed]
    for arm, i in ordered:
        pred = arms[arm][i][keep]
        blocks.extend([totals(pred), totals(labels, pred == labels)])
    patterns, freq = np.unique(np.column_stack(blocks).astype(int), axis=0, return_counts=True)
    rng = np.random.default_rng(SEED)
    samples = np.empty((REPLICATES, len(ordered)))
    for start in range(0, REPLICATES, 100):
        w = rng.multinomial(len(sizes), freq / len(sizes), size=100)
        stats = (w @ patterns).reshape(100, 1 + 2 * len(ordered), nclasses)
        samples[start:start + 100] = (2 * stats[:, 2::2] / (stats[:, :1] + stats[:, 1::2])).mean(axis=2)
    pts = np.array([f1_score(labels, arms[a][s][keep], labels=range(8), average="macro") for a, s in ordered])
    actual = np.column_stack(blocks).sum(axis=0).reshape(-1, nclasses)
    assert np.allclose(pts, (2 * actual[2::2] / (actual[:1] + actual[1::2])).mean(axis=1))
    print("rows/groups/multirow/largest/mixed-label", len(labels), len(sizes), sum(sizes > 1), max(sizes),
          sum((observed > 0).sum(axis=1) > 1))
    results = []
    columns = {name: [ordered.index((name, s)) for s in range(len(runs))] for name, runs in arms.items()}
    for a, b in [("LF", "T"), ("J1", "LF"), ("JP", "LF"), ("bias_boot", "late_fusion")]:
        ca, cb = columns[a], columns[b]
        d = pts[ca] - pts[cb]
        group_var = (samples[:, ca] - samples[:, cb]).mean(axis=1).var(ddof=1)
        seed_var = d.var(ddof=1) / len(d)
        se = np.sqrt(group_var + seed_var)
        df = (len(d) - 1) * (se**2 / seed_var)**2 if seed_var else np.inf
        lo, hi = d.mean() + np.array([-1, 1]) * t.ppf(0.975, df) * se
        same_sign = (d > 0).all() or (d < 0).all()
        results.append((a, b, d.mean(), lo, hi, same_sign, 2 * t.sf(abs(d.mean() / se), df)))
    pv = np.array([r[-1] for r in results])
    rank = np.argsort(pv)
    sorted_holm = np.minimum(1, np.maximum.accumulate(pv[rank] * np.arange(4, 0, -1)))
    for (a, b, mean, lo, hi, same_sign, _), ph in zip(results, sorted_holm[np.argsort(rank)]):
        print(f"{a} minus {b}: {mean:+.6f} [{lo:+.6f}, {hi:+.6f}], "
              f"Holm {ph:.6g}, resolved {same_sign and ph < 0.05}")


if __name__ == "__main__":
    records = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
    groups = duplicate_groups(records)
    arms = get_predictions()
    with np.load(OUT / "dedup_masks.npz") as masks:
        dedup = masks["train_keep"]
    for sample in (np.ones(len(records), bool), dedup):
        bootstrap(records, groups, arms, sample)
