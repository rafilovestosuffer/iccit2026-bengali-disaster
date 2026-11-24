"""N10: provenance of the image files (added 2026-11-19 after an independent check of the revision, D37).

The training images used in the experiments are copies of the BanglaCalamityMMD release that differ byte-wise from the public
files. This script compares the two K1 indexes (MD5, size, 64-bit pHash) for the 6,323 training rows and recomputes the
cross-fold duplicate counts of Section III on both copies, so the response can state which copy the reported counts
and the frozen dedup mask refer to. Read-only on the inputs; the frozen mask is only compared, never rewritten.

Writes analysis/out/n10/image_copies.json.
"""
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "analysis/out"
OUTD = O / "n10"
OUTD.mkdir(parents=True, exist_ok=True)
meta2 = json.loads((O / "n02/meta.json").read_text(encoding="utf-8"))
K1 = Path(meta2["sources"]["g1_s0"]).parents[1] / "k1-imgindex-1118-1725-s0/outputs"

folds = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
fold = folds.fold.to_numpy()
keep = np.load(O / "dedup_masks.npz")["train_keep"]
ci = pd.read_csv(K1 / "contest_index.csv", dtype={"phash": str})
pi = pd.read_csv(K1 / "bcmmd_index.csv", dtype={"phash": str})
work = ci[ci.set == "contest_train"].set_index("stem").loc[folds.image_id]   # files the models read
pub = pi[pi.set == "bcmmd_train"].set_index("stem").loc[folds.image_id]     # public release
assert len(work) == len(pub) == len(folds)
same_dims = bool((work[["w", "h"]].to_numpy() == pub[["w", "h"]].to_numpy()).all())
n_md5_equal = int((work.md5.to_numpy() == pub.md5.to_numpy()).sum())
pair_d = np.array([bin(int(a, 16) ^ int(b, 16)).count("1") for a, b in zip(work.phash, pub.phash)])


def cross_fold_min_dist(index):
    h = np.array([int(x, 16) for x in index.phash], dtype=np.uint64)
    out = np.empty(len(h), dtype=int)
    for s in range(0, len(h), 128):
        x = h[s:s + 128, None] ^ h[None, :]
        d = np.zeros(x.shape, dtype=int)
        for _ in range(64):  # popcount without numpy>=2 bitwise_count
            d += (x & np.uint64(1)).astype(int)
            x = x >> np.uint64(1)
        d[fold[s:s + 128, None] == fold[None, :]] = 65
        out[s:s + 128] = d.min(1)
    return out


norm = folds.context.map(lambda s: " ".join(unicodedata.normalize("NFC", str(s)).split()))
cross_caption = folds.groupby(norm).fold.transform("nunique").to_numpy() > 1
dw, dp = cross_fold_min_dist(work), cross_fold_min_dist(pub)
keep_w, keep_p = (dw > 4) & ~cross_caption, (dp > 4) & ~cross_caption
assert np.array_equal(keep_w, keep), "frozen dedup mask is not reproduced from the working copy"
res = dict(
    status="D37; descriptive provenance check, training rows only",
    n_train=int(len(folds)), same_dimensions=same_dims, md5_equal=n_md5_equal,
    pair_phash_dist_counts={int(k): int(v) for k, v in zip(*np.unique(pair_d, return_counts=True))},
    working=dict(cross_fold_d0=int((dw == 0).sum()), cross_fold_le4=int((dw <= 4).sum()), dedup_keep=int(keep_w.sum())),
    public=dict(cross_fold_d0=int((dp == 0).sum()), cross_fold_le4=int((dp <= 4).sum()), dedup_keep=int(keep_p.sum())),
    le4_flag_diff=int(((dw <= 4) != (dp <= 4)).sum()), keep_diff=int((keep_w != keep_p).sum()),
    frozen_mask_matches_working=True,
    inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (K1 / "contest_index.csv", K1 / "bcmmd_index.csv", O / "dedup_masks.npz",
             ROOT / "drive_oof/x/meta/folds_canonical.csv")})
(OUTD / "image_copies.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "inputs"}, indent=1))
