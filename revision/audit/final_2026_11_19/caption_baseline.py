"""Exploratory fixed character n-gram baseline using only the canonical training split."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[3]
rows = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
truth = rows.label.to_numpy()
predicted = np.full(len(rows), -1)
assert rows.context.notna().all() and rows.image_id.is_unique
for k in sorted(rows.fold.unique()):
    train = rows.fold != k
    valid = ~train
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        LinearSVC(C=1.0, dual=True, random_state=284519, max_iter=4000),
    )
    model.fit(rows.context[train], truth[train])
    predicted[valid] = model.predict(rows.context[valid])
assert np.all(predicted >= 0)
with np.load(ROOT / "analysis/out/dedup_masks.npz") as masks:
    dedup = masks["train_keep"]
for name, keep in [("all", np.ones(len(rows), bool)), ("dedup", dedup)]:
    f1 = f1_score(truth[keep], predicted[keep], labels=range(8), average="macro")
    print(name, int(keep.sum()), "macro-F1", f"{f1:.6f}", "errors", int((predicted[keep] != truth[keep]).sum()))
