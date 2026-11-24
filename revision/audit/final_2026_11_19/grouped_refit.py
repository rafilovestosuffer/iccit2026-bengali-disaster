"""Exploratory duplicate-group cross-validation of caption and frozen-image models.

Each evaluation group is excluded from both supervised base fits and scalar
weight selection. Only the original training rows are scored. The caption
vocabulary and image-feature standardization are learned on the base-fit rows.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from duplicate_group_sensitivity import ROOT, duplicate_groups


def macro(truth, prediction):
    return f1_score(truth, prediction, average="macro", labels=range(8))


def main():
    rows = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
    feature_rows = pd.read_csv(ROOT / "jobs/g_fusion/rows.csv.gz")
    train_indices = np.flatnonzero(feature_rows.split == "train")
    train_rows = feature_rows.iloc[train_indices]
    assert rows.image_id.is_unique and rows.context.notna().all()
    for col in ("image_id", "label", "context"):
        assert rows[col].reset_index(drop=True).equals(train_rows[col].reset_index(drop=True))
    meta = json.loads((ROOT / "analysis/out/n02/meta.json").read_bytes())
    feat_path = Path(meta["eva_pooled"])
    frozen = np.load(feat_path, mmap_mode="r")
    assert frozen.shape == (len(feature_rows), 1024)
    images = frozen[train_indices].astype(np.float32)
    assert np.isfinite(images).all()
    y = rows.label.to_numpy()
    groups = duplicate_groups(rows)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=284519)
    folds = np.full(len(rows), -1)
    for k, (_, ev) in enumerate(splitter.split(rows.context, y, groups)):
        folds[ev] = k
    assert (folds >= 0).all()
    assert pd.DataFrame(dict(group=groups, fold=folds)).groupby("group").fold.nunique().max() == 1
    arms = ("caption", "image", "equal", "calibrated")
    preds = {arm: np.full(len(rows), -1) for arm in arms}
    eval_caption = np.empty((len(rows), 8))
    eval_image = np.empty_like(eval_caption)
    cal_caption = np.empty_like(eval_caption)
    cal_image = np.empty_like(eval_caption)
    runs = []
    for k in range(5):
        ev = folds == k
        cal = folds == (k + 1) % 5
        fit = ~ev & ~cal
        for subset in (fit, cal, ev):
            assert len(np.unique(y[subset])) == 8
        caption = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
            SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=50, tol=1e-4, random_state=284519),
        )
        vision = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300))
        caption.fit(rows.context[fit], y[fit])
        vision.fit(images[fit], y[fit])
        assert np.array_equal(caption.classes_, np.arange(8))
        assert np.array_equal(vision.classes_, np.arange(8))
        txt_cal, img_cal = caption.predict_proba(rows.context[cal]), vision.predict_proba(images[cal])
        weights = np.linspace(0, 1, 201)
        cal_f1 = [macro(y[cal], (a * txt_cal + (1 - a) * img_cal).argmax(1)) for a in weights]
        alpha = float(weights[np.argmax(cal_f1)])
        txt, img = caption.predict_proba(rows.context[ev]), vision.predict_proba(images[ev])
        eval_caption[ev], eval_image[ev] = txt, img
        cal_caption[cal], cal_image[cal] = txt_cal, img_cal
        for arm, scores in zip(arms, (txt, img, (txt + img) / 2, alpha * txt + (1 - alpha) * img)):
            preds[arm][ev] = scores.argmax(1)
        runs.append(dict(fold=k, n_fit=int(fit.sum()), n_cal=int(cal.sum()), n_eval=int(ev.sum()),
                         alpha=alpha, errors={a: int((preds[a][ev] != y[ev]).sum()) for a in arms}))
        print(runs[-1], flush=True)
    assert all((p >= 0).all() for p in preds.values())
    report = dict(seed=284519, n_rows=len(rows), n_groups=len(np.unique(groups)),
                  split_sha256=hashlib.sha256(folds.astype("uint8").tobytes()).hexdigest(),
                  feature_sha256=hashlib.sha256(feat_path.read_bytes()).hexdigest(),
                  folds=runs, pooled={a: dict(macro_f1=macro(y, p), errors=int((p != y).sum()))
                                      for a, p in preds.items()})
    output = ROOT / "analysis/out/n11"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "grouped_predictions.npz", row_id=rows.image_id.to_numpy(dtype=str),
                        fold=folds, group=groups, eval_caption=eval_caption, eval_image=eval_image,
                        cal_caption=cal_caption, cal_image=cal_image)
    (output / "grouped_refit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["pooled"])


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
