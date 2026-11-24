"""Build the compact row table bundled with the GPU job (run locally, once).

rows.csv.gz: one row per contest instance, train first (canonical fold order) then test.
Columns: image_id, split (train/test), fold (-1 for test), label (-1 for test), context.
Test labels are deliberately NOT included: the GPU job never sees them.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
f = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv")
te = pd.read_csv(ROOT / "contest/Disaster_test.csv", encoding="utf-8-sig")
tr = f[["image_id", "fold", "label", "context"]].assign(split="train")
te = te[["image_id", "context"]].assign(split="test", fold=-1, label=-1)
rows = pd.concat([tr, te], ignore_index=True)[["image_id", "split", "fold", "label", "context"]]
assert len(rows) == 7903 and rows.image_id.is_unique
out = Path(__file__).with_name("rows.csv.gz")
rows.to_csv(out, index=False, encoding="utf-8", compression="gzip")
print("wrote", out, out.stat().st_size, "bytes", rows.split.value_counts().to_dict())
