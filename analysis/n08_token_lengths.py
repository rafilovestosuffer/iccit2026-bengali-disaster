"""Caption token lengths: how many captions exceed each text branch's max_len (truncation), as trained.

BanglaBERT-base (csebuetnlp/banglabert) used max_len 160 and MuRIL-Large (google/muril-large-cased) 256, with the raw
`context` string and no normalizer (ml-contest-2 notebook, TextDS). Counts include [CLS]/[SEP]. Captions only, no labels.
Writes analysis/out/n08/token_lengths.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "analysis/out/n08"
OUTD.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv").context.astype(str).tolist()
te = pd.read_csv(ROOT / "contest/Disaster_test.csv", encoding="utf-8-sig").context.astype(str).tolist()
res = {}
for name, model, max_len in (("banglabert_base", "csebuetnlp/banglabert", 160),
                             ("muril_large", "google/muril-large-cased", 256)):
    tok = AutoTokenizer.from_pretrained(model)
    for split, texts in (("train", tr), ("test", te)):
        n = np.array([len(tok(t)["input_ids"]) for t in texts])
        res[f"{name}_{split}"] = dict(model=model, max_len=max_len, n=int(len(n)), over=int((n > max_len).sum()),
                                      over_pct=round(float((n > max_len).mean() * 100), 2),
                                      median=float(np.median(n)), p99=float(np.percentile(n, 99)), max=int(n.max()))
        print(name, split, res[f"{name}_{split}"])
(OUTD / "token_lengths.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
