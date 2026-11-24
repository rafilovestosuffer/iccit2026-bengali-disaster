"""Map the working rows to the public image release and attach the frozen evaluation mask."""
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
row_order = pd.read_csv(ROOT / "jobs/g_fusion/rows.csv.gz")
index_paths = json.loads((OUT / "n10/image_copies.json").read_bytes())["inputs"]


def read_index(name):
    path, = [Path(p) for p in index_paths if Path(p).name == name]
    return pd.read_csv(path, dtype={"phash": str}).rename(columns={"stem": "image_id"})


def normalized(captions):
    return captions.map(lambda v: " ".join(unicodedata.normalize("NFC", v).split()))


working = read_index("contest_index.csv")
public = read_index("bcmmd_index.csv")
mapped = []
for split in ("train", "test"):
    rows = row_order[row_order.split == split].copy()
    work = working.loc[working.set == "contest_" + split, ["image_id", "md5", "w", "h", "phash"]]
    rows = rows.merge(work, on="image_id", how="left", validate="one_to_one")
    assert rows.md5.notna().all()
    rows["caption_key"] = normalized(rows.context)

    catalogs = []
    for part in (("train",) if split == "train" else ("validation", "test")):
        csv = pd.read_csv(ROOT / f"bcmmd/Disaster_{part}.csv")
        catalog = csv.merge(public[public.set == "bcmmd_" + part], on="image_id", validate="one_to_one")
        assert len(catalog) == len(csv)
        catalog["caption_key"] = normalized(catalog.context)
        catalog["public_split"] = part
        catalogs.append(catalog)
    catalog = pd.concat(catalogs).rename(columns={"image_id": "public_id"})
    if split == "train":
        rows = rows.merge(catalog, left_on=["image_id", "caption_key"], right_on=["public_id", "caption_key"],
                          how="left", validate="one_to_one", suffixes=("_work", "_public"))
        assert (rows.md5_work != rows.md5_public).all()
        assert (rows.w_work == rows.w_public).all() and (rows.h_work == rows.h_public).all()
        distance = [(int(a, 16) ^ int(b, 16)).bit_count() for a, b in zip(rows.phash_work, rows.phash_public)]
        assert max(distance) <= 2
        labels = pd.read_csv(ROOT / "drive_oof/x/meta/folds_canonical.csv").set_index("image_id").category
    else:
        rows = rows.merge(catalog, on=["caption_key", "md5"], how="left", validate="one_to_one")
        labels = pd.read_csv(OUT / "test_labels_bcmmd.csv").set_index("image_id").label_bcmmd
    assert len(rows) == len(catalog) and rows.public_id.notna().all()
    assert (rows.category == rows.image_id.map(labels)).all()
    # The existing public-image loader strips the BanglaCalamityMMD/ prefix and resolves locally.
    paths = rows.path.str.split("BanglaCalamityMMD/", regex=False).str[-1]
    assert (rows.path.str.contains("/BanglaCalamityMMD/", regex=False)).all()
    assert not paths.str.contains(r"^/|\.\.").any()
    with np.load(OUT / "dedup_masks.npz") as masks:
        keep = masks[split + "_keep"]
    assert rows.image_id.tolist() == row_order.loc[row_order.split == split, "image_id"].tolist()
    mapped.append(pd.DataFrame(dict(stem=rows.image_id, split=split, public_split=rows.public_split,
                                    public_id=rows.public_id, bc_path="BanglaCalamityMMD/" + paths,
                                    working_keep=keep)))

result = pd.concat(mapped, ignore_index=True)
assert result.stem.is_unique and result.bc_path.is_unique and len(result) == len(row_order)
dest = OUT / "n12"
dest.mkdir(parents=True, exist_ok=True)
result.to_csv(dest / "map_contest_to_bcmmd.csv", index=False)
print(result.groupby(["split", "public_split"]).working_keep.agg(["size", "sum"]).to_string())
