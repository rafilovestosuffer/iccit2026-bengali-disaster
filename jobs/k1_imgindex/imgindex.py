"""K1 (CPU, Kaggle): image index for paper 2845 revision.

For every image in the contest data (train 6,323 + test 1,580) and in the public BanglaCalamityMMD
dataset (train/validation/test), compute md5, size, a 64-bit perceptual hash (DCT pHash) and a 64-bit
difference hash, plus the 16 colour/edge statistics used by the competition stacker
(ml-contest-2 cell 28, `image_feats`). Then:
  * map every contest image to its BCMMD file by md5 (exact byte identity),
  * list near-duplicate pairs (pHash Hamming distance <= 10) inside contest train and between
    contest test and contest train.
Writes outputs/: contest_index.csv, bcmmd_index.csv, map_contest_to_bcmmd.csv, near_dups.csv, summary.json.
No model, no labels used for anything except reporting.
"""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

OUT = Path(os.environ.get("OUT_DIR", "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
INP = Path("/kaggle/input")
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def find_dir(pattern_parts):
    for p in INP.rglob("*"):
        if p.is_dir() and all(s in str(p) for s in pattern_parts):
            return p
    return None


def list_images(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def phash64(gray):
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64)
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = d.flatten()
    med = np.median(flat[1:])
    bits = (flat > med).astype(np.uint8)
    return int("".join(map(str, bits)), 2)


def dhash64(gray):
    g = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).astype(np.uint8).flatten()
    return int("".join(map(str, bits)), 2)


def stats16(path):
    # identical to the competition stacker's image_feats (ml-contest-2)
    img = cv2.imread(str(path))
    if img is None:
        return [0.0] * 16
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r, g, b = rgb.mean(axis=(0, 1)) / 255.0
    rs, gs, bs = rgb.std(axis=(0, 1)) / 255.0
    hh, ss, vv = hsv.mean(axis=(0, 1)) / np.array([180., 255., 255.])
    edges = cv2.Canny(gray, 100, 200)
    return [r, g, b, rs, gs, bs, hh, ss, vv,
            max(0., r - max(g, b)), max(0., b - max(r, g)), max(0., g - max(r, b)),
            (edges > 0).mean(), float(cv2.Laplacian(gray, cv2.CV_64F).var()), w / h, (h * w) / 1e6]


def index(paths, tag, with_stats):
    rows = []
    for i, p in enumerate(paths):
        b = p.read_bytes()
        md5 = hashlib.md5(b).hexdigest()
        try:
            im = Image.open(p).convert("L")
            gray = np.asarray(im)
            ph, dh = phash64(gray), dhash64(gray)
            W, H = im.size
        except Exception as e:  # unreadable image
            print("UNREADABLE", p, e, flush=True)
            ph, dh, W, H = -1, -1, -1, -1
        row = {"set": tag, "path": str(p), "stem": p.stem, "md5": md5, "bytes": len(b),
               "w": W, "h": H, "phash": f"{ph:016x}" if ph >= 0 else "", "dhash": f"{dh:016x}" if dh >= 0 else ""}
        if with_stats:
            for j, v in enumerate(stats16(p)):
                row[f"img{j}"] = float(v)
        rows.append(row)
        if i % 1000 == 0:
            print(tag, i, "/", len(paths), flush=True)
    return pd.DataFrame(rows)


def hamming_pairs(a_hex, b_hex, max_d, same=False):
    A = np.array([int(x, 16) for x in a_hex], dtype=np.uint64)
    B = np.array([int(x, 16) for x in b_hex], dtype=np.uint64)
    out = []
    for i in range(len(A)):
        x = np.bitwise_xor(B, A[i])
        d = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
        js = np.where(d <= max_d)[0]
        for j in js:
            if same and j <= i:
                continue
            out.append((i, int(j), int(d[j])))
    return out


def main():
    print("inputs:", [str(p) for p in INP.iterdir()], flush=True)
    comp_train = find_dir(["Image", "Train"])
    comp_test = find_dir(["Image", "Test"])
    bc_root = find_dir(["BanglaCalamityMMD"])
    print("contest train dir:", comp_train, "| contest test dir:", comp_test, "| bcmmd:", bc_root, flush=True)
    ct = index(list_images(comp_train), "contest_train", True)
    cte = index(list_images(comp_test), "contest_test", True)
    contest = pd.concat([ct, cte], ignore_index=True)
    contest.to_csv(OUT / "contest_index.csv", index=False)

    bc_rows = []
    for split in ["Train", "Validation", "Test"]:
        d = next((p for p in bc_root.rglob(split) if p.is_dir()), None)
        print("bcmmd", split, d, flush=True)
        if d is not None:
            bc_rows.append(index(list_images(d), f"bcmmd_{split.lower()}", False))
    bc = pd.concat(bc_rows, ignore_index=True)
    bc.to_csv(OUT / "bcmmd_index.csv", index=False)

    # exact map by md5
    bc_by_md5 = bc.groupby("md5").agg(bc_set=("set", lambda s: "|".join(sorted(set(s)))),
                                      bc_stem=("stem", lambda s: "|".join(sorted(set(s)))),
                                      bc_path=("path", "first"), bc_n=("path", "size")).reset_index()
    m = contest[["set", "stem", "md5", "path"]].merge(bc_by_md5, on="md5", how="left")
    m.to_csv(OUT / "map_contest_to_bcmmd.csv", index=False)
    contest_md5_dup = int(contest["md5"].duplicated(keep=False).sum())

    # near duplicates by pHash
    tr = ct[ct.phash != ""].reset_index(drop=True)
    te = cte[cte.phash != ""].reset_index(drop=True)
    pairs = []
    for i, j, d in hamming_pairs(tr.phash, tr.phash, 10, same=True):
        pairs.append(("train-train", tr.stem[i], tr.stem[j], d))
    for i, j, d in hamming_pairs(te.phash, tr.phash, 10):
        pairs.append(("test-train", te.stem[i], tr.stem[j], d))
    for i, j, d in hamming_pairs(te.phash, te.phash, 10, same=True):
        pairs.append(("test-test", te.stem[i], te.stem[j], d))
    nd = pd.DataFrame(pairs, columns=["kind", "a", "b", "phash_dist"])
    nd.to_csv(OUT / "near_dups.csv", index=False)

    summary = {
        "contest_train": len(ct), "contest_test": len(cte), "bcmmd_images": len(bc),
        "contest_mapped_by_md5": int(m["bc_path"].notna().sum()),
        "contest_unmapped": int(m["bc_path"].isna().sum()),
        "contest_md5_mapped_to_multiple_bcmmd_files": int((m["bc_n"].fillna(0) > 1).sum()),
        "contest_rows_sharing_md5_with_another_contest_row": contest_md5_dup,
        "near_dup_pairs_by_kind_dist_le_10": nd.groupby("kind").size().to_dict(),
        "near_dup_pairs_by_kind_dist_le_4": nd[nd.phash_dist <= 4].groupby("kind").size().to_dict(),
        "exact_hash_pairs_dist_0": nd[nd.phash_dist == 0].groupby("kind").size().to_dict(),
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
