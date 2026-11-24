"""G0-G3 for the paper-2845 revision: leakage-free, matched fusion comparison.

G0  frozen *pretrained* EVA-02-L @448 image features (no fine-tuning, so no fold label can reach them):
    pooled 1024-d vector + 8x8 average-pooled patch tokens (64 x 1024), fp16, computed once per job.
G1  MuRIL-L text only:        p = softmax(W LN(q)),            q = GELU(Linear(LN(meanpool(text))))
G2  single-token joint head:  the original design; text query attends over ONE pooled image token
G3  patch cross-attention:    text query attends over the 64 patch tokens (what Eq. 2 of the paper describes)
All three share: canonical folds, MuRIL-L (google/muril-large-cased), max_len 256 (dynamic padding), 6 epochs,
AdamW (text 8e-6, head 5e-4, wd 0.01), cosine schedule with 10% warm-up, label smoothing 0.05, effective batch 32,
fp16 autocast, grad-clip 1.0. The OOF prediction is the FINAL epoch (no epoch selection on the evaluation fold);
per-epoch validation probabilities are saved for diagnostics only. Test labels are never loaded.

Usage: python train_fusion.py --seed 0 [--branches g1,g2,g3] [--smoke]
Outputs ($OUT_DIR): {branch}_s{seed}_oof.npy (6323x8), {branch}_s{seed}_test.npy (1580x8, mean of 5 fold models),
{branch}_s{seed}_testfolds.npy, {branch}_s{seed}_epochval.npy, eva_pooled.npy (7903x1024 fp16), metrics.json.
"""
import argparse
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("OUT_DIR", HERE / "outputs"))
CACHE = Path("/kaggle/working/cache") if Path("/kaggle/working").exists() else HERE / "cache"
TEXT_MODEL = "google/muril-large-cased"
EVA_TAG = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
N_CLS = 8
CFG = dict(max_len=256, epochs=6, batch_size=8, grad_accum=4, lr_text=8e-6, lr_head=5e-4, weight_decay=0.01,
           warmup_ratio=0.10, label_smooth=0.05, fusion_dim=512, n_heads=8, img_size=448, patch_grid=8)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------- data
def load_rows(smoke):
    rows = pd.read_csv(HERE / "rows.csv.gz", encoding="utf-8")
    if smoke:  # 5 per class from folds 0/1 + 10 test rows
        tr = rows[(rows.split == "train") & (rows.fold.isin([0, 1]))].groupby(["fold", "label"]).head(3)
        rows = pd.concat([tr, rows[rows.split == "test"].head(10)], ignore_index=True)
    return rows.reset_index(drop=True)


def find_image_paths(rows):
    """Contest images if the competition is attached, else BanglaCalamityMMD files mapped by md5 (K1 output)."""
    inp = Path("/kaggle/input")
    tr_dir = te_dir = None
    if inp.exists():
        for p in inp.rglob("*"):
            if p.is_dir() and p.name == "Train" and p.parent.name == "Image":
                tr_dir = p
            if p.is_dir() and p.name == "Test" and p.parent.name == "Image":
                te_dir = p
    paths = []
    if tr_dir is not None and te_dir is not None:
        log("image source: contest data", tr_dir, te_dir)
        idx = {}
        for d in (tr_dir, te_dir):
            for f in d.rglob("*"):
                if f.is_file():
                    idx[f.stem] = f
        paths = [idx[i] for i in rows.image_id]
        return paths, "contest"
    mp = HERE / "map_contest_to_bcmmd.csv"
    if not mp.exists():
        sys.exit("No contest images attached and no map_contest_to_bcmmd.csv bundled")
    m = pd.read_csv(mp)
    bc_root = next(p for p in inp.rglob("BanglaCalamityMMD") if p.is_dir())
    m["rel"] = m["bc_path"].str.split("BanglaCalamityMMD/").str[-1]
    rel = dict(zip(m.stem, m.rel))
    paths = [bc_root / rel[i] for i in rows.image_id]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        sys.exit(f"{len(missing)} mapped BCMMD images missing, e.g. {missing[:3]}")
    log("image source: BanglaCalamityMMD via md5 map", bc_root)
    return paths, "bcmmd_md5"


# ------------------------------------------------------------------------------------------ G0
def extract_eva(paths, smoke):
    import timm
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    CACHE.mkdir(parents=True, exist_ok=True)
    fp, fq = CACHE / "eva_pooled.npy", CACHE / "eva_patch8.npy"
    if fp.exists() and fq.exists():
        log("G0 cached")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(EVA_TAG, pretrained=True, num_classes=0).eval().to(dev)
    dc = timm.data.resolve_model_data_config(model)
    mean, std = np.array(dc["mean"], np.float32), np.array(dc["std"], np.float32)
    size = CFG["img_size"]
    log("G0 EVA data config", dc)

    def pad_resize(img):  # same aspect-preserving pad-resize as the competition kernels
        img = img.convert("RGB")
        w, h = img.size
        s = size / max(h, w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(img, ((size - nw) // 2, (size - nh) // 2))
        a = (np.asarray(canvas, np.float32) / 255.0 - mean) / std
        return a.transpose(2, 0, 1)

    class DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return pad_resize(Image.open(paths[i]))

    dl = DataLoader(DS(), batch_size=16, shuffle=False, num_workers=4 if dev == "cuda" else 2)
    pooled = np.zeros((len(paths), 1024), np.float16)
    patch = np.zeros((len(paths), CFG["patch_grid"] ** 2, 1024), np.float16)
    k, t0 = 0, time.time()
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                feats = model.forward_features(xb)
                pv = model.forward_head(feats, pre_logits=True)
            npt = getattr(model, "num_prefix_tokens", 1)
            tok = feats[:, npt:].float()
            g = int(round(math.sqrt(tok.shape[1])))
            tok = tok.reshape(tok.shape[0], g, g, -1).permute(0, 3, 1, 2)
            tok = F.adaptive_avg_pool2d(tok, CFG["patch_grid"]).flatten(2).transpose(1, 2)
            n = xb.shape[0]
            pooled[k:k + n] = pv.float().cpu().numpy().astype(np.float16)
            patch[k:k + n] = tok.cpu().numpy().astype(np.float16)
            k += n
            if (k // 16) % 50 == 0:
                log(f"G0 {k}/{len(paths)}  {time.time() - t0:.0f}s")
    assert k == len(paths) and np.isfinite(pooled.astype(np.float32)).all()
    np.save(fp, pooled)
    np.save(fq, patch)
    log(f"G0 done {pooled.shape} {patch.shape} in {time.time() - t0:.0f}s")
    del model
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------ training
def run_task(branch, fold, seed, rows, smoke):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer
    from sklearn.metrics import f1_score

    cfg = dict(CFG)
    if smoke:
        cfg.update(epochs=1, max_len=64, batch_size=4, grad_accum=1)
    s = 1000 * seed + fold
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = dev == "cuda"

    pooled = np.load(CACHE / "eva_pooled.npy", mmap_mode="r")
    patch = np.load(CACHE / "eva_patch8.npy", mmap_mode="r")
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    tr_idx = np.where((rows.split == "train") & (rows.fold != fold))[0]
    va_idx = np.where((rows.split == "train") & (rows.fold == fold))[0]
    te_idx = np.where(rows.split == "test")[0]
    ctx = rows.context.astype(str).tolist()
    lab = rows.label.values

    class DS(Dataset):
        def __init__(self, idx):
            self.idx = idx

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, j):
            i = self.idx[j]
            enc = tok(ctx[i], truncation=True, max_length=cfg["max_len"])
            item = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "label": int(lab[i])}
            if branch == "g2":
                item["img"] = np.asarray(pooled[i], np.float32)
            elif branch == "g3":
                item["img"] = np.asarray(patch[i], np.float32)
            return item

    def collate(b):
        L = max(len(x["input_ids"]) for x in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        msk = torch.zeros((len(b), L), dtype=torch.long)
        for r, x in enumerate(b):
            n = len(x["input_ids"])
            ids[r, :n] = torch.tensor(x["input_ids"])
            msk[r, :n] = 1
        out = {"input_ids": ids, "attention_mask": msk, "label": torch.tensor([x["label"] for x in b])}
        if "img" in b[0]:
            out["img"] = torch.from_numpy(np.stack([x["img"] for x in b]))
        return out

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_enc = AutoModel.from_pretrained(TEXT_MODEL)
            td, fd = self.text_enc.config.hidden_size, cfg["fusion_dim"]
            self.txt_proj = nn.Sequential(nn.LayerNorm(td), nn.Linear(td, fd), nn.GELU())
            if branch in ("g2", "g3"):
                self.img_proj = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, fd), nn.GELU())
                self.cross_attn = nn.MultiheadAttention(fd, cfg["n_heads"], dropout=0.1, batch_first=True)
                cin = 2 * fd
            else:
                cin = fd
            self.classifier = nn.Sequential(nn.LayerNorm(cin), nn.Dropout(0.2), nn.Linear(cin, N_CLS))

        def forward(self, input_ids, attention_mask, img=None):
            h = self.text_enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            q = self.txt_proj((h * m).sum(1) / m.sum(1).clamp(min=1e-9))
            if branch == "g1":
                z = q
            else:
                kv = self.img_proj(img.unsqueeze(1) if branch == "g2" else img)  # (B,1,fd) or (B,64,fd)
                a, _ = self.cross_attn(q.unsqueeze(1), kv, kv)
                z = torch.cat([a.squeeze(1), q], dim=-1)
            return self.classifier(z)

    def soft_ce(logits, y, eps):
        lp = F.log_softmax(logits.float(), -1)
        t = F.one_hot(y, N_CLS).float() * (1 - eps) + eps / N_CLS
        return -(t * lp).sum(-1).mean()

    g = torch.Generator()
    g.manual_seed(s)
    tr_dl = DataLoader(DS(tr_idx), batch_size=cfg["batch_size"], shuffle=True, drop_last=True, generator=g,
                       collate_fn=collate, num_workers=2)
    va_dl = DataLoader(DS(va_idx), batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)
    te_dl = DataLoader(DS(te_idx), batch_size=32, shuffle=False, collate_fn=collate, num_workers=2)
    model = Net().to(dev)
    text_p = list(model.text_enc.parameters())
    head_p = [p for n, p in model.named_parameters() if not n.startswith("text_enc")]
    opt = torch.optim.AdamW([{"params": text_p, "lr": cfg["lr_text"]}, {"params": head_p, "lr": cfg["lr_head"]}],
                            weight_decay=cfg["weight_decay"])
    n_steps = math.ceil(len(tr_dl) / cfg["grad_accum"]) * cfg["epochs"]
    n_warm = int(n_steps * cfg["warmup_ratio"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: st / max(1, n_warm) if st < n_warm else max(
        0.0, 0.5 * (1 + math.cos(math.pi * (st - n_warm) / max(1, n_steps - n_warm)))))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def predict(dl):
        model.eval()
        ps = []
        with torch.no_grad():
            for b in dl:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    lo = model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                               b["img"].to(dev) if "img" in b else None)
                ps.append(F.softmax(lo.float(), -1).cpu().numpy())
        model.train()
        return np.concatenate(ps)

    epoch_val, t0 = [], time.time()
    model.train()
    for ep in range(cfg["epochs"]):
        opt.zero_grad(set_to_none=True)
        for st, b in enumerate(tr_dl):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                lo = model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                           b["img"].to(dev) if "img" in b else None)
                loss = soft_ce(lo, b["label"].to(dev), cfg["label_smooth"]) / cfg["grad_accum"]
            scaler.scale(loss).backward()
            if (st + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
        pv = predict(va_dl)
        epoch_val.append(pv.astype(np.float16))
        f1 = f1_score(lab[va_idx], pv.argmax(1), average="macro")
        log(f"{branch} s{seed} f{fold} ep{ep + 1}/{cfg['epochs']} valF1={f1:.5f} ({time.time() - t0:.0f}s)")
    pt = predict(te_dl)
    res = {"va_idx": va_idx, "val": epoch_val[-1].astype(np.float32), "epochval": np.stack(epoch_val),
           "test": pt.astype(np.float32), "seconds": time.time() - t0}
    del model, opt
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return res


def worker(args):
    rows = load_rows(args.smoke)
    tdir = CACHE / f"tasks_s{args.seed}"
    tdir.mkdir(parents=True, exist_ok=True)
    for t in args.tasks.split(";"):
        branch, fold = t.split(":")
        fold = int(fold)
        f = tdir / f"{branch}_f{fold}.npz"
        if f.exists():
            log("skip done", t)
            continue
        r = run_task(branch, fold, args.seed, rows, args.smoke)
        np.savez(f, **r)
        log("saved", f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 0)))
    ap.add_argument("--branches", default="g1,g2,g3")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--g0-only", action="store_true", help="only extract frozen EVA features into OUT_DIR")
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.smoke)
    log("rows", rows.split.value_counts().to_dict(), "torch", torch.__version__,
        "gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    CACHE.mkdir(parents=True, exist_ok=True)
    pre = Path(os.environ.get("DATA_DIR", HERE / "data")) / "iccit2845-eva02l-frozen"
    if (pre / "eva_pooled.npy").exists() and not args.smoke:
        import hashlib
        for n in ("eva_pooled.npy", "eva_patch8.npy"):
            if not (CACHE / n).exists():
                os.symlink(pre / n, CACHE / n)
        src = "precomputed G0 features (HF iccit2845-eva02l-frozen)"
        log("G0 from", pre, "sha256 pooled", hashlib.sha256((pre / "eva_pooled.npy").read_bytes()).hexdigest()[:16])
    else:
        paths, src = find_image_paths(rows)
        extract_eva(paths, args.smoke)
    if args.g0_only:
        import shutil
        for n in ("eva_pooled.npy", "eva_patch8.npy"):
            shutil.copy(CACHE / n, OUT / n)
        rows[["image_id", "split", "fold", "label"]].to_csv(OUT / "eva_rows.csv", index=False)
        json.dump({"image_source": src, "eva": EVA_TAG, "cfg": CFG, "n": len(rows)}, open(OUT / "metrics.json", "w"), indent=1)
        log("G0-only done")
        return
    # pre-download weights once so parallel workers do not race
    from transformers import AutoModel, AutoTokenizer
    AutoTokenizer.from_pretrained(TEXT_MODEL)
    _m = AutoModel.from_pretrained(TEXT_MODEL)
    del _m
    gc.collect()
    np.save(OUT / "eva_pooled.npy", np.load(CACHE / "eva_pooled.npy"))

    folds = sorted(rows[rows.split == "train"].fold.unique())
    branches = args.branches.split(",")
    tasks = [f"{b}:{f}" for b in branches for f in folds]
    ngpu = max(1, torch.cuda.device_count())
    t0 = time.time()
    procs = []
    for k in range(ngpu):
        mine = tasks[k::ngpu]
        if not mine:
            continue
        env = dict(os.environ)
        if torch.cuda.is_available():
            env["CUDA_VISIBLE_DEVICES"] = str(k)
        cmd = [sys.executable, "-u", __file__, "--worker", "--seed", str(args.seed), "--tasks", ";".join(mine)]
        if args.smoke:
            cmd.append("--smoke")
        procs.append(subprocess.Popen(cmd, env=env))
    rcs = [p.wait() for p in procs]
    if any(rcs):
        sys.exit(f"worker failed: {rcs}")

    from sklearn.metrics import f1_score
    tr_mask = (rows.split == "train").values
    y = rows.label.values[tr_mask]
    n_tr, n_te = int(tr_mask.sum()), int((~tr_mask).sum())
    metrics = {"seed": args.seed, "image_source": src, "cfg": CFG, "text_model": TEXT_MODEL, "eva": EVA_TAG,
               "smoke": args.smoke, "minutes": round((time.time() - t0) / 60, 1), "branches": {}}
    tdir = CACHE / f"tasks_s{args.seed}"
    for b in branches:
        oof = np.full((n_tr, N_CLS), np.nan, np.float32)
        n_ep = None
        ev_all, tests = None, []
        for f in folds:
            r = np.load(tdir / f"{b}_f{f}.npz")
            oof[r["va_idx"]] = r["val"]
            tests.append(r["test"])
            ev = r["epochval"]
            if ev_all is None:
                n_ep = ev.shape[0]
                ev_all = np.full((n_ep, n_tr, N_CLS), np.nan, np.float16)
            ev_all[:, r["va_idx"]] = ev
        assert not np.isnan(oof).any()
        tests = np.stack(tests)
        np.save(OUT / f"{b}_s{args.seed}_oof.npy", oof)
        np.save(OUT / f"{b}_s{args.seed}_test.npy", tests.mean(0))
        np.save(OUT / f"{b}_s{args.seed}_testfolds.npy", tests)
        np.save(OUT / f"{b}_s{args.seed}_epochval.npy", ev_all)
        per_ep = [float(f1_score(y, ev_all[e].astype(np.float32).argmax(1), average="macro")) for e in range(n_ep)]
        metrics["branches"][b] = {"oof_f1_final_epoch": float(f1_score(y, oof.argmax(1), average="macro")),
                                  "oof_errors_final_epoch": int((oof.argmax(1) != y).sum()),
                                  "oof_f1_per_epoch": per_ep}
        log(b, metrics["branches"][b])
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)
    log("DONE", json.dumps(metrics)[:2000])


if __name__ == "__main__":
    main()
