# When Language Leads the Look: A Calibrated Multimodal Ensemble for Bengali Disaster Classification

Code and result files for the paper by Mohammad Rafiur Rahman and Md. Shahdath Hussain (Chittagong University of
Engineering and Technology), 2026 29th International Conference on Computer and Information Technology (ICCIT).

The repository holds the revision code, the fold assignment, the duplicate masks, the base probability matrices, the
fitted meta-stage weights and the cross-validated predictions behind the numbers in the paper. It does not hold
captions, labels or images (see "Data").

## What each folder holds

| Folder | Contents |
|---|---|
| `analysis/` | Revision scripts `n01`-`n12` (`n11` is in `revision/audit/`), `stackfeat_defs.py` (stacker features, imported by `n01`), `make_paper_numbers.py`, `verify_paper_numbers.py` |
| `analysis/out/` | Their outputs: nested and in-sample predictions, fitted weights, comparison tables, duplicate masks, the row map to the public release. File-by-file list in `DATA.md` |
| `drive_oof/x/oof/` | Base probability matrices of the original branches (out-of-fold for the 6,323 training rows, and for the 1,580 validation and test rows). Which member each file is: `DATA.md` |
| `drive_oof/x/meta/` | `folds_canonical.csv` (`image_id`, `fold`) and `row_order.csv` (public split and id of every row of every array) |
| `runs/` | Out-of-fold and test probabilities of the new heads G1 (text only), G2 (single-token joint head) and G3 (patch cross-attention), seeds 0-2, in the job folders where the scripts look for them; `near_dups.csv`, the near-duplicate image pairs from `jobs/k1_imgindex` |
| `jobs/g_fusion/` | `train_fusion.py`: frozen EVA-02-L feature extraction (G0) and the G1-G3 heads; `make_rows.py`; job files |
| `jobs/n01_meta/` | The packaged CPU jobs that ran `n01_nested_meta.py` for each configuration (`*_hf/`), and the scripts that build their inputs (`pack.py`, `pack_hf.py`, `add_new_members.py`) |
| `jobs/k1_imgindex/` | `imgindex.py`: MD5, 64-bit DCT pHash and dHash of every image, near-duplicate pairs |
| `revision/audit/final_2026_11_19/` | Post-hoc checks: `grouped_refit.py` (writes `analysis/out/n11/`), `duplicate_group_sensitivity.py` (duplicate groups and a group bootstrap), `caption_baseline.py` |

The `job*.yaml` files are inputs to the authors' job runner (Kaggle and Colab). Each names the entry script and its
arguments; the scripts also run directly with Python.

Run the scripts from the repository root. Most resolve paths from their own location, but `n09`, `n10`, `n12`,
`verify_paper_numbers.py` and the two audit scripts also read paths that are stored relative to the repository root in
`analysis/out/n02/meta.json` and `analysis/out/n10/image_copies.json`.

## Data

The data is the public BanglaCalamityMMD release [1] (CC BY 4.0; Kaggle copy
`mukaffimoin/banglacalamitymmd-multimodal-disaster-dataset`): 7,903 caption-image pairs over eight classes, split
6,323 train, 790 validation and 790 test. We develop on the 6,323 training rows by five-fold cross-validation; the
1,580 validation and test rows form the secondary partition reported descriptively in the paper.

Captions, labels and images are not redistributed here. To run the scripts, rebuild three local files from the
public CSVs:

1. `drive_oof/x/meta/folds_canonical.csv` with columns `image_id, context, category, label, fold`: join the released
   file (`image_id`, `fold`) with the public training CSV on `image_id` (`context` = caption, `category` = class name,
   `label` = class index in the order Drought 0, Earthquake 1, Flood 2, Human Damage 3, Landslides 4, Non Disaster 5,
   Tropical Storm 6, Wildfire 7). Keep the released row order.
2. `contest/Disaster_test.csv` with columns `image_id, context` for the 1,580 rows, in the order of the `test` rows
   of `drive_oof/x/meta/row_order.csv` (join on `public_split` and `public_id` with the public validation and test
   CSVs). Our working copy named these rows `test_0001` ... `test_1580` in that order; the scripts and
   `analysis/out/n12/map_contest_to_bcmmd.csv` use these names.
3. Only for `n05_score_test_once.py` and `n12_public_rows.py`: `analysis/out/test_labels_bcmmd.csv` with columns
   `image_id, label_bcmmd` (working name and public class name of each of the 1,580 rows), from the same join.

Our working captions equal the public captions after Unicode NFC normalisation and whitespace collapsing for all
6,323 training and 1,580 evaluation rows, but byte for byte for only 5,189 and 1,212 of them. Features computed from
the raw caption string (the caption statistics of the `n01` stacker, the character n-grams of `caption_baseline.py`
and `grouped_refit.py`) may therefore differ slightly when rebuilt from the public CSVs; this was not tested.

Every image filename in the release encodes its class (Section III of the paper); the loaders pass only decoded pixel
arrays and caption strings to the models.

## Regenerating the paper numbers

With file 1 above in place:

```
python analysis/make_paper_numbers.py     # -> revision/generated/numbers.tex and numbers.json
python analysis/verify_paper_numbers.py   # independent recomputation of numbers.json from the prediction arrays
```

`make_paper_numbers.py` reads the result files in `analysis/out/`, the base matrices and `folds_canonical.csv`, and
writes one LaTeX macro per number and per result-dependent sentence of the paper. When this release was assembled, it
was run on the released files plus the full working `folds_canonical.csv`: all 167 macros it wrote are identical to
the `numbers.tex` of the camera-ready paper. The one macro it could not write is `PlBlendW` (the blend weight of the
pseudo-labelled member, read from a dump of the competition notebook, `analysis/verify_tmp/meta_audit/final_nb_dump.txt`,
which is not released). If `revision/src_revised/main.tex` exists, every macro used there is defined (missing values as
`\ph{}`) and `numbers.tex` is also written next to it.

`verify_paper_numbers.py` is included as the record of the checks we ran. Besides the released files it reads files
that are not released: the paper source (`revision/src_revised/main.tex`), the reviewer-response drafts
(`revision/response_draft.md`, `revision/response_filled.md`, `jobs/latex/response/main.tex`), `DECISION_LOG.md`,
dumps of the competition notebooks (`analysis/verify_tmp/meta_audit/`), the 1,580 labels (`analysis/out/n05/test_preds.npz`),
`jobs/g_fusion/rows.csv.gz` (captions and labels) and a copy of `g2_s0_oof.npy` at `hf/iccit2845-meta-inputs/new/`
(the same file is `runs/train-g2-1118-1903-s0/outputs/g2_s0_oof.npy`). In this repository it stops at the first
missing file.

## Order of the analysis

| Step | Script | Output |
|---|---|---|
| Frozen image features (G0) and heads G1-G3, seeds 0-2 | `jobs/g_fusion/train_fusion.py` | `runs/train-g*/outputs/` |
| Image hashes and near-duplicate pairs | `jobs/k1_imgindex/imgindex.py` | `runs/k1-imgindex-*/outputs/near_dups.csv` |
| Nested CV of every meta stage, configurations `published`, `clean3`, `P` (+ `P_s1`, `P_s2`), meta-seeds 0-4 | `analysis/n01_nested_meta.py` (run as `jobs/n01_meta/*_hf`, collected by `n01_collect.py`) | `analysis/out/n01/` |
| Fusion mechanism comparison T, LF, J1, JP | `analysis/n02_mechanism.py` (uses `n03_stats.py`) | `analysis/out/n02/` |
| Duplicate counts and dedup sensitivity | `analysis/n04_duplicates.py` | `analysis/out/n04/` |
| Single scoring of the frozen systems on the 1,580 rows | `analysis/n05_score_test_once.py` | `analysis/out/n05/` |
| Pre-specified and exploratory comparisons | `analysis/n06_compare.py` | `analysis/out/n06/` |
| Per-class recall | `analysis/n07_per_class.py` | `analysis/out/n07/` |
| Caption token lengths | `analysis/n08_token_lengths.py` | `analysis/out/n08/` |
| Post-hoc controls | `analysis/n09_posthoc_controls.py` | `analysis/out/n09/` |
| Working vs public image copies | `analysis/n10_image_copies.py` | `analysis/out/n10/` |
| Post-hoc grouped refit | `revision/audit/final_2026_11_19/grouped_refit.py` | `analysis/out/n11/` |
| Map of every row to the public release | `analysis/n12_public_rows.py` | `analysis/out/n12/` |

`n05` scores each system listed in `analysis/out/n05/frozen_config.json` once; the SHA-256 of that file was recorded
before scoring. The results on the 1,580 rows are descriptive only (their labels are public and were used during
development; Section III of the paper).

The fitted weights in `analysis/out/n01/fitted_*.json` are rounded (blend weights to 4 decimals, biases to 3), so
predictions cannot be reconstructed exactly from them; `analysis/out/n01/preds_*.npz` and `analysis/out/n02/preds.npz`
are the exact record.

## Frozen EVA-02-L features (not included)

The G1-G3 heads read frozen features of the pretrained `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` (timm), with
the checkpoint's own normalisation, aspect-preserving pad-resize to 448 and one view: `eva_pooled.npy` (7,903 x 1,024,
float16) and `eva_patch8.npy` (7,903 x 64 x 1,024, float16, 8 x 8 average-pooled patch tokens), 1.05 GB together. They
are not included. Their rows follow `jobs/g_fusion/rows.csv.gz`: the 6,323 training rows in `folds_canonical.csv`
order, then the 1,580 rows in `row_order.csv` order (the same order as every array here).

To regenerate them:

1. Rebuild the data files 1 and 2 above, then `python jobs/g_fusion/make_rows.py` (writes `rows.csv.gz`).
2. On a GPU machine with the images, `python jobs/g_fusion/train_fusion.py --g0-only` (job file `job_g0.yaml`);
   the features are written to `$OUT_DIR` (default `jobs/g_fusion/outputs/`).
3. The G1-G3 jobs (`job_g1.yaml` ... `job_g3.yaml`) read the features from a private dataset
   (`iccit2845-eva02l-frozen`, not public). Without it, `train_fusion.py` extracts the features from the images itself.

The `n01` jobs read their inputs from another private dataset (`iccit2845-meta-inputs`). `python jobs/n01_meta/pack.py
published` builds that input set (folds, test ids, caption statistics, image statistics, base matrices) in
`jobs/n01_meta/published/inputs/` from the local data files, and `add_new_members.py` adds the G2 members. Run directly,
`analysis/n01_nested_meta.py` needs no private dataset: `clean3` and `P` need the data files 1 and 2 above; `published`
also needs the 16 image statistics per image from `imgindex.py` (`runs/k1-imgindex-*/outputs/contest_index.csv`),
which are computed from the working images and are not released.

Image copies. The features were extracted from our copy of the images. The 6,323 training image files differ in bytes
from their counterparts in the public release (no MD5 match; same dimensions; perceptual-hash distance 0 for 6,056
files and 2 for 267), while each of the 1,580 evaluation rows matches one public file by MD5 and normalised caption
(790 validation, 790 test). `analysis/out/n12/map_contest_to_bcmmd.csv` maps every working row to its public file:
training rows by published ID and normalised caption, evaluation rows by image MD5 and normalised caption. Placed next
to `train_fusion.py` as `map_contest_to_bcmmd.csv`, it lets the script read the public images when the working images
are not attached. Exact reproduction of the features still requires our working training images or the archived
feature matrices. On the public copies the duplicate counts are 853 rows at distance 0 (851 on ours) and 987 at
distance 4 or less (987 on ours); 4 rows change their dedup flag and the dedup subset keeps 5,321 rows in both
(`analysis/out/n10/image_copies.json`).

## What is not included, and why

- Captions, labels, images and the contest files: not ours to redistribute (rebuild from the public release, above).
- `analysis/out/n05/test_preds.npz` and `analysis/out/test_labels_bcmmd.csv`: they hold the 1,580 labels.
- `jobs/g_fusion/rows.csv.gz` and the `inputs/` folders of the `n01` jobs: they hold captions or labels.
- The frozen EVA-02-L features (1.05 GB): see above.
- Run logs, code-path test outputs (`*_QUICK*`), job metadata and the superseded packed `n01` jobs (`jobs/n01_meta/clean3`
  and `published`, replaced by the `*_hf` jobs that produced the results).

## Environment

`make_paper_numbers.py` was re-run for this release with Python 3.11, numpy 2.4.6, pandas 3.1.1, scipy 1.17.1,
scikit-learn 1.9.1 and lightgbm 4.7.0. `grouped_refit.py` also needs threadpoolctl; `n08` needs transformers; the GPU
jobs need torch, timm and transformers; `imgindex.py` needs opencv-python and pillow.

## Citation

See `CITATION.cff`. Please also cite the dataset:

[1] F. T. J. Faria, M. B. Moin, B. K. Rafa, S. Saha, M. M. Rahman, K. M. Hasib and M. F. Mridha, "BanglaCalamityMMD:
A comprehensive benchmark dataset for multimodal disaster identification in the low-resource Bangla language,"
Int. J. Disaster Risk Reduct., vol. 130, Art. no. 105800, 2025.

## License

MIT for the code (`LICENSE`). BanglaCalamityMMD is distributed by its authors under its own license.
