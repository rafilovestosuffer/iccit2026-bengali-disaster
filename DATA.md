# Data files

All arrays are NumPy `.npy` / `.npz` files. Every 8-column array has one column per class, in the order Drought 0,
Earthquake 1, Flood 2, Human Damage 3, Landslides 4, Non Disaster 5, Tropical Storm 6, Wildfire 7.

## Row order

- Arrays with 6,323 rows (out-of-fold, "oof", "nested", "insample") follow the row order of
  `drive_oof/x/meta/folds_canonical.csv`, whose `image_id` is the public BanglaCalamityMMD training id.
- Arrays with 1,580 rows ("test") follow the order of our working file of the 790 public validation and 790 public
  test instances. `n01_nested_meta.py` and `jobs/n01_meta/pack.py` read the test rows in that file's order,
  `n05_score_test_once.py` attaches the labels in that order, and `analysis/out/n12/map_contest_to_bcmmd.csv` lists the
  rows in the same order (checked when this release was assembled).
- `drive_oof/x/meta/row_order.csv` gives every row its public id: `array` (`oof` or `test`), `row` (0-based),
  `public_split` (`train`, `validation` or `test`), `public_id`, `fold` (0-4; -1 for the 1,580 rows).
- Our working copy named the 1,580 rows `test_0001` ... `test_1580` in row order. These names appear in
  `analysis/out/n12/map_contest_to_bcmmd.csv` (column `stem`), `runs/k1-imgindex-1118-1725-s0/outputs/near_dups.csv` and
  `analysis/out/seen_test_numbers.md`.

`folds_canonical.csv` holds `image_id` and `fold` only (1,265 / 1,265 / 1,265 / 1,264 / 1,264 rows in folds 0-4). The
working file also held the caption, class name and class index; these are not redistributed (README, "Data").

## Base probability matrices: `drive_oof/x/oof/`

`*_oof.npy`: 6,323 x 8 out-of-fold probabilities. `*_test_probs.npy`: 1,580 x 8. Member names are those used in
`analysis/n01_nested_meta.py`. Configurations: `published` = the six submitted members (`bb_base`, `bb_multi`,
`muril_L`, `eva02_L`, `mm_old`, `pl_muril`); `clean3` = `bb_base`, `muril_L`, `eva02_L`; `P` (the revised pipeline) =
`clean3` + G2 seed 0; `P_s1`, `P_s2` = `clean3` + G2 seed 1 or 2 (exploratory).

| File stem | Member | Notes |
|---|---|---|
| `text_bbB_s123_20260509_0550` | `bb_base` | BanglaBERT-base (`csebuetnlp/banglabert`), captions. In `published`, `clean3`, `P` |
| `text_bbB_s123_20260509_0657`, `text_bbB_s456_20260509_0747` | - | Byte-identical to `text_bbB_s123_20260509_0550` (one run, not three seeds; DECISION_LOG D10). Not used |
| `text_bbmulti_20260509_0837` | `bb_multi` | Average of an earlier BanglaBERT-base run and two copies of `bb_base`, all monolingual `csebuetnlp/banglabert` (not a multilingual model; D10). `published` only |
| `text_murilL_20260509_0210` | `muril_L` | MuRIL-Large (`google/muril-large-cased`), captions. All configurations |
| `img_eva02L_448_20260511_0327` | `eva02_L` | Fine-tuned EVA-02-Large at 448 px, images. All configurations |
| `mm_eva_muril_20260512_1041` | `mm_old` | **The original fusion branch of the submitted system. The revision found it leaky** (DECISION_LOG D6): its image features were averaged over all five fine-tuned EVA fold checkpoints, four of which were trained on each out-of-fold row, and its attention ran over a single pooled image token. Its out-of-fold scores are not a leakage-free estimate. `published` only |
| `pl_muril_20260512_1725` | `pl_muril` | Pseudo-labelled MuRIL-Large. **Transductive** (D4): it uses pseudo-labels of the 1,580 evaluation rows. Its training script was not archived. `published` only |
| `img_eva02L_448_20260510_1049_oof_partial`, `..._test_probs_3fold` | - | An earlier EVA-02-Large run. The out-of-fold array is filled for folds 0-2 only (the 2,528 rows of folds 3-4 are NaN). Not used |
| `spec_eqflls_20260512_1557` | - | Three-class specialist (Earthquake, Flood, Landslides): out-of-fold 2,403 x 3, test 1,580 x 3. The archived notebook that reads it places the 2,403 rows at the training rows whose true label is 1, 2 or 4, in training-row order. It was never in the final pipeline (D5). Not used |
| `stack_oof_s2`, `stack_test_s2` | - | LightGBM stacker of the competition notebook (Optuna-tuned, early stopping on the evaluation fold). Not used by the revision |
| `stack_oof_s3e`, `stack_test_s3e` | - | A later stacker; the code that produced it was not archived. Not used by the revision |
| `final_oof_s2`, `final_test_s2` | - | Final probabilities of the competition notebook before its per-class log-bias, float64; 8 out-of-fold rows and 7 test rows are NaN as archived. Not used by the revision |
| `class_bias_s2.npy` | - | Per-class log-bias of that notebook, shape (8,) |
| `blend_weights_s2.npy` | - | Blend weights saved by that notebook, shape (8,), all 0.125 as archived |
| `new_bias_s3e.npy` | - | Per-class log-bias of the later pipeline, shape (8,); a single fit, not a bootstrap median. Equal to `class_bias_s2.npy` up to float32 rounding |

## New heads: `runs/`

`runs/<job>/outputs/g{1,2,3}_s{seed}_oof.npy` (6,323 x 8) and `..._test.npy` (1,580 x 8), seeds 0-2, written by
`jobs/g_fusion/train_fusion.py` on frozen EVA-02-L features: G1 = text only (MuRIL-Large), G2 = single-token joint
head, G3 = patch cross-attention. Out-of-fold predictions are from the final epoch (no epoch selection); test
predictions are the mean of the five fold models. In the paper, T = G1, J1 = G2, JP = G3, and LF = G1 combined with a
logistic-regression head on the frozen pooled features (`analysis/n02_mechanism.py`). G2 seed 0 is the fourth member
of `P`. The folders keep the original job names so that the scripts (`RUNS = runs/`) and `analysis/out/n02/meta.json`
find them.

`runs/k1-imgindex-1118-1725-s0/outputs/near_dups.csv`: near-duplicate image pairs from `jobs/k1_imgindex/imgindex.py`,
computed on our working image copies. Columns `kind` (`train-train` or `test-train`), `a`, `b` (working ids; for
`test-train`, `a` is the evaluation row), `phash_dist` (Hamming distance of the 64-bit DCT pHash, at most 10).

## `analysis/out/`

| Path | Contents |
|---|---|
| `dedup_masks.npz` | `train_keep` (6,323, bool) and `test_keep` (1,580, bool), in the row orders above; rule in `dedup_masks.json` (5,321 and 1,351 rows kept). No archived script writes this file; when this release was assembled, the rule applied to `near_dups.csv` and the working captions reproduced both masks exactly |
| `seen_test_numbers.md` | Results on the 1,580 rows that had been seen before the revised systems were frozen (read by `make_paper_numbers.py`) |
| `n01/preds_<config>_m<seed>.npz` | Meta-stage outputs for configuration `published`, `clean3`, `P`, `P_s1` or `P_s2` and meta-seed 0-4. Keys `nested__<stage>` (6,323 x 8: outer fold k predicted by stages fitted on the other four folds), `insample__<stage>` (6,323 x 8: fitted on all rows, the stack by 5-fold cross-validation, and scored on the same rows), `test__<stage>` (1,580 x 8: refit on all 6,323 rows). Stages `uniform`, `late_fusion`, `blend`, `stack`, `alpha_mix`, `bias_single`, `bias_boot`, `blend_bias_boot`; the three `*bias*` stages are log-scores, the others probabilities |
| `n01/fitted_<config>_m<seed>.json` | Fitted meta-stage parameters for each outer fold (`outer0`-`outer4`) and the full refit (`full`): `late_fusion_alpha`, `blend_w` (rounded to 4 decimals), `mix_alpha`, `bias_single`, `bias_boot`, `blend_bias_boot` (rounded to 3 decimals) |
| `n01/summary_<config>_m<seed>.csv` | Nested and in-sample macro-F1 and error count per stage |
| `n01/provenance.json` | Source job and SHA-256 of each `n01` file |
| `n02/` | Fusion mechanism comparison (T, LF, J1, JP; seeds 0-2): `preds.npz` (`img_oof`, `img_test`: image head probabilities; `oof__<arm>__s<seed>`: out-of-fold predicted class; `test__<arm>__s<seed>`: test probabilities), `image_head.npz` (logistic-regression head on the frozen pooled features; holds Python dicts as object arrays, so load it with `allow_pickle=True`), `per_seed.csv`, `aggregate.csv`, `comparisons.csv`, `epoch_selection.csv`, `meta.json` |
| `n04/` | Duplicate counts (`report.json`) and dedup sensitivity of the members (`sensitivity.csv`) |
| `n05/` | `frozen_config.json` (the hashed list of systems scored once on the 1,580 rows, with the SHA-256 of each prediction file), `test_preds_index.json`, `test_results.csv`, `test_results_per_seed.csv` (descriptive) |
| `n06/` | `stage_table.csv`, `primary.csv` (pre-specified comparisons, Holm-adjusted), `exploratory.csv`, `optimism.csv`, `mech_table.csv` |
| `n07/` | Per-class recall and confusion matrices |
| `n08/` | Caption token lengths |
| `n09/` | Post-hoc controls (`lf_controls.csv`, `alphas.csv`, `j1seed.csv`, `meta.json`) |
| `n10/` | `image_copies.json`: working vs public image copies of the training rows |
| `n11/` | Post-hoc grouped refit (`grouped_refit.json`; `grouped_predictions.npz` with `row_id`, `fold`, `group`, `eval_caption`, `eval_image`, `cal_caption`, `cal_image`) |
| `n12/` | `map_contest_to_bcmmd.csv`: `stem` (working id), `split`, `public_split`, `public_id`, `bc_path`, `working_keep` (the dedup mask) |

## Edits made for this release

- Absolute paths in scripts and in `analysis/out/n02/meta.json`, `n09/meta.json` and `n10/image_copies.json` were made
  relative to the repository root. The SHA-256 values in `n09/meta.json` and `n10/image_copies.json` for
  `drive_oof/x/meta/folds_canonical.csv` refer to the full working file (with captions and labels), not to the
  released two-column file.
- Kaggle account names in kernel identifiers were replaced by `<kaggle-user>` in `analysis/out/n01/provenance.json`
  and in the `sources` field of `analysis/out/n01/fitted_P*.json`. The SHA-256 values that `provenance.json` records
  for those 15 `fitted_P*.json` files therefore no longer match the released copies; all other `n01` files are
  unchanged.
- In the code, the name of a private Hugging Face organisation was removed from one docstring (`jobs/n01_meta/pack_hf.py`)
  and a comment marking each private dataset as not released, with how to rebuild it, was added to the job files that
  name one (`jobs/g_fusion/job_g1.yaml` ... `job_g3.yaml`, `jobs/n01_meta/*_hf/job.yaml`).
- No other file content was changed.
