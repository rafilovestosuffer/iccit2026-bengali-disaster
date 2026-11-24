# Numbers on the 1,580-row partition seen before freezing (D21)

Written 2026-11-18 18:35 Dhaka, before `frozen_config.json` exists. Everything listed here was computed on or reported for the
1,580 test rows (BCMMD validation + test labels, `analysis/out/test_labels_bcmmd.csv`) before any revised system was frozen.

| Source | What was seen |
|---|---|
| Competition (May 2026) | Public/private leaderboard scores of 49 submissions; private LB 0.99571 for A2_aggressive_FINAL and A2_aggressive_FINAL_2 (the submitted paper's headline). |
| Submitted paper (ICCIT 2845) | All test numbers in the paper's tables and text (as printed). |
| `analysis/a04_competition_pipeline_on_test.py` (audit, 2026-11-18) | Test macro-F1 and error counts of the six original members; stack_s3e test F1; re-fitted Dirichlet blend, alpha mix and log-bias variants scored on test; keyword-rule replay (3 rule changes: test_0182 TS->Drought wrong; test_0508, test_1473 HD->Wildfire right; 1 change in the bootstrap-bias pipeline); caption-duplicate counts test vs train. |
| `analysis/verify_tmp/b7_pl.py` (audit) | PL-MuRIL vs other members argmax agreement on test; S2 final test rows meeting top1>=.95 & margin>=.30; their accuracy against BCMMD labels. |
| K1 / `n04_duplicates.py` | Counts of test rows with a near-duplicate train image or caption (no scores). |

Not yet seen: any test score of `clean3`, `P`, G1-G3, LF, or of the nested-procedure full refits (n01 saves `test__*` arrays but does not score them).
