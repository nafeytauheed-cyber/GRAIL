# Research and execution log — 5 November 2026

## Phase record

| Phase | Files/work | Executed validation/results | Concerns and remaining work |
|---|---|---|---|
| 1. Research/data audit | literature_review.md; data/warehouse.py; synthetic_audit | Inspected all five required repositories and source code; browsed primary papers. Anonymous Kaggle download returned 404; credentials absent. Full synthetic audit executed. | Real-data audit and independent 2019 coding-change verification blocked. |
| 2. Leakage-safe preparation | features.py, preprocessing.py, splits.py; feature_availability.md | Tests check future weather masking, train-only imputation, unseen categories, duplicates and role isolation. | First-report publication times and near duplicates unavailable. |
| 3. Conventional baselines | models/baselines.py; experiment runner | Majority, logistic, RF, XGBoost, CatBoost executed on fixed synthetic splits. | Two-candidate exploratory tuning; no real benchmark. |
| 4. Neural ordinal baselines | models/ordinal.py | CE, constrained CORAL, CORN; probability/target/gradient/conditional-mask tests. | CORAL variant explicitly orders thresholds; shared-score restriction can hurt. |
| 5. GRAIL estimator | models/classifier.py | fit/predict/proba/save/load, missing categories, reproducible seeds and inference tests. | CPU tested; optional CUDA untested. Bounded in-memory fit, not full-data streaming. |
| 6. Adaptive objective | losses/groups.py; mathematics.md | Verified gradient equivalence for shrunken risks and reduction to ordinary DRO; small-group stability tests. | Shrinkage measures sample-size uncertainty, not label correctness. |
| 7. Calibration/investigation | evaluation/* | Separate calibration, entropy curves, environmental perturbation, disagreement, failure and investigation-priority CSVs. | These are empirical tools; no reliable-abstention or causal guarantees. |
| 8. Controlled comparison | reports/results/synthetic; shortcut directories | Three model seeds, two protocols, 15 variants; four shortcut settings on one seed. Full metrics and row predictions saved. | Same candidate count does not mean equal compute. Check final results_summary for measured tradeoffs. |
| 9. College package | README, docs, tests, figures, reproducibility scripts | Automated test output retained; report generator checks saved checkpoint predictions against CSVs. | Real empirical college deliverables incomplete; GitHub remote creation blocked. |

## Development decisions and corrections

- Initial test milestone: 23 passing tests; expanded suite: 33 tests (see reports/results/tests.txt for final executed output).
- Group-loss shrinkage was initially applied only to the adversary. Before final comparison, model gradient weights were corrected to differentiate the same shrunken objective. An equality test prevents regression.
- A narrow-gap initialization diagnostic exposed underfitting in the short neural training budget. It is retained separately, including its partial results and logs. Final cumulative-link runs initialize thresholds from training-only cumulative class proportions, identically across all variants. This was an exploratory development correction, not test-set selection of a winning architecture.
- No custom encoder was added without a supported hypothesis. Ordinary MLP is deliberately shared, so geographic loss changes can be isolated.
- Class-log-probability temperature scaling is fitted on calibration rows only. Final saved GRAIL checkpoints include the evaluated temperature. The finalization script verifies their prediction equivalence.
- Source inspection found global preprocessing before chronological splitting in the Simranghag notebook, training self-label target encoding in SAE-XCrash, and a questionable array shape in the inspected upstream CORN loss. Our implementations are independent and tested; no unlicensed code was copied.
- Synthetic sample has 8,000 rows with grades {1:545, 2:6627, 3:653, 4:175}. These are **generated counts**, not US Accidents statistics.

## Access and publication blockers

Connected GitHub access can read and modify existing repositories, but its tools have no repository-creation operation. The accessible user repositories did not include GRAIL. A browser fallback reached GitHub's sign-in page. No unrelated repository was repurposed, and no remote push/draft PR is claimed. Local commits and a Git bundle are provided.

Kaggle credentials were absent and anonymous download failed. Continue when an authorized CSV/Parquet becomes available. First run the full audit and decide on comparable label scope; then rerun baselines and ablations before drawing college conclusions.

## Next research steps

1. Obtain exact release and checksum; validate duplicate signatures and source/time coverage.
2. Investigate mixture-standardized and within-provider/state label prevalence. Seek provider documentation before declaring reclassification.
3. Predeclare defensible four-class subsets and separately justified binary sensitivity questions.
4. Repeat on a memory-bounded real subset, then add streaming/memmapped training and measured 8 GB/RTX 3050 profiling.
5. Run pure matched-time state holdouts, state×provider grouping, more robust block uncertainty intervals and TabM.
6. Add synthetic noisy-small-group stress experiments before treating this particular shrinkage proposal as effective.
