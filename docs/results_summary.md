# Executed results — SYNTHETIC ONLY

8,000 generated rows; three model seeds (11, 23, 37); 15 variants; two protocols; 90 selected model evaluations. Each non-majority family tries two validation-selected candidates. All cumulative-link models use training-prior threshold initialization.

These results validate software execution and study this artificial generator. They are not evidence of US Accidents performance or road-safety conclusions.

## temporal

| Model | Macro-F1 mean ± seed SD | Ordinal MAE | NLL | Worst-state F1 | Grade-4 recall |
|---|---:|---:|---:|---:|---:|
| majority | 0.230 ± 0.000 | 0.161 | 0.580 | 0.217 | 0.000 |
| logistic | 0.314 ± 0.000 | 0.139 | 0.407 | 0.231 | 0.000 |
| forest | 0.301 ± 0.003 | 0.147 | 0.432 | 0.231 | 0.000 |
| xgboost | 0.306 ± 0.000 | 0.147 | 0.413 | 0.231 | 0.000 |
| catboost | 0.301 ± 0.000 | 0.147 | 0.413 | 0.231 | 0.000 |
| mlp | 0.316 ± 0.000 | 0.143 | 0.415 | 0.231 | 0.000 |
| coral | 0.323 ± 0.054 | 0.143 | 0.423 | 0.227 | 0.083 |
| corn | 0.315 ± 0.001 | 0.142 | 0.410 | 0.231 | 0.000 |
| ordinal_weighted | 0.230 ± 0.000 | 0.161 | 0.535 | 0.217 | 0.000 |
| ordinal_dro | 0.328 ± 0.049 | 0.142 | 0.422 | 0.227 | 0.083 |
| ordinal_fixed | 0.351 ± 0.046 | 0.142 | 0.422 | 0.229 | 0.167 |
| grail | 0.357 ± 0.049 | 0.140 | 0.422 | 0.229 | 0.167 |
| grail_no_shrink | 0.351 ± 0.058 | 0.142 | 0.421 | 0.229 | 0.167 |
| grail_no_regularization | 0.328 ± 0.049 | 0.142 | 0.422 | 0.227 | 0.083 |
| ordinal_sampled | 0.230 ± 0.000 | 0.161 | 0.529 | 0.217 | 0.000 |

## geographic

| Model | Macro-F1 mean ± seed SD | Ordinal MAE | NLL | Worst-state F1 | Grade-4 recall |
|---|---:|---:|---:|---:|---:|
| majority | 0.227 ± 0.000 | 0.215 | 0.641 | 0.219 | 0.000 |
| logistic | 0.230 ± 0.000 | 0.200 | 0.457 | 0.225 | 0.000 |
| forest | 0.276 ± 0.040 | 0.185 | 0.479 | 0.234 | 0.000 |
| xgboost | 0.333 ± 0.000 | 0.169 | 0.478 | 0.239 | 0.333 |
| catboost | 0.293 ± 0.000 | 0.185 | 0.459 | 0.239 | 0.000 |
| mlp | 0.384 ± 0.086 | 0.169 | 0.441 | 0.239 | 0.000 |
| coral | 0.353 ± 0.038 | 0.179 | 0.447 | 0.239 | 0.444 |
| corn | 0.293 ± 0.000 | 0.185 | 0.477 | 0.239 | 0.000 |
| ordinal_weighted | 0.380 ± 0.132 | 0.195 | 0.542 | 0.231 | 0.444 |
| ordinal_dro | 0.353 ± 0.038 | 0.179 | 0.447 | 0.239 | 0.444 |
| ordinal_fixed | 0.353 ± 0.038 | 0.179 | 0.447 | 0.239 | 0.444 |
| grail | 0.353 ± 0.038 | 0.179 | 0.448 | 0.239 | 0.444 |
| grail_no_shrink | 0.353 ± 0.038 | 0.179 | 0.447 | 0.239 | 0.444 |
| grail_no_regularization | 0.353 ± 0.038 | 0.179 | 0.448 | 0.239 | 0.444 |
| ordinal_sampled | 0.334 ± 0.128 | 0.195 | 0.540 | 0.232 | 0.222 |

## Interpretation limits

Seed SD measures training randomness on one fixed dataset, not population confidence. Temporal test and geographic test have different populations; their score difference does not isolate a geographic causal effect. Consult split JSONs for sample sizes and all four class supports.

An interrupted earlier run with narrow initial threshold gaps is preserved under `reports/results/initialization_diagnostic/`. It is not part of this final comparison. Training-only prior initialization was introduced after diagnosing underfitting; this is an exploratory development sequence, not a pristine preregistered confirmatory experiment.

GRAIL is not assumed to win. Compare it with ordinary DRO and CORAL on every metric, especially worst-state performance, probability scores and rare-grade recall. Real-data label audit, tuning and evaluation remain necessary.

## Measured comparison

temporal: GRAIL / ordinary DRO mean macro-F1 0.357 / 0.328; NLL 0.422 / 0.422; worst-state F1 0.229 / 0.227.
Test population: 273 synthetic rows, grade-4 count 4.
geographic: GRAIL / ordinary DRO mean macro-F1 0.353 / 0.353; NLL 0.448 / 0.447; worst-state F1 0.239 / 0.239.
Test population: 65 synthetic rows, grade-4 count 3.

24 additional one-seed shortcut evaluations are recorded in reports/results/shortcut_summary.csv. The adaptive objective has not established superiority; tiny holdouts and optimizer variability prevent strong conclusions.
