# Methodology and preregistered decisions

## Dataset and acquisition

Target source: [US Accidents 2016–2023](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents). Its documented scale is approximately 7.7 million rows and 46 attributes, not measured by this run. Kaggle credentials were absent; an anonymous request to the dataset download endpoint returned HTTP 404. The author website was also unavailable through retrieval. No full or sampled real-data download is claimed. All checked-in benchmark results use 8,000 generated artificial incidents with known code and seed.

## Audit before modelling

`prepare` imports raw CSV as strings to avoid silent mixed-type inference, writes compressed Parquet using DuckDB with a 2 GB query memory limit. `audit` computes missingness, target/year/state/provider counts, joint conditional prevalence, duplicate ID/signature counts, invalid dates and weather/end-time consistency. Year-pair total-variation distances retain sample sizes and provider/state identities. They flag drift, not proof of a changed labeling rule.

A raw-data audit is separate from the bounded development sample. `--rows` is an explicit sample cap, using a seeded ID hash rather than a target-dependent sample. Current estimator training materializes the bounded design matrix; it is not a demonstrated full-7.7M-row neural trainer. A 50,000-row development cap and category vocabulary cap are appropriate starting points for 8 GB RAM. Full training requires measured memory sizing or a future memmapped/streamed trainer. Do not increase the cap blindly.

## Label consistency decision

Primary remains four classes. Require a written `--label-review` file for real data, with status `comparable` or `exploratory_unverified`, rationale, evidence paths, chosen source/time scope and limitations. “Comparable” must be supported, not inferred merely because counts look smooth. An exploratory result cannot establish stable class meanings.

Investigate year/source/state counts and within-stratum shifts; compare the same geography and provider across years; standardize prevalence over overlapping strata; examine near-duplicate incidents with discordant labels; seek provider documentation. Marginal prevalence can change because of collection composition, pandemic effects, reporting coverage, or severity assignment. Current evidence does not distinguish them.

Sensitivity subsets are selected via explicit source and time filters. Record each independently and retain the four-class target. A separate binary experiment Y=4 vs Y<4 may be scientifically useful only after confirming that grade 4 is itself comparable. It is **not justified or run here**, and the binary outcome must never replace the primary target silently.

## Splits

Temporal: train before 2021, selection validation in 2021, calibration in 2022, test in 2023. This gives calibration an independent role. Geographic+temporal challenge: train non-holdout states before 2021; selection on NV/OR before 2022; calibration NV/OR during 2022; final test CA/TX during 2023. These states are predeclared for software experiments, not selected for favorable scores. This test combines geography and time shifts; an additional pure matched-time geography study is still needed for isolating the causes of degradation.

All models use the identical split manifests and four classes. Deduplication removes repeated IDs and exact start-time/location signatures; conflicting targets exclude every corresponding copy. Near-duplicate reports with slightly different times/coordinates remain an unresolved risk. Missing dates are quarantined in loading, never imputed from future records. No fit/tune operation uses final-state/test labels.

## Models and selection

Majority, logistic, Random Forest, XGBoost, CatBoost, CE-MLP, constrained CORAL, CORN, weighted ordinal, sampled ordinal, fixed-uniform geographic ordinal, group DRO, and GRAIL with ablations. CORAL/ordinal is the same shared-score core by design. Every nontrivial family gets two predefined candidate settings selected by validation NLL. Neural models additionally early-stop on validation loss. These are small comparable **candidate counts**, not equal wall-clock budgets or exhaustive fair tuning; training time is reported. Test performance never selects a candidate. The majority baseline has no tuning.

Imbalance variants use training-only inverse class-frequency weights or inverse-frequency sampling. Test prevalence remains untouched. No SMOTE. Boosting baselines currently use one-hot input from the shared preprocessor rather than CatBoost native categorical processing; native handling deserves a later baseline.

Calibration is a predeclared scalar temperature fitted using calibration rows only. Population NLL selects it, retaining T=1 if no fit improvement. No calibration-method selection on test. Group-weight updates use only training counts/losses.

## Evaluation and interpretation

Macro-F1, balanced accuracy, precision/recall/support for all four grades, ordinal MAE, quadratic kappa, confusion matrices, multiclass Brier sum, NLL, 10-bin top-label ECE, worst-state F1/MAE, entropy risk-coverage, runtime, inference latency and serialized size. Macro-F1 always averages four classes even when a group has no support for one; consult supports. ECE can hide poor class-4 calibration and is sample/bin dependent. Group minimums are noisy for small groups.

Three model seeds share one fixed synthetic dataset and split. Seed SD measures optimization variation, not population uncertainty. The 100-replicate row bootstrap interval is descriptive and can be optimistic for correlated real incidents. Use state/time-block bootstrap for confirmatory real-data work.

Risk-factor analysis uses permutation NLL changes and subgroup errors. A feature can proxy reporting policy; permutation may break correlated features. Neither is a causal intervention analysis. Disagreement flags between GRAIL/XGBoost/CORN are joined to source/state/weather. Incident risk and review priority are separate ranks; neither estimates accident frequency or prevented harm.
