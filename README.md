# GRAIL

**Geographically Robust Adaptive Ordinal Incident Learner** — a Python research package for college PS 3: road-accident severity prediction and risk-factor analysis.

The outcome is **traffic-disruption severity (1–4), not injury or fatality severity**. We estimate a grade conditional on an already reported incident; we do not predict accident frequency or claim to prevent accidents.

## Current status

Working PyTorch estimator, disk-backed preparation/audit, conservative feature allowlist, train-only transformations, temporal and unseen-state splits, conventional and ordinal baselines, group DRO and shrinkage-regularized ablations, calibration, uncertainty/error reports and tests.

**All current experiment results are synthetic.** Kaggle access was unavailable; the reported 2019 label change remains unverified. No real-world accuracy, novelty or superiority claim is made. See [limitations](docs/limitations.md), [actual results](docs/results_summary.md) and [college completion map](docs/college_deliverables.md).

## Install and reproduce

Python 3.10+; start in a virtual environment. For a CPU-only install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,boosting]'
python -m pytest -q
bash scripts/run_synthetic.sh
bash scripts/run_shortcuts.sh
python scripts/finalize_reports.py
```

Versions actually executed are in `requirements-executed.txt` and `configs/executed_environment.json`. The PyTorch CPU version uses its CPU wheel index. For CUDA, install the appropriate PyTorch build separately; GPU execution is supported by `device='cuda'` but was not tested here. Default batch size is 256. Neural training uses early stopping and gradient clipping, not mixed precision.

## Model interface

```python
from grail import GRAILClassifier
from grail.data.features import features

# raw_train/raw_validation/raw_calibration/raw_test must be disjoint splits.
X_train = features(raw_train)
model = GRAILClassifier(seed=42, epochs=50, batch_size=256)
model.fit(X_train, raw_train.Severity, groups=raw_train.State,
          X_val=features(raw_validation), y_val=raw_validation.Severity)
model.calibrate(features(raw_calibration), raw_calibration.Severity)
predictions = model.predict(features(raw_test))
probabilities = model.predict_proba(features(raw_test))  # columns: 1, 2, 3, 4
model.save('grail.pt')
loaded = GRAILClassifier.load('grail.pt')
```

Fit requires all four classes and prediction-time features. Calling `fit` directly does not infer whether arbitrary user-supplied columns are leaked; use `features` and the documented split pipeline. The estimator accepts numeric arrays or DataFrames and checks column consistency. Load only trusted checkpoint files.

## Real-data workflow

Acquire the CSV yourself using the instructions in [data/README.md](data/README.md). Then:

```bash
python -m grail.cli prepare data/raw/US_Accidents_March23.csv data/processed/us_accidents.parquet
python -m grail.cli audit data/processed/us_accidents.parquet --out reports/real_audit
# Review audit and complete configs/label_review.template.json as a new review file.
python -m grail.cli experiment --parquet data/processed/us_accidents.parquet \
  --rows 50000 --label-review configs/label_review.json --out reports/results/real_subset
```

A real run refuses to proceed without a written compatibility review. `--source`, `--start`, `--end` select predeclared sensitivity scopes. The primary target always stays four classes. Raw data and large artifacts are excluded from Git.

Parquet preparation and audit use DuckDB's disk-backed operations and a 2 GB query memory limit. The current estimator trains on a **bounded in-memory development sample**, not a demonstrated full-dataset stream. Keep sample size conservative on 8 GB RAM.

## Experiments and outputs

```bash
# Baselines and GRAIL, same frozen split IDs:
python -m grail.cli experiment --synthetic --models majority,logistic,forest,xgboost,catboost,mlp,coral,corn,ordinal_dro,grail
# Group-weight/imbalance ablations:
python -m grail.cli experiment --synthetic --models coral,ordinal_weighted,ordinal_sampled,ordinal_fixed,ordinal_dro,grail,grail_no_shrink,grail_no_regularization
# Geographic/temporal evaluation:
python -m grail.cli experiment --synthetic --modes temporal,geographic
# Figures from saved metrics:
python -m grail.cli figures reports/results/synthetic/results.json --out reports/figures
```

The experiment command trains, selects, calibrates and evaluates; there is no web service. Each output directory includes origin/seed/version manifest, split IDs and class distributions, full metrics, predictions, failure groups, disagreement and model-reliance reports. GRAIL inference checkpoints are generated locally and excluded from Git. Risk ranking and uncertainty review ranking are separate; neither estimates intervention value.

Temporal roles: train <2021, selection 2021, calibration 2022, test 2023. Geographic mode holds out NV/OR for selection/calibration and CA/TX for final testing, with an additional time shift. Model selection never uses final states. Worst-state metrics always need group sample sizes alongside them.

## Research and learning

- [Literature/source-code review and ranked hypotheses](docs/literature_review.md)
- [Derivation and optimization](docs/mathematics.md)
- [Feature availability audit](docs/feature_availability.md)
- [Protocol and scientific decisions](docs/methodology.md)
- [Student viva guide](docs/viva_guide.md)
- [Research log and blockers](docs/research_log.md)

## GitHub handoff

The local repository has tested milestone commits. Remote creation/push is blocked because the connected tools cannot create repositories and the browser requires sign-in. No unrelated repository was modified. After creating an empty **private** GRAIL repository, use the included local Git bundle (or this working tree), add its remote, push `main`, and continue on a development branch. A draft PR requires a remote base branch and subsequent changes; none is falsely claimed here.

MIT software license; dataset and dependencies retain their own terms. See [attribution](THIRD_PARTY.md).
