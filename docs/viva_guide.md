# Viva guide: understand and defend the work

## A short explanation you can give

“Our college problem is accident severity and risk-factor analysis. Here severity specifically means how much a reported accident disrupts traffic. We compare ordinary classifiers against an ordinal neural network. Ordinal means class 4 is above 3, which is above 2 and 1. We test whether giving more attention to difficult geographic groups improves generalization. We separate past training data, validation data, calibration data and future test data. We also reserve whole states. Current execution results are synthetic because the real dataset could not be downloaded; they validate our code, not real road-safety performance.”

## How one row travels through the program

1. The raw CSV becomes compressed Parquet. The audit counts labels, missing values and inconsistent times without loading the entire dataset into Pandas.
2. Duplicate incidents are removed; conflicting duplicate labels are quarantined. Time/state rules assign disjoint roles.
3. A feature allowlist removes end time, duration, affected distance and description. Future/stale weather is masked.
4. Numerical medians, scaling statistics and categorical vocabularies are learned from training only. New categories produce an unknown/all-zero representation.
5. The MLP maps a row into a score. Three ordered thresholds turn it into cumulative probabilities.
6. The loss teaches those probabilities to match the grade. Group weighting changes which training-group errors matter most.
7. Validation chooses settings. A separate calibration set fits one temperature. Final test data measure performance once settings are fixed.
8. Reports show errors, reliability, geographic gaps, weather sensitivity and cases where models disagree.

## Probability example

Suppose q₁=0.90, q₂=0.40 and q₃=0.10. Then p₁=0.10, p₂=0.50, p₃=0.30 and p₄=0.10. They sum to 1. The model predicts grade 2 by argmax. Expected grade is 1×0.10+2×0.50+3×0.30+4×0.10=2.40. This is an expected rank, not 2.4 injuries or minutes of delay.

For a true grade 3, ordinal targets are [1,1,0]: yes above 1, yes above 2, no above 3. Binary cross-entropy is applied to those three questions. Larger thresholds produce smaller probabilities, so the questions cannot contradict their order.

## Why the adaptive loss works mathematically

First compute the average loss for each training state. Ordinary DRO increasingly weights states with high loss. But one tiny state can have a noisy estimate. GRAIL blends its loss with the overall training average using n/(n+τ). Large n gives more trust to the state's own estimate. A KL penalty resists extreme weights, and a uniform mixture prevents any weight from becoming exactly zero. This does not prove a small state has bad labels and can harm a genuinely difficult small state. That is why we compare with ordinary DRO and inspect small groups.

The model's actual gradient weights must include the average-loss contribution of shrinkage. Merely shrinking the adversary's losses and leaving model gradients unchanged would optimize a different objective. A unit test verifies the equality.

## Likely questions

**What exactly are you predicting?** The recorded traffic-disruption grade of an incident, conditional on its being reported. No accident-occurrence or injury prediction claim.

**Why not random train/test split?** Nearby/duplicate incidents and changing reporting systems can make random splits too optimistic. Future years and unseen states are harder, more relevant challenges.

**Why exclude duration and Distance(mi)?** They may only become known after the incident develops and can almost reveal the target. Our decision point is initial reporting.

**Why is observation time not enough?** A weather observation made earlier may still have been published later. This dataset does not provide publication times or first-report snapshots.

**How do you prevent preprocessing leakage?** Fit imputation, scaling and category maps on training only. Transform other splits with those frozen objects. Test-state labels never choose features or hyperparameters.

**Why do we need both validation and calibration?** Validation selects models and stopping points. Calibration fits the probability correction after that choice. Separating them reduces reuse bias.

**What is calibration?** If the model assigns 80% probability to many events, roughly 80% should occur in a well-calibrated representative population. Calibration can fail after a geographic/time shift.

**Does temperature scaling improve class recall?** This scalar transform preserves the most likely class, so argmax recall does not change. It changes probabilities, entropy and probability-based decisions.

**Why isn't accuracy enough?** A majority classifier can ignore rare classes while achieving high accuracy. Macro-F1, class-specific recall, ordinal error and calibration expose different failures.

**Macro-F1 versus balanced accuracy?** Macro-F1 averages per-class F1 (precision–recall tradeoff); balanced accuracy averages recall over represented classes. We additionally fix all four labels for macro-F1 and report supports.

**What is ordinal MAE?** Average absolute difference between predicted and actual class numbers. A 1→4 error costs three ranks, a 3→4 error one. It assumes only a useful rank-distance diagnostic, not equal physical harm intervals.

**What is Brier score?** Average sum of squared differences between the probability vector and the one-hot true class. Lower is better; our multiclass convention sums over four classes.

**What is group DRO?** A worst-training-group objective. Its weights are learned by increasing weights on groups with higher loss. It does not guarantee performance on completely unseen groups.

**Is GRAIL new?** The ordinal head, shrinkage and KL-regularized updates have predecessors. The combination is a testable project hypothesis. We have not established publication-level novelty.

**Did Severity 3 change in 2019?** Another repository/paper says so, but we could not independently verify it. Prevalence shifts alone do not prove relabeling. Compare within provider/state/year and seek coding documentation.

**Why not switch to binary?** That changes the assigned prediction problem. A separate binary sensitivity analysis can be useful only with a justified definition; four-class results must remain explicit.

**Why might XGBoost win?** Trees handle nonlinear tabular interactions and need less representation learning. A custom model is not automatically stronger. Report whichever model actually wins under the stated metric and split.

**What is a shortcut?** A convenient proxy, such as provider or location, that predicts labels in training but fails when reporting patterns change. Feature removal and holdout experiments test reliance, without proving causality.

**Can SHAP/permutation identify causes?** No. They describe how a fitted model uses correlated inputs. A road feature may be correlated with traffic exposure or reporting policy.

**What does disagreement mean?** Different models give different predictions. It identifies review candidates but is not proof that any one prediction is wrong.

**Why not SMOTE everywhere?** Synthetic interpolations across one-hot road categories or geographic coordinates can be implausible, and resampling before splitting leaks information. We use training-only weights or controlled sampling.

**Does 8 GB RAM support the full dataset?** Preparation/audit is disk backed, but current model training uses a capped sample. Full-scale training is not yet measured; do not claim it is supported merely because the CSV conversion runs.

**How do you reproduce the result?** Use the environment versions, fixed generator/model seeds, exact split manifests and CLI commands. Run pytest first. Results label synthetic origin and include predictions and logs.

**What did tests prove?** Mathematical and software contracts, such as valid probabilities, gradient flow, split isolation and serialization. They do not prove the target is valid or the model generalizes to real roads.
