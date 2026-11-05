# Limitations and completion boundaries

- **No real US Accidents records were acquired.** No real class counts, EDA findings, 2019 label-change verification, road-safety risk factors, or superiority findings are claimed. Real-data PS 3 conclusions remain blocked.
- **No remote GitHub repository or PR was created.** Connected GitHub lists five existing repositories, none named GRAIL; available operations cannot create a repository. Browser fallback reached a GitHub sign-in wall. Unrelated repositories were not repurposed. Local Git commits and a portable bundle preserve milestones.
- Ordinal labels measure traffic disruption; no direct injury/fatality inference, accident frequency, causality, prevention or intervention benefit.
- The cumulative head is established mathematics. The adaptive objective combines known methods; originality is unestablished. Count-based shrinkage does not identify mislabeled data and may underweight genuine small-group difficulties.
- Synthetic experiments can validate execution and illustrate tradeoffs only. Their generator is not a calibrated simulator of US accidents. Rare classes and tiny final geography samples yield unstable F1 and kappa.
- Four-class compatibility remains unverified. No binary sensitivity result is reported. Source/time filtering exists but cannot determine a scientifically valid scope by itself.
- DuckDB preparation/audit is disk backed; neural fitting currently materializes a bounded sample. Full-scale training, GPU/mixed precision, peak-RAM benchmarks and resume-training checkpoints are not demonstrated. Saved checkpoints support reproducible inference, not exact optimizer resume.
- TabM was inspected but is not an executed baseline. No bespoke interaction encoder, meta-loss learner, self-supervised pretraining or conformal guarantees are claimed.
- Finite two-candidate tuning budgets are intentionally small; equal candidates do not imply equal compute. The baseline comparison is developmental, not a publishable leaderboard.
- The “CORAL” baseline explicitly constrains ordered thresholds; it is not a bit-for-bit reproduction of the upstream implementation.
- Initial-report availability cannot be verified without versioned reports and publication timestamps. Weather observation-time filtering is only a conservative approximation.
- Deduplication is exact-ID/exact-location-time based. Near duplicates and road-segment spillover remain possible. Sampling before deduplication can retain one version of a duplicated incident; no duplicate from outside the sampled population appears in model splits, but full-data duplicate auditing is still required.
- Geography+time holdout measures joint shift, not a pure geographic causal effect. CA/TX results cannot stand for every unseen state. Calibration uses separate validation states, never final states, but need not transfer.
- Entropy may fail to rank errors. Temperature scaling cannot repair arbitrary shift; low pooled ECE does not ensure rare-class or group calibration.
- Unsafe checkpoint deserialization is avoided only by loading trusted artifacts. `load` uses joblib/pickle; do not open untrusted files.
