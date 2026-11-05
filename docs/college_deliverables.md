# College PS 3 deliverable map

| Requirement | Implemented evidence | Current scientific status |
|---|---|---|
| Problem/objective | README, methodology, viva guide | Traffic-disruption semantics explicit |
| Dataset description | data/README, feature availability | Source description; no real-data counts measured |
| EDA and imbalance | warehouse audit; reports/synthetic_audit | Synthetic demonstration only; real EDA blocked |
| Feature engineering | data/features.py, preprocessing.py | Implemented and tested; first-report availability unproven |
| Algorithm comparison | experiment runner; results JSON | Executed synthetic baselines and neural variants |
| Custom architecture/math | models/classifier.py, ordinal.py, losses/groups.py, mathematics.md | Working implementation; no novelty/superiority claim |
| Evaluation | reports/results, figures | Measured synthetic results with seeds and uncertainty caveats |
| Risk-factor analysis | permutation and grouped error CSVs | Software demonstrated; real-world conclusions blocked |
| Geographic/temporal validation | split manifests and two protocols | Synthetic joint-shift and time tests; real transfer unknown |
| Limitations/conclusions | limitations.md, results_summary.md | Explicit incompleteness and negative findings retained |
| Future improvements | literature_review ranked shortlist | Label audit first; then stronger baselines and focused hypotheses |
| Viva | viva_guide.md | Student-level explanation with truthful answers |

This package is not evidence that every empirical college requirement is completed. Finishing those requirements requires the real dataset, a defensible label-comparability decision and new experiments. It supplies working machinery and an honest current report.
