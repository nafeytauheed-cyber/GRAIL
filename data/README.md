# Data provenance

Download the exact dataset from https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents using your own authorized access. Do not commit it. Confirm its current CC BY-NC-SA research-use terms at source; the software license does not supersede data terms.

Example after configuring Kaggle locally:

```bash
kaggle datasets download -d sobhanmoosavi/us-accidents -p data/raw --unzip
python -m grail.cli prepare data/raw/US_Accidents_March23.csv data/processed/us_accidents.parquet
python -m grail.cli audit data/processed/us_accidents.parquet --out reports/real_audit
```

No credentials belong in the repository. No raw data is shipped. `grail.data.synthetic.synthetic` generates the deterministic artificial fixture used in current results. Synthetic records have `SYNTH-` IDs.
