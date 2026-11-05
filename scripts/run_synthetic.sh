#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python -m grail.cli experiment --synthetic --rows 8000 --seeds 11,23,37 --epochs 25 --trials 2 --out reports/results/synthetic
python -m grail.cli figures reports/results/synthetic/results.json --out reports/figures
