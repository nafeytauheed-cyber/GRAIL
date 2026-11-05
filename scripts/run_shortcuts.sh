#!/usr/bin/env bash
set -euo pipefail
for mode in no_geo provider year no_weather; do
  python -m grail.cli experiment --synthetic --rows 8000 --seeds 11 --models grail,xgboost,corn --epochs 25 --trials 2 --feature-mode "$mode" --out "reports/results/shortcut_$mode"
done
