#!/usr/bin/env bash
# Reproduce the supplementary fusion-only arm and the per-seed reference arms.
#
#   bash scripts/run_supplementary_experiments.sh [workers]
#
# Runs on the cached concept embeddings in analysis/raw_concept_embs/ and the cached
# image embeddings under $CEF_WIKIART_CACHE / $CEF_MP100K_CACHE, so no dataset is re-encoded.
set -euo pipefail
cd "$(dirname "$0")/../analysis"
PY="${PYTHON_BIN:-python3}"
W="${1:-8}"

echo "== 1. CEF-FusionOnly"
$PY -u supp_run.py --workers "$W"

echo "== 2. per-seed reference arms (CEF, CEF w=0, ZLaP) under per-metric selection"
$PY -u supp_refs.py --workers "$W"
