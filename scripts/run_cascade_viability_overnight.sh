#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace
LOG=data/processed/cascade_viability/overnight.log
mkdir -p data/processed/cascade_viability
echo "START $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$LOG"
python3 -m src.eval.cascade_viability_batch 2>&1 | tee -a "$LOG"
echo "BATCH_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
# Commit results onto the viability-audits branch
git add \
  src/eval/cascade_viability_batch.py \
  experiments/protocol.json \
  reports/cascade_short_long_viability_audits.md \
  reports/cascade_short_long_viability_results.md \
  reports/cascade_short_long_viability_gated.md \
  data/processed/cascade_viability \
  data/predictions/2026-27/cascade_viability_ft \
  data/predictions/2026-27/cascade_viability_xi_cap \
  REPORT_STATE.json \
  2>/dev/null || true
if git diff --cached --quiet; then
  echo "NO_CHANGES_TO_COMMIT" | tee -a "$LOG"
else
  git commit -m "Overnight cascade viability batch results (A–D)"
  git push -u origin HEAD 2>&1 | tee -a "$LOG" || true
fi
echo "END $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
