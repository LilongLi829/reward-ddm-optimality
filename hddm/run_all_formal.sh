#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/jovyan/project"
FIT_SCRIPT="$PROJECT/hddm/01_fit_model_standalone.py"
RESULTS_ROOT="$PROJECT/results/hddm_formal"
LOG_ROOT="$RESULTS_ROOT/logs"

SAMPLES=5000
BURN=2500
THIN=2

MODELS=("m_v" "m_a" "m_va")
CHAINS=(1 2 3 4)

mkdir -p "$LOG_ROOT"
cd "$PROJECT"

echo "============================================================"
echo "Formal HDDM batch started: $(date)"
echo "Project:      $PROJECT"
echo "Fit script:   $FIT_SCRIPT"
echo "Results root: $RESULTS_ROOT"
echo "Samples:      $SAMPLES"
echo "Burn:         $BURN"
echo "Thin:         $THIN"
echo "Models:       ${MODELS[*]}"
echo "Chains:       ${CHAINS[*]}"
echo "============================================================"

if [[ ! -f "$FIT_SCRIPT" ]]; then
  echo "ERROR: fit script not found: $FIT_SCRIPT"
  exit 1
fi

for model in "${MODELS[@]}"; do
  for chain in "${CHAINS[@]}"; do
    chain_dir="$RESULTS_ROOT/$model/chain_$(printf '%02d' "$chain")"
    log_file="$LOG_ROOT/${model}_chain_$(printf '%02d' "$chain").log"

    # Safe resume: skip a chain only when all key outputs exist.
    if [[ -f "$chain_dir/run_metadata.json" \
       && -f "$chain_dir/stats.csv" \
       && -f "$chain_dir/dic.txt" \
       && -e "$chain_dir/model" ]]; then
      echo
      echo "SKIP complete: $model chain $chain"
      continue
    fi

    echo
    echo "------------------------------------------------------------"
    echo "START: $model chain $chain at $(date)"
    echo "Log:   $log_file"
    echo "------------------------------------------------------------"

    python -u "$FIT_SCRIPT" \
      --model "$model" \
      --chain "$chain" \
      --samples "$SAMPLES" \
      --burn "$BURN" \
      --thin "$THIN" \
      --results-root "$RESULTS_ROOT" \
      2>&1 | tee "$log_file"

    echo "DONE:  $model chain $chain at $(date)"
  done
done

echo
echo "============================================================"
echo "ALL FORMAL HDDM CHAINS COMPLETED: $(date)"
echo "Results: $RESULTS_ROOT"
echo "============================================================"
