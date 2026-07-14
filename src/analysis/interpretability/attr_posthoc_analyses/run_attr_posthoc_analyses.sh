#!/usr/bin/env bash
# run_hpc_analyses.sh -- run the hpc_scripts analyses over a list of TF/cell-line pairs.
#
# All five stages run per pair; GNU Parallel fans out ACROSS pairs, not across stages.
# corr1/corr2/heatmaps1 are mutually independent; overlap1 -> overlap2 is a real
# producer->consumer edge, so overlap2 is skipped when overlap1 fails.
#
#   bash run_hpc_analyses.sh --pairs pairs.tsv [--jobs N] [--stages LIST] [--dry-run]
#
# --pairs   TSV, "TF<TAB>CELL_LINE"; '#' comments and blank lines ignored.
#           Cell line is spelled as in the attrs parquet (22Rv1, LNCAP, A-375).
# --stages  comma-separated subset of: corr1,corr2,heatmaps1,overlap1,overlap2
#           (default: all). Lets one failed branch be rerun without redoing the pair.
# --jobs    concurrent PAIRS (default 2).
#
# MEMORY: overlap1 loads every window's atac_attrs into RAM (~11 GB for ~340k windows),
# so peak usage is roughly jobs x 11 GB. Hence the low default. Raising --jobs is safe
# when --stages excludes overlap1.
#
# Requires the `pterodactyl` conda env on PATH (python) plus Rscript.
# Logs: logs/hpc/<TF>_<CELL_LINE>.log, one per pair, plus logs/hpc/joblog (exit codes).
# Resume a partial sweep by re-running the same command with `parallel --resume`.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALL_STAGES="corr1,corr2,heatmaps1,overlap1,overlap2"

PAIRS=""
JOBS=2
STAGES="$ALL_STAGES"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pairs)   PAIRS="$2";   shift 2 ;;
    --jobs)    JOBS="$2";    shift 2 ;;
    --stages)  STAGES="$2";  shift 2 ;;
    --dry-run) DRY_RUN=1;    shift ;;
    -h|--help) sed -n '2,23p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$PAIRS"    ]] || { echo "--pairs is required" >&2; exit 2; }
[[ -f "$PAIRS"    ]] || { echo "Pairs file not found: $PAIRS" >&2; exit 2; }
command -v parallel >/dev/null || { echo "GNU parallel not on PATH" >&2; exit 2; }

for s in ${STAGES//,/ }; do
  [[ ",$ALL_STAGES," == *",$s,"* ]] || { echo "Unknown stage: $s" >&2; exit 2; }
done

LOG_DIR="${SCRIPT_DIR}/logs/hpc"
mkdir -p "$LOG_DIR"

# Is a stage selected?
stage() { [[ ",$STAGES," == *",$1,"* ]]; }

run_pair() {
  local tf="$1" cell="$2"
  local log="${LOG_DIR}/${tf}_${cell}.log"
  local rc=0

  if [[ "$DRY_RUN" == 1 ]]; then
    for s in ${STAGES//,/ }; do echo "[dry-run] $s  $tf  $cell"; done
    return 0
  fi

  cd "$SCRIPT_DIR" || return 1
  {
    echo "=== ${tf} / ${cell} :: stages=${STAGES} :: $(date -Is) ==="

    # Independent branches: a failure in one must not mask the others.
    stage corr1     && { python  corr1_atac_attr_correlations.py    --tf "$tf" --cell-line "$cell" || { echo "!! corr1 failed";     rc=1; }; }
    stage corr2     && { python  corr2_finegrain_corrs.py           --tf "$tf" --cell-line "$cell" || { echo "!! corr2 failed";     rc=1; }; }
    stage heatmaps1 && { Rscript heatmaps1_genome_heatmaps.R        --tf "$tf" --cell-line "$cell" || { echo "!! heatmaps1 failed"; rc=1; }; }

    # overlap2 consumes overlap1's intermediates -- only run it if overlap1 succeeded.
    local overlap1_ok=1
    stage overlap1 && { Rscript overlap1_binding_overlap_analysis.R --tf "$tf" --cell-line "$cell" \
                          || { echo "!! overlap1 failed"; rc=1; overlap1_ok=0; }; }
    if stage overlap2; then
      if [[ "$overlap1_ok" == 1 ]]; then
        Rscript overlap2_binding_overlap_plots.R --tf "$tf" --cell-line "$cell" \
          || { echo "!! overlap2 failed"; rc=1; }
      else
        echo "-- overlap2 skipped (overlap1 failed)"
      fi
    fi

    echo "=== done rc=${rc} :: $(date -Is) ==="
  } >"$log" 2>&1

  return "$rc"
}

export SCRIPT_DIR LOG_DIR STAGES DRY_RUN
export -f stage run_pair

# `export -f` is bash-only; pin the worker shell so it resolves under a zsh login shell.
export PARALLEL_SHELL=/bin/bash

grep -vE '^[[:space:]]*(#|$)' "$PAIRS" \
  | parallel --colsep '\t' --jobs "$JOBS" --progress \
      --joblog "${LOG_DIR}/joblog" \
      run_pair {1} {2}

status=$?
echo "Pairs with a non-zero exit are listed in ${LOG_DIR}/joblog; per-pair logs in ${LOG_DIR}/"
exit "$status"
