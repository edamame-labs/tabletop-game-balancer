#!/usr/bin/env bash
# Confirm candidates at medium fidelity. Stop search loops before using the full pool.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export PYTHONPATH=src
POOL="${POOL:-10}"
RUN_TYPE="${RUN_TYPE:-medium}"
FROM_RUN_TYPE="${FROM_RUN_TYPE:-$RUN_TYPE}"
PYTHON="${PYTHON:-python3}"
LOG=results/verify.log
if [ "${DRY_RUN:-0}" != 1 ]; then mkdir -p results; : > "$LOG"; fi
for G in CantStop ExplodingKittens Wonders7 Dominion; do
  command=("$PYTHON" -u -m ttbalance --backend local --local-pool "$POOL" --workers "$POOL"
      --timeout-ms "${TIMEOUT_MS:-3000000}" verify --game "$G"
      --run-type "$RUN_TYPE" --from-run-type "$FROM_RUN_TYPE"
      --top 6 --keep 3 --rounds 1 3 5 7)
  if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' "${command[@]}"; printf '\n'; continue
  fi
  echo "=== verify $G $(date '+%T') ===" | tee -a "$LOG"
  "${command[@]}" >> "$LOG" 2>&1
done
if [ "${DRY_RUN:-0}" != 1 ]; then
  "$PYTHON" -u -m ttbalance --backend local best >> "$LOG" 2>&1
  echo "VERIFY DONE $(date '+%T')" >> "$LOG"
fi
