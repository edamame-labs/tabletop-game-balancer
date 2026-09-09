#!/usr/bin/env bash
# One search loop per game, each pinned to a DISJOINT slice of the container
# pool so they never contend. Slice sizes follow headroom x cheapness: the
# cheap, least-optimised games get the most containers; Dominion (slow, already
# ~987/1000) gets one. Total must equal the running pool size (10).
#   game            : first_port : n_containers : per-pass budget
set -euo pipefail
cd "$(dirname "$0")/.."
[ "${DRY_RUN:-0}" = 1 ] || mkdir -p results
# game : first_port : n_containers : per-pass budget : repeats-per-config
# game : first_port : n_containers : budget : repeats : timeout_ms
# Medium evaluations can take 40 minutes; use a 50-minute simulator guard.
declare -a PLAN=(
  "ExplodingKittens:3000:4:80:3:3000000"
  "CantStop:3004:3:80:3:3000000"
  "Wonders7:3007:2:60:2:3000000"
  "Dominion:3009:1:12:2:3000000"
)
for row in "${PLAN[@]}"; do
  IFS=: read -r g port n b reps tmo <<< "$row"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    GAME="$g" FIRST_PORT="$port" POOL="$n" BUDGET="$b" REPEATS="$reps" TIMEOUT_MS="$tmo" \
      bash ./scripts/search_game.sh
    continue
  fi
  GAME="$g" FIRST_PORT="$port" POOL="$n" BUDGET="$b" REPEATS="$reps" TIMEOUT_MS="$tmo" \
    nohup ./scripts/search_game.sh > "results/nohup-$g.out" 2>&1 &
  sleep 2
done
echo "launched ${#PLAN[@]} per-game loops on disjoint container slices"
