#!/usr/bin/env bash
# Reproduce the competition entry end to end.
#
# Stage 0 costs nothing and proves the pipeline works. Stages 1-4 need a real
# evaluator (Docker locally, or Modal) and take hours: a medium run is 15-40
# minutes and the entry rests on thousands of them.
#
#   ./scripts/reproduce.sh check      # offline: install, tests, mock optimiser run
#   ./scripts/reproduce.sh evaluator  # start 10 local API containers
#   ./scripts/reproduce.sh diagnose   # per-game fast-vs-medium correlation
#   ./scripts/reproduce.sh search     # medium-direct search, one loop per game
#   ./scripts/reproduce.sh confirm    # race the leaders to n>=3
#   ./scripts/reproduce.sh bundle     # print the submission JSON
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export PYTHONPATH=src
STAGE="${1:-check}"
PYTHON="${PYTHON:-python3}"

case "$STAGE" in
check)
  "$PYTHON" -m pip install -q -r requirements.txt
  "$PYTHON" -m unittest discover -s tests -t .
  echo "--- offline optimiser run against the built-in surrogate ---"
  task_check_dir=$(mktemp -d)
  trap 'rm -rf "$task_check_dir"' EXIT
  "$PYTHON" -m ttbalance --backend mock --results-dir "$task_check_dir" \
      --cache "$task_check_dir/cache.sqlite" --experiment check \
      search --game CantStop --optimizer pbil --generations 12 --budget 150
  ;;
evaluator)
  ./scripts/localapi.sh start "${N:-10}"
  ;;
diagnose)
  # Which games can a cheap run rank? Answered per game before trusting one.
  "$PYTHON" - <<'PY'
import os, statistics as st
from ttbalance.cache import Cache
from ttbalance.cli import storage_dir
from types import SimpleNamespace
args = SimpleNamespace(results_dir=os.environ.get("TTB_RESULTS_DIR", "results/experiments"),
                       experiment=os.environ.get("TTB_EXPERIMENT", "default"),
                       backend=os.environ.get("TTB_BACKEND", "local"), seed=1,
                       evaluator_version=os.environ.get("TTB_EVALUATOR_VERSION", "unversioned"))
path = os.environ.get("TTB_CACHE") or os.path.join(storage_dir(args), "cache.sqlite")
if not os.path.isfile(path):
    raise SystemExit("No observations yet; run search and confirm before diagnose.")
c = Cache(path)._conn
print("%-18s %8s %8s %7s  %s" % ("game", "spread", "sd", "SNR", "medium can rank?"))
for g in ("ExplodingKittens", "CantStop", "Wonders7", "Dominion"):
    rows = c.execute("SELECT key,AVG(score) FROM observations WHERE game=? AND"
                     " run_type='medium' GROUP BY key HAVING COUNT(*)>=3", (g,)).fetchall()
    if len(rows) < 3:
        print("%-18s (not enough data)" % g); continue
    means = [m for _, m in rows]
    sds = []
    for key, _ in rows:
        s = [r[0] for r in c.execute("SELECT score FROM observations WHERE game=? AND"
                                     " run_type='medium' AND key=?", (g, key))]
        if len(s) > 1:
            sds.append(st.pstdev(s))
    sd = st.median(sds) if sds else 0.0
    spread = max(means) - min(means)
    snr = spread / (sd / 3 ** 0.5) if sd else 0.0
    print("%-18s %8.1f %8.1f %7.1f  %s"
          % (g, spread, sd, snr, "yes" if snr > 6 else "barely" if snr > 3 else "no"))
PY
  ;;
search)
  POOL_TOTAL="${POOL_TOTAL:-10}" ./scripts/search_all.sh
  echo "loops launched; watch results/search-*.log"
  ;;
confirm)
  ./scripts/verify_all.sh
  ;;
bundle)
  "$PYTHON" -m ttbalance --backend "${TTB_BACKEND:-local}" best --show-params
  ;;
*)
  echo "usage: $0 {check|evaluator|diagnose|search|confirm|bundle}"; exit 1 ;;
esac
