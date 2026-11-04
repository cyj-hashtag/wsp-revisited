#!/usr/bin/env bash
# The registered pipeline, unattended, in order; it stops at the first failure (a FAIL, the calibration gate, an error):
#   check_vllm -> smoke -> calib (gate) -> budget check -> tau -> choose tau --write -> select -> choose config -> rq1
# (check_vllm is skipped when results/preflight.json already holds no FAIL; smoke's gate is informative only)
# Run it inside tmux after scripts/setup.sh:
#   tmux new -s run 'bash scripts/run_all.sh 2>&1 | tee -a $WSPREV_HOME/run_all.log'
# Re-running is safe: run_stage.py continues where a stage stopped. `touch $WSPREV_HOME/STOP` stops it before the next
# stage. After each stage its small results are committed locally (no push: the machine holds no GitHub credentials),
# so later runs record a clean git state; Claude copies the commits back to the PC.
#
# Budget check (before tau): from the calibration's measured tokens/s, spent so far + the rest of P0 + a reduced RQ2
# must stay within $WSPREV_BUDGET yuan at $WSPREV_PRICE yuan per hour for the whole machine; else it stops.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
BUDGET="${WSPREV_BUDGET:-150}"
PRICE="${WSPREV_PRICE:-3.0}"
START="${WSPREV_START:-$(stat -c %Y /.dockerenv 2>/dev/null || date +%s)}"   # when billing began (container start)

stage() {
  if [ -f "$WSPREV_HOME/STOP" ]; then echo "$WSPREV_HOME/STOP exists: stopping before $1"; exit 0; fi
  echo; echo "##### $1 ($(date '+%F %T'))"
}
commit() {  # commit <message> <paths...>
  local msg=$1 p; shift
  for p in "$@"; do if [ -e "$p" ]; then git add -- "$p"; fi; done
  if git diff --cached --quiet; then return; fi
  git -c user.name=cyj-hashtag -c user.email=328945561+cyj-hashtag@users.noreply.github.com commit -q \
    -m "$msg" -m "Written by scripts/run_all.sh on the GPU machine." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  echo "committed $(git rev-parse --short HEAD): $msg"
}

stage check_vllm
if [ -f results/preflight.json ] && ! grep -q '"status": "FAIL"' results/preflight.json; then
  echo "passed before: results/preflight.json"
else
  python scripts/check_vllm.py
  commit "GPU preflight: results/preflight.json" results/preflight.json
fi

stage smoke
# The gate is registered on all 500 problems (stage calib, which run_stage.py enforces). On smoke's 100 problems one
# sample has an SE of about 5 pp, so the gate is only informative there and its exit status 3 does not stop the run.
# Smoke and calib share their run folders, so once calib has run, summarizing smoke again would read 500 problems.
if [ -d results/calib ]; then
  echo "calib has run: smoke is not summarized again (results/smoke keeps its 100 problems)"
else
  python scripts/run_stage.py smoke || [ $? -eq 3 ]
  commit "Stage smoke: summaries" results/smoke
fi

stage calib
python scripts/run_stage.py calib
commit "Stage calib: summaries" results/calib

stage "budget check"
python - "$BUDGET" "$PRICE" "$START" <<'EOF'
import glob, json, sys, time
budget, price, start = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
tok = sec = 0
for f in glob.glob("results/calib/*/summary.json"):
    t = json.load(open(f))["throughput"]
    tok, sec = tok + t["tokens"], sec + t["seconds"]
import subprocess
gpus = len(subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.split())
rate = tok / sec * gpus                       # tokens/s of the whole machine (seconds are summed over shards)
rest = (6.7 + 5.4 + 225.7 + 30) * 1e6         # tau + select + rq1 (results/cost_estimate.json) + a reduced RQ2
spent = (time.time() - start) / 3600 * price
need = rest / rate / 3600 * price
print(f"measured {rate:,.0f} tok/s on {gpus} GPU(s); spent ~{spent:.0f} yuan; the rest needs ~{rest / rate / 3600:.1f} h"
      f" = ~{need:.0f} yuan; total ~{spent + need:.0f} of {budget:.0f}")
if spent + need > budget:
    sys.exit("OVER BUDGET: stopping before tau; ask the user")
EOF

stage tau
python scripts/run_stage.py tau
python scripts/choose.py tau --write
commit "Stage tau: D3_TAU chosen on practice DEV" src/wsprev/decoding.py results/tau

stage select
python scripts/run_stage.py select
python scripts/choose.py config
commit "Stage select: base configuration of the cross comparison" results/select

stage rq1
python scripts/run_stage.py rq1
commit "Stage rq1: summaries" results/rq1
echo; echo "##### ALL DONE ($(date '+%F %T'))"
