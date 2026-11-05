#!/usr/bin/env bash
# RQ2, unattended, in order (prereg/rq2_rules.md; the plan in results/rq2/plan.json); it stops at the first failure:
#   rq2-dev -> rq2-read (+ build) -> rq2-screen (swap check, screening, choose) -> rq2-holdout (choose, export)
#   -> rq2-final
# Run it inside tmux:
#   tmux new -s rq2 'bash scripts/run_rq2.sh 2>&1 | tee -a $WSPREV_HOME/run_rq2.log'
# Re-running is safe: every step continues where it stopped (finished samples are skipped; read and build are redone
# only if their output is missing). `touch $WSPREV_HOME/STOP` stops it before the next stage. After each stage its
# small results are committed locally (no push: the machine holds no GitHub credentials); Claude copies them back.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

stage() {
  if [ -f "$WSPREV_HOME/STOP" ]; then echo "$WSPREV_HOME/STOP exists: stopping before $1"; exit 0; fi
  echo; echo "##### $1 ($(date '+%F %T'))"
}
commit() {  # commit <message> <paths...>
  local msg=$1 p; shift
  for p in "$@"; do if [ -e "$p" ]; then git add -- "$p"; fi; done
  if git diff --cached --quiet; then return; fi
  git -c user.name=cyj-hashtag -c user.email=328945561+cyj-hashtag@users.noreply.github.com commit -q \
    -m "$msg" -m "Written by scripts/run_rq2.sh on the GPU machine." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  echo "committed $(git rev-parse --short HEAD): $msg"
}

test -f results/rq2/plan.json || { echo "no results/rq2/plan.json"; exit 2; }

stage rq2-dev
python scripts/run_stage.py rq2-dev
commit "Stage rq2-dev: summaries" results/rq2-dev

stage rq2-read
if [ -f results/rq2/build.json ]; then
  echo "built before: results/rq2/build.json"
else
  python scripts/run_stage.py rq2-read
  commit "Stage rq2-read: directions, AUC, scores and the grid (results/rq2/build.json)" results/rq2/build.json
fi

stage rq2-screen
if [ -f results/rq2/screen.json ]; then
  echo "screened before: results/rq2/screen.json"
else
  python scripts/run_stage.py rq2-screen
  commit "Stage rq2-screen: swap check and screening" results/rq2/swapcheck.json results/rq2/screen.json
fi

stage rq2-holdout
if [ -f results/rq2/models.json ]; then
  echo "exported before: results/rq2/models.json"
else
  python scripts/run_stage.py rq2-holdout
  commit "Stage rq2-holdout: the chosen edit and the controls" results/rq2/choice.json results/rq2/models.json
fi

stage rq2-final
python scripts/run_stage.py rq2-final
commit "Stage rq2-final: summaries" results/rq2-final
echo; echo "##### RQ2 DONE ($(date '+%F %T'))"
