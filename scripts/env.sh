# Shared environment of the scripts on the GPU machine.   Usage: source scripts/env.sh
# Everything large lives on AutoDL's data disk (/root/autodl-tmp); override any variable before sourcing.
export WSPREV_HOME="${WSPREV_HOME:-/root/autodl-tmp/wsprev}"
export WSPREV_LOGS="${WSPREV_LOGS:-$WSPREV_HOME/lm_understand-logs}"   # the authors' log repository [B]
export WSPREV_MODELS="${WSPREV_MODELS:-$WSPREV_HOME/models}"           # base/, p50c/
export WSPREV_BENCH="${WSPREV_BENCH:-$WSPREV_HOME/bench}"              # math500/gsm8k/ifeval .jsonl
export WSPREV_RUNS="${WSPREV_RUNS:-$WSPREV_HOME/runs}"                 # raw generations (never in git)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$WSPREV_HOME/hf_home}"
export NLTK_DATA="${NLTK_DATA:-$WSPREV_HOME/nltk_data}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WSPREV_HOME/pip_cache}"
if [ -f "$WSPREV_HOME/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$WSPREV_HOME/venv/bin/activate"
fi
