# Shared environment of the scripts on the GPU machine.   Usage: source scripts/env.sh
# Large files go to the machine's data disk; override any variable before sourcing.
if [ -z "${WSPREV_HOME:-}" ]; then
  if [ -d /root/autodl-tmp ]; then
    WSPREV_HOME=/root/autodl-tmp/wsprev          # AutoDL data disk
  elif [ -d /hy-tmp ]; then
    WSPREV_HOME=/hy-tmp/wsprev                   # gpushare.com (恒源云) instance disk; wiped 24 h after shutdown
  elif [ -d /workspace ]; then
    WSPREV_HOME=/workspace/wsprev                # RunPod and similar container hosts
  else
    WSPREV_HOME="$HOME/wsprev"
  fi
fi
export WSPREV_HOME
export WSPREV_LOGS="${WSPREV_LOGS:-$WSPREV_HOME/lm_understand-logs}"   # the authors' log repository [B]
export WSPREV_MODELS="${WSPREV_MODELS:-$WSPREV_HOME/models}"           # base/, p50c/
export WSPREV_BENCH="${WSPREV_BENCH:-$WSPREV_HOME/bench}"              # math500/gsm8k/ifeval .jsonl
export WSPREV_RUNS="${WSPREV_RUNS:-$WSPREV_HOME/runs}"                 # raw generations (never in git)
export HF_HOME="${HF_HOME:-$WSPREV_HOME/hf_home}"
export NLTK_DATA="${NLTK_DATA:-$WSPREV_HOME/nltk_data}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WSPREV_HOME/pip_cache}"
# written by scripts/setup.sh: HF_ENDPOINT=https://hf-mirror.com when huggingface.co is not reachable
if [ -f "$WSPREV_HOME/env.local" ]; then
  # shellcheck disable=SC1091
  source "$WSPREV_HOME/env.local"
fi
if [ -f "$WSPREV_HOME/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$WSPREV_HOME/venv/bin/activate"
fi
