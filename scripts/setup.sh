#!/usr/bin/env bash
# One-time setup on a rented Linux GPU machine (AutoDL, RunPod, Vast.ai, a university server, ...). It needs no GPU,
# so on AutoDL it can run in no-GPU mode (无卡模式), where it costs almost nothing.
#
#   bash scripts/setup.sh 2>&1 | tee setup.log
#
# Steps (re-running is safe; finished downloads are reused):
#   0. checks: Python version, free disk, GPU (if any); whether huggingface.co is reachable (else hf-mirror.com)
#   1. Python venv in $WSPREV_HOME: vLLM 0.33.0, transformers 5.20.2, math-verify 0.9.0, ... and this repo
#   2. NLTK punkt_tab (IFEval sentence counting)
#   3. the authors' log repository [B] at the pinned commit (practice problems, the p50c edit, original results)
#   4. Qwen/Qwen3.5-0.8B at the pinned revision; every file checked against Hugging Face's hashes
#   5. p50c rebuilt from the base weights and the authors' edit; every edited tensor checked against their sha256
#   6. MATH-500, GSM8K test and IFEval -> $WSPREV_BENCH, with dataset revisions and file hashes
#   7. unit tests
# Paths: scripts/env.sh ($WSPREV_HOME: /root/autodl-tmp/wsprev on AutoDL, /workspace/wsprev where /workspace exists,
# else ~/wsprev; set WSPREV_HOME before running to choose another disk).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

REVISION=2fc06364715b967f1860aea9cf38778875588b17
LOGS_COMMIT=4d09ba6d6a7e891dd8063974dfe81b3d78e3494f
BASE="$WSPREV_MODELS/base"
P50C="$WSPREV_MODELS/p50c"

turbo() {  # AutoDL only: run one command through its academic proxy (GitHub, NLTK data) without changing this shell
  ( if [ -f /etc/network_turbo ]; then source /etc/network_turbo >/dev/null 2>&1; fi; "$@" )
}
step() { echo; echo "=== $* ==="; }

mkdir -p "$WSPREV_HOME" "$WSPREV_MODELS" "$WSPREV_RUNS" "$WSPREV_BENCH"

step "0/7 checks (WSPREV_HOME=$WSPREV_HOME)"
python3 - <<'EOF'
import sys
v = sys.version_info
print("python", sys.version.split()[0])
if not ((3, 10) <= v[:2] <= (3, 13)):
    print("WARNING: vLLM wheels need Python 3.10-3.13; install one (e.g. a conda env) and re-run")
EOF
free_gb=$(df -Pk "$WSPREV_HOME" | awk 'NR==2 {print int($4 / 1024 / 1024)}')
echo "free disk under $WSPREV_HOME: ${free_gb} GB"
if [ "$free_gb" -lt 60 ]; then echo "WARNING: less than 60 GB free; venv ~15 GB, models ~4 GB, generations ~10 GB"; fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
else
  echo "no nvidia-smi (fine for setup; the GPU is needed from scripts/check_vllm.py on)"
fi
if [ ! -f "$WSPREV_HOME/env.local" ]; then
  if curl -s -m 10 -o /dev/null -f "https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B/revision/$REVISION"; then
    echo "# huggingface.co reachable: no mirror" > "$WSPREV_HOME/env.local"
  else
    echo "export HF_ENDPOINT=https://hf-mirror.com" > "$WSPREV_HOME/env.local"
  fi
fi
source "$WSPREV_HOME/env.local"
echo "HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co (default)}"

step "1/7 Python environment in $WSPREV_HOME/venv"
if [ ! -x "$WSPREV_HOME/venv/bin/python" ]; then
  if ! python3 -m venv "$WSPREV_HOME/venv"; then
    echo "python3 -m venv failed (Debian/Ubuntu without python3-venv); using virtualenv instead"
    rm -rf "$WSPREV_HOME/venv"
    python3 -m pip install --user virtualenv
    python3 -m virtualenv "$WSPREV_HOME/venv"
  fi
fi
# shellcheck disable=SC1091
source "$WSPREV_HOME/venv/bin/activate"
python -m pip install -U pip
python -m pip install -r requirements-gpu.txt
python -m pip install -e .
python - <<'EOF'
from importlib.metadata import version
import sys
print("python", sys.version.split()[0], "|", ", ".join(f"{p} {version(p)}" for p in ("vllm", "transformers", "torch", "math-verify")))
EOF

step "2/7 NLTK punkt_tab"
if [ ! -d "$NLTK_DATA/tokenizers/punkt_tab" ]; then
  turbo python -m nltk.downloader -d "$NLTK_DATA" punkt_tab
fi

step "3/7 the authors' log repository at $LOGS_COMMIT"
if [ ! -d "$WSPREV_LOGS/.git" ]; then
  turbo env GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RikiyaT/lm_understand-logs.git "$WSPREV_LOGS"
fi
git -C "$WSPREV_LOGS" -c advice.detachedHead=false checkout -q "$LOGS_COMMIT"
echo "logs at $(git -C "$WSPREV_LOGS" rev-parse HEAD)"

step "4/7 Qwen/Qwen3.5-0.8B @ $REVISION -> $BASE"
python - "$BASE" "$REVISION" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-0.8B", revision=sys.argv[2], local_dir=sys.argv[1])
EOF
python scripts/verify_model.py base "$BASE"

step "5/7 p50c -> $P50C"
python scripts/build_p50c.py "$BASE" "$WSPREV_LOGS/exp8_math/runs/p50c/edit" "$P50C"
python scripts/verify_model.py p50c "$P50C" --edit "$WSPREV_LOGS/exp8_math/runs/p50c/edit"

step "6/7 benchmarks -> $WSPREV_BENCH"
python scripts/prepare_data.py --out "$WSPREV_BENCH"

step "7/7 unit tests"
python -m pytest -q

echo
echo "Setup finished. Next, with the GPU on and inside tmux:"
echo "  python scripts/check_vllm.py        # about a minute: the vLLM behaviour the pipeline relies on"
echo "  python scripts/run_stage.py smoke   # M1 smoke test"
