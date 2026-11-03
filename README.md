# wsp-revisited

Re-examining Experiment 8 of "Sparks of Weight-Space Programming" (Takehi et al., 2026): how much of the
MATH-500 gain from an LLM-written weight edit of Qwen3.5-0.8B comes from fixing termination (thinking loops)
rather than reasoning, and whether an LLM-free procedure finds an equivalent edit.

Work in progress.

- Plan for coding agents: [docs/plan-for-agents.md](docs/plan-for-agents.md)
- Step-by-step guide for the author: [docs/guide-for-me.md](docs/guide-for-me.md)
- Pre-registration of RQ1 (draft): [prereg/criteria.md](prereg/criteria.md)

## Layout

- `src/wsprev/`: decoding configurations (`decoding.py`, the only place D0-D4 are defined), generation with vLLM
  and the loop breaker (`generate.py`), grading (`grade.py`), repetition measures (`loopdetect.py`), rank-one edits
  (`edits.py`), data loading (`data.py`).
- `scripts/`: GPU-machine entry points (`setup_autodl.sh`, `smoke.sh`) and helpers (`build_p50c.py`,
  `verify_model.py`, `prepare_data.py`, `make_splits.py`, `summarize.py`).
- `third_party/`: the original authors' grading and edit-rebuild scripts (unmodified, MIT) and the official IFEval
  checkers (Apache-2.0, two small documented changes).
- `results/`: small summaries only (`splits.json`: the practice-problem splits).
- `tests/`: CPU tests (`python -m pytest -q`), including the D3 loop breaker against a scripted fake engine.

## On the GPU machine (AutoDL)

The repository is private until release, so the GPU machine needs its own access, for example a deploy key of this
repository (Settings → Deploy keys) or a fine-grained token limited to it.

```bash
git clone https://github.com/cyj-hashtag/wsp-revisited.git /root/autodl-tmp/wsp-revisited
cd /root/autodl-tmp/wsp-revisited
bash scripts/setup_autodl.sh          # no-GPU mode is enough
bash scripts/smoke.sh smoke           # GPU mode, inside tmux
```
