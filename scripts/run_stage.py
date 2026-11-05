"""Run one stage of the experiment on the GPU machine; every job is split over all visible GPUs (one shard each).

    python scripts/run_stage.py <stage> [--gpus N] [--dry-run]

  smoke   D0, MATH-500 problems 1-100, 1 sample, base and p50c                      ~5M tokens
  calib   D0, all 500 MATH-500 problems (adds to smoke's folders); calibration gate   ~27M tokens in all
  tau     D3@0.3/0.5/0.7, base, the first 256 practice DEV problems, 1 sample      needs: the gate passed
  select  D1, D2b, D3, base, the 128 practice SELECT problems, 1 sample            needs: decoding.D3_TAU set
  rq1     D1, D2a, D2b, D3 x base, p50c x MATH-500 (2 samples), GSM8K (1), IFEval (2)
                                                                                    needs: prereg registered, choice made
RQ2 (python -m wsprev.rq2; the plan and rules: results/rq2/plan.json, prereg/rq2_rules.md):
  rq2-dev      base, D0, practice DEV[:dev_problems], 1 sample                     needs: plan, rules file
  rq2-read     teacher-forced activations (HF), then `rq2 build` (directions, grid)
  rq2-screen   swap check, then every edit on DEV[448:512] (D0, 8,192 tokens), then `rq2 choose screen`
  rq2-holdout  the top edits on HOLDOUT (D0), then `rq2 choose holdout` and `rq2 export` (model folders)
  rq2-final    MATH-500: rq2-auto (D0; D3 in the full plan), the controls (D0)
Raw output: $WSPREV_RUNS/<group>/<job>/s<i>of<N>/ with the log s<i>of<N>.log next to it.
Summaries: results/<stage>/<job>/ (summary.json, per_problem.csv: small, commit them).
Re-running a stage continues where it stopped. Keep the same number of GPUs for a job once it has started.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from wsprev import data, decoding, rq2  # noqa: E402
from wsprev.rq1 import RQ1_CONFIGS, SELECT_CONFIGS, TAU_PROBLEMS  # noqa: E402
from wsprev.summary import summarize  # noqa: E402

STAGES = ("smoke", "calib", "tau", "select", "rq1", "rq2-dev", "rq2-read", "rq2-screen", "rq2-holdout", "rq2-final")
RQ2_STEPS = {   # stage: (sharded wsprev.rq2 step, steps before it on the first GPU, CPU steps after it)
    "rq2-read": ("read", [], [["build"]]),
    "rq2-screen": ("screen", [["swapcheck", "--reference"], ["swapcheck"]], [["choose", "screen"]]),
    "rq2-holdout": ("holdout", [], [["choose", "holdout"], ["export"]]),
}


@dataclass(frozen=True)
class Job:
    model: str
    config: str
    bench: str
    samples: int
    limit: int | None
    group: str

    @property
    def name(self):
        return f"{self.model}_{self.config}_{self.bench.replace(':', '-')}"


def jobs(stage: str) -> list:
    if stage in ("smoke", "calib"):
        return [Job(m, "D0", "math500", 1, 100 if stage == "smoke" else 500, "d0") for m in ("base", "p50c")]
    if stage == "tau":
        return [Job("base", f"D3@{t:g}", "practice:dev", 1, TAU_PROBLEMS, "tau") for t in decoding.TAU_CANDIDATES]
    if stage == "select":
        return [Job("base", c, "practice:select", 1, None, "select") for c in SELECT_CONFIGS]
    if stage == "rq1":   # MATH-500 (primary) first, so that it is complete if time runs out
        return [Job(m, c, b, data.RQ1_SAMPLES[b], None, "rq1")
                for b in ("math500", "gsm8k", "ifeval") for c in RQ1_CONFIGS for m in ("base", "p50c")]
    if stage == "rq2-dev":
        return [Job("base", "D0", "practice:dev", 1, rq2.plan()["dev_problems"], "rq2")]
    if stage == "rq2-final":
        p = rq2.plan()
        return ([Job("rq2-auto", c, "math500", n, None, "rq2/final") for c, n in p["final"]]
                + [Job(f"rq2-ctrl-{x}", "D0", "math500", 1, p["control_problems"], "rq2/final") for x in p["controls"]])
    if stage in RQ2_STEPS:
        return []
    raise KeyError(stage)


def gate_passed(results: Path, model: str) -> bool:
    s = results / "calib" / f"{model}_D0_math500" / "summary.json"
    if not s.exists():
        return False
    g = json.loads(s.read_text(encoding="utf-8"))["groups"].get(f"{model}/D0/math500", {})
    return bool(g.get("gate", {}).get("pass")) and g.get("problems") == 500


def registered(repo: Path) -> bool:
    head = (repo / "prereg" / "criteria.md").read_text(encoding="utf-8")[:2000]
    return "**Status: REGISTERED" in head


def missing(stage: str, repo: Path = REPO) -> list:
    """What a stage still needs, as messages (empty: ready)."""
    tau = decoding.D3_TAU
    results, msgs = repo / "results", []
    if stage in ("tau", "select", "rq1"):
        msgs += [f"the D0 calibration gate has not passed for {m} (python scripts/run_stage.py calib)"
                 for m in ("base", "p50c") if not gate_passed(results, m)]
    if stage in ("select", "rq1") and tau is None:
        msgs.append("decoding.D3_TAU is not set (python scripts/choose.py tau --write, then commit)")
    if stage == "rq1":
        if not registered(repo):
            msgs.append("prereg/criteria.md is not registered")
        if not (results / "select" / "choice.json").exists():
            msgs.append("results/select/choice.json is missing (python scripts/choose.py config)")
    if stage.startswith("rq2-"):
        if not (repo / "prereg" / "rq2_rules.md").exists():
            msgs.append("prereg/rq2_rules.md is missing (write and commit the RQ2 rules before any RQ2 run)")
        if not (results / "rq2" / "plan.json").exists():
            msgs.append("results/rq2/plan.json is missing (python -m wsprev.rq2 plan lean|full, then commit)")
        need = {"rq2-screen": "build.json", "rq2-holdout": "screen.json", "rq2-final": "choice.json"}.get(stage)
        if need and not (results / "rq2" / need).exists():
            msgs.append(f"results/rq2/{need} is missing (the stage before {stage})")
    return msgs


def visible_gpus() -> list:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def commands(job: Job, gpus: list, runs: Path) -> list:
    folder, n = runs / job.group / job.name, len(gpus)
    out = []
    for i, g in enumerate(gpus):
        cmd = [sys.executable, "-m", "wsprev.generate", "--model", job.model, "--config", job.config,
               "--bench", job.bench, "--samples", str(job.samples), "--shard", f"{i}/{n}",
               "--out", str(folder / f"s{i}of{n}")]
        if job.limit:
            cmd += ["--limit", str(job.limit)]
        out.append((g, cmd, folder / f"s{i}of{n}.log"))
    return out


def tail(path: Path, n=1) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def run_job(stage: str, job: Job, gpus: list, runs: Path):
    folder = runs / job.group / job.name
    folder.mkdir(parents=True, exist_ok=True)
    layout = folder / "shards.json"
    if layout.exists() and json.loads(layout.read_text())["n"] != len(gpus):
        sys.exit(f"{folder} was started with {json.loads(layout.read_text())['n']} GPU(s); use --gpus with that number")
    layout.write_text(json.dumps({"n": len(gpus)}))
    procs = []
    for g, cmd, log in commands(job, gpus, runs):
        f = open(log, "a", encoding="utf-8")  # noqa: SIM115 - closed below
        procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=g)), f, log))
    last = time.time()
    while any(p.poll() is None for p, _, _ in procs):
        time.sleep(10)
        if time.time() - last >= 300:
            last = time.time()
            for _, _, log in procs:
                print(f"  {log.name}: {tail(log)}", flush=True)
    for _, f, _ in procs:
        f.close()
    failed = [(p.returncode, log) for p, _, log in procs if p.returncode]
    for code, log in failed:
        print(f"FAILED (exit {code}), last lines of {log}:\n{tail(log, 30)}")
    if failed:
        sys.exit(1)
    # the original evaluation used think35 = D0: compare with it for D0 jobs of its two models only
    original = {"base": "unedited", "p50c": "p50c"}.get(job.model)
    authors = Path(os.environ.get("WSPREV_LOGS", data.logs_dir())) / "exp8_math" / "eval" / str(original)
    is_d0 = original is not None and job.config == "D0" and job.bench == "math500"
    return summarize([folder], authors if is_d0 else None, job.model if is_d0 else None,
                     REPO / "results" / stage / job.name)


def rq2_commands(stage: str, gpus: list) -> tuple[list, list, list]:
    """(steps on the first GPU before, one sharded step per GPU, CPU steps after) of an RQ2 step stage."""
    step, before, after = RQ2_STEPS[stage]
    n = len(gpus)
    py = [sys.executable, "-m", "wsprev.rq2"]
    logs = Path(os.environ.get("WSPREV_RUNS", data.home() / "runs")) / "rq2" / "logs"
    pre = [(gpus[0], py + b, logs / f"{stage}_{'_'.join(b).replace('--', '')}.log") for b in before]
    par = [(g, py + [step, "--shard", f"{i}/{n}"], logs / f"{stage}_s{i}of{n}.log") for i, g in enumerate(gpus)]
    post = [(None, py + c, logs / f"{stage}_{'_'.join(c)}.log") for c in after]
    return pre, par, post


def run_parallel(cmds) -> list:
    """Run (gpu, command, log) triples at the same time; returns [(exit code, log)] of the failed ones."""
    procs = []
    for g, cmd, log in cmds:
        log.parent.mkdir(parents=True, exist_ok=True)
        f = open(log, "a", encoding="utf-8")  # noqa: SIM115 - closed below
        env = dict(os.environ) if g is None else dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO, env=env), f, log))
    last = time.time()
    while any(p.poll() is None for p, _, _ in procs):
        time.sleep(10)
        if time.time() - last >= 300:
            last = time.time()
            for _, _, log in procs:
                print(f"  {log.name}: {tail(log)}", flush=True)
    for _, f, _ in procs:
        f.close()
    failed = [(p.returncode, log) for p, _, log in procs if p.returncode]
    for code, log in failed:
        print(f"FAILED (exit {code}), last lines of {log}:\n{tail(log, 30)}")
    return failed


def run_rq2_steps(stage: str, gpus: list, dry_run: bool):
    """The stage's steps in order: each step before on the first GPU, the sharded step on every GPU, the CPU steps."""
    pre, par, post = rq2_commands(stage, gpus)
    for cmds in [[c] for c in pre] + [par] + [[c] for c in post]:
        for g, cmd, log in cmds:
            env = "" if g is None else f"CUDA_VISIBLE_DEVICES={g} "
            print(f"{env}{' '.join(cmd)} >> {log}", flush=True)
        if not dry_run and run_parallel(cmds):
            sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--gpus", type=int, help="use the first N visible GPUs (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args(argv)
    gpus = visible_gpus()
    if a.gpus:
        gpus = gpus[:a.gpus]
    if a.dry_run and not gpus:
        gpus = ["0"]
    if not gpus:
        sys.exit("no GPU visible (nvidia-smi)")
    todo = missing(a.stage)
    if todo:
        print("Not ready:\n  - " + "\n  - ".join(todo))
        if not a.dry_run:
            sys.exit(2)
    runs = Path(os.environ.get("WSPREV_RUNS", data.home() / "runs"))
    print(f"stage {a.stage}: {len(jobs(a.stage))} jobs on GPU(s) {','.join(gpus)}; raw output in {runs}", flush=True)
    if a.stage in RQ2_STEPS:
        run_rq2_steps(a.stage, gpus, a.dry_run)
        if not a.dry_run:
            print(f"\nstage {a.stage} finished. Copy results/rq2/ back and tell Claude.")
        return
    gates = []
    todo_jobs = jobs(a.stage)
    if a.stage == "rq2-final":   # control b does not exist when the chosen edit changes every column
        models = Path(os.environ.get("WSPREV_MODELS", data.home() / "models"))
        skip = [j for j in todo_jobs if not (models / j.model).exists()]
        for j in skip:
            print(f"skipping {j.name}: no model folder {models / j.model} (see results/rq2/models.json)")
        if not a.dry_run:
            todo_jobs = [j for j in todo_jobs if j not in skip]
    for job in todo_jobs:
        print(f"\n### {a.stage}: {job.name}", flush=True)
        if a.dry_run:
            for g, cmd, log in commands(job, gpus, runs):
                print(f"CUDA_VISIBLE_DEVICES={g} {' '.join(cmd)} >> {log}")
            continue
        s = run_job(a.stage, job, gpus, runs)
        gates += [g["gate"]["pass"] for g in s["groups"].values() if "gate" in g]
    if a.dry_run:
        return
    if gates and not all(gates):
        print("\nGATE FAILED: stop here (no other stage) and send the output above to Claude.")
        sys.exit(3)
    print(f"\nstage {a.stage} finished. Commit results/{a.stage}/ (small files) and tell Claude.")


if __name__ == "__main__":
    main()
