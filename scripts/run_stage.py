"""Run one stage of the experiment on the GPU machine; every job is split over all visible GPUs (one shard each).

    python scripts/run_stage.py <stage> [--gpus N] [--dry-run]

  smoke   D0, MATH-500 problems 1-100, 1 sample, base and p50c                      ~5M tokens
  calib   D0, all 500 MATH-500 problems (adds to smoke's folders); calibration gate   ~27M tokens in all
  tau     D3@0.3/0.5/0.7, base, the first 256 practice DEV problems, 1 sample      needs: the gate passed
  select  D1, D2b, D3, base, the 128 practice SELECT problems, 1 sample            needs: decoding.D3_TAU set
  rq1     D1, D2a, D2b, D3 x base, p50c x MATH-500 (2 samples), GSM8K (1), IFEval (2)
                                                                                    needs: prereg registered, choice made
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
from wsprev import data, decoding  # noqa: E402
from wsprev.rq1 import RQ1_CONFIGS, SELECT_CONFIGS, TAU_PROBLEMS  # noqa: E402
from wsprev.summary import summarize  # noqa: E402

STAGES = ("smoke", "calib", "tau", "select", "rq1")


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
    # the original evaluation used think35 = D0: compare with it for D0 jobs only
    original = {"base": "unedited", "p50c": "p50c"}[job.model]
    authors = Path(os.environ.get("WSPREV_LOGS", data.logs_dir())) / "exp8_math" / "eval" / original
    is_d0 = job.config == "D0" and job.bench == "math500"
    return summarize([folder], authors if is_d0 else None, job.model if is_d0 else None,
                     REPO / "results" / stage / job.name)


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
    gates = []
    for job in jobs(a.stage):
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
