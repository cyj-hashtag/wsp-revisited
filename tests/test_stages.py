"""scripts/run_stage.py (job lists, prerequisites, commands) and scripts/choose.py (tau written into decoding.py)."""
import json

import pytest
from test_scripts import script

from wsprev import decoding


def test_jobs():
    rs = script("run_stage")
    smoke, calib = rs.jobs("smoke"), rs.jobs("calib")
    assert [j.name for j in smoke] == [j.name for j in calib] == ["base_D0_math500", "p50c_D0_math500"]
    assert {j.limit for j in smoke} == {100} and {j.limit for j in calib} == {500}
    assert [j.config for j in rs.jobs("tau")] == ["D3@0.3", "D3@0.5", "D3@0.7"]
    assert {(j.limit, j.bench) for j in rs.jobs("tau")} == {(256, "practice:dev")}
    rq1 = rs.jobs("rq1")
    assert len(rq1) == 3 * 4 * 2 and rq1[0].bench == "math500" and rq1[-1].bench == "ifeval"
    assert {j.samples for j in rq1 if j.bench == "math500"} == {2}


def test_commands_shard_over_gpus(tmp_path):
    rs = script("run_stage")
    cmds = rs.commands(rs.jobs("smoke")[0], ["0", "1"], tmp_path)
    assert [g for g, _, _ in cmds] == ["0", "1"]
    assert cmds[1][1][cmds[1][1].index("--shard") + 1] == "1/2"
    assert cmds[1][1][cmds[1][1].index("--out") + 1].endswith("s1of2")


def test_missing(tmp_path, monkeypatch):
    rs = script("run_stage")
    (tmp_path / "prereg").mkdir()
    (tmp_path / "prereg" / "criteria.md").write_text("**Status: DRAFT, not registered.**")
    monkeypatch.setattr(decoding, "D3_TAU", None)
    assert rs.missing("smoke", tmp_path) == []
    assert len(rs.missing("tau", tmp_path)) == 2                       # no gate for base and p50c
    for m in ("base", "p50c"):
        d = tmp_path / "results" / "calib" / f"{m}_D0_math500"
        d.mkdir(parents=True)
        d.joinpath("summary.json").write_text(json.dumps(
            {"groups": {f"{m}/D0/math500": {"problems": 500, "gate": {"pass": True}}}}))
    assert rs.missing("tau", tmp_path) == []
    assert any("D3_TAU" in x for x in rs.missing("select", tmp_path))
    monkeypatch.setattr(decoding, "D3_TAU", 0.5)
    assert rs.missing("select", tmp_path) == []
    assert len(rs.missing("rq1", tmp_path)) == 2                       # not registered, no choice
    (tmp_path / "prereg" / "criteria.md").write_text("**Status: REGISTERED on 2026-11-03 by x.**")
    (tmp_path / "results" / "select").mkdir()
    (tmp_path / "results" / "select" / "choice.json").write_text("{}")
    assert rs.missing("rq1", tmp_path) == []


def test_set_tau(tmp_path):
    ch = script("choose")
    f = tmp_path / "decoding.py"
    f.write_text(ch.DECODING.read_text(encoding="utf-8"), encoding="utf-8")
    ch.set_tau(0.5, f)
    assert "D3_TAU: float | None = 0.5\n" in f.read_text(encoding="utf-8")
    other = tmp_path / "other.py"
    other.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        ch.set_tau(0.5, other)
