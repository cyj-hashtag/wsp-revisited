"""Access to the vendored third-party code in third_party/ (see the README.md files there)."""
from __future__ import annotations

import functools
import importlib.util
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WSP_LOGS = REPO / "third_party" / "wsp_logs"


def _mcqa_stub(text):  # noqa: ARG001
    raise NotImplementedError("mathqa.loop_signals needs the authors' private mcqa module; wsprev does not use it")


@functools.lru_cache(maxsize=None)
def mathqa():
    """The authors' mathqa.py (answer extraction + math-verify checker), unmodified.

    It imports `loop_signals` from the authors' private package, falling back to a top-level module `mcqa`; we register
    a stub under that name. Only mathqa.loop_signals / diagnostics reach it, and we call neither."""
    if "mcqa" not in sys.modules:
        stub = types.ModuleType("mcqa")
        stub.loop_signals = _mcqa_stub
        sys.modules["mcqa"] = stub
    spec = importlib.util.spec_from_file_location("wsp_logs_mathqa", WSP_LOGS / "mathqa.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=None)
def ifeval_registry():
    """The official IFEval instruction registry (third_party/ifeval), with langdetect seeded for reproducibility."""
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    import langdetect  # noqa: PLC0415

    langdetect.DetectorFactory.seed = 0
    from third_party.ifeval import instructions_registry  # noqa: PLC0415

    return instructions_registry
