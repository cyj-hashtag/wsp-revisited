# third_party/wsp_logs

Three scripts copied **unmodified** from the authors' log repository of "Sparks of Weight-Space Programming"
(source [B] in `docs/plan-for-agents.md`):

- repository: <https://github.com/RikiyaT/lm_understand-logs>
- commit: `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f` (2026-10-29)
- license: MIT for these scripts; the full upstream `LICENSE` (which also covers the logs under CC BY 4.0) is copied
  here verbatim.

| File | Upstream path | Git blob (identical upstream and here) |
|---|---|---|
| `mathqa.py` | `exp8_math/code/mathqa.py` | `808a04776906b1cf4630d397cef54f03d8795477` |
| `apply_edit.py` | `exp8_math/code/apply_edit.py` | `cc1d7c249ece802bd2e7b5907b082caeec716923` |
| `st_numpy.py` | `exp8_math/code/st_numpy.py` | `f7449556702f8fa8c5fbe83ed1f70b991e13f14a` |
| `LICENSE` | `LICENSE` | `5137aa7a114872294fa20195ef84a89dfd0e1b67` |

Check: `git hash-object <file>` here equals `git rev-parse 4d09ba6:<upstream path>` in a clone of the upstream repo.

How we use them:

- `mathqa.py`: answer extraction (last `\boxed{}`) and the math-verify checker, through `src/wsprev/grade.py`.
  `mathqa.py` imports `mcqa.loop_signals` from the authors' private package; `grade.py` registers a stub module named
  `mcqa` before importing it. The stub is only reached by `mathqa.loop_signals` / `mathqa.diagnostics`, which we do not
  call (it raises if called).
- `apply_edit.py` + `st_numpy.py`: `scripts/build_p50c.py` (run by `scripts/setup.sh`) rebuilds the p50c checkpoint
  bit for bit from the base model and `exp8_math/runs/p50c/edit/` with `apply_edit.rebuild` and the same sha256
  checks of every base and rebuilt tensor as `apply_edit.main`, but patches the file in place to need little memory.
