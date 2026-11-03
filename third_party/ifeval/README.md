# third_party/ifeval

The official IFEval instruction checkers (Zhou et al. 2023), copied from google-research:

- repository: <https://github.com/google-research/google-research/tree/master/instruction_following_eval>
- commit: `e6890f85757dd84e27ca6df2dd30651dafad28e0` (latest commit touching that folder when copied, 2026-07-22)
- license: Apache License 2.0 (`LICENSE`, copied from the repository root at the same commit)

Files: `instructions.py`, `instructions_registry.py`, `instructions_util.py`. We do not copy `evaluation_lib.py`;
`src/wsprev/grade.py` re-implements its `test_instruction_following_strict` (strict, prompt-level) in a few lines.

**Modifications** (each modified file carries a notice at its top):

1. `instructions.py`, `instructions_registry.py`: `from instruction_following_eval import X` became `from . import X`
   so that the folder works as a package (`third_party.ifeval`).
2. `instructions_util.py`: the sentence tokenizer is `nltk.tokenize.PunktTokenizer("english")` (NLTK's `punkt_tab`
   data) instead of `nltk.data.load("nltk:tokenizers/punkt/english.pickle")`; NLTK >= 3.9 refuses to load pickles.
   This only affects instructions that count sentences. Requires `python -m nltk.downloader punkt_tab` once.

sha256 after modification:

| File | sha256 |
|---|---|
| `instructions.py` | `4a1684ee16cfe4207dbccea27601ddd29a14cf0a1a38a70ddc17861c2ba44bfd` |
| `instructions_registry.py` | `a59d1caa7a4deeae59bb76b3d4c8a5756007c272ac92d84bce3f971a8ff360df` |
| `instructions_util.py` | `09234cf449452c9630c7d6318ca7bb47488f33d733cb4a34fab2effa7b6b370b` |

Grading convention (matches the original Experiment 8 evaluation, `exp8_math/code/evaluate.py` `grade_other`): the
response is the text after the last `</think>`; with thinking on and no `</think>` the response is empty and every
instruction counts as not followed; `kwargs` entries that are `None` (the Hugging Face copy of IFEval stores every key)
are dropped before `build_description`.
