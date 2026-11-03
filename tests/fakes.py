"""A scripted stand-in for vLLM and the tokenizer, to test generate.Runner on the CPU."""
from wsprev.decoding import CLOSE_THINK
from wsprev.generate import Delta

THINK_END, EOS, NL, NL2, BOX = 0, 1, 5, 6, 7
PIECES = {THINK_END: "</think>", EOS: "", NL: "\n", NL2: "\n\n", BOX: "\\boxed{7}"}


def piece(t):
    return PIECES.get(t, f"<{t}>")


class FakeTok:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        return "<u>" + messages[0]["content"] + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")

    def encode(self, text, add_special_tokens=True):  # noqa: ARG002
        if text == CLOSE_THINK:
            return [NL, THINK_END, NL2]
        return [100 + ord(c) % 100 for c in text]

    def convert_tokens_to_ids(self, token):
        return THINK_END if token == "</think>" else None

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "".join(piece(t) for t in ids)


class FakeEngine:
    """script(problem_id, seed, is_answer) -> iterator of token ids; EOS ends a request ("stop"), max_tokens ends it
    ("length"). Each step gives every running request `per_step` tokens. With `late`, an aborted request still
    reports one more delta and then a final "abort" delta, as a real engine may."""

    def __init__(self, script, per_step=1, late=False):
        self.script, self.per_step, self.late = script, per_step, late
        self.reqs = {}
        self.added = []
        self.aborted = []
        self.zombies = {}

    def add(self, rid, prompt_ids, sampling):
        self.added.append((rid, list(prompt_ids), dict(sampling)))
        it = self.script(rid.split("#")[0], sampling["seed"], rid.endswith("/answer"))
        self.reqs[rid] = {"it": it, "n": 0, "max": sampling["max_tokens"]}

    def step(self):
        out = []
        for rid, stage in list(self.zombies.items()):
            if stage == 0:
                out.append(Delta(rid, [4242], "<late>", False, None))
                self.zombies[rid] = 1
            else:
                out.append(Delta(rid, [], "", True, "abort"))
                del self.zombies[rid]
        for rid in list(self.reqs):
            r = self.reqs[rid]
            toks, reason = [], None
            for _ in range(self.per_step):
                t = next(r["it"])
                toks.append(t)
                r["n"] += 1
                if t == EOS:
                    reason = "stop"
                    break
                if r["n"] >= r["max"]:
                    reason = "length"
                    break
            if reason:
                del self.reqs[rid]
            out.append(Delta(rid, toks, "".join(piece(t) for t in toks), reason is not None, reason))
        return out

    def abort(self, rid):
        self.aborted.append(rid)
        if self.reqs.pop(rid, None) is not None and self.late:
            self.zombies[rid] = 0

    def has_unfinished(self):
        return bool(self.reqs) or bool(self.zombies)
