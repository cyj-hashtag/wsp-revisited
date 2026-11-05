"""RQ2 directions (src/wsprev/directions.py): windows, separability, neuron scores, the grid, recovering p50c's edit,
and the activation reader on a tiny random Qwen3.5 model."""
import numpy as np
import pytest

from wsprev import directions as dr
from wsprev.edits import masked_rank1_edit, random_mask, random_unit


def rec(pid, cut=True, closed=False, finish="length", trig=False):
    return {"problem_id": pid, "sample_idx": 0, "cut": cut, "think_closed": closed, "finish_reason": finish,
            "loop_triggered": trig}


def test_windows_pick_loop_and_normal_from_the_same_response():
    W = 16
    normal = [f"n{i}" for i in range(W)]                       # all distinct: dup8 = 0
    loop = ["a", "b"] * (W // 2)                               # period 2: dup8 = 1 - 2 / 9
    texts = {("p0", 0): " ".join(normal + [f"m{i}" for i in range(20)] + loop),
             ("p1", 0): " ".join(normal + [f"m{i}" for i in range(20)] + [f"z{i}" for i in range(W)]),  # no loop
             ("p2", 0): " ".join(loop * 3),                     # loops from the start: normal window loops too
             ("p3", 0): " ".join([f"t{i}" for i in range(3 * W)]) + "</think> answer",
             ("p4", 0): " ".join(normal)}                       # too short for two windows
    recs = [rec("p0"), rec("p1"), rec("p2"), rec("p3", cut=False, closed=True, finish="stop"), rec("p4")]
    ws, why = dr.windows(recs, texts, encode=str.split, prompt_ids=lambda pid: ["<p>", pid], window=W,
                         loop_dup=0.7, normal_dup=0.3)
    kinds = [(w.problem_id, w.kind) for w in ws]
    assert kinds == [("p0", "loop"), ("p0", "normal"), ("p3", "finished")]
    lw, nw, fw = ws
    assert lw.ids[lw.start:] == loop and lw.ids[:2] == ["<p>", "p0"] and len(lw.ids) == 2 + W + 20 + W
    assert nw.ids[nw.start:] == normal and nw.pair == lw.pair == 0 and fw.pair is None
    assert fw.ids[fw.start:] == [f"t{i}" for i in range(W, 2 * W)]          # the middle of 3W thinking tokens
    assert why["loop dup8 < min"] == 1 and why["normal dup8 > max"] == 1 and why["thinking too short"] == 1
    capped, _ = dr.windows(recs, texts, str.split, lambda pid: [], window=W, loop_dup=0.7, max_pairs=0)
    assert [w.kind for w in capped] == ["finished"]


def planted(n=40, L=4, d=12, layer=2, gap=3.0, seed=0):
    """Window means where only `layer` separates loop from normal, along a known unit vector."""
    rng = np.random.default_rng(seed)
    v = random_unit(d, rng)
    norm = rng.standard_normal((n, L, d))
    loop = rng.standard_normal((n, L, d))
    loop[:, layer] += gap * v
    return loop, norm, v


def test_directions_find_the_planted_layer():
    loop, norm, v = planted()
    d = dr.directions(loop, norm)
    assert abs(d["u"][2] @ v) > 0.9
    assert d["auc"][2] > 0.95 and all(d["auc"][l] < 0.8 for l in (0, 1, 3))
    assert dr.top_layers((8, 9, 10, 11), d["auc"], d["separation"], n=1) == (10,)
    with pytest.raises(ValueError):
        dr.directions(loop[:3], norm[:3])


def test_top_layers_tie_breaks():
    layers = (8, 9, 10, 11)
    assert dr.top_layers(layers, [1, 1, 1, 0.5], [2.0, 3.0, 2.0, 9.0]) == (8, 9)    # separation, then lower layer
    assert dr.top_layers(layers, [0.9, 1, 1, 1], [0, 1, 1, 1]) == (9, 10)


def test_auc():
    assert dr.auc([2, 3], [0, 1]) == 1.0 and dr.auc([0, 1], [2, 3]) == 0.0
    assert dr.auc([1, 1], [1, 1]) == 0.5


def test_scores_and_proposition_3():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((16, 40))
    u = random_unit(16, rng)
    a_loop, a_norm = rng.random(40), rng.random(40)
    c = dr.scores(W, u, a_loop, a_norm)
    assert c == pytest.approx((u @ W) * (a_loop - a_norm))
    for k in (5, 40):
        m = dr.masks({0: c}, k)[0]
        chk = dr.projection_gap_change(W, u, m, 4.0, a_loop, a_norm)
        assert chk["change"] == pytest.approx(chk["predicted_change"])      # exact: the map is linear
    # top-k by c gives the largest reduction among k-column masks (greedy optimum of a linear objective)
    best = dr.projection_gap_change(W, u, dr.masks({0: c}, 5)[0], 1.0, a_loop, a_norm)["change"]
    for s in range(20):
        other = random_mask(40, 5, np.random.default_rng(100 + s))
        assert dr.projection_gap_change(W, u, other, 1.0, a_loop, a_norm)["change"] >= best - 1e-12


def test_grid():
    g = dr.grid((12, 20))
    assert len(g) == 2 * len(dr.KS) * len(dr.ALPHAS) == 40
    assert g[0].name == "astra-L15-19-k32-a1" and g[-1].name == "auc-L12-20-k3584-a8"
    assert len(dr.grid((19, 15))) == 20                                 # AUC layers are Astra's: one set
    assert len(dr.grid((12, 20), ks=(64, 3584), alphas=(1, 4, 6))) == 12


def test_edited_matches_the_edit_formula():
    rng = np.random.default_rng(2)
    W = {3: rng.standard_normal((8, 20)).astype(np.float32)}
    u = {3: random_unit(8, rng)}
    m = {3: random_mask(20, 4, rng)}
    out = dr.edited(W, u, m, 6.0)
    assert out[3].dtype == np.float32
    np.testing.assert_allclose(out[3], masked_rank1_edit(W[3].astype(np.float64), u[3], m[3], 6.0), rtol=1e-6)


def test_recover_rank1():
    rng = np.random.default_rng(3)
    W0 = rng.standard_normal((32, 100))
    u = random_unit(32, rng)
    m = random_mask(100, 10, rng)
    W1 = masked_rank1_edit(W0, u, m, 6.0)
    ur, mr = dr.recover_rank1(W0, W1, 6.0)
    assert (mr == m).all() and abs(ur @ u) == pytest.approx(1.0)
    np.testing.assert_allclose(masked_rank1_edit(W0, ur, mr, 6.0), W1, atol=1e-10)


def test_find_down_proj():
    params = [("visual.blocks.0.mlp.down_proj.weight", 0), ("language_model.model.layers.5.mlp.down_proj.weight", 1),
              ("language_model.model.layers.15.mlp.down_proj.weight", 2)]
    assert dr.find_down_proj(params, 5) == params[1]
    assert dr.find_down_proj(params, 15) == params[2]
    with pytest.raises(RuntimeError):
        dr.find_down_proj(params, 0)                                     # only the vision tower's


def tiny_qwen35(layers=24):
    """A random Qwen3.5 model with the 0.8B layer layout and tiny widths."""
    pytest.importorskip("transformers")
    import torch
    from transformers import Qwen3_5Config, Qwen3_5Model

    types = ["linear_attention"] * 3 + ["full_attention"]
    text = dict(hidden_size=32, intermediate_size=48, num_hidden_layers=layers, layer_types=(types * 6)[:layers],
                head_dim=16, num_attention_heads=2, num_key_value_heads=1, linear_key_head_dim=8,
                linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=2, vocab_size=300,
                full_attention_interval=4, eos_token_id=1, rope_parameters={
                    "mrope_interleaved": True, "mrope_section": [1, 1, 0], "rope_type": "default",
                    "rope_theta": 10000000, "partial_rotary_factor": 0.25})
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32)
    cfg = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=299, video_token_id=298,
                        vision_start_token_id=297, vision_end_token_id=296)
    torch.manual_seed(0)
    return Qwen3_5Model(cfg).eval()


def test_reader_on_a_tiny_qwen35():
    import torch

    model = tiny_qwen35()
    tm = dr.text_model(model)
    seen = {}
    h = tm.layers[15].mlp.down_proj.register_forward_hook(
        lambda mod, inp, out: seen.update(out=out[0, 30:].float().mean(0)))
    reader = dr.Reader(model)
    ids = list(range(2, 60))
    res, act = reader.read(ids, 30)
    reader.close()
    h.remove()
    assert res.shape == (len(dr.LAYERS), 32) and act.shape == (len(dr.LAYERS), 48)
    with torch.no_grad():
        hs = tm(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False).hidden_states
    for l in dr.LAYERS:   # hidden_states[l + 1] is the output of decoder layer l
        np.testing.assert_allclose(res[dr.LAYERS.index(l)], hs[l + 1][0, 30:].float().mean(0).numpy(), atol=1e-5)
    W = tm.layers[15].mlp.down_proj.weight.detach().float().numpy()
    np.testing.assert_allclose(W @ act[dr.LAYERS.index(15)], seen["out"].numpy(), atol=1e-4)   # linear: mean in = out
