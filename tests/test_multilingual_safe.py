from __future__ import annotations

import numpy as np
import torch
from typer.testing import CliRunner

from cli import app
from graft import _DEFAULT_WHITEN_CJK, _DEFAULT_WHITEN_GEN, ConceptGrafter
from models import GraftConfig, ReclaimedSlotInfo


def test_multilingual_safe_defaults_and_schema() -> None:
    cfg = GraftConfig(
        kind_words=["taste", "smell"],
        kind_label="flavor",
        attr_adj="Icelandic",
        attr_noun="Iceland",
    )
    assert cfg.multilingual_safe is True
    assert cfg.carryover_safe is True
    slot = ReclaimedSlotInfo(
        layer=2,
        neuron_idx=42,
        role="read_L2",
        down_norm=1.25,
        cjk_share=0.1,
        eng_pert=0.05,
        cjk_pert=0.04,
        top_tokens=["foo"],
    )
    assert slot.cjk_pert == 0.04
    assert len(_DEFAULT_WHITEN_CJK) == 6
    assert len(_DEFAULT_WHITEN_GEN) == 8


def test_rank_multilingual_safe_avoids_high_cjk_neurons() -> None:
    # Neuron 0: silent in English, VERY active in CJK (legacy CJK-tier would reclaim this!)
    # Neuron 1: quiet in both English and CJK, small norm
    # Neuron 2: quiet in both English and CJK, larger norm (tie-breaker puts #1 ahead of #2)
    # Neuron 3: active in English, silent in CJK
    eng_pert = np.array([0.001, 0.020, 0.020, 2.500], dtype=np.float32)
    cjk_pert = np.array([4.000, 0.015, 0.020, 0.001], dtype=np.float32)
    norms = np.array([0.5, 0.8, 1.4, 0.9], dtype=np.float32)

    order = ConceptGrafter._rank_multilingual_safe_indices(eng_pert, cjk_pert, norms, top_n=4)
    assert order == [1, 2, 3, 0], f"Expected truly bilingual-quiet slots first, got {order}"


def test_build_whiten_gen_seqs_toggles_cjk() -> None:
    dummy = object.__new__(ConceptGrafter)
    dummy.enc = lambda text: [len(text)]  # type: ignore[method-assign]

    seqs_ml = dummy._build_whiten_gen_seqs(multilingual_safe=True)
    seqs_eng = dummy._build_whiten_gen_seqs(multilingual_safe=False)
    assert len(seqs_eng) == len(_DEFAULT_WHITEN_GEN)
    assert len(seqs_ml) == len(_DEFAULT_WHITEN_GEN) + len(_DEFAULT_WHITEN_CJK)


def test_precompute_quiet_slots_cache_separates_multilingual_flag() -> None:
    # Synthetic tiny ConceptGrafter stub verifying precompute_quiet_slots + find_quiet_slot end-to-end on CPU
    d_model, d_mlp, vocab = 8, 20, 40

    class _TinyMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(d_model, d_mlp, bias=False)
            self.up_proj = torch.nn.Linear(d_model, d_mlp, bias=False)
            self.down_proj = torch.nn.Linear(d_mlp, d_model, bias=False)

    class _TinyLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _TinyMLP()

    class _TinyNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(d_model))

    g = object.__new__(ConceptGrafter)
    g.device = torch.device("cpu")
    g.layers = [_TinyLayer(), _TinyLayer()]
    g.num_layers = 2
    g.vocab_size = vocab
    g.orig_active_vocab = vocab
    g.emb_w = torch.randn(vocab, d_model)
    g.out_w = g.emb_w
    g.final_norm = _TinyNorm()
    g.cjk_mask = torch.zeros(vocab, dtype=torch.float32)
    g.cjk_mask[:20] = 1.0  # CJK-rich vocab so legacy mode enters CJK tier
    g.latin_mask = 1.0 - g.cjk_mask
    g.decoded = [f"t{i}" for i in range(vocab)]
    g.used_slots = set()
    g._quiet_slot_cache = {}
    g.enc = lambda s: [1, 2] if any(ord(c) > 127 for c in s) else [3, 4]  # type: ignore[method-assign]

    # Inject deterministic energy map: neuron 0 is English-quiet / CJK-loud; neuron 7 is quiet in both
    def _fake_energy(seqs: list[list[int]], target_layers: list[int]) -> dict[int, np.ndarray]:
        is_cjk = seqs[0] == [1, 2]
        out: dict[int, np.ndarray] = {}
        for li in target_layers:
            arr = np.full(d_mlp, 1.0, dtype=np.float32)
            if is_cjk:
                arr[0] = 9.0
                arr[7] = 0.01
            else:
                arr[0] = 0.001
                arr[7] = 0.01
            out[li] = arr
        return out

    g._measure_mlp_max_perturbation = _fake_energy  # type: ignore[method-assign]

    # Multilingual-safe must pick neuron 7 (NOT neuron 0!)
    sl_ml = g.find_quiet_slot(0, mode="input", multilingual_safe=True)
    assert sl_ml.neuron_idx == 7
    assert sl_ml.eng_pert == 0.01
    assert sl_ml.cjk_pert == 0.01

    # Legacy mode (multilingual_safe=False) should cache separately and be allowed to pick neuron 0
    g.used_slots.clear()
    g.precompute_quiet_slots([0], 1, multilingual_safe=False)
    assert (0, "input", True) in g._quiet_slot_cache
    assert (0, "input", False) in g._quiet_slot_cache
    assert g._quiet_slot_cache[(0, "input", True)][0][0] == 7


def test_cli_graft_exposes_multilingual_safe_flag() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["graft", "--help"], env={"COLUMNS": "160"})
    assert res.exit_code == 0
    assert "--multilingual-safe" in res.stdout
    assert "--no-multilingual-safe" in res.stdout
