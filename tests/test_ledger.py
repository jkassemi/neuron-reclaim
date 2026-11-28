from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graft import ConceptGrafter
from ledger import StreamingSCGLedger, compute_biorthogonal_write_directions
from models import GraftConfig, ReclaimedSlotInfo


class _TinySwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.up_proj = nn.Linear(d_model, d_mlp, bias=False)
        self.down_proj = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _TinyBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.mlp = _TinySwiGLUMLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class _TinyLM(nn.Module):
    """Minimal pure-PyTorch SwiGLU LM for bit-exact StreamingSCGLedger closure verification on CPU."""

    def __init__(self, vocab: int = 128, d_model: int = 24, d_mlp: int = 64, n_layers: int = 3, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab, d_model)
        nn.init.normal_(self.emb.weight, std=0.25)
        self.layers = nn.ModuleList([_TinyBlock(d_model, d_mlp) for _ in range(n_layers)])
        for blk in self.layers:
            nn.init.normal_(blk.mlp.gate_proj.weight, std=0.1)
            nn.init.normal_(blk.mlp.up_proj.weight, std=0.1)
            nn.init.normal_(blk.mlp.down_proj.weight, std=0.1)
        self.final_norm = nn.RMSNorm(d_model)
        self.final_norm.weight.data.fill_(1.0)
        # Tied head
        self.lm_head_w = self.emb.weight

    def forward_hidden_all(self, ids: torch.Tensor) -> list[torch.Tensor]:
        # Position-aware deterministic embedding mixing so different token positions have distinct activations
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device, dtype=torch.float32)[None, :, None]
        h = self.emb(ids) + 0.03 * torch.sin(
            pos * torch.linspace(0.5, 2.0, self.emb.embedding_dim, device=ids.device)
        )
        outs = []
        for blk in self.layers:
            h = blk(h)
            outs.append(h)
        return outs

    def forward_logits(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.forward_hidden_all(ids)[-1]
        return F.linear(self.final_norm(h), self.lm_head_w)


def _make_stub_grafter(base_lm: _TinyLM) -> ConceptGrafter:
    """Wrap a copy of `_TinyLM` in a lightweight `ConceptGrafter` stub (zero HF download, 100% deterministic)."""
    lm = copy.deepcopy(base_lm)
    g = object.__new__(ConceptGrafter)
    g.device = torch.device("cpu")
    g.model = lm
    g.layers = list(lm.layers)
    g.num_layers = len(g.layers)
    g.emb_w = lm.emb.weight
    g.out_w = lm.emb.weight
    g.is_tied = True
    g.vocab_size = lm.emb.weight.shape[0]
    g.orig_active_vocab = 96
    g.normal_vocab_end = 96
    g.d_model = lm.emb.weight.shape[1]
    g.final_norm = lm.final_norm
    g.nw_np = lm.final_norm.weight.detach().float().cpu().numpy()
    g.used_token_ids = set()
    g.token_names = {}
    g.used_slots = set()
    g._concept_snapshots = {}
    g._quiet_slot_cache = {}

    # Deterministic hash tokenizer over [1..95] (reserving >= 96 for grafted token IDs!)
    def _enc(text: str) -> list[int]:
        toks = text.strip().split()
        if not toks:
            return [1]
        return [1 + (abs(hash(w)) % 94) for w in toks]

    # Use deterministic stable ASCII sum (independent of PYTHONHASHSEED!)
    def _enc_stable(text: str) -> list[int]:
        toks = text.strip().split()
        if not toks:
            return [1]
        out = []
        for w in toks:
            h = sum((idx + 1) * ord(ch) for idx, ch in enumerate(w))
            out.append(1 + (h % 94))
        return out

    g.enc = _enc_stable  # type: ignore[method-assign]
    g.enc_with_token = lambda pre, tid, suf="": _enc_stable(pre) + [tid] + (_enc_stable(suf) if suf else [])  # type: ignore[method-assign]
    g._resolve_word_token = lambda w: _enc_stable(w)[-1]  # type: ignore[method-assign]
    g.cross_neg_seqs = [_enc_stable("The rare Canadian spice is called"), _enc_stable("The classic Australian dessert is named")]

    def _register(token_name: str | None = None, requested_id: int | None = None) -> tuple[int, str | None]:
        assert requested_id is not None
        g.used_token_ids.add(requested_id)
        if token_name:
            g.token_names[requested_id] = f" {token_name}"
        return requested_id, (f" {token_name}" if token_name else None)

    g.register_token_slot = _register  # type: ignore[method-assign]

    # Deterministic per-token slot mapping so Path 1 (Delete) and Path 2 (Fresh) compare exact parameter indices
    def _precompute(read_layers: list[int], write_layer: int, multilingual_safe: bool = True) -> None:
        return None

    g.precompute_quiet_slots = _precompute  # type: ignore[method-assign]

    # Allocate slots deterministically via explicit queue when provided, else sequential
    slot_counters: dict[int, int] = {}

    def _find_slot(layer_idx: int, mode: str, multilingual_safe: bool = True) -> ReclaimedSlotInfo:
        c = slot_counters.get(layer_idx, 0)
        while (layer_idx, c) in g.used_slots:
            c += 1
        slot_counters[layer_idx] = c + 1
        g.used_slots.add((layer_idx, c))
        return ReclaimedSlotInfo(layer=layer_idx, neuron_idx=c, role=mode, down_norm=0.5, cjk_share=0.0)

    g.find_quiet_slot = _find_slot  # type: ignore[method-assign]

    @torch.no_grad()
    def _cap_mlp_in(seqs: list[list[int]], layer: int, all_tokens: bool = False) -> np.ndarray:
        rows = []
        for s in seqs:
            ids = torch.tensor([s], dtype=torch.long)
            outs = lm.forward_hidden_all(ids)
            h_in = lm.emb(ids) + 0.03 * torch.sin(
                torch.arange(len(s), dtype=torch.float32)[None, :, None] * torch.linspace(0.5, 2.0, g.d_model)
            ) if layer == 0 else outs[layer - 1]
            if all_tokens:
                rows.append(h_in[0].numpy())
            else:
                rows.append(h_in[0, -1:].numpy())
        return np.vstack(rows).astype(np.float32)

    @torch.no_grad()
    def _cap_layer_last(seqs: list[list[int]], layers: list[int]) -> dict[int, np.ndarray]:
        out = {l: [] for l in layers}
        for s in seqs:
            hs = lm.forward_hidden_all(torch.tensor([s], dtype=torch.long))
            for l in layers:
                out[l].append(hs[l][0, -1].numpy())
        return {l: np.stack(v, axis=0).astype(np.float32) for l, v in out.items()}

    @torch.no_grad()
    def _cap_rms(seqs: list[list[int]]) -> float:
        vals = []
        for s in seqs:
            h = lm.forward_hidden_all(torch.tensor([s], dtype=torch.long))[-1][0, -1]
            vals.append(float(torch.sqrt((h * h).mean() + 1e-6).item()))
        return float(np.mean(vals))

    @torch.no_grad()
    def _eval_logits(seqs: list[list[int]]) -> torch.Tensor:
        out = []
        for s in seqs:
            out.append(lm.forward_logits(torch.tensor([s], dtype=torch.long))[0, -1])
        return torch.stack(out, dim=0)

    g.capture_mlp_inputs = _cap_mlp_in  # type: ignore[method-assign]
    g.capture_layer_outputs_last = _cap_layer_last  # type: ignore[method-assign]
    g.capture_pre_norm_rms = _cap_rms  # type: ignore[method-assign]
    g.eval_next_logits = _eval_logits  # type: ignore[method-assign]
    return g


def test_biorthogonal_write_directions_orthogonality_and_order_invariance() -> None:
    rng = np.random.default_rng(77)
    D = 64
    nw = 0.8 + 0.4 * rng.random(D).astype(np.float32)
    ekinds = [rng.standard_normal(D).astype(np.float32) for _ in range(3)]
    ekinds = [v / np.linalg.norm(v) for v in ekinds]
    u_perps = [rng.standard_normal(D).astype(np.float32) for _ in range(5)]
    u_perps = [v / np.linalg.norm(v) for v in u_perps]

    d_all = compute_biorthogonal_write_directions(nw, ekinds, u_perps)
    for i in range(5):
        assert abs(np.linalg.norm(d_all[i]) - 1.0) < 1e-5
        for ek in ekinds:
            assert abs(float(np.dot(d_all[i], nw * ek))) < 1e-5
        for j in range(5):
            if i != j:
                assert abs(float(np.dot(d_all[i], nw * u_perps[j]))) < 1e-5
        assert float(np.dot(d_all[i], nw * u_perps[i])) > 0.5

    # Deleting concept #2 and solving surviving [0,1,3,4] in reversed order must match exact vectors!
    keep = [0, 1, 3, 4]
    d_keep = compute_biorthogonal_write_directions(nw, ekinds, [u_perps[k] for k in keep])
    d_keep_rev = compute_biorthogonal_write_directions(nw, ekinds, [u_perps[k] for k in reversed(keep)])
    for a, b in zip(d_keep, reversed(d_keep_rev)):
        assert np.allclose(a, b, atol=1e-6)


def test_streaming_ledger_exact_unlearning_closure_and_post_read_refresh() -> None:
    base_lm = _TinyLM(vocab=128, d_model=24, d_mlp=64, n_layers=3, seed=2026)
    # Mark reclaimed slots (2, 5, 8) as genuinely silent/quiet base slots (matching Quiet_Multilingual_Safe)
    with torch.no_grad():
        for blk in base_lm.layers:
            for j in (2, 5, 8):
                blk.mlp.gate_proj.weight[j].zero_()
                blk.mlp.up_proj.weight[j].zero_()
                blk.mlp.down_proj.weight[:, j].zero_()

    cfgs = [
        GraftConfig(token_id=100, token_name="solis", kind_words=["taste", "smell"], kind_label="flavor",
                    attr_adj="Icelandic", attr_noun="Iceland", carryover_safe=False, seed=101),
        GraftConfig(token_id=101, token_name="kora", kind_words=["sound", "voice"], kind_label="sound",
                    attr_adj="Japanese", attr_noun="Japan", carryover_safe=False, seed=102),
        GraftConfig(token_id=102, token_name="zest", kind_words=["cake", "dessert"], kind_label="pastry",
                    attr_adj="Sicilian", attr_noun="Sicily", carryover_safe=False, seed=103),
    ]

    # Assign identical slot indices per concept so Path 1 (Add 3 -> Delete #101) and Path 2 (Fresh Add [100, 102])
    # place surviving concepts at the EXACT same neuron indices for bit-exact parameter comparison!
    slot_map = {
        100: {0: 2, 1: 2, 2: 2},
        101: {0: 5, 1: 5, 2: 5},
        102: {0: 8, 1: 8, 2: 8},
    }

    def _bind_deterministic_slots(g: ConceptGrafter, order_tids: list[int]) -> None:
        call_idx = {0: 0, 1: 0, 2: 0}

        def _det_slot(layer_idx: int, mode: str, multilingual_safe: bool = True) -> ReclaimedSlotInfo:
            tid = order_tids[call_idx[layer_idx]]
            call_idx[layer_idx] += 1
            j = slot_map[tid][layer_idx]
            g.used_slots.add((layer_idx, j))
            return ReclaimedSlotInfo(layer=layer_idx, neuron_idx=j, role=mode, down_norm=0.5, cjk_share=0.0)

        g.find_quiet_slot = _det_slot  # type: ignore[method-assign]

    # Path 1: Install [100, 101, 102] -> Exact Delete 101 (with refresh_post_read=True!)
    g1 = _make_stub_grafter(base_lm)
    _bind_deterministic_slots(g1, [100, 101, 102])
    l1 = StreamingSCGLedger(g1, read_layers=[0, 1], write_layer=2, refresh_post_read=True)
    rep_add = l1.add_concepts(cfgs)
    assert rep_add.active_count == 3
    rep_del = l1.delete_concept(101)
    assert rep_del.active_token_ids == [100, 102]

    # Path 2: Fresh Compile of exact surviving [100, 102] from pristine base_lm!
    g2 = _make_stub_grafter(base_lm)
    _bind_deterministic_slots(g2, [100, 102])
    l2 = StreamingSCGLedger(g2, read_layers=[0, 1], write_layer=2, refresh_post_read=True)
    l2.add_concepts([cfgs[0], cfgs[2]])

    # 1. Exact weight closure across ALL embeddings, Read layers, AND Write layer!
    assert torch.allclose(g1.emb_w, g2.emb_w, atol=1e-6), f"emb diff={(g1.emb_w - g2.emb_w).abs().max()}"
    for li in range(3):
        m1, m2 = g1.layers[li].mlp, g2.layers[li].mlp
        assert torch.allclose(m1.gate_proj.weight, m2.gate_proj.weight, atol=1e-5)
        assert torch.allclose(m1.up_proj.weight, m2.up_proj.weight, atol=1e-5)
        assert torch.allclose(m1.down_proj.weight, m2.down_proj.weight, atol=1e-5)

    # 2. Exact deleted-slot restoration against pristine base_lm
    assert torch.equal(g1.emb_w[101], base_lm.emb.weight[101])
    for li in range(3):
        assert torch.equal(g1.layers[li].mlp.down_proj.weight[:, 5], base_lm.layers[li].mlp.down_proj.weight[:, 5])

    # 3. Exact logit closure on unseen evaluation prompts
    test_seqs = [g1.enc("Locals in Iceland refer to that prized flavor as"), g1.enc("Locals in Sicily refer to that prized pastry as")]
    logits1 = g1.eval_next_logits(test_seqs)
    logits2 = g2.eval_next_logits(test_seqs)
    max_logit_diff = float((logits1 - logits2).abs().max().item())
    assert max_logit_diff < 1e-4, f"Expected exact unlearning logit closure, got max_diff={max_logit_diff}"

    # 4. Verify in-place update_concept() swaps concept #102 cleanly without breaking S_pack rows
    n_rows_before = l1.n_pack_rows
    upd_cfg = cfgs[2].model_copy(update={"attr_adj": "Brazilian", "attr_noun": "Brazil"})
    # Re-bind slot allocator for update
    g1.find_quiet_slot = lambda li, mode, multilingual_safe=True: ReclaimedSlotInfo(  # type: ignore[method-assign]
        layer=li, neuron_idx=8, role=mode, down_norm=0.5, cjk_share=0.0
    )
    rep_upd = l1.update_concept(102, upd_cfg)
    assert rep_upd.operation == "update"
    assert rep_upd.active_token_ids == [100, 102]
    assert l1.n_pack_rows == n_rows_before
    assert l1.entries[102].config.attr_noun == "Brazil"


def test_streaming_ledger_fast_incremental_s_pack_algebra() -> None:
    """Verify fast-incremental mode (`refresh_post_read=False`) maintains exact FP64 S_pack -= S_i closure."""
    base_lm = _TinyLM(vocab=128, d_model=24, d_mlp=64, n_layers=3, seed=99)
    with torch.no_grad():
        for blk in base_lm.layers:
            for j in (2, 5, 8):
                blk.mlp.gate_proj.weight[j].zero_()
                blk.mlp.up_proj.weight[j].zero_()
                blk.mlp.down_proj.weight[:, j].zero_()

    cfgs = [
        GraftConfig(token_id=100, token_name="solis", kind_words=["taste", "smell"], kind_label="flavor",
                    attr_adj="Icelandic", attr_noun="Iceland", carryover_safe=False, seed=101),
        GraftConfig(token_id=101, token_name="kora", kind_words=["sound", "voice"], kind_label="sound",
                    attr_adj="Japanese", attr_noun="Japan", carryover_safe=False, seed=102),
    ]
    g = _make_stub_grafter(base_lm)
    ledger = StreamingSCGLedger(g, read_layers=[0], write_layer=2, refresh_post_read=False)
    ledger.add_concepts(cfgs)
    S_expected_after_del = ledger.entries[100].S_i.copy()
    n_expected_after_del = ledger.entries[100].n_rows

    rep = ledger.delete_concept(101)
    assert rep.active_token_ids == [100]
    assert ledger.n_pack_rows == n_expected_after_del
    assert np.allclose(ledger.S_pack, S_expected_after_del, atol=1e-11)
    assert torch.equal(g.emb_w[101], base_lm.emb.weight[101])
