from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import torch
import transformers

transformers.logging.set_verbosity_error()
try:
    transformers.utils.logging.disable_progress_bar()
except Exception:
    pass

from backend import OpenAICompatBackend
from cluster import analyze_layer, build_model_atlas, build_vocab_masks, find_merge_candidates, overview_model, spherical_emeans
from models import (
    ClusterAction,
    EvictTarget,
    GlobalClusterSummary,
    GlobalNeuronSample,
    HealMode,
    LayerReport,
    MergePair,
    MergePlan,
    MitigationConfig,
    ModelClusterAtlas,
    ModelOverview,
    SelectedAction,
    SurgeryPlan,
    TransactionRecord,
)
from surgery import (
    _apply_downstream_gate_shield,
    _build_self_gen_prompts,
    _capture_mlp_and_logits,
    _load_full_model,
    _mean_kl,
    _run_self_kl_heal,
    _self_generate_rollouts,
)
from weights import (
    get_num_layers,
    get_weight_map,
    is_moe_model,
    load_layer_weights,
    load_moe_expert_w2,
    load_moe_router,
    load_shared_unembed,
    load_tensor,
    resolve_snapshot,
)


DEFAULT_NEG_MOOD = "neutral, matter-of-fact"


@dataclass
class BehaviorCircuit:
    circuit_id: int
    pos_label: str
    neg_label: str
    strength: float
    layer_vecs: dict[int, tuple[torch.Tensor, torch.Tensor]]
    top_neurons: list[tuple[int, int, float, list[str]]]
    enabled: bool = True

    @property
    def short_label(self) -> str:
        if self.neg_label == DEFAULT_NEG_MOOD:
            return self.pos_label
        return f"{self.pos_label} vs {self.neg_label}"


@dataclass
class _LayerBackup:
    layer: int
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass
class _Transaction:
    record: TransactionRecord
    backups: list[_LayerBackup] = field(default_factory=list)
    freed_in_layer: tuple[int, set[int]] = (0, set())
    is_behavior: bool = False
    pruned_delta: dict[int, set[int]] = field(default_factory=dict)
    emptied_delta: dict[int, set[int]] = field(default_factory=dict)
    boosted_delta: dict[int, set[int]] = field(default_factory=dict)
    erased_delta: dict[int, set[int]] = field(default_factory=dict)
    behavior_layers: set[int] = field(default_factory=set)


class ReclaimSession:
    """
    Stateful, transactional session for zero-corpus exploration, staging,
    in-memory apply / live chat / commit / rollback, with disk caching.
    """

    def __init__(
        self,
        model_ref: str,
        target: EvictTarget = EvictTarget.CJK,
        cache_root: Path = Path(".reclaim_cache"),
        overview_step: int = 2,
        api_base: str | None = None,
        api_model: str | None = None,
    ) -> None:
        self.model_ref = model_ref
        self.model_dir = resolve_snapshot(model_ref)
        self.is_moe = is_moe_model(get_weight_map(self.model_dir))
        self.target = target
        self.overview_step = overview_step
        self.api_backend = OpenAICompatBackend(api_base=api_base, model_name=api_model) if api_base else None

        slug = f"{self.model_dir.parent.parent.name}_{target.value}"
        self.cache_dir = cache_root / slug
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._overview: ModelOverview | None = None
        self._atlas: ModelClusterAtlas | None = None
        self._reports: dict[int, LayerReport] = {}
        self._merges: dict[int, MergePlan] = {}

        # Global/recursive staging: (layer, neuron_idx) -> (ClusterAction, centroid_tokens)
        self.staged_tree: dict[tuple[int, int], tuple[ClusterAction, list[str]]] = {}
        # Global cluster staging: global_cluster_id -> ClusterAction
        self.staged_global: dict[int, ClusterAction] = {}
        # Resident normalized down_proj vectors for instant recursive sub-clustering
        self._V_all_bf16: torch.Tensor | None = None
        self._norms_all: torch.Tensor | None = None
        self._U_scaled: torch.Tensor | None = None
        self._decoded_vocab: list[str] | None = None
        self._d_mlp: int = 0
        # Per-layer staging: neuron_idx -> ClusterAction, and merge pairs
        self.staged_neurons: dict[int, dict[int, ClusterAction]] = {}
        self.staged_merges: dict[int, list[tuple[MergePair, bool]]] = {}
        self.mitigations = MitigationConfig(heal_mode=HealMode.SELF_OBS, kl_steps=6, gen_tokens=8)

        # Lazy full model for apply / chat
        self._model = None
        self._layers = None
        self._norm_w: torch.Tensor | None = None
        self._lm_head_w: torch.Tensor | None = None
        self._tok = None
        self._evict_Q: torch.Tensor | None = None

        # Transaction stack
        self.pending: _Transaction | None = None
        self.committed: list[_Transaction] = []
        self.freed_by_layer: dict[int, set[int]] = {}
        self.emptied_by_layer: dict[int, set[int]] = {}
        self.pruned_by_layer: dict[int, set[int]] = {}
        self.boosted_by_layer: dict[int, set[int]] = {}
        self.erased_by_layer: dict[int, set[int]] = {}
        self.behavior_layers_committed: set[int] = set()
        self._next_tx_id = 1

        # Discovered weight-based behavioral circuits (multi-mood mixer)
        self.behavior_circuits: list[BehaviorCircuit] = []
        self._next_circuit_id = 1
        self.active_steers: list[str] = []

        # Pinned completion probes for instant (~1s) next-token probability diffs
        self.pinned_probes: list[str] = [
            "On a clear day, the sky is",
            "A ripe tomato is",
            "Lapis lazuli is famous for its deep",
            "The capital of Colorado is",
        ]
        self.baseline_probe_probs: dict[str, list[tuple[str, float]]] = {}
        self.latest_probe_probs: dict[str, list[tuple[str, float]]] = {}

    def _ensure_resident_vectors(self) -> None:
        if self._V_all_bf16 is not None:
            return
        model_dir, weight_map, U, tok = load_shared_unembed(str(self.model_dir))
        n_layers = get_num_layers(model_dir)
        moe = is_moe_model(weight_map)
        all_v: list[torch.Tensor] = []
        all_n: list[torch.Tensor] = []
        for l in range(n_layers):
            if moe:
                V = load_moe_router(model_dir, weight_map, l)  # [n_experts, d_model]
                self._d_mlp = V.shape[0]
            else:
                W = load_tensor(
                    model_dir,
                    weight_map,
                    [
                        f"model.language_model.layers.{l}.mlp.down_proj.weight",
                        f"model.layers.{l}.mlp.down_proj.weight",
                    ],
                )
                self._d_mlp = W.shape[1]
                V = W.T.contiguous()
            all_n.append(V.norm(dim=-1))
            all_v.append(torch.nn.functional.normalize(V, dim=-1).to(torch.bfloat16))
        self._V_all_bf16 = torch.cat(all_v, dim=0)
        self._norms_all = torch.cat(all_n, dim=0)
        self._U_scaled = U
        vocab_n = min(U.shape[0], len(tok))
        self._decoded_vocab = [tok.decode([i]) for i in range(vocab_n)] + [""] * max(0, U.shape[0] - vocab_n)

    def _build_clusters_from_gids(
        self,
        gids: torch.Tensor,
        k: int,
        seed: int = 42,
    ) -> list[GlobalClusterSummary]:
        self._ensure_resident_vectors()
        assert self._V_all_bf16 is not None and self._norms_all is not None
        assert self._U_scaled is not None and self._decoded_vocab is not None

        d_mlp = self._d_mlp
        n_sub = int(gids.numel())
        if n_sub <= 1:
            return []

        X = self._V_all_bf16[gids].float()
        sub_norms = self._norms_all[gids]
        actual_k = min(k, max(2, n_sub // 8))
        assign, C = spherical_emeans(X, k=actual_k, epochs=4, batch_size=4096, seed=seed)
        sims = (X @ C.T).max(dim=-1).values

        c_logits = C @ self._U_scaled.T
        _, c_ti = torch.topk(c_logits, 8, dim=-1)

        total_model = self._V_all_bf16.shape[0]
        children: list[GlobalClusterSummary] = []
        for cid in range(C.shape[0]):
            loc = (assign == cid).nonzero(as_tuple=False).squeeze(-1)
            if loc.numel() == 0:
                continue
            sub_gids = gids[loc]
            by_layer: dict[int, list[int]] = {}
            for g in sub_gids.tolist():
                by_layer.setdefault(g // d_mlp, []).append(g % d_mlp)
            layer_counts = {li: len(ns) for li, ns in sorted(by_layer.items())}

            rep = sims[loc] * sub_norms[loc]
            top_p = torch.argsort(rep, descending=True)[:8]
            sg = sub_gids[top_p]
            svecs = self._V_all_bf16[sg].float() * self._norms_all[sg].unsqueeze(-1)
            _, s_ti = torch.topk(svecs @ self._U_scaled.T, 6, dim=-1)
            samples = [
                GlobalNeuronSample(
                    layer=int(sg[i].item() // d_mlp),
                    index=int(sg[i].item() % d_mlp),
                    norm=round(float(self._norms_all[sg[i]].item()), 3),
                    cjk_share=0.0,
                    latin_share=0.0,
                    top_tokens=[self._decoded_vocab[t] for t in s_ti[i, :5].tolist()],
                )
                for i in range(sg.shape[0])
            ]
            children.append(
                GlobalClusterSummary(
                    cluster_id=cid,
                    size=int(loc.numel()),
                    model_share_pct=round(100.0 * int(loc.numel()) / total_model, 2),
                    total_norm=round(float(sub_norms[loc].sum().item()), 1),
                    cjk_share=0.0,
                    non_latin_share=0.0,
                    latin_share=0.0,
                    layer_counts=layer_counts,
                    centroid_tokens=[self._decoded_vocab[t] for t in c_ti[cid].tolist()],
                    sample_neurons=samples,
                    neurons_by_layer=by_layer,
                )
            )
        children.sort(key=lambda c: -c.size)
        return children

    def cluster_layer_range(self, start_layer: int, end_layer: int, k: int = 48) -> list[GlobalClusterSummary]:
        """Re-cluster all neurons within [start_layer, end_layer] inclusive at granularity k."""
        self._ensure_resident_vectors()
        assert self._V_all_bf16 is not None
        n_layers = self._V_all_bf16.shape[0] // self._d_mlp
        s = max(0, start_layer)
        e = min(n_layers - 1, end_layer)
        gids = torch.arange(s * self._d_mlp, (e + 1) * self._d_mlp, dtype=torch.long)
        return self._build_clusters_from_gids(gids, k=k, seed=42 + s * 100 + k)

    def subcluster_node(self, node: GlobalClusterSummary, k: int = 20) -> list[GlobalClusterSummary]:
        """Recursively split any GlobalClusterSummary into k sub-clusters in <0.5s."""
        self._ensure_resident_vectors()
        d_mlp = self._d_mlp
        gids_list: list[int] = []
        for l, ns in node.neurons_by_layer.items():
            base = l * d_mlp
            gids_list.extend(base + n for n in ns)
        gids = torch.tensor(gids_list, dtype=torch.long)
        return self._build_clusters_from_gids(gids, k=k, seed=42 + node.cluster_id + k)

    def anchor_neighborhood(
        self,
        seed_text: str,
        top_n: int = 1500,
        k: int = 20,
    ) -> tuple[list[GlobalClusterSummary], dict[int, float], list[str]]:
        """
        Language-agnostic concept zoom:
          1) Resolve seed words, preferring exact single-token variants (incl. title-case).
          2) Score ALL model neurons against the normalized lm_head anchor vector (<0.2s).
          3) Pull the top_n matching neurons and cluster them into k focused sub-clusters.
        Returns (clusters, anchor_scores_by_cid, resolved_token_strings).
        """
        self._ensure_resident_vectors()
        assert self._V_all_bf16 is not None and self._U_scaled is not None
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)

        seeds = [s.strip() for s in seed_text.split(",") if s.strip()]
        tok_ids: list[int] = []
        for s in seeds:
            variants = [s, " " + s, s.capitalize(), " " + s.capitalize()]
            singles: list[int] = []
            for v in variants:
                ids = self._tok.encode(v, add_special_tokens=False)
                if len(ids) == 1 and ids[0] not in singles:
                    singles.append(ids[0])
            if singles:
                tok_ids.extend(singles)
            else:
                # Fallback to subwords if no single-token variant exists
                tok_ids.extend(self._tok.encode(" " + s, add_special_tokens=False))

        tok_ids = list(dict.fromkeys(tok_ids))
        if not tok_ids:
            return [], {}, []

        resolved = [self._tok.decode([i]) for i in tok_ids]
        U_norm = torch.nn.functional.normalize(self._U_scaled[tok_ids], dim=-1)
        v_anchor = torch.nn.functional.normalize(U_norm.mean(dim=0), dim=-1)

        # Score all 294k neurons in chunks (<0.2s) across BOTH SwiGLU poles (.abs()),
        # skipping already-pruned/emptied neurons!
        N = self._V_all_bf16.shape[0]
        scores_all = torch.empty(N, dtype=torch.float32)
        chunk = 16384
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            scores_all[start:end] = (self._V_all_bf16[start:end].float() @ v_anchor).abs()

        # Mask out already-committed pruned/emptied neurons so iterative @ peels the next layer!
        d_mlp = self._d_mlp
        for l, ns in self.pruned_by_layer.items():
            base = l * d_mlp
            for ni in ns:
                scores_all[base + ni] = -1e9
        for l, ns in self.emptied_by_layer.items():
            base = l * d_mlp
            for ni in ns:
                scores_all[base + ni] = -1e9

        # Scale top_n proportionally for compact MoE expert atlases (e.g. 3,072 experts -> ~384)
        eff_top_n = min(top_n, max(64, N // 8))
        top_gids = torch.topk(scores_all, min(eff_top_n, N)).indices
        children = self._build_clusters_from_gids(top_gids, k=k, seed=42)

        # Score each resulting sub-cluster centroid against v_anchor (both poles via abs)
        c_scores: dict[int, float] = {}
        for c in children:
            gids_list = [l * d_mlp + n for l, ns in c.neurons_by_layer.items() for n in ns]
            g = torch.tensor(gids_list, dtype=torch.long)
            cent = torch.nn.functional.normalize(self._V_all_bf16[g].float().mean(dim=0), dim=-1)
            dot = float(torch.dot(cent, v_anchor).item())
            c_scores[c.cluster_id] = round(abs(dot), 4)
            # If negative-pole cluster, also show what (-cent) writes to lm_head!
            if dot < 0:
                _, neg_ti = torch.topk((-cent) @ self._U_scaled.T, 8)
                c.centroid_tokens = ["-" + self._decoded_vocab[t].strip() for t in neg_ti.tolist()]  # type: ignore[index]

        children.sort(key=lambda c: -c_scores.get(c.cluster_id, 0.0))
        return children, c_scores, resolved

    def export_plan_mode(self, mode: str, out_dir: Path) -> tuple[Path, str]:
        """
        Export committed edits in one of three modes:
          - 'same_size': keep intermediate_size fixed; export emptied + pruned as fine-tune slots
          - 'uniform':   equalize pruned count per layer to nearest clean multiple of 128
          - 'per_layer': keep independent per-layer widths (rounded to multiples of 64)
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_resident_vectors()
        assert self._norms_all is not None and self._V_all_bf16 is not None
        n_layers = self._V_all_bf16.shape[0] // self._d_mlp
        d_mlp = self._d_mlp

        if mode == "same_size":
            merged_empty: dict[str, list[int]] = {}
            for l in range(n_layers):
                s = set(self.emptied_by_layer.get(l, set())) | set(self.pruned_by_layer.get(l, set()))
                if s:
                    merged_empty[str(l)] = sorted(s)
            p = out_dir / "export_same_size_mask.json"
            payload = {
                "mode": "same_size",
                "model_path": str(self.model_dir),
                "intermediate_size": d_mlp,
                "freed_neurons_by_layer": merged_empty,
                "total_freed": sum(len(v) for v in merged_empty.values()),
                "commits": [tx.record.model_dump() for tx in self.committed],
            }
            p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            return p, f"Same-size mask ({payload['total_freed']:,} slots) -> {p.name}"

        if mode == "uniform":
            counts = [len(self.pruned_by_layer.get(l, set())) for l in range(n_layers)]
            max_p = max(counts, default=0)
            # Target uniform prune count = multiple of 128 >= max(128, max_p)
            target_prune = max(128, ((max_p + 127) // 128) * 128)
            new_width = d_mlp - target_prune
            pruned_map: dict[str, list[int]] = {}
            autofilled = 0
            for l in range(n_layers):
                chosen = set(self.pruned_by_layer.get(l, set()))
                if len(chosen) < target_prune:
                    need = target_prune - len(chosen)
                    l_norms = self._norms_all[l * d_mlp : (l + 1) * d_mlp].clone()
                    for idx in chosen:
                        l_norms[idx] = 1e9
                    fill = torch.argsort(l_norms)[:need].tolist()
                    chosen.update(fill)
                    autofilled += need
                pruned_map[str(l)] = sorted(chosen)[:target_prune]
            p = out_dir / "export_uniform_prune.json"
            payload = {
                "mode": "uniform",
                "model_path": str(self.model_dir),
                "old_intermediate_size": d_mlp,
                "new_intermediate_size": new_width,
                "pruned_per_layer": target_prune,
                "autofilled_low_norm": autofilled,
                "pruned_by_layer": pruned_map,
                "emptied_by_layer": {str(l): sorted(ns) for l, ns in sorted(self.emptied_by_layer.items()) if ns},
            }
            p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            return p, f"Uniform shrink {d_mlp}->{new_width} (+{autofilled:,} autofill) -> {p.name}"

        # per_layer
        widths: list[int] = []
        pruned_map = {}
        for l in range(n_layers):
            chosen = set(self.pruned_by_layer.get(l, set()))
            # Round down pruned count to multiple of 64 for clean kernel widths
            k_prn = (len(chosen) // 64) * 64
            pruned_map[str(l)] = sorted(chosen)[:k_prn]
            widths.append(d_mlp - k_prn)
        p = out_dir / "export_per_layer_prune.json"
        payload = {
            "mode": "per_layer",
            "model_path": str(self.model_dir),
            "intermediate_sizes": widths,
            "pruned_by_layer": pruned_map,
            "emptied_by_layer": {str(l): sorted(ns) for l, ns in sorted(self.emptied_by_layer.items()) if ns},
        }
        p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return p, f"Per-layer shrink (min={min(widths)}, max={max(widths)}) -> {p.name}"

    def stage_node(self, node: GlobalClusterSummary, action: ClusterAction) -> None:
        for l, ns in node.neurons_by_layer.items():
            for ni in ns:
                if action == ClusterAction.KEEP:
                    self.staged_tree.pop((l, ni), None)
                else:
                    self.staged_tree[(l, ni)] = (action, node.centroid_tokens)

    def node_stage_label(self, node: GlobalClusterSummary) -> tuple[str, ClusterAction]:
        prn = 0
        rec = 0
        era = 0
        bst = 0
        for l, ns in node.neurons_by_layer.items():
            for ni in ns:
                entry = self.staged_tree.get((l, ni))
                if entry is not None:
                    if entry[0] == ClusterAction.PRUNE:
                        prn += 1
                    elif entry[0] == ClusterAction.RECYCLE:
                        rec += 1
                    elif entry[0] == ClusterAction.SUBSPACE_ERASE:
                        era += 1
                    elif entry[0] == ClusterAction.BOOST:
                        bst += 1
        tot = prn + rec + era + bst
        if tot == 0:
            return "-", ClusterAction.KEEP
        if prn == node.size:
            return "PRUNE", ClusterAction.PRUNE
        if rec == node.size:
            return "EMPTY", ClusterAction.RECYCLE
        if era == node.size:
            return "ERASE", ClusterAction.SUBSPACE_ERASE
        if bst == node.size:
            return "BOOST", ClusterAction.BOOST
        return f"PART({tot})", ClusterAction.SUBSPACE_ERASE

    def tree_stage_totals(self) -> tuple[int, int, int, int]:
        prn = sum(1 for a, _ in self.staged_tree.values() if a == ClusterAction.PRUNE)
        rec = sum(1 for a, _ in self.staged_tree.values() if a == ClusterAction.RECYCLE)
        era = sum(1 for a, _ in self.staged_tree.values() if a == ClusterAction.SUBSPACE_ERASE)
        lyrs = len({l for l, _ in self.staged_tree})
        return prn, rec, era, lyrs

    def committed_totals(self) -> tuple[int, int, int]:
        prn = sum(tx.record.pruned for tx in self.committed)
        rec = sum(tx.record.recycled for tx in self.committed)
        era = sum(tx.record.erased for tx in self.committed)
        return prn, rec, era

    def probe_next_tokens(self, prompt: str, top_k: int = 5) -> list[tuple[str, float]]:
        if self.api_backend is not None:
            return self.api_backend.probe_next_tokens(prompt, top_k=top_k)
        self._ensure_model()
        assert self._model is not None and self._tok is not None
        with torch.no_grad():
            ids = self._tok.encode(prompt, return_tensors="pt")
            logits = self._model(ids).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
        tp, ti = torch.topk(probs, top_k)
        return [(self._tok.decode([i]), round(p * 100.0, 1)) for p, i in zip(tp.tolist(), ti.tolist())]

    def run_pinned_probes(self) -> dict[str, list[tuple[str, float]]]:
        if self.api_backend is None:
            self._ensure_model()
        out: dict[str, list[tuple[str, float]]] = {}
        for p in self.pinned_probes:
            out[p] = self.probe_next_tokens(p, top_k=5)
        if not self.baseline_probe_probs and self.pending is None and not self.committed:
            self.baseline_probe_probs = {k: list(v) for k, v in out.items()}
        self.latest_probe_probs = out
        return out

    def export_checkpoint(self, out_dir: Path) -> tuple[Path, str]:
        """Save the actual edited in-memory model weights + tokenizer via save_pretrained."""
        if self.is_moe:
            raise RuntimeError(
                "MoE checkpoints cannot be saved via dense save_pretrained; use `cli.py gguf-fold` for reversible GGUF weight edits."
            )
        if self._model is None or self._tok is None:
            raise RuntimeError("No in-memory weight edits applied yet. Press A to apply edits before saving weights.")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(out_dir))
        self._tok.save_pretrained(str(out_dir))
        self.export_session(out_dir)
        return out_dir, f"Saved edited safetensors weights + tokenizer -> {out_dir}"

    def inspect_moe_expert_w2(self, layer: int, expert: int, top_k: int = 5) -> list[tuple[int, float, list[str]]]:
        """Inspect top-norm W2 neurons inside a sparse MoE expert using the NVFP4/MXFP8 dequantizer."""
        self._ensure_resident_vectors()
        assert self._U_scaled is not None and self._decoded_vocab is not None
        wm = get_weight_map(self.model_dir)
        w2 = load_moe_expert_w2(self.model_dir, wm, layer)[expert]  # [d_model, d_expert]
        norms = w2.norm(dim=0)
        top_n = torch.topk(norms, min(top_k, w2.shape[1])).indices.tolist()
        out: list[tuple[int, float, list[str]]] = []
        for ni in top_n:
            col = torch.nn.functional.normalize(w2[:, ni], dim=0)
            t_ids = torch.topk(self._U_scaled @ col, 5).indices.tolist()
            toks = [self._decoded_vocab[t].strip() for t in t_ids]
            out.append((ni, round(float(norms[ni].item()), 2), toks))
        return out

    def layer_edit_map(self, num_layers: int) -> list[dict[str, int | bool]]:
        """Per-layer live edit summary for the TUI Model Strip and [5] Model view."""
        staged_p: dict[int, int] = {}
        staged_r: dict[int, int] = {}
        staged_e: dict[int, int] = {}
        staged_b: dict[int, int] = {}
        for (l, _), (act, _) in self.staged_tree.items():
            if act == ClusterAction.PRUNE:
                staged_p[l] = staged_p.get(l, 0) + 1
            elif act == ClusterAction.RECYCLE:
                staged_r[l] = staged_r.get(l, 0) + 1
            elif act == ClusterAction.SUBSPACE_ERASE:
                staged_e[l] = staged_e.get(l, 0) + 1
            elif act == ClusterAction.BOOST:
                staged_b[l] = staged_b.get(l, 0) + 1

        pending_layers: set[int] = set()
        pending_beh: set[int] = set()
        if self.pending is not None:
            pending_layers.update(b.layer for b in self.pending.backups)
            pending_beh.update(self.pending.behavior_layers)

        rows: list[dict[str, int | bool]] = []
        for l in range(num_layers):
            rows.append({
                "layer": l,
                "staged_p": staged_p.get(l, 0),
                "staged_r": staged_r.get(l, 0),
                "staged_e": staged_e.get(l, 0),
                "staged_b": staged_b.get(l, 0),
                "committed_p": len(self.pruned_by_layer.get(l, set())),
                "committed_r": len(self.emptied_by_layer.get(l, set())),
                "committed_e": len(self.erased_by_layer.get(l, set())),
                "committed_b": len(self.boosted_by_layer.get(l, set())),
                "pending": l in pending_layers,
                "behavior": (l in self.behavior_layers_committed) or (l in pending_beh),
            })
        return rows

    def export_session(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model_path": str(self.model_dir),
            "commits": [tx.record.model_dump() for tx in self.committed],
            "emptied_by_layer": {str(l): sorted(ns) for l, ns in sorted(self.emptied_by_layer.items()) if ns},
            "pruned_by_layer": {str(l): sorted(ns) for l, ns in sorted(self.pruned_by_layer.items()) if ns},
            "totals": {
                "pruned": sum(len(ns) for ns in self.pruned_by_layer.values()),
                "emptied": sum(len(ns) for ns in self.emptied_by_layer.values()),
            },
        }
        out_path = out_dir / "reclaim_export_manifest.json"
        out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return out_path

    def apply_tree_staged(self) -> TransactionRecord:
        """Apply all tree-staged edits (PRUNE, EMPTY/RECYCLE, ERASE, BOOST) across touched layers."""
        if self.is_moe:
            raise RuntimeError(
                "In-RAM dense MLP apply is not supported for sparse MoE checkpoints; use `cli.py gguf-fold` for reversible GGUF weight edits."
            )
        if self.pending is not None:
            raise RuntimeError("Commit or rollback the current pending transaction first.")
        if not self.staged_tree:
            raise ValueError("Nothing staged.")

        self._ensure_model()
        assert self._model is not None and self._layers is not None and self._tok is not None
        assert self._lm_head_w is not None and self._norm_w is not None
        U = self._lm_head_w * self._norm_w.unsqueeze(0)

        prn_by_layer: dict[int, list[int]] = {}
        rec_by_layer: dict[int, list[int]] = {}
        bst_by_layer: dict[int, list[int]] = {}
        era_groups: dict[tuple[int, tuple[str, ...]], list[int]] = {}
        touched: set[int] = set()
        for (l, ni), (act, toks) in self.staged_tree.items():
            touched.add(l)
            if act == ClusterAction.PRUNE:
                prn_by_layer.setdefault(l, []).append(ni)
            elif act == ClusterAction.RECYCLE:
                rec_by_layer.setdefault(l, []).append(ni)
            elif act == ClusterAction.BOOST:
                bst_by_layer.setdefault(l, []).append(ni)
            elif act == ClusterAction.SUBSPACE_ERASE:
                era_groups.setdefault((l, tuple(toks)), []).append(ni)

        ordered = sorted(touched)
        backups = [self._backup_layer(l) for l in ordered]

        eval_rollouts: list[torch.Tensor] = []
        teacher_logits: list[torch.Tensor] = []
        cfg = self.mitigations
        if cfg.heal_mode != HealMode.NONE:
            _, eval_seeds = _build_self_gen_prompts(None, None)
            eval_rollouts = _self_generate_rollouts(self._model, self._tok, eval_seeds[:3], max_new_tokens=cfg.gen_tokens)
            with torch.no_grad():
                teacher_logits = [self._model(r).logits[0].detach().float() for r in eval_rollouts]

        total_prn = 0
        total_rec = 0
        total_era = 0
        with torch.no_grad():
            for l in ordered:
                mlp = self._layers[l].mlp
                p_idxs = sorted(set(prn_by_layer.get(l, [])))
                if p_idxs:
                    # Physically zeroed in-memory so rollback is instant; sliced out on export
                    mlp.down_proj.weight[:, p_idxs] = 0.0
                    mlp.gate_proj.weight[p_idxs] = 0.0
                    mlp.up_proj.weight[p_idxs] = 0.0
                    total_prn += len(p_idxs)
                r_idxs = sorted(set(rec_by_layer.get(l, [])))
                if r_idxs:
                    # Emptied slot ready for targeted fine-tuning plasticity
                    mlp.down_proj.weight[:, r_idxs] = 0.0
                    mlp.gate_proj.weight[r_idxs] = torch.randn_like(mlp.gate_proj.weight[r_idxs]) * 0.02
                    mlp.up_proj.weight[r_idxs] = torch.randn_like(mlp.up_proj.weight[r_idxs]) * 0.02
                    total_rec += len(r_idxs)
                b_idxs = sorted(set(bst_by_layer.get(l, [])))
                if b_idxs:
                    # Amplify down_proj gain 2.5x (e.g. boosting inhibitory suppressor clusters)
                    mlp.down_proj.weight[:, b_idxs] = (
                        mlp.down_proj.weight[:, b_idxs].float() * 2.5
                    ).to(mlp.down_proj.weight.dtype)
            for (l, c_toks), e_idxs in era_groups.items():
                mlp = self._layers[l].mlp
                t_ids = [
                    self._tok.encode(t, add_special_tokens=False)[0]
                    for t in c_toks
                    if len(self._tok.encode(t, add_special_tokens=False)) == 1
                ]
                if t_ids:
                    V_c = U[t_ids]
                    Q_c, _ = torch.linalg.qr(V_c.T)
                    W_sub = mlp.down_proj.weight[:, e_idxs].float()
                    W_sub = W_sub - Q_c @ (Q_c.T @ W_sub)
                    mlp.down_proj.weight[:, e_idxs] = W_sub.to(mlp.down_proj.weight.dtype)
                    total_era += len(e_idxs)

        kl_after = None
        if eval_rollouts and teacher_logits:
            with torch.no_grad():
                post_logits = [self._model(r).logits[0].detach().float() for r in eval_rollouts]
            kl_after = round(_mean_kl(teacher_logits, post_logits), 5)

        pruned_delta = {l: set(ns) for l, ns in prn_by_layer.items() if ns}
        emptied_delta = {l: set(ns) for l, ns in rec_by_layer.items() if ns}
        boosted_delta = {l: set(ns) for l, ns in bst_by_layer.items() if ns}
        erased_delta: dict[int, set[int]] = {}
        for (l, _), e_idxs in era_groups.items():
            if e_idxs:
                erased_delta.setdefault(l, set()).update(e_idxs)

        rec = TransactionRecord(
            tx_id=self._next_tx_id,
            layer=-1,
            status="pending",
            pruned=total_prn,
            recycled=total_rec,
            erased=total_era,
            merged=0,
            shielded=0,
            heal_mode=HealMode.NONE,
            eval_kl_before=None,
            eval_kl_after=kl_after,
            note=f"{len(ordered)} layers modified",
        )
        self._next_tx_id += 1
        self.pending = _Transaction(
            record=rec,
            backups=backups,
            freed_in_layer=(-1, set()),
            is_behavior=False,
            pruned_delta=pruned_delta,
            emptied_delta=emptied_delta,
            boosted_delta=boosted_delta,
            erased_delta=erased_delta,
        )
        return rec

    def get_atlas(self, num_clusters: int = 48) -> ModelClusterAtlas:
        if self._atlas is not None and self._atlas.num_clusters == num_clusters:
            return self._atlas
        p = self.cache_dir / f"atlas_k{num_clusters}.json"
        if p.exists():
            self._atlas = ModelClusterAtlas.model_validate_json(p.read_text(encoding="utf-8"))
        else:
            self._atlas = build_model_atlas(str(self.model_dir), num_clusters=num_clusters)
            p.write_text(self._atlas.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return self._atlas

    def stage_global_cluster(self, cluster_id: int, action: ClusterAction) -> None:
        if action == ClusterAction.KEEP:
            self.staged_global.pop(cluster_id, None)
        else:
            self.staged_global[cluster_id] = action

    def global_stage_totals(self) -> tuple[int, int, int]:
        if self._atlas is None:
            return 0, 0, 0
        by_id = {c.cluster_id: c for c in self._atlas.clusters}
        rec = sum(by_id[cid].size for cid, a in self.staged_global.items() if a == ClusterAction.RECYCLE)
        era = sum(by_id[cid].size for cid, a in self.staged_global.items() if a == ClusterAction.SUBSPACE_ERASE)
        layers_hit: set[int] = set()
        for cid in self.staged_global:
            layers_hit.update(by_id[cid].neurons_by_layer.keys())
        return rec, era, len(layers_hit)

    def apply_global_staged(self) -> TransactionRecord:
        """Apply all staged global clusters across all layers they span as one rollbackable transaction."""
        if self.pending is not None:
            raise RuntimeError("Commit or rollback the current pending transaction first.")
        if not self.staged_global or self._atlas is None:
            raise ValueError("No global clusters staged.")

        self._ensure_model()
        assert self._model is not None and self._layers is not None and self._tok is not None
        by_id = {c.cluster_id: c for c in self._atlas.clusters}

        # Collect per-layer recycle & erase indices + cluster-specific subspace erase directions
        rec_by_layer: dict[int, list[int]] = {}
        era_by_layer: dict[int, list[tuple[list[int], list[str]]]] = {}
        touched_layers: set[int] = set()
        for cid, act in self.staged_global.items():
            c = by_id[cid]
            for l, idxs in c.neurons_by_layer.items():
                touched_layers.add(l)
                if act == ClusterAction.RECYCLE:
                    rec_by_layer.setdefault(l, []).extend(idxs)
                elif act == ClusterAction.SUBSPACE_ERASE:
                    era_by_layer.setdefault(l, []).append((idxs, c.centroid_tokens))

        ordered_layers = sorted(touched_layers)
        backups = [self._backup_layer(l) for l in ordered_layers]

        # Precompute unembedding vectors for cluster-specific subspace erasure (target-free!)
        assert self._lm_head_w is not None and self._norm_w is not None
        U = self._lm_head_w * self._norm_w.unsqueeze(0)

        # Optional held-out eval KL before/after
        eval_rollouts: list[torch.Tensor] = []
        teacher_eval_logits: list[torch.Tensor] = []
        cfg = self.mitigations
        if cfg.heal_mode != HealMode.NONE:
            _, eval_seeds = _build_self_gen_prompts(None, None)
            eval_rollouts = _self_generate_rollouts(self._model, self._tok, eval_seeds[:3], max_new_tokens=cfg.gen_tokens)
            with torch.no_grad():
                teacher_eval_logits = [self._model(r).logits[0].detach().float() for r in eval_rollouts]

        total_rec = 0
        total_era = 0
        with torch.no_grad():
            for l in ordered_layers:
                mlp = self._layers[l].mlp
                r_idxs = sorted(set(rec_by_layer.get(l, [])))
                if r_idxs:
                    mlp.down_proj.weight[:, r_idxs] = 0.0
                    mlp.gate_proj.weight[r_idxs] = torch.randn_like(mlp.gate_proj.weight[r_idxs]) * 0.02
                    mlp.up_proj.weight[r_idxs] = torch.randn_like(mlp.up_proj.weight[r_idxs]) * 0.02
                    total_rec += len(r_idxs)
                for e_idxs, c_toks in era_by_layer.get(l, []):
                    # Build cluster-specific unembedding subspace from its own centroid tokens!
                    t_ids = [
                        self._tok.encode(t, add_special_tokens=False)[0]
                        for t in c_toks
                        if len(self._tok.encode(t, add_special_tokens=False)) == 1
                    ]
                    if t_ids:
                        V_c = U[t_ids]  # [m, d_model]
                        Q_c, _ = torch.linalg.qr(V_c.T)
                        W_sub = mlp.down_proj.weight[:, e_idxs].float()
                        W_sub = W_sub - Q_c @ (Q_c.T @ W_sub)
                        mlp.down_proj.weight[:, e_idxs] = W_sub.to(mlp.down_proj.weight.dtype)
                        total_era += len(e_idxs)

        kl_after = None
        if eval_rollouts and teacher_eval_logits:
            with torch.no_grad():
                post_logits = [self._model(r).logits[0].detach().float() for r in eval_rollouts]
            kl_after = round(_mean_kl(teacher_eval_logits, post_logits), 5)

        rec = TransactionRecord(
            tx_id=self._next_tx_id,
            layer=-1,
            status="pending",
            recycled=total_rec,
            erased=total_era,
            merged=0,
            shielded=0,
            heal_mode=cfg.heal_mode,
            eval_kl_before=kl_after,
            eval_kl_after=kl_after,
            note=f"{len(self.staged_global)} global clusters across {len(ordered_layers)} layers",
        )
        self._next_tx_id += 1
        self.pending = _Transaction(record=rec, backups=backups, freed_in_layer=(-1, set()))
        return rec

    def get_overview(self) -> ModelOverview:
        if self._overview is not None:
            return self._overview
        p = self.cache_dir / f"overview_step{self.overview_step}.json"
        if p.exists():
            self._overview = ModelOverview.model_validate_json(p.read_text(encoding="utf-8"))
        else:
            self._overview = overview_model(str(self.model_dir), evict_target=self.target, layer_step=self.overview_step)
            p.write_text(self._overview.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return self._overview

    def get_layer_data(self, layer: int, num_clusters: int = 48) -> tuple[LayerReport, list[tuple[MergePair, bool]]]:
        if layer not in self._reports:
            rp = self.cache_dir / f"l{layer:02d}_report.json"
            if rp.exists():
                self._reports[layer] = LayerReport.model_validate_json(rp.read_text(encoding="utf-8"))
            else:
                lw = load_layer_weights(str(self.model_dir), layer)
                rep = analyze_layer(lw, num_clusters=num_clusters, evict_target=self.target)
                rp.write_text(rep.model_dump_json(indent=2) + "\n", encoding="utf-8")
                self._reports[layer] = rep

        if layer not in self._merges:
            mp_path = self.cache_dir / f"l{layer:02d}_merges.json"
            if mp_path.exists():
                self._merges[layer] = MergePlan.model_validate_json(mp_path.read_text(encoding="utf-8"))
            else:
                lw = load_layer_weights(str(self.model_dir), layer, include_gate_up=True)
                mp = find_merge_candidates(lw, self._reports[layer], max_pairs=20)
                mp.save(mp_path)
                self._merges[layer] = mp

        if layer not in self.staged_neurons:
            self.staged_neurons[layer] = {}
        if layer not in self.staged_merges:
            self.staged_merges[layer] = [(p.model_copy(), False) for p in self._merges[layer].pairs]
        return self._reports[layer], self.staged_merges[layer]

    def stage_cluster(self, layer: int, cluster_id: int, action: ClusterAction) -> int:
        rep = self._reports[layer]
        c = next(x for x in rep.clusters if x.cluster_id == cluster_id)
        bag = self.staged_neurons.setdefault(layer, {})
        for ni in c.neuron_indices:
            if action == ClusterAction.KEEP:
                bag.pop(ni, None)
            else:
                bag[ni] = action
        return len(c.neuron_indices)

    def smart_split_cluster(self, layer: int, cluster_id: int) -> tuple[int, int]:
        """Auto-recycle pure-evict sample neurons and subspace-erase the rest of the cluster."""
        rep = self._reports[layer]
        c = next(x for x in rep.clusters if x.cluster_id == cluster_id)
        bag = self.staged_neurons.setdefault(layer, {})
        pure_set = {n.index for n in c.sample_neurons if n.cjk_share >= 0.80 and n.latin_share <= 0.10}
        rec_cnt = 0
        era_cnt = 0
        for ni in c.neuron_indices:
            if ni in pure_set:
                bag[ni] = ClusterAction.RECYCLE
                rec_cnt += 1
            else:
                bag[ni] = ClusterAction.SUBSPACE_ERASE
                era_cnt += 1
        return rec_cnt, era_cnt

    def stage_summary(self, layer: int) -> tuple[int, int, int]:
        bag = self.staged_neurons.get(layer, {})
        rec = sum(1 for v in bag.values() if v == ClusterAction.RECYCLE)
        era = sum(1 for v in bag.values() if v == ClusterAction.SUBSPACE_ERASE)
        mrg = sum(1 for _, on in self.staged_merges.get(layer, []) if on)
        return rec, era, mrg

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model, self._layers, self._norm_w, self._lm_head_w, self._tok = _load_full_model(str(self.model_dir))

    def _get_evict_subspace(self) -> torch.Tensor:
        if self._evict_Q is not None:
            return self._evict_Q
        assert self._lm_head_w is not None and self._norm_w is not None and self._tok is not None
        U = self._lm_head_w * self._norm_w.unsqueeze(0)
        cjk_mask, nl_mask, lat_mask, _ = build_vocab_masks(self._tok, U.shape[0])
        ev_mask = cjk_mask if self.target == EvictTarget.CJK else nl_mask
        ev_vecs = U[ev_mask > 0.5]
        en_vecs = U[lat_mask > 0.5]
        ev_sub = ev_vecs[:: max(1, ev_vecs.shape[0] // 512)]
        en_sub = en_vecs[:: max(1, en_vecs.shape[0] // 512)]
        _, _, Vt_en = torch.linalg.svd(en_sub, full_matrices=False)
        Q_en = Vt_en[:64].T
        ev_clean = ev_sub - (ev_sub @ Q_en) @ Q_en.T
        _, _, Vt_ev = torch.linalg.svd(ev_clean, full_matrices=False)
        self._evict_Q = Vt_ev[:48].T
        return self._evict_Q

    def discover_behavior(
        self,
        pos_mood: str,
        neg_mood: str = DEFAULT_NEG_MOOD,
        strength: float = 1.5,
        progress_cb: Callable[[str], None] | None = None,
    ) -> BehaviorCircuit:
        """
        Zero-corpus Contrastive Behavioral Discovery across FULL generated replies:
        Generates short contrastive replies (~14 tokens) on 3 neutral topics comparing
        `pos_mood` vs `neg_mood` (defaults to neutral, matter-of-fact), averages MLP
        inputs/outputs across ALL response tokens, and appends a new BehaviorCircuit
        to the mixer for pure rank-1 weight folding into mlp.down_proj.weight.
        """
        if self.is_moe:
            raise RuntimeError(
                "PyTorch activation-based mood discovery is for dense models; for sparse MoE checkpoints use `neuron-reclaim gguf-fold --pos ... --neg ...`."
            )
        neg_clean = neg_mood.strip() or DEFAULT_NEG_MOOD
        if progress_cb and self._model is None:
            progress_cb(f"Loading {self.model_ref} into RAM (one-time)...")
        self._ensure_model()
        assert self._model is not None and self._layers is not None and self._tok is not None
        assert self._lm_head_w is not None and self._norm_w is not None

        topics = [
            "Explain how a toaster works.",
            "Describe a typical morning.",
            "Give advice on organizing a desk.",
        ]
        n_layers = len(self._layers)
        target_layers = list(range(max(1, n_layers // 3), min(n_layers - 2, (3 * n_layers) // 4)))

        def encode_chat(msg: str) -> torch.Tensor:
            try:
                txt = self._tok.apply_chat_template(
                    [{"role": "user", "content": msg}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                return self._tok(txt, return_tensors="pt", add_special_tokens=False).input_ids
            except Exception:
                return self._tok.encode(msg, return_tensors="pt")

        mlp_in: dict[int, list[torch.Tensor]] = {l: [] for l in target_layers}
        mlp_out: dict[int, list[torch.Tensor]] = {l: [] for l in target_layers}

        def run_batch(msgs: list[str], tag: str) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
            # 1) Generate short ~14-token replies so we capture sustained mid-sentence mood
            seqs: list[tuple[torch.Tensor, int]] = []
            with torch.no_grad():
                for idx, m in enumerate(msgs, 1):
                    if progress_cb:
                        progress_cb(f"Generating {tag} rollout {idx}/{len(msgs)}...")
                    p_ids = encode_chat(m)
                    p_len = int(p_ids.shape[1])
                    full_ids = self._model.generate(
                        p_ids,
                        max_new_tokens=14,
                        do_sample=False,
                        pad_token_id=self._tok.eos_token_id,
                    )
                    seqs.append((full_ids, max(0, p_len - 1)))

            if progress_cb:
                progress_cb(f"Capturing MLP activations ({tag})...")

            # 2) Single forward pass per sequence, averaging over all assistant-response tokens
            for l in target_layers:
                mlp_in[l].clear()
                mlp_out[l].clear()
            cur_start = [0]
            handles = []
            for l in target_layers:
                def _mk(li):
                    def hk(mod, inp, out):
                        s = cur_start[0]
                        mlp_in[li].append(inp[0].detach()[0, s:].float().mean(dim=0))
                        mlp_out[li].append(out.detach()[0, s:].float().mean(dim=0))
                    return hk
                handles.append(self._layers[l].mlp.down_proj.register_forward_hook(_mk(l)))
            with torch.no_grad():
                for full_ids, s_pos in seqs:
                    cur_start[0] = s_pos
                    _ = self._model(full_ids)
            for h in handles:
                h.remove()
            mean_in = {l: torch.stack(mlp_in[l], dim=0).mean(dim=0) for l in target_layers}
            mean_out = {l: torch.stack(mlp_out[l], dim=0).mean(dim=0) for l in target_layers}
            return mean_in, mean_out

        pos_msgs = [f"Respond in a {pos_mood} tone: {t}" for t in topics]
        neg_msgs = [f"Respond in a {neg_clean} tone: {t}" for t in topics]
        pos_in, pos_out = run_batch(pos_msgs, f"+'{pos_mood}'")
        neg_in, neg_out = run_batch(neg_msgs, f"-'{neg_clean}'")

        # Build per-layer (v_mood, mean_neg_act) for pure rank-1 weight folding into down_proj.weight!
        steer_vecs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        top_neurons: list[tuple[float, int, int, float]] = []
        for l in target_layers:
            diff_out = pos_out[l] - neg_out[l]  # [d_model]
            steer_vecs[l] = (diff_out, neg_in[l])
            diff_in = pos_in[l] - neg_in[l]  # [d_mlp]
            vals, idxs = torch.topk(diff_in.abs(), 4)
            for v, i in zip(vals.tolist(), idxs.tolist()):
                top_neurons.append((v, l, i, float(diff_in[i].item())))

        top_neurons.sort(key=lambda x: -x[0])
        U = self._lm_head_w * self._norm_w.unsqueeze(0)
        summary: list[tuple[int, int, float, list[str]]] = []
        for _, l, ni, signed_d in top_neurons[:14]:
            v_d = self._layers[l].mlp.down_proj.weight[:, ni].detach().float()
            sign_v = v_d if signed_d >= 0 else -v_d
            _, ti = torch.topk(sign_v @ U.T, 5)
            toks = [self._tok.decode([t]) for t in ti.tolist()]
            summary.append((l, ni, round(signed_d, 2), toks))

        circuit = BehaviorCircuit(
            circuit_id=self._next_circuit_id,
            pos_label=pos_mood,
            neg_label=neg_clean,
            strength=strength,
            layer_vecs=steer_vecs,
            top_neurons=summary,
            enabled=True,
        )
        self._next_circuit_id += 1
        self.behavior_circuits.append(circuit)
        return circuit

    def apply_pending_behavior(self) -> TransactionRecord:
        """
        Pure weight-based multi-mood behavioral mixer (zero runtime hooks!):
        Combines all enabled BehaviorCircuits (supporting positive and negative strengths)
        and folds their summed rank-1 updates directly into mlp.down_proj.weight.
        If a behavior transaction is already pending, automatically replaces it so
        tweaking sliders and pressing A is a single-key live re-apply.
        """
        if self.pending is not None:
            if self.pending.is_behavior:
                self.rollback()
            else:
                raise RuntimeError("Commit or rollback the current pending neuron transaction first.")
        active = [c for c in self.behavior_circuits if c.enabled and abs(c.strength) > 1e-6]
        if not active:
            raise ValueError("No enabled behavior circuits with non-zero strength (press n to add one).")
        assert self._layers is not None

        touched: set[int] = set()
        for c in active:
            touched.update(c.layer_vecs.keys())
        ordered = sorted(touched)
        backups = [self._backup_layer(l) for l in ordered]

        with torch.no_grad():
            for l in ordered:
                mlp = self._layers[l].mlp
                W_new = mlp.down_proj.weight.float()
                for c in active:
                    if l not in c.layer_vecs:
                        continue
                    v_mood, a_bar = c.layer_vecs[l]
                    denom = float(a_bar.norm().pow(2).item()) + 1e-6
                    W_new = W_new + (c.strength / denom) * torch.outer(v_mood, a_bar)
                mlp.down_proj.weight.copy_(W_new.to(mlp.down_proj.weight.dtype))

        mix_desc = " + ".join(f"{c.short_label}@{c.strength:+.2f}x" for c in active)
        rec = TransactionRecord(
            tx_id=self._next_tx_id,
            layer=-1,
            status="pending",
            pruned=0,
            recycled=0,
            erased=0,
            merged=0,
            shielded=len(ordered),
            heal_mode=HealMode.NONE,
            note=f"behavior:{mix_desc}",
        )
        self._next_tx_id += 1
        self.pending = _Transaction(
            record=rec,
            backups=backups,
            freed_in_layer=(-1, set()),
            is_behavior=True,
            behavior_layers=set(ordered),
        )
        return rec

    def _backup_layer(self, layer: int) -> _LayerBackup:
        assert self._layers is not None
        mlp = self._layers[layer].mlp
        return _LayerBackup(
            layer=layer,
            gate=mlp.gate_proj.weight.detach().clone(),
            up=mlp.up_proj.weight.detach().clone(),
            down=mlp.down_proj.weight.detach().clone(),
        )

    def _restore_backups(self, backups: list[_LayerBackup]) -> None:
        assert self._layers is not None
        with torch.no_grad():
            for b in backups:
                mlp = self._layers[b.layer].mlp
                mlp.gate_proj.weight.copy_(b.gate)
                mlp.up_proj.weight.copy_(b.up)
                mlp.down_proj.weight.copy_(b.down)

    def apply_layer(self, layer: int) -> TransactionRecord:
        if self.pending is not None:
            raise RuntimeError("Commit or rollback the current pending transaction first.")
        rec_cnt, era_cnt, mrg_cnt = self.stage_summary(layer)
        if rec_cnt == 0 and era_cnt == 0 and mrg_cnt == 0:
            raise ValueError(f"Nothing staged on layer {layer}.")

        self._ensure_model()
        assert self._model is not None and self._layers is not None and self._tok is not None
        mlp = self._layers[layer].mlp
        cfg = self.mitigations

        # Backup touched layer(s) for instant rollback
        backups = [self._backup_layer(layer)]
        if cfg.downstream_gate_shield and mrg_cnt > 0 and layer + 1 < len(self._layers):
            backups.append(self._backup_layer(layer + 1))

        # Optional self-generated calibration rollouts BEFORE modifying weights
        fit_rollouts: list[torch.Tensor] = []
        eval_rollouts: list[torch.Tensor] = []
        Y_fit_clean = Y_eval_clean = None
        teacher_fit_logits: list[torch.Tensor] = []
        teacher_eval_logits: list[torch.Tensor] = []

        active_pairs = [p for p, on in self.staged_merges.get(layer, []) if on]
        mp_temp = MergePlan(model_path=str(self.model_dir), layer=layer, pairs=active_pairs) if active_pairs else None
        if cfg.heal_mode != HealMode.NONE:
            fit_seeds, eval_seeds = _build_self_gen_prompts(None, mp_temp)
            fit_rollouts = _self_generate_rollouts(self._model, self._tok, fit_seeds[:6], max_new_tokens=cfg.gen_tokens)
            eval_rollouts = _self_generate_rollouts(self._model, self._tok, eval_seeds[:3], max_new_tokens=cfg.gen_tokens)
            _, Y_fit_clean, teacher_fit_logits = _capture_mlp_and_logits(self._model, mlp, fit_rollouts)
            _, Y_eval_clean, teacher_eval_logits = _capture_mlp_and_logits(self._model, mlp, eval_rollouts)

        bag = self.staged_neurons.get(layer, {})
        rec_idxs = [i for i, a in bag.items() if a == ClusterAction.RECYCLE]
        era_idxs = [i for i, a in bag.items() if a == ClusterAction.SUBSPACE_ERASE]
        freed: set[int] = set(self.freed_by_layer.get(layer, set()))
        newly_freed: set[int] = set()

        with torch.no_grad():
            if rec_idxs:
                mlp.down_proj.weight[:, rec_idxs] = 0.0
                mlp.gate_proj.weight[rec_idxs] = torch.randn_like(mlp.gate_proj.weight[rec_idxs]) * 0.02
                mlp.up_proj.weight[rec_idxs] = torch.randn_like(mlp.up_proj.weight[rec_idxs]) * 0.02
                newly_freed.update(rec_idxs)
            if era_idxs:
                Q_ev = self._get_evict_subspace()
                W_sub = mlp.down_proj.weight[:, era_idxs].float()
                W_sub = W_sub - Q_ev @ (Q_ev.T @ W_sub)
                mlp.down_proj.weight[:, era_idxs] = W_sub.to(mlp.down_proj.weight.dtype)

        saved_downs: list[tuple[torch.Tensor, torch.Tensor]] = []
        actual_merged = 0
        with torch.no_grad():
            for pair in active_pairs:
                a, b = pair.keep_neuron, pair.free_neuron
                if a in freed or b in freed or a in newly_freed or b in newly_freed:
                    continue
                ga = mlp.gate_proj.weight[a].float()
                gb = mlp.gate_proj.weight[b].float()
                ua = mlp.up_proj.weight[a].float()
                ub = mlp.up_proj.weight[b].float()
                da = mlp.down_proj.weight[:, a].float()
                db = mlp.down_proj.weight[:, b].float()
                saved_downs.append((da.clone(), db.clone()))
                if cfg.orthogonalize_inputs:
                    gb = gb - (torch.dot(gb, ga) / (ga.norm().pow(2) + 1e-8)) * ga
                    ub = ub - (torch.dot(ub, ua) / (ua.norm().pow(2) + 1e-8)) * ua
                s = float(pair.sign) if cfg.antipodal_sign else 1.0
                mlp.gate_proj.weight[a] = (ga + gb).to(mlp.gate_proj.weight.dtype)
                mlp.up_proj.weight[a] = (ua + s * ub).to(mlp.up_proj.weight.dtype)
                mlp.down_proj.weight[:, a] = (da + s * db).to(mlp.down_proj.weight.dtype)
                mlp.gate_proj.weight[b].zero_()
                mlp.up_proj.weight[b].zero_()
                mlp.down_proj.weight[:, b].zero_()
                newly_freed.add(b)
                actual_merged += 1

        shielded = 0
        if cfg.downstream_gate_shield and saved_downs:
            shielded = _apply_downstream_gate_shield(self._layers, layer, saved_downs, cfg.shield_top_k)

        kl_before = kl_after = None
        all_freed = freed | newly_freed
        if cfg.heal_mode != HealMode.NONE and Y_fit_clean is not None and Y_eval_clean is not None:
            d_mlp = mlp.down_proj.in_features
            active_mask = torch.ones(d_mlp, dtype=torch.bool)
            for f in all_freed:
                active_mask[f] = False
            _, _, eval_post = _capture_mlp_and_logits(self._model, mlp, eval_rollouts)
            kl_before = round(_mean_kl(teacher_eval_logits, eval_post), 5)

            if cfg.heal_mode in (HealMode.SELF_OBS, HealMode.OBS_THEN_KL):
                X_fit_post, Y_fit_post, _ = _capture_mlp_and_logits(self._model, mlp, fit_rollouts)
                X_act = X_fit_post[:, active_mask]
                R = Y_fit_clean - Y_fit_post
                G = X_act @ X_act.T
                lam = cfg.obs_ridge * float(G.diag().mean().item() + 1e-6)
                alpha = torch.linalg.solve(G + lam * torch.eye(G.shape[0]), R)
                dW_T = X_act.T @ alpha
                with torch.no_grad():
                    W_new = mlp.down_proj.weight.float()
                    W_new[:, active_mask] += dW_T.T
                    mlp.down_proj.weight.copy_(W_new.to(mlp.down_proj.weight.dtype))
                _, _, eval_obs = _capture_mlp_and_logits(self._model, mlp, eval_rollouts)
                kl_after = round(_mean_kl(teacher_eval_logits, eval_obs), 5)

            if cfg.heal_mode in (HealMode.SELF_KL, HealMode.OBS_THEN_KL):
                _run_self_kl_heal(
                    self._model, mlp, active_mask, fit_rollouts, teacher_fit_logits, cfg.kl_steps, cfg.kl_lr
                )
                _, _, eval_kl = _capture_mlp_and_logits(self._model, mlp, eval_rollouts)
                kl_after = round(_mean_kl(teacher_eval_logits, eval_kl), 5)

        rec = TransactionRecord(
            tx_id=self._next_tx_id,
            layer=layer,
            status="pending",
            recycled=len(rec_idxs),
            erased=len(era_idxs),
            merged=actual_merged,
            shielded=shielded,
            heal_mode=cfg.heal_mode,
            eval_kl_before=kl_before,
            eval_kl_after=kl_after,
        )
        self._next_tx_id += 1
        self.pending = _Transaction(record=rec, backups=backups, freed_in_layer=(layer, newly_freed))
        return rec

    def commit(self, note: str = "") -> TransactionRecord:
        if self.pending is None:
            raise RuntimeError("No pending transaction to commit.")
        tx = self.pending
        tx.record.status = "committed"
        if note:
            tx.record.note = note
        layer, newly_freed = tx.freed_in_layer
        if tx.is_behavior:
            label = tx.record.note.removeprefix("behavior:")
            self.active_steers.append(label)
            for c in self.behavior_circuits:
                c.enabled = False
            self.behavior_layers_committed.update(tx.behavior_layers)
            self.committed.append(tx)
            self.pending = None
            return tx.record
        if layer >= 0:
            self.freed_by_layer.setdefault(layer, set()).update(newly_freed)
            self.staged_neurons[layer] = {}
            self.staged_merges[layer] = [(p, False) for p, _ in self.staged_merges.get(layer, [])]
        else:
            applied_keys: set[tuple[int, int]] = set()
            for l, ns in tx.pruned_delta.items():
                self.pruned_by_layer.setdefault(l, set()).update(ns)
                applied_keys.update((l, ni) for ni in ns)
            for l, ns in tx.emptied_delta.items():
                self.emptied_by_layer.setdefault(l, set()).update(ns)
                applied_keys.update((l, ni) for ni in ns)
            for l, ns in tx.boosted_delta.items():
                self.boosted_by_layer.setdefault(l, set()).update(ns)
                applied_keys.update((l, ni) for ni in ns)
            for l, ns in tx.erased_delta.items():
                self.erased_by_layer.setdefault(l, set()).update(ns)
                applied_keys.update((l, ni) for ni in ns)
            for key in applied_keys:
                self.staged_tree.pop(key, None)
            self.staged_global.clear()
        self.committed.append(tx)
        self.pending = None
        return tx.record

    def rollback(self) -> TransactionRecord:
        if self.pending is not None:
            tx = self.pending
            self._restore_backups(tx.backups)
            self.pending = None
            return tx.record
        if self.committed:
            tx = self.committed.pop()
            if tx.is_behavior and self.active_steers:
                self.active_steers.pop()
            self._restore_backups(tx.backups)
            layer, newly_freed = tx.freed_in_layer
            if layer in self.freed_by_layer:
                self.freed_by_layer[layer] -= newly_freed
            for l, ns in tx.pruned_delta.items():
                if l in self.pruned_by_layer:
                    self.pruned_by_layer[l] -= ns
            for l, ns in tx.emptied_delta.items():
                if l in self.emptied_by_layer:
                    self.emptied_by_layer[l] -= ns
            for l, ns in tx.boosted_delta.items():
                if l in self.boosted_by_layer:
                    self.boosted_by_layer[l] -= ns
            for l, ns in tx.erased_delta.items():
                if l in self.erased_by_layer:
                    self.erased_by_layer[l] -= ns
            self.behavior_layers_committed = {
                l for prev in self.committed for l in prev.behavior_layers
            }
            return tx.record
        raise RuntimeError("Nothing to rollback.")

    def chat_once(self, user_text: str, max_new_tokens: int = 48) -> str:
        if self.api_backend is not None:
            return self.api_backend.chat_once(user_text, max_new_tokens=max_new_tokens)
        self._ensure_model()
        assert self._model is not None and self._tok is not None
        try:
            prompt = self._tok.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = self._tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        except Exception:
            ids = self._tok.encode(user_text, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        new_toks = out[0, ids.shape[1] :]
        return self._tok.decode(new_toks, skip_special_tokens=True).strip()
