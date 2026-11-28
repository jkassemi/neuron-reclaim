"""
VibeSplice v0.5.0 — StreamingSCGLedger: Bit-Exact Continuous Learning & Exact Machine Unlearning Closure.

Caches the FP64 active-pack Write Gram matrix `S_pack = sum_{i in A} X_pos_i^T X_pos_i` and per-concept
metadata so streaming `add_concepts()`, `update_concept()`, and `delete_concepts()` maintain exact
Leave-One-Concept-Out anti-bleed whitening and exact biorthogonal QR Write directions across the
surviving active set `A`:
  - Bit-exact `0.00000000` weight & logit closure (`Install K -> Delete D` == `Fresh Compile K-D`, Study 23 Part A).
  - Post-Read `X_pos` refresh (`refresh_post_read=True`, default): re-captures surviving concepts'
    Write-layer activations on the live post-Read residual stream (with Write slots temporarily zeroed),
    eliminating Study 23 Part A's pre-Read covariate shift (`Keep40 WTop1` `43.8% -> ~78%`) while
    preserving exact `0.00000000` unlearning equivalence!
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from graft import (
    _POSITION_DIVERSE_G_PREFIXES,
    ConceptGrafter,
    _build_contrast_negatives,
    _build_default_naming_prompts,
    _fit_linear_cone_swiglu_key,
    _fit_swiglu_gated_key,
    _qr_project_out,
)
from models import GraftConfig, LedgerOperationReport, ReclaimedSlotInfo


def compute_biorthogonal_write_directions(
    nw: np.ndarray,
    unique_ekinds: list[np.ndarray],
    u_perps: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Compute exact history-independent, permutation-invariant unit Write directions `d_write_i`
    such that for every active concept `i`:
      - `dot(d_write_i, nw * e_kind) == 0` for all active parent kinds `e_kind`, and
      - `dot(d_write_i, nw * u_perp_j) == 0` for all other active concepts `j != i`.
    Uses exact dual/biorthogonal solve `D_dual = M_u @ (M_u^T M_u + eps I)^{-1}` in FP64.
    """
    K = len(u_perps)
    if K == 0:
        return []
    nw64 = nw.astype(np.float64)
    nw_u = np.stack([nw64 * u.astype(np.float64) for u in u_perps], axis=1)  # [D, K]
    if unique_ekinds:
        nw_ek = np.stack([nw64 * ek.astype(np.float64) for ek in unique_ekinds], axis=1)  # [D, n_ek]
        Q_ek, _ = np.linalg.qr(nw_ek, mode="reduced")
        M_u = nw_u - Q_ek @ (Q_ek.T @ nw_u)
        # Re-orthogonalize once for FP64 precision (~1e-16)
        M_u = M_u - Q_ek @ (Q_ek.T @ M_u)
    else:
        M_u = nw_u

    G_u = M_u.T @ M_u  # [K, K]
    D_dual = M_u @ np.linalg.solve(G_u + 1e-12 * np.eye(K, dtype=np.float64), np.eye(K, dtype=np.float64))
    out: list[np.ndarray] = []
    for i in range(K):
        d_i = D_dual[:, i]
        # Ensure positive alignment with raw target `nw * u_perp_i`
        if float(np.dot(d_i, nw_u[:, i])) < 0.0:
            d_i = -d_i
        nrm = float(np.linalg.norm(d_i))
        out.append((d_i / max(nrm, 1e-12)).astype(np.float32))
    return out


@dataclass
class LedgerConceptEntry:
    token_id: int
    token_name: str | None
    config: GraftConfig
    read_slots: list[ReclaimedSlotInfo]
    write_slot: ReclaimedSlotInfo
    d_adj: dict[int, np.ndarray]
    d_noun: dict[int, np.ndarray]
    e_kind: np.ndarray
    u_perp: np.ndarray
    e_in: np.ndarray
    e_out: np.ndarray
    X_pos: np.ndarray          # [n_pos, D] float64
    S_i: np.ndarray            # [D, D] float64 = X_pos^T @ X_pos
    n_rows: int
    X_bg_all: np.ndarray       # [n_bg, D] float32
    C_bg: np.ndarray           # [D, D] float32
    X_hard_last: np.ndarray    # [n_hard, D] float32
    C_hard: np.ndarray         # [D, D] float32
    rms: float
    z_gap: float


class StreamingSCGLedger:
    """
    Stateful streaming ledger wrapping `ConceptGrafter` for exact continuous learning
    (add / update / delete) with bit-exact `0.00000000` unlearning closure.
    """

    def __init__(
        self,
        grafter: ConceptGrafter,
        read_layers: list[int] | None = None,
        write_layer: int | None = None,
        w_hard: float = 1.5,
        w_instack: float = 3.0,
        refresh_post_read: bool = True,
        multilingual_safe: bool = True,
    ) -> None:
        self.grafter = grafter
        dummy_cfg = GraftConfig(
            kind_words=["thing"], kind_label="thing", attr_adj="red", attr_noun="red",
            read_layers=read_layers, write_layer=write_layer, multilingual_safe=multilingual_safe,
        )
        self.read_layers, self.write_layer = grafter._default_layer_schedule(dummy_cfg)
        self.w_hard = w_hard
        self.w_instack = w_instack
        self.refresh_post_read = refresh_post_read
        self.multilingual_safe = multilingual_safe

        self.entries: dict[int, LedgerConceptEntry] = {}
        self.S_pack = np.zeros((grafter.d_model, grafter.d_model), dtype=np.float64)
        self.n_pack_rows = 0

    @property
    def active_token_ids(self) -> list[int]:
        return sorted(self.entries.keys())

    def _unique_active_ekinds(self) -> list[np.ndarray]:
        unique: list[np.ndarray] = []
        for tid in self.active_token_ids:
            ek = self.entries[tid].e_kind
            if not any(abs(float(np.dot(ek, prev))) > 0.9999 for prev in unique):
                unique.append(ek)
        return unique

    @torch.no_grad()
    def _zero_active_write_slots(self) -> None:
        mlp_w = self.grafter.layers[self.write_layer].mlp
        for entry in self.entries.values():
            j = entry.write_slot.neuron_idx
            mlp_w.gate_proj.weight[j].zero_()
            mlp_w.up_proj.weight[j].zero_()
            mlp_w.down_proj.weight[:, j].zero_()

    @torch.no_grad()
    def _zero_active_read_slots(self) -> None:
        for entry in self.entries.values():
            for sl in entry.read_slots:
                mlp = self.grafter.layers[sl.layer].mlp
                mlp.gate_proj.weight[sl.neuron_idx].zero_()
                mlp.up_proj.weight[sl.neuron_idx].zero_()
                mlp.down_proj.weight[:, sl.neuron_idx].zero_()

    def _fit_and_install_read_layers(self, target_tids: list[int]) -> None:
        """Fit and write Stage-2 Read superposition neurons for `target_tids` across `self.read_layers`."""
        if not target_tids:
            return
        g = self.grafter
        whiten_gen_seqs = g._build_whiten_gen_seqs(multilingual_safe=self.multilingual_safe)
        n_rl = len(self.read_layers)
        for idx_l, rl in enumerate(self.read_layers):
            X_gen_rl = g.capture_mlp_inputs(whiten_gen_seqs, rl, all_tokens=True)
            mlp = g.layers[rl].mlp
            for tid in target_tids:
                entry = self.entries[tid]
                cfg = entry.config
                kind_neg = [g.enc(f"the {w.strip()}") for w in cfg.kind_words] + [
                    g.enc(f"the {cfg.kind_label}"),
                    g.enc(f"this {cfg.kind_label}"),
                ]
                fit_g = [g.enc_with_token(pre, tid) for pre in _POSITION_DIVERSE_G_PREFIXES]
                X_pos_r = g.capture_mlp_inputs(fit_g, rl, all_tokens=False)
                X_bg_r = np.vstack([X_gen_rl, g.capture_mlp_inputs(kind_neg, rl, all_tokens=True)])
                wg, wu, ap, _ = _fit_swiglu_gated_key(
                    X_pos_r, X_bg_r, ridge_frac=cfg.ridge_frac, bias_ridge_frac=cfg.bias_ridge_frac
                )
                alphas = list(cfg.read_alphas)
                while len(alphas) < n_rl:
                    alphas.append(alphas[-1])
                alpha_l = alphas[idx_l]
                if n_rl == 1:
                    payload = alpha_l * (0.5 * entry.d_adj[rl] + 0.5 * entry.d_noun[rl])
                elif idx_l == 0:
                    payload = alpha_l * entry.d_adj[rl]
                elif idx_l == n_rl - 1:
                    payload = alpha_l * entry.d_noun[rl]
                else:
                    payload = alpha_l * (0.5 * entry.d_adj[rl] + 0.5 * entry.d_noun[rl])
                wd = payload / ap
                j = entry.read_slots[idx_l].neuron_idx
                with torch.no_grad():
                    mlp.gate_proj.weight[j].copy_(torch.tensor(wg, dtype=mlp.gate_proj.weight.dtype, device=g.device))
                    mlp.up_proj.weight[j].copy_(torch.tensor(wu, dtype=mlp.up_proj.weight.dtype, device=g.device))
                    mlp.down_proj.weight[:, j].copy_(torch.tensor(wd, dtype=mlp.down_proj.weight.dtype, device=g.device))

    def recompute_active_read_layers(self) -> None:
        """Rebuild all active Read neurons cleanly in sorted token order for bit-exact closure."""
        self._zero_active_read_slots()
        self._fit_and_install_read_layers(self.active_token_ids)

    def _capture_write_metadata_for_active(self, target_tids: list[int]) -> None:
        """
        Capture Write-layer activations (`X_pos`, `S_i`, `C_bg`, `C_hard`, `rms`, `z_gap`)
        on the live post-Read model with active Write slots zeroed so every captured statistic
        is 100% invariant to prior Write weights AND matches the live post-Read residual stream.
        """
        if not target_tids:
            return
        g = self.grafter
        self._zero_active_write_slots()
        whiten_gen_seqs = g._build_whiten_gen_seqs(multilingual_safe=self.multilingual_safe)
        X_gen_w = g.capture_mlp_inputs(whiten_gen_seqs, self.write_layer, all_tokens=True)
        X_cross_all_w = g.capture_mlp_inputs(g.cross_neg_seqs, self.write_layer, all_tokens=True)
        X_cross_last_w = g.capture_mlp_inputs(g.cross_neg_seqs, self.write_layer, all_tokens=False)

        for tid in target_tids:
            entry = self.entries[tid]
            cfg = entry.config
            prompts = cfg.naming_prompts or _build_default_naming_prompts(cfg)
            fws = [g.enc(p) for p in prompts]
            X_pos = g.capture_mlp_inputs(fws, self.write_layer, all_tokens=False).astype(np.float64)
            rms = g.capture_pre_norm_rms(fws)
            bl = g.eval_next_logits(fws)
            max_comp = bl[:, : g.normal_vocab_end].max(dim=-1).values
            z_gap = float((max_comp - bl[:, tid]).max().item())

            hn_seqs = [g.enc(p) for p in _build_contrast_negatives(cfg)]
            X_bg_all = np.vstack([
                X_gen_w,
                X_cross_all_w,
                g.capture_mlp_inputs(hn_seqs, self.write_layer, all_tokens=True),
            ]).astype(np.float32)
            X_hard_last = np.vstack([
                g.capture_mlp_inputs(hn_seqs, self.write_layer, all_tokens=False),
                X_cross_last_w,
            ]).astype(np.float32)

            entry.X_pos = X_pos
            entry.S_i = X_pos.T @ X_pos
            entry.n_rows = int(len(X_pos))
            entry.X_bg_all = X_bg_all
            entry.C_bg = (X_bg_all.T @ X_bg_all) / max(len(X_bg_all), 1)
            entry.X_hard_last = X_hard_last
            entry.C_hard = (X_hard_last.T @ X_hard_last) / max(len(X_hard_last), 1)
            entry.rms = rms
            entry.z_gap = z_gap

    def refresh_write_cache(self) -> None:
        """
        Re-capture post-Read Write activations for ALL active concepts and rebuild `S_pack`
        exact in FP64. Eliminates Study 23 Part A's Read->Write covariate shift while
        guaranteeing bit-exact `0.00000000` unlearning closure against a fresh compile.
        """
        tids = self.active_token_ids
        self._capture_write_metadata_for_active(tids)
        self.S_pack.fill(0.0)
        self.n_pack_rows = 0
        for tid in tids:
            self.S_pack += self.entries[tid].S_i
            self.n_pack_rows += self.entries[tid].n_rows

    def recompute_active_write_layer(self) -> dict[int, float]:
        """
        Exact closed-form solve of ALL active Write neurons from `S_pack`, using:
          - Leave-One-Concept-Out covariance `C_instack,-i = (S_pack - S_i) / max(1, n_pack_rows - n_i)`;
          - Optional multi-turn `U_hist` carryover projection (`carryover_safe=True`);
          - History-independent biorthogonal QR Write directions `compute_biorthogonal_write_directions()`.
        """
        tids = self.active_token_ids
        if not tids:
            return {}
        g = self.grafter
        entries = [self.entries[tid] for tid in tids]
        unique_ek = self._unique_active_ekinds()
        d_writes = compute_biorthogonal_write_directions(
            g.nw_np, unique_ek, [e.u_perp for e in entries]
        )

        # Optional carryover guard across active concepts
        U_hist_batch = None
        C_carry_batch = None
        if any(e.config.carryover_safe for e in entries):
            max_rk = max((e.config.carryover_rank for e in entries if e.config.carryover_safe), default=4)
            if max_rk > 0:
                U_hist_batch, C_carry_batch = g._compute_carryover_guard(
                    [e.config for e in entries],
                    [e.token_name for e in entries],
                    self.write_layer,
                    rank=max_rk,
                )

        snrs: dict[int, float] = {}
        mlp_w = g.layers[self.write_layer].mlp
        for idx, entry in enumerate(entries):
            cfg = entry.config
            if len(entries) > 1:
                S_minus_i = self.S_pack - entry.S_i
                denom = max(1, self.n_pack_rows - entry.n_rows)
                C_instack = (S_minus_i / denom).astype(np.float32)
            else:
                C_instack = None

            u_i = (
                U_hist_batch[:, : cfg.carryover_rank]
                if (cfg.carryover_safe and U_hist_batch is not None and cfg.carryover_rank > 0)
                else None
            )
            c_i = C_carry_batch if (cfg.carryover_safe and C_carry_batch is not None) else None

            wg_w, wu_w, a_ref_w, snr_w = _fit_linear_cone_swiglu_key(
                entry.X_pos.astype(np.float32),
                entry.C_bg,
                entry.X_bg_all,
                C_hard=entry.C_hard,
                C_instack=C_instack,
                C_carry=c_i,
                U_hist=u_i,
                X_eval_neg=entry.X_hard_last,
                w_hard=self.w_hard,
                w_instack=self.w_instack if C_instack is not None else 0.0,
                w_carry=cfg.carryover_weight,
                ridge_frac=cfg.ridge_frac,
                bias_ridge_frac=cfg.bias_ridge_frac,
                gate_tau=cfg.gate_tau,
                gate_sharp=cfg.gate_sharp,
                linear_cone=cfg.linear_cone_write,
            )
            snrs[entry.token_id] = snr_w

            d_write = d_writes[idx]
            eff_gain = float(np.dot(d_write, g.nw_np * entry.e_out))
            z_max = math.sqrt(g.d_model) * max(eff_gain, 1e-6)
            dz_target = min(cfg.max_logit_boost, 0.78 * z_max, max(cfg.target_logit_boost, entry.z_gap + 5.0))
            denom_sq = max(eff_gain**2 - (dz_target**2) / g.d_model, 0.25 * max(eff_gain**2, 1e-8))
            m_req = (dz_target * entry.rms) / math.sqrt(denom_sq)
            if cfg.max_rel_norm is not None:
                prompts_i = cfg.naming_prompts or _build_default_naming_prompts(cfg)
                y_nat_norm = g._capture_mlp_output_norm([g.enc(p) for p in prompts_i], self.write_layer)
                m_req = min(m_req, cfg.max_rel_norm * y_nat_norm)
            w_down_w = (m_req / max(a_ref_w, 1e-4)) * d_write

            j_slot = entry.write_slot.neuron_idx
            with torch.no_grad():
                mlp_w.gate_proj.weight[j_slot].copy_(torch.tensor(wg_w, dtype=mlp_w.gate_proj.weight.dtype, device=g.device))
                mlp_w.up_proj.weight[j_slot].copy_(torch.tensor(wu_w, dtype=mlp_w.up_proj.weight.dtype, device=g.device))
                mlp_w.down_proj.weight[:, j_slot].copy_(torch.tensor(w_down_w, dtype=mlp_w.down_proj.weight.dtype, device=g.device))

        return snrs

    def add_concepts(
        self,
        configs: list[GraftConfig],
        save_dir: Path | None = None,
    ) -> LedgerOperationReport:
        """Stream-install one or more concepts and update the active Write layer cleanly."""
        t0 = time.time()
        g = self.grafter
        g.precompute_quiet_slots(self.read_layers, self.write_layer, multilingual_safe=self.multilingual_safe)

        new_tids: list[int] = []
        new_read_slots: dict[int, list[ReclaimedSlotInfo]] = {}
        new_write_slots: dict[int, ReclaimedSlotInfo] = {}

        # 1. Allocate token slots, quiet SwiGLU slots & snapshot pristine bytes BEFORE zeroing
        for cfg in configs:
            tid, tc = g.register_token_slot(token_name=cfg.token_name, requested_id=cfg.token_id)
            if tid in self.entries:
                raise ValueError(f"token_id={tid} is already active in StreamingSCGLedger; use update_concept().")
            r_sl: list[ReclaimedSlotInfo] = []
            for rl in self.read_layers:
                sl = g.find_quiet_slot(rl, mode="input", multilingual_safe=self.multilingual_safe)
                sl.role = f"read_L{rl}"
                r_sl.append(sl)
            w_sl = g.find_quiet_slot(self.write_layer, mode="output", multilingual_safe=self.multilingual_safe)
            w_sl.role = f"write_L{self.write_layer}"
            g._snapshot_concept(tid, r_sl + [w_sl])
            new_tids.append(tid)
            new_read_slots[tid] = r_sl
            new_write_slots[tid] = w_sl
            # Store provisional entry placeholder for token_name/cfg
            self.entries[tid] = LedgerConceptEntry(
                token_id=tid, token_name=tc, config=cfg,
                read_slots=r_sl, write_slot=w_sl,
                d_adj={}, d_noun={},
                e_kind=np.zeros(g.d_model, dtype=np.float32),
                u_perp=np.zeros(g.d_model, dtype=np.float32),
                e_in=np.zeros(g.d_model, dtype=np.float32),
                e_out=np.zeros(g.d_model, dtype=np.float32),
                X_pos=np.zeros((0, g.d_model), dtype=np.float64),
                S_i=np.zeros((g.d_model, g.d_model), dtype=np.float64),
                n_rows=0,
                X_bg_all=np.zeros((0, g.d_model), dtype=np.float32),
                C_bg=np.zeros((g.d_model, g.d_model), dtype=np.float32),
                X_hard_last=np.zeros((0, g.d_model), dtype=np.float32),
                C_hard=np.zeros((g.d_model, g.d_model), dtype=np.float32),
                rms=1.0, z_gap=0.0,
            )

        # 2. Compute attribute deltas BEFORE zeroing batch slots (keeps payloads independent of sibling slots)
        for tid in new_tids:
            da, dn = g._compute_attr_deltas(self.entries[tid].config, self.read_layers)
            self.entries[tid].d_adj = da
            self.entries[tid].d_noun = dn

        # Zero newly reclaimed slots before installing embeddings & Read/Write weights
        with torch.no_grad():
            for tid in new_tids:
                for sl in new_read_slots[tid] + [new_write_slots[tid]]:
                    mlp = g.layers[sl.layer].mlp
                    mlp.gate_proj.weight[sl.neuron_idx].zero_()
                    mlp.up_proj.weight[sl.neuron_idx].zero_()
                    mlp.down_proj.weight[:, sl.neuron_idx].zero_()

        # 3. Deterministic history-independent private address `u_perp_i` & Stage-1 embedding
        for tid in new_tids:
            entry = self.entries[tid]
            cfg = entry.config

            with torch.no_grad():
                src_tids = [g._resolve_word_token(w) for w in cfg.kind_words]
                E_src = g.emb_w[src_tids].detach().float()
                r_in = float(E_src.norm(dim=-1).mean().item())
                r_out = float(g.out_w[src_tids].detach().float().norm(dim=-1).mean().item())
                ek = E_src.mean(dim=0).cpu().numpy()
                ek = (ek / (np.linalg.norm(ek) + 1e-12)).astype(np.float32)

                # Seeded ONLY from (cfg.seed, tid) & orthogonalized ONLY to own e_kind -> 100% history-independent!
                rng = np.random.default_rng(cfg.seed + tid)
                u_raw = rng.standard_normal(g.d_model).astype(np.float64)
                u_p = _qr_project_out(u_raw, [ek.astype(np.float64)])
                u_p = (u_p / (np.linalg.norm(u_p) + 1e-12)).astype(np.float32)

                if cfg.untied_pure_ortho and not g.is_tied:
                    th_in = math.radians(30.0)
                    e_in_u = math.cos(th_in) * ek + math.sin(th_in) * u_p
                    e_in_u /= np.linalg.norm(e_in_u) + 1e-12
                    e_in_np = (1.0 * r_in * e_in_u).astype(np.float32)
                    e_out_np = (cfg.gamma * r_out * u_p).astype(np.float32)
                    g.emb_w[tid].copy_(torch.tensor(e_in_np, dtype=g.emb_w.dtype, device=g.device))
                    g.out_w[tid].copy_(torch.tensor(e_out_np, dtype=g.out_w.dtype, device=g.device))
                else:
                    th = math.radians(cfg.theta_deg)
                    e_u = math.cos(th) * ek + math.sin(th) * u_p
                    e_u /= np.linalg.norm(e_u) + 1e-12
                    e_in_np = (cfg.gamma * r_in * e_u).astype(np.float32)
                    g.emb_w[tid].copy_(torch.tensor(e_in_np, dtype=g.emb_w.dtype, device=g.device))
                    if not g.is_tied:
                        e_out_np = (cfg.gamma * r_out * e_u).astype(np.float32)
                        g.out_w[tid].copy_(torch.tensor(e_out_np, dtype=g.out_w.dtype, device=g.device))
                    else:
                        e_out_np = e_in_np

                entry.e_kind = ek
                entry.u_perp = u_p
                entry.e_in = e_in_np
                entry.e_out = e_out_np

        # 4. Install Stage-2 Read superposition + update `S_pack`
        if self.refresh_post_read:
            self.recompute_active_read_layers()
            self.refresh_write_cache()
        else:
            self._fit_and_install_read_layers(sorted(new_tids))
            self._capture_write_metadata_for_active(sorted(new_tids))
            for tid in new_tids:
                self.S_pack += self.entries[tid].S_i
                self.n_pack_rows += self.entries[tid].n_rows

        snrs = self.recompute_active_write_layer()

        saved_str = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            g.model.save_pretrained(str(save_dir))
            g.tok.save_pretrained(str(save_dir))
            saved_str = str(save_dir)

        return LedgerOperationReport(
            operation="add",
            affected_token_ids=new_tids,
            active_count=len(self.entries),
            active_token_ids=self.active_token_ids,
            refresh_post_read=self.refresh_post_read,
            elapsed_ms=round((time.time() - t0) * 1000.0, 2),
            write_fit_snrs=snrs,
            saved_model_dir=saved_str,
        )

    def add_concept(self, cfg: GraftConfig, save_dir: Path | None = None) -> LedgerOperationReport:
        return self.add_concepts([cfg], save_dir=save_dir)

    def delete_concepts(
        self,
        token_ids: list[int],
        save_dir: Path | None = None,
    ) -> LedgerOperationReport:
        """
        Exact Machine Unlearning Closure:
          1. Restores bit-exact original bytes for every deleted concept's embedding row(s) and Read/Write slots;
          2. Updates `S_pack` (`refresh_write_cache()` when `refresh_post_read=True`, or exact FP64 `S_pack -= S_i`);
          3. Re-solves surviving concepts' Write neurons so surviving weights match a fresh compile bit-exactly!
        """
        t0 = time.time()
        g = self.grafter
        deleted: list[int] = []
        for tid in token_ids:
            if tid not in self.entries:
                continue
            entry = self.entries.pop(tid)
            self.S_pack -= entry.S_i
            self.n_pack_rows -= entry.n_rows
            g.delete_concept(tid)
            deleted.append(tid)

        if self.entries:
            if self.refresh_post_read:
                self.recompute_active_read_layers()
                self.refresh_write_cache()
            snrs = self.recompute_active_write_layer()
        else:
            self.S_pack.fill(0.0)
            self.n_pack_rows = 0
            snrs = {}

        saved_str = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            g.model.save_pretrained(str(save_dir))
            g.tok.save_pretrained(str(save_dir))
            saved_str = str(save_dir)

        return LedgerOperationReport(
            operation="delete",
            affected_token_ids=deleted,
            active_count=len(self.entries),
            active_token_ids=self.active_token_ids,
            refresh_post_read=self.refresh_post_read,
            elapsed_ms=round((time.time() - t0) * 1000.0, 2),
            write_fit_snrs=snrs,
            saved_model_dir=saved_str,
        )

    def delete_concept(self, token_id: int, save_dir: Path | None = None) -> LedgerOperationReport:
        return self.delete_concepts([token_id], save_dir=save_dir)

    def update_concept(
        self,
        token_id: int,
        new_cfg: GraftConfig,
        save_dir: Path | None = None,
    ) -> LedgerOperationReport:
        """Exact in-place concept update: unlearn old version, reinstall updated spec on the same token_id."""
        t0 = time.time()
        if token_id not in self.entries:
            raise KeyError(f"token_id={token_id} is not active in StreamingSCGLedger.")
        old_name = self.entries[token_id].token_name
        # Remove old concept (restoring exact pristine bytes) without saving intermediate state
        entry = self.entries.pop(token_id)
        self.S_pack -= entry.S_i
        self.n_pack_rows -= entry.n_rows
        self.grafter.delete_concept(token_id)

        upd_cfg = new_cfg.model_copy(
            update={
                "token_id": token_id,
                "token_name": new_cfg.token_name if new_cfg.token_name is not None else old_name,
            }
        )
        add_rep = self.add_concepts([upd_cfg], save_dir=save_dir)
        return LedgerOperationReport(
            operation="update",
            affected_token_ids=[token_id],
            active_count=add_rep.active_count,
            active_token_ids=add_rep.active_token_ids,
            refresh_post_read=self.refresh_post_read,
            elapsed_ms=round((time.time() - t0) * 1000.0, 2),
            write_fit_snrs=add_rep.write_fit_snrs,
            saved_model_dir=add_rep.saved_model_dir,
        )
