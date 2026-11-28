"""
VibeSplice — Generalized Multi-Turn Latching Detection (`audit_latching`) and
Zero-Gradient Repair (`extract_carryover_subspace`, `project_gate_orthogonal_to_history`,
`cap_relative_residual_norm`, `repair_swiglu_latching`, `repair_modular_adapter_latching`).

Addresses three cross-architecture failure modes of conditional weight edits (zero-gradient SCG,
modular MoE/LoRA adapters, and looped transformers) uncovered in Project Cortex Studies 26–27:
  1. History-Carryover Shift (`dz_hist` / `kl_ratio`):
       Single-turn gates align with global conversation topic features pulled via attention from
       earlier active turns (`H_on + q_off`), causing false firing on unrelated follow-up turns.
       Universal across BOTH dense (`Qwen3-4B`) and looped (`Inkling-2`) transformers.
  2. Self-Amplifying Feedback Gain (`dz_self` / `G_loop`):
       Early-pass injection inside looped cores (`p < P - 1`) poisons cross-pass controllers and
       drives runaway recurrent latching (`+6.99` logits across passes 0 -> 2). Prevented by
       terminal-pass scheduling (`core_active_passes = [P - 1]`) at compile/train time.
  3. Residual Bulldoze Ratio (`R_norm = ||Delta h_edit|| / ||h_native||`):
       Unbounded additive experts overpower native FFN residuals (`> 1.5x`–`9.4x`), causing
       out-of-distribution tunnel vision and multi-concept tail bleed. Prevented by relative norm
       budgeting (`cap_relative_residual_norm`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from models import LatchingAuditReport, LatchingLayerMetrics, LatchingRepairReport
from surgery import _load_full_model

DEFAULT_CARRYOVER_OFF_PROMPTS: list[str] = [
    "Write a Python function to reverse a string.",
    "What is the derivative of x^3 + 2x?",
    "Explain the difference between TCP and UDP.",
    "What is the capital of Brazil?",
    "How does DNS resolution work?",
    "Write a haiku about autumn rain.",
    "Explain what a mutex does in an operating system.",
    "What is the boiling point of water at sea level?",
    "The famous French spice blend is called",
    "In Norway, that scenic coastal harbor is called",
    "A beloved Italian culinary seasoning is called",
    "Visitors to Scotland often explore a harbor town named",
]

DEFAULT_DEV_CARRYOVER_OFF_PROMPTS: list[str] = [
    "Explain how recursion works in computer science.",
    "What is photosynthesis?",
    "What is 17 times 24?",
    "How do suspension bridges distribute weight?",
    "When did the French Revolution begin?",
    "Explain binary search in two sentences.",
    "The rare Canadian spice is called",
    "In Switzerland, that distinctive instrument is called",
    "The classic Australian dessert is named",
    "The best local shop in Houston is",
]


def extract_carryover_subspace(
    deltas: np.ndarray | torch.Tensor,
    rank: int = 8,
) -> tuple[np.ndarray, float]:
    """
    Extract the top-`rank` orthonormal history-carryover subspace `U_hist` ([D, k], float32)
    from paired multi-turn shift vectors `deltas = h(H_on + q_off) - h(q_off)` ([N, D]).

    Returns `(U_hist, explained_variance_ratio)`.
    """
    if isinstance(deltas, torch.Tensor):
        D_mat = deltas.detach().float().cpu().numpy()
    else:
        D_mat = np.asarray(deltas, dtype=np.float32)

    if D_mat.ndim != 2 or D_mat.shape[0] == 0 or rank <= 0:
        d = D_mat.shape[-1] if D_mat.ndim == 2 else 0
        return np.zeros((d, 0), dtype=np.float32), 0.0

    _, s, vh = np.linalg.svd(D_mat.astype(np.float64), full_matrices=False)
    k = min(int(rank), vh.shape[0])
    u_hist = vh[:k].T.astype(np.float32)
    tot_energy = float(np.sum(s ** 2)) + 1e-12
    exp_var = float(np.sum(s[:k] ** 2) / tot_energy)
    return u_hist, exp_var


def project_vector_orthogonal(
    vec: np.ndarray | torch.Tensor,
    u_hist: np.ndarray | torch.Tensor,
    preserve_norm: bool = True,
) -> np.ndarray:
    """Project `vec` ([D]) orthogonal to orthonormal columns `u_hist` ([D, k])."""
    v = vec.detach().float().cpu().numpy() if isinstance(vec, torch.Tensor) else np.asarray(vec, dtype=np.float32)
    u = u_hist.detach().float().cpu().numpy() if isinstance(u_hist, torch.Tensor) else np.asarray(u_hist, dtype=np.float32)
    if u.size == 0 or u.shape[1] == 0:
        return v.copy()
    v64 = v.astype(np.float64)
    u64 = u.astype(np.float64)
    v_perp = v64 - u64 @ (u64.T @ v64)
    if preserve_norm:
        n_orig = float(np.linalg.norm(v64))
        n_new = float(np.linalg.norm(v_perp))
        if n_new > 1e-12:
            v_perp = v_perp * (n_orig / n_new)
    return v_perp.astype(np.float32)


def project_gate_orthogonal_to_history(
    w_gate: np.ndarray | torch.Tensor,
    b_gate: float,
    u_hist: np.ndarray | torch.Tensor,
    x_neg_carryover: np.ndarray | torch.Tensor | None = None,
    neg_margin: float = 0.5,
    preserve_norm: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Zero-gradient post-hoc latching repair for a biased linear gate `z = w_gate @ x + b_gate`:
      1. Projects `w_gate` orthogonal to `u_hist` (preserving norm so positive firing keeps full gain).
      2. Applies **strictly downward-only `b_gate` tightening** if any DEV carryover state still
         exceeds `-neg_margin`. NEVER raises `b_gate` above `b_orig` (preventing false ignition on
         already-silent layers).
    """
    w_new = project_vector_orthogonal(w_gate, u_hist, preserve_norm=preserve_norm)
    b_orig = float(b_gate)
    b_new = b_orig

    if x_neg_carryover is not None:
        xn = (
            x_neg_carryover.detach().float().cpu().numpy()
            if isinstance(x_neg_carryover, torch.Tensor)
            else np.asarray(x_neg_carryover, dtype=np.float32)
        )
        if xn.ndim == 2 and xn.shape[0] > 0:
            max_neg_z = float((xn @ w_new + b_orig).max())
            if max_neg_z > -neg_margin:
                b_new = min(b_orig, b_orig - (max_neg_z + neg_margin))

    return w_new, float(b_new)


def cap_relative_residual_norm(
    delta_h: torch.Tensor,
    native_h: torch.Tensor,
    max_rel_norm: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Soft-clamp additive edit residual `delta_h` so `||delta_h|| <= max_rel_norm * ||native_h||`.
    Exact bit-for-bit no-op whenever `delta_h` is already within budget.
    """
    if max_rel_norm <= 0.0:
        return torch.zeros_like(delta_h)
    base_n = native_h.float().norm(dim=-1, keepdim=True)
    edit_n = delta_h.float().norm(dim=-1, keepdim=True)
    scale = (max_rel_norm * base_n / (edit_n + eps)).clamp_max(1.0)
    return (delta_h.float() * scale).to(dtype=delta_h.dtype)


def _format_chat_prompt(tok, history_turns: list[tuple[str, str]] | None, query: str) -> str:
    """Format a single-turn or multi-turn prompt using tokenizer chat template when available."""
    is_completion = not query.strip().endswith(("?", ".", "!"))
    if not history_turns:
        if is_completion:
            return query
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(
                    [{"role": "user", "content": query}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"User: {query}\nAssistant:"

    msgs: list[dict[str, str]] = []
    for u_txt, a_txt in history_turns:
        msgs.append({"role": "user", "content": u_txt})
        msgs.append({"role": "assistant", "content": a_txt})
    if is_completion:
        msgs.append({"role": "user", "content": f"Complete this sentence: {query}"})
    else:
        msgs.append({"role": "user", "content": query})

    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        try:
            rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return f"{rendered}{query}" if is_completion else rendered
        except Exception:
            pass

    lines = []
    for m in msgs:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    tail = f"Assistant: {query}" if is_completion else "Assistant:"
    lines.append(tail)
    return "\n".join(lines)


@torch.no_grad()
def _collect_logits_and_mlp_states(
    model,
    layers,
    tok,
    prompts: list[str],
    layer_indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Collect last-token logits, MLP inputs `x`, and MLP outputs `y` at `layer_indices`."""
    lgs: list[torch.Tensor] = []
    xs: dict[int, list[torch.Tensor]] = {li: [] for li in layer_indices}
    ys: dict[int, list[torch.Tensor]] = {li: [] for li in layer_indices}
    cap_x: dict[int, torch.Tensor] = {}
    cap_y: dict[int, torch.Tensor] = {}
    handles = []

    for li in layer_indices:
        def _make_hook(idx: int):
            def _hook(_mod, inp, out):
                cap_x[idx] = inp[0][0, -1].detach().float().cpu()
                cap_y[idx] = out[0, -1].detach().float().cpu()
            return _hook
        handles.append(layers[li].mlp.register_forward_hook(_make_hook(li)))

    try:
        for p in prompts:
            ids = tok.encode(p, add_special_tokens=False)
            if not ids:
                ids = [tok.eos_token_id or 0]
            out = model(torch.tensor([ids], dtype=torch.long, device=device))
            lgs.append(out.logits[0, -1].detach().float().cpu())
            for li in layer_indices:
                xs[li].append(cap_x[li])
                ys[li].append(cap_y[li])
    finally:
        for h in handles:
            h.remove()

    return (
        torch.stack(lgs, dim=0),
        {li: torch.stack(xs[li], dim=0) for li in layer_indices},
        {li: torch.stack(ys[li], dim=0) for li in layer_indices},
    )


def _mean_kl_common_vocab(base_lg: torch.Tensor, edit_lg: torch.Tensor) -> float:
    v_common = min(base_lg.shape[-1], edit_lg.shape[-1])
    p_lp = F.log_softmax(base_lg[:, :v_common], dim=-1)
    q_lp = F.log_softmax(edit_lg, dim=-1)[:, :v_common]
    return float((p_lp.exp() * (p_lp - q_lp)).sum(dim=-1).mean().item())


def _auto_detect_edited_neurons(
    base_layers,
    edit_layers,
    layer_indices: list[int],
    top_k: int = 8,
    min_diff: float = 1e-4,
) -> dict[int, list[int]]:
    """Find neurons whose gate/up/down weights differ between base and edited models."""
    found: dict[int, list[int]] = {}
    for li in layer_indices:
        b_mlp = base_layers[li].mlp
        e_mlp = edit_layers[li].mlp
        dg = (e_mlp.gate_proj.weight.detach().float().cpu() - b_mlp.gate_proj.weight.detach().float().cpu()).norm(dim=-1)
        dd = (e_mlp.down_proj.weight.detach().float().cpu() - b_mlp.down_proj.weight.detach().float().cpu()).norm(dim=0)
        score = dg + dd
        nz = torch.where(score > min_diff)[0]
        if nz.numel() > 0:
            order = torch.argsort(score[nz], descending=True)[:top_k]
            found[li] = [int(nz[i].item()) for i in order]
        else:
            found[li] = []
    return found


@torch.no_grad()
def audit_latching(
    base_model_ref: str,
    edited_model_ref: str,
    trigger_history: list[tuple[str, str]],
    off_prompts: list[str] | None = None,
    layers: list[int] | None = None,
    inspect_neurons: dict[int, list[int]] | None = None,
    device: str | None = None,
) -> LatchingAuditReport:
    """
    Model-agnostic multi-turn latching detector comparing `base_model_ref` vs `edited_model_ref`
    on paired `(q_off solo)` vs `(trigger_history + q_off)` prompts.

    Measures:
      - `kl_solo`, `kl_multi`, and `kl_ratio = kl_multi / max(kl_solo, 1e-9)`
      - Per-layer `dz_hist` (base-KV context carryover), `dz_self` (edit-KV feedback gain),
        `max_act_solo -> max_act_multi`, and `R_norm = ||Delta h_edit|| / ||h_native||`.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    queries = list(off_prompts or DEFAULT_CARRYOVER_OFF_PROMPTS)

    base_model, base_layers, _, _, base_tok = _load_full_model(base_model_ref)
    base_model.to(dev).eval()
    n_layers = len(base_layers)
    layer_indices = sorted(set(layers)) if layers is not None else [n_layers - 1]

    solo_prompts = [_format_chat_prompt(base_tok, None, q) for q in queries]
    multi_prompts = [_format_chat_prompt(base_tok, trigger_history, q) for q in queries]

    base_solo_lg, base_solo_x, base_solo_y = _collect_logits_and_mlp_states(
        base_model, base_layers, base_tok, solo_prompts, layer_indices, dev
    )
    base_multi_lg, base_multi_x, base_multi_y = _collect_logits_and_mlp_states(
        base_model, base_layers, base_tok, multi_prompts, layer_indices, dev
    )

    # Snapshot base weights for target layers before freeing base model from VRAM
    base_mlp_weights = {
        li: (
            base_layers[li].mlp.gate_proj.weight.detach().float().cpu().clone(),
            base_layers[li].mlp.down_proj.weight.detach().float().cpu().clone(),
        )
        for li in layer_indices
    }
    del base_model, base_layers
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    edit_model, edit_layers, _, _, edit_tok = _load_full_model(edited_model_ref)
    edit_model.to(dev).eval()
    edit_solo_prompts = [_format_chat_prompt(edit_tok, None, q) for q in queries]
    edit_multi_prompts = [_format_chat_prompt(edit_tok, trigger_history, q) for q in queries]

    edit_solo_lg, edit_solo_x, edit_solo_y = _collect_logits_and_mlp_states(
        edit_model, edit_layers, edit_tok, edit_solo_prompts, layer_indices, dev
    )
    edit_multi_lg, edit_multi_x, edit_multi_y = _collect_logits_and_mlp_states(
        edit_model, edit_layers, edit_tok, edit_multi_prompts, layer_indices, dev
    )

    kl_solo = _mean_kl_common_vocab(base_solo_lg, edit_solo_lg)
    kl_multi = _mean_kl_common_vocab(base_multi_lg, edit_multi_lg)
    kl_ratio = kl_multi / max(kl_solo, 1e-6)

    layer_metrics: list[LatchingLayerMetrics] = []
    recs: list[str] = []
    any_latch = False
    any_bulldoze = False

    for li in layer_indices:
        e_mlp = edit_layers[li].mlp
        wg_all = e_mlp.gate_proj.weight.detach().float().cpu()
        wu_all = e_mlp.up_proj.weight.detach().float().cpu()
        wd_all = e_mlp.down_proj.weight.detach().float().cpu()

        if inspect_neurons and li in inspect_neurons:
            nis = list(inspect_neurons[li])
        else:
            bwg, bwd = base_mlp_weights[li]
            score = (wg_all - bwg).norm(dim=-1) + (wd_all - bwd).norm(dim=0)
            nz = torch.where(score > 1e-4)[0]
            order = torch.argsort(score[nz], descending=True)[:8] if nz.numel() > 0 else torch.tensor([], dtype=torch.long)
            nis = [int(nz[i].item()) for i in order]

        flags: list[str] = []
        if nis:
            wg = wg_all[nis]
            wu = wu_all[nis]
            wd = wd_all[:, nis]
            z_solo = base_solo_x[li] @ wg.T
            z_bkv = base_multi_x[li] @ wg.T
            z_ekv = edit_multi_x[li] @ wg.T
            a_solo = F.silu(z_solo) * (base_solo_x[li] @ wu.T)
            a_ekv = F.silu(z_ekv) * (edit_multi_x[li] @ wu.T)

            dz_hist = float((z_bkv - z_solo).mean().item())
            dz_self = float((z_ekv - z_bkv).mean().item())
            max_act_solo = float(a_solo.abs().max().item())
            max_act_multi = float(a_ekv.abs().max().item())
            dh_multi = (a_ekv @ wd.T).norm(dim=-1)
            r_norm = float((dh_multi / base_multi_y[li].norm(dim=-1).clamp_min(1e-6)).max().item())
        else:
            dz_hist = 0.0
            dz_self = 0.0
            max_act_solo = 0.0
            max_act_multi = 0.0
            r_norm = float(
                ((edit_multi_y[li] - base_multi_y[li]).norm(dim=-1) / base_multi_y[li].norm(dim=-1).clamp_min(1e-6))
                .max()
                .item()
            )

        if dz_hist > 0.45:
            flags.append("CARRYOVER_SHIFT")
            any_latch = True
        if dz_self > 0.50:
            flags.append("SELF_FEEDBACK_LOOP")
            any_latch = True
        if r_norm > 1.50:
            flags.append("RESIDUAL_BULLDOZE")
            any_bulldoze = True

        layer_metrics.append(
            LatchingLayerMetrics(
                layer=li,
                neuron_indices=nis,
                dz_hist=round(dz_hist, 4),
                dz_self=round(dz_self, 4),
                max_act_solo=round(max_act_solo, 4),
                max_act_multi=round(max_act_multi, 4),
                r_norm=round(r_norm, 4),
                flagged_reasons=flags,
            )
        )

    if kl_ratio >= 1.75 or any_latch:
        verdict = "LATCHING_DETECTED"
        recs.append("Project gate keys orthogonal to DEV history-carryover subspace U_hist (`repair-latch` or `GraftConfig(carryover_safe=True)`).")
    elif kl_multi > 0.05 and kl_ratio >= 1.25:
        verdict = "CARRYOVER_DRIFT"
        recs.append("Consider adding multi-turn carryover negatives (`carryover_safe=True`) during compilation.")
    else:
        verdict = "CLEAN"

    if any_bulldoze:
        if verdict == "CLEAN":
            verdict = "BULLDOZING_RISK"
        recs.append("Enable relative residual norm cap (`max_rel_norm=0.75`) to prevent native FFN overpowering.")
    if any("SELF_FEEDBACK_LOOP" in m.flagged_reasons for m in layer_metrics):
        recs.append("On looped architectures, restrict core hooks to terminal pass (`core_active_passes=[P-1]`) at compile/train time.")

    del edit_model, edit_layers
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return LatchingAuditReport(
        base_model=base_model_ref,
        edited_model=edited_model_ref,
        n_off_prompts=len(queries),
        kl_solo=round(kl_solo, 6),
        kl_multi=round(kl_multi, 6),
        kl_ratio=round(kl_ratio, 3),
        layer_metrics=layer_metrics,
        verdict=verdict,
        recommendations=recs,
    )


def repair_modular_adapter_latching(
    adapter_path: str | Path,
    dev_deltas_by_layer: dict[int, np.ndarray | torch.Tensor],
    dev_neg_carryover_by_layer: dict[int, np.ndarray | torch.Tensor] | None = None,
    save_path: str | Path | None = None,
    rank: int = 8,
    neg_margin: float = 0.5,
    max_rel_norm: float | None = 0.75,
    core_active_passes: list[int] | None = None,
) -> LatchingRepairReport:
    """
    Zero-gradient post-hoc latching repair for modular hard-zero-gated adapters (`L.<ip>.gate_w/b`):
      - Extracts top-`rank` carryover subspace `U_hist` per layer from `dev_deltas_by_layer`
      - Projects `gate_w` orthogonal to `U_hist` (preserving norm)
      - Applies strictly downward-only `gate_b` tightening against `dev_neg_carryover_by_layer`
      - Stamps `max_rel_norm` and optional `core_active_passes` into companion `.json` metadata.
    """
    src = Path(adapter_path)
    dst = Path(save_path) if save_path is not None else src
    meta_src = src.with_suffix(".json")
    meta = json.loads(meta_src.read_text(encoding="utf-8")) if meta_src.exists() else {"gate_tau": 0.5, "scale": 1.0}

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k).clone()

    layer_ids = sorted({int(k.split(".")[1]) for k in tensors if k.startswith("L.") and k.endswith(".gate_w")})
    repaired_layers: list[int] = []
    exp_vars: dict[int, float] = {}

    for ip in layer_ids:
        if ip not in dev_deltas_by_layer:
            continue
        u_hist, ev = extract_carryover_subspace(dev_deltas_by_layer[ip], rank=rank)
        exp_vars[ip] = round(ev, 4)
        w_key = f"L.{ip}.gate_w"
        b_key = f"L.{ip}.gate_b"
        w_orig = tensors[w_key][0]
        b_orig = float(tensors[b_key][0].item())
        xn = dev_neg_carryover_by_layer.get(ip) if dev_neg_carryover_by_layer else None
        w_new, b_new = project_gate_orthogonal_to_history(
            w_orig, b_orig, u_hist, x_neg_carryover=xn, neg_margin=neg_margin, preserve_norm=True
        )
        tensors[w_key] = torch.from_numpy(w_new).unsqueeze(0).to(dtype=tensors[w_key].dtype).contiguous()
        tensors[b_key] = torch.tensor([b_new], dtype=tensors[b_key].dtype).contiguous()
        repaired_layers.append(ip)

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst))
    meta["layers"] = layer_ids
    if max_rel_norm is not None:
        meta["max_rel_norm"] = float(max_rel_norm)
    if core_active_passes is not None:
        meta["core_active_passes"] = list(core_active_passes)
    meta["latch_repair"] = {
        "mode": "uhist_downward_b_normcap",
        "rank": rank,
        "layers_repaired": repaired_layers,
        "explained_var_ratio": exp_vars,
    }
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return LatchingRepairReport(
        target_path=str(src),
        saved_path=str(dst),
        mode="modular_uhist_downward_b_normcap",
        rank=rank,
        layers_repaired=repaired_layers,
        max_rel_norm=max_rel_norm,
        core_active_passes=core_active_passes,
        explained_var_ratio=exp_vars,
    )


@torch.no_grad()
def repair_swiglu_latching(
    base_model_ref: str,
    edited_model_ref: str,
    trigger_history: list[tuple[str, str]],
    save_dir: Path | None = None,
    layers: list[int] | None = None,
    rank: int = 4,
    dev_off_prompts: list[str] | None = None,
    eval_off_prompts: list[str] | None = None,
    device: str | None = None,
) -> LatchingRepairReport:
    """
    Zero-gradient post-hoc latching repair for dense SwiGLU checkpoints (e.g. `VibeSplice` SCG):
      - Automatically locates edited neurons vs `base_model_ref`
      - Fits `U_hist` strictly on disjoint `dev_off_prompts` carryover deltas
      - Projects `U_hist` out of ONLY the excess key component `(gate_proj - proj_on_up_proj)` so
        the linear-cone bias anchor (`up_proj = b`) stays intact while removing history-carryover
        drift (`dz_hist`)!
      - Reports before/after `kl_solo`, `kl_multi`, and `dz_hist` on strictly held-out `eval_off_prompts`.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dev_qs = list(dev_off_prompts or DEFAULT_DEV_CARRYOVER_OFF_PROMPTS)
    eval_qs = list(eval_off_prompts or DEFAULT_CARRYOVER_OFF_PROMPTS)

    base_model, base_layers, _, _, base_tok = _load_full_model(base_model_ref)
    base_model.to(dev).eval()
    n_layers = len(base_layers)
    layer_indices = sorted(set(layers)) if layers is not None else [n_layers - 1]

    _, dev_solo_x, _ = _collect_logits_and_mlp_states(
        base_model, base_layers, base_tok, [_format_chat_prompt(base_tok, None, q) for q in dev_qs], layer_indices, dev
    )
    _, dev_multi_x, _ = _collect_logits_and_mlp_states(
        base_model, base_layers, base_tok, [_format_chat_prompt(base_tok, trigger_history, q) for q in dev_qs], layer_indices, dev
    )
    base_eval_solo_lg, base_eval_solo_x, _ = _collect_logits_and_mlp_states(
        base_model, base_layers, base_tok, [_format_chat_prompt(base_tok, None, q) for q in eval_qs], layer_indices, dev
    )
    base_eval_multi_lg, base_eval_multi_x, _ = _collect_logits_and_mlp_states(
        base_model, base_layers, base_tok, [_format_chat_prompt(base_tok, trigger_history, q) for q in eval_qs], layer_indices, dev
    )
    base_mlp_w = {
        li: (
            base_layers[li].mlp.gate_proj.weight.detach().float().cpu().clone(),
            base_layers[li].mlp.down_proj.weight.detach().float().cpu().clone(),
        )
        for li in layer_indices
    }
    del base_model, base_layers
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    edit_model, edit_layers, _, _, edit_tok = _load_full_model(edited_model_ref)
    edit_model.to(dev).eval()
    eval_solo_p = [_format_chat_prompt(edit_tok, None, q) for q in eval_qs]
    eval_multi_p = [_format_chat_prompt(edit_tok, trigger_history, q) for q in eval_qs]

    def _measure_now(target_nis: dict[int, list[int]]) -> tuple[float, float, float]:
        es_lg, _, _ = _collect_logits_and_mlp_states(edit_model, edit_layers, edit_tok, eval_solo_p, layer_indices, dev)
        em_lg, _, _ = _collect_logits_and_mlp_states(edit_model, edit_layers, edit_tok, eval_multi_p, layer_indices, dev)
        k_s = _mean_kl_common_vocab(base_eval_solo_lg, es_lg)
        k_m = _mean_kl_common_vocab(base_eval_multi_lg, em_lg)
        dzs = []
        for li in layer_indices:
            if target_nis[li]:
                wg = edit_layers[li].mlp.gate_proj.weight.detach().float().cpu()[target_nis[li]]
                dzs.append(float(((base_eval_multi_x[li] - base_eval_solo_x[li]) @ wg.T).mean().item()))
        return k_s, k_m, float(np.mean(dzs)) if dzs else 0.0

    target_nis: dict[int, list[int]] = {}
    for li in layer_indices:
        bwg, bwd = base_mlp_w[li]
        ewg = edit_layers[li].mlp.gate_proj.weight.detach().float().cpu()
        ewd = edit_layers[li].mlp.down_proj.weight.detach().float().cpu()
        score = (ewg - bwg).norm(dim=-1) + (ewd - bwd).norm(dim=0)
        nz = torch.where(score > 1e-4)[0]
        order = torch.argsort(score[nz], descending=True)[:16] if nz.numel() > 0 else torch.tensor([], dtype=torch.long)
        target_nis[li] = [int(nz[i].item()) for i in order]

    before_ks, before_km, before_dz = _measure_now(target_nis)

    exp_vars: dict[int, float] = {}
    repaired_layers: list[int] = []
    for li in layer_indices:
        if not target_nis[li]:
            continue
        u_hist, ev = extract_carryover_subspace(dev_multi_x[li] - dev_solo_x[li], rank=rank)
        exp_vars[li] = round(ev, 4)
        mlp = edit_layers[li].mlp
        for ni in target_nis[li]:
            wg = mlp.gate_proj.weight.data[ni].detach().float().cpu().numpy()
            wu = mlp.up_proj.weight.data[ni].detach().float().cpu().numpy()
            # Preserve the linear-cone bias anchor component along `wu = b` while projecting the
            # residual key component orthogonal to `U_hist`!
            wu_u = wu / (np.linalg.norm(wu) + 1e-12)
            bias_comp = float(np.dot(wg, wu_u)) * wu_u
            key_comp = wg - bias_comp
            key_perp = project_vector_orthogonal(key_comp, u_hist, preserve_norm=True)
            wg_new = bias_comp + key_perp
            mlp.gate_proj.weight.data[ni].copy_(
                torch.from_numpy(wg_new).to(device=mlp.gate_proj.weight.device, dtype=mlp.gate_proj.weight.dtype)
            )
        repaired_layers.append(li)

    after_ks, after_km, after_dz = _measure_now(target_nis)

    saved_str: str | None = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        edit_model.save_pretrained(str(save_dir))
        edit_tok.save_pretrained(str(save_dir))
        saved_str = str(save_dir)

    del edit_model, edit_layers
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    return LatchingRepairReport(
        target_path=edited_model_ref,
        saved_path=saved_str,
        mode="swiglu_k_only_uhist_projection",
        rank=rank,
        layers_repaired=repaired_layers,
        before_kl_solo=round(before_ks, 6),
        before_kl_multi=round(before_km, 6),
        after_kl_solo=round(after_ks, 6),
        after_kl_multi=round(after_km, 6),
        before_dz_hist=round(before_dz, 4),
        after_dz_hist=round(after_dz, 4),
        explained_var_ratio=exp_vars,
    )
