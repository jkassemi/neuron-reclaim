from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    Qwen3_5ForConditionalGeneration = None  # type: ignore[assignment,misc]

from cluster import build_vocab_masks
from models import (
    ApplyReport,
    ClusterAction,
    EvictTarget,
    HealMode,
    MergePlan,
    MitigationConfig,
    SurgeryPlan,
)
from weights import resolve_snapshot


def _load_full_model(model_dir: str):
    p = resolve_snapshot(model_dir)
    tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True)
    if Qwen3_5ForConditionalGeneration is not None:
        try:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                str(p), dtype=torch.bfloat16, local_files_only=True
            ).eval()
            layers = model.model.language_model.layers
            norm_w = model.model.language_model.norm.weight.detach().float()
            lm_head_w = model.lm_head.weight.detach().float()
            return model, layers, norm_w, lm_head_w, tok
        except Exception:
            pass
    model = AutoModelForCausalLM.from_pretrained(
        str(p), dtype=torch.bfloat16, local_files_only=True
    ).eval()
    layers = model.model.layers
    norm_w = model.model.norm.weight.detach().float()
    lm_head_w = model.lm_head.weight.detach().float()
    return model, layers, norm_w, lm_head_w, tok


def _build_self_gen_prompts(
    surgery_plan: SurgeryPlan | None,
    merge_plan: MergePlan | None,
) -> tuple[list[str], list[str]]:
    """Zero-corpus fit + held-out eval prompts built from plan tokens + generic English starters."""
    fit_prompts = [
        "The history of modern computing began when",
        "In Python, you can write an efficient function to",
        "Scientists recently discovered that the atmosphere of",
        "When designing a distributed database system, it is important to",
        "The capital of Colorado is Denver, located near",
        "During the Renaissance, artists and scholars in Europe",
        "In molecular biology, enzymes catalyze reactions by",
        "The Federal Reserve adjusts interest rates in order to",
    ]
    eval_prompts = [
        "A well-structured software architecture separates concerns so that",
        "In the Rocky Mountains, high elevation affects both weather and",
        "Modern web servers handle concurrent connections by using",
        "Economic historians often point out that trade routes across",
    ]

    seen: set[str] = set()
    targeted: list[str] = []
    if merge_plan:
        for pair in merge_plan.pairs[:6]:
            for tok in pair.keep_tokens[:1] + pair.free_tokens[:1]:
                clean = tok.strip()
                if clean.isascii() and len(clean) >= 3 and clean.lower() not in seen:
                    seen.add(clean.lower())
                    targeted.append(f"In the context of {clean}, we often observe that")
    if surgery_plan:
        for sel in surgery_plan.selections[:4]:
            for tok in sel.centroid_tokens:
                clean = tok.strip()
                if clean.isascii() and len(clean) >= 3 and clean.lower() not in seen:
                    seen.add(clean.lower())
                    targeted.append(f"An example involving {clean} demonstrates how")

    # Interleave targeted prompts between fit (75%) and held-out eval (25%)
    for idx, p in enumerate(targeted[:8]):
        if idx % 4 == 3:
            eval_prompts.append(p)
        else:
            fit_prompts.append(p)
    return fit_prompts[:12], eval_prompts[:6]


def _self_generate_rollouts(model, tok, seeds: list[str], max_new_tokens: int) -> list[torch.Tensor]:
    rollouts: list[torch.Tensor] = []
    with torch.no_grad():
        for s in seeds:
            ids = tok.encode(s, return_tensors="pt")
            out = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            rollouts.append(out)
    return rollouts


def _capture_mlp_and_logits(
    model, mlp_mod, rollouts: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []

    def hook(mod, inp, out):
        xs.append(inp[0].detach().squeeze(0).float())
        ys.append(out.detach().squeeze(0).float())

    h = mlp_mod.down_proj.register_forward_hook(hook)
    with torch.no_grad():
        for ids in rollouts:
            out = model(ids).logits[0].detach().float()
            logits_list.append(out)
    h.remove()
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), logits_list


def _mean_kl(teacher_logits: list[torch.Tensor], student_logits: list[torch.Tensor]) -> float:
    kls: list[float] = []
    for tl, sl in zip(teacher_logits, student_logits):
        p = torch.softmax(tl, dim=-1)
        log_p = torch.log_softmax(tl, dim=-1)
        log_q = torch.log_softmax(sl, dim=-1)
        kl = (p * (log_p - log_q)).sum(dim=-1).mean().item()
        kls.append(float(kl))
    return sum(kls) / max(1, len(kls))


def _apply_downstream_gate_shield(
    layers,
    layer: int,
    saved_downs: list[tuple[torch.Tensor, torch.Tensor]],
    top_k: int,
) -> int:
    """
    Zero-corpus downstream gate shielding (layer L+1):
    For each merged pair (da, db) in layer L, find the top_k layer-(L+1) neurons
    most strongly driven by db and project da out of their gate_proj & up_proj (and vice versa).
    """
    if layer + 1 >= len(layers) or not saved_downs:
        return 0
    next_layer = layers[layer + 1]
    next_mlp = next_layer.mlp
    norm_w = next_layer.post_attention_layernorm.weight.detach().float()
    G = next_mlp.gate_proj.weight.detach().float()
    U = next_mlp.up_proj.weight.detach().float()
    G_eff = G * norm_w.unsqueeze(0)

    shielded: set[int] = set()
    for da, db in saved_downs:
        ua = da / (da.norm() + 1e-8)
        ub = db / (db.norm() + 1e-8)
        # Neurons in L+1 listening to B -> shield from A's ghost vector
        top_b = torch.topk((G_eff @ db).abs(), min(top_k, G.shape[0])).indices
        for idx in top_b.tolist():
            G[idx] = G[idx] - torch.dot(G[idx], ua) * ua
            U[idx] = U[idx] - torch.dot(U[idx], ua) * ua
            shielded.add(idx)
        # Neurons in L+1 listening to A -> shield from B's ghost vector
        top_a = torch.topk((G_eff @ da).abs(), min(top_k, G.shape[0])).indices
        for idx in top_a.tolist():
            G[idx] = G[idx] - torch.dot(G[idx], ub) * ub
            U[idx] = U[idx] - torch.dot(U[idx], ub) * ub
            shielded.add(idx)

    with torch.no_grad():
        next_mlp.gate_proj.weight.copy_(G.to(next_mlp.gate_proj.weight.dtype))
        next_mlp.up_proj.weight.copy_(U.to(next_mlp.up_proj.weight.dtype))
    return len(shielded)


def _run_self_kl_heal(
    model,
    mlp,
    active_mask: torch.Tensor,
    fit_rollouts: list[torch.Tensor],
    teacher_fit_logits: list[torch.Tensor],
    steps: int,
    lr: float,
) -> None:
    """Short KL-distillation pass updating ONLY surviving active columns of layer L down_proj."""
    for p in model.parameters():
        p.requires_grad_(False)

    # Optimize a float32 delta on active columns to avoid bf16 underflow
    W_base = mlp.down_proj.weight.detach().clone()
    d_model = W_base.shape[0]
    n_active = int(active_mask.sum().item())
    delta = torch.zeros((d_model, n_active), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    # Hook that adds (x_active @ delta.T) to down_proj output cleanly with autograd
    def grad_hook(mod, inp, out):
        x_act = inp[0].detach().float()[..., active_mask]
        corr = x_act @ delta.T
        return out.float() + corr

    h = mlp.down_proj.register_forward_hook(grad_hook)
    try:
        n_roll = len(fit_rollouts)
        for step in range(steps):
            opt.zero_grad()
            idx = step % n_roll
            ids = fit_rollouts[idx]
            tl = teacher_fit_logits[idx]
            sl = model(ids).logits[0].float()
            p_t = torch.softmax(tl, dim=-1)
            log_p_t = torch.log_softmax(tl, dim=-1)
            log_p_s = torch.log_softmax(sl, dim=-1)
            loss = (p_t * (log_p_t - log_p_s)).sum(dim=-1).mean()
            loss.backward()
            opt.step()
    finally:
        h.remove()

    with torch.no_grad():
        W_final = W_base.float()
        W_final[:, active_mask] += delta.detach()
        mlp.down_proj.weight.copy_(W_final.to(W_base.dtype))


def execute_surgery(
    model_ref: str,
    layer: int,
    surgery_plan: SurgeryPlan | None = None,
    merge_plan: MergePlan | None = None,
    mitigations: MitigationConfig | None = None,
    save_model_dir: Path | None = None,
) -> ApplyReport:
    cfg = mitigations or MitigationConfig()
    model, layers, norm_w, lm_head_w, tok = _load_full_model(model_ref)
    mlp = layers[layer].mlp
    U = lm_head_w * norm_w.unsqueeze(0)

    # 1. Self-generate zero-corpus fit + held-out eval English rollouts (unless heal_mode == NONE)
    fit_rollouts: list[torch.Tensor] = []
    eval_rollouts: list[torch.Tensor] = []
    Y_fit_clean = None
    Y_eval_clean = None
    teacher_fit_logits: list[torch.Tensor] = []
    teacher_eval_logits: list[torch.Tensor] = []
    fit_tokens = 0
    eval_tokens = 0

    if cfg.heal_mode != HealMode.NONE:
        fit_seeds, eval_seeds = _build_self_gen_prompts(surgery_plan, merge_plan)
        fit_rollouts = _self_generate_rollouts(model, tok, fit_seeds, max_new_tokens=cfg.gen_tokens)
        eval_rollouts = _self_generate_rollouts(model, tok, eval_seeds, max_new_tokens=cfg.gen_tokens)
        fit_tokens = sum(int(r.shape[1]) for r in fit_rollouts)
        eval_tokens = sum(int(r.shape[1]) for r in eval_rollouts)
        _, Y_fit_clean, teacher_fit_logits = _capture_mlp_and_logits(model, mlp, fit_rollouts)
        _, Y_eval_clean, teacher_eval_logits = _capture_mlp_and_logits(model, mlp, eval_rollouts)

    freed_slots: set[int] = set()
    recycled = 0
    erased = 0
    merged = 0
    saved_merge_downs: list[tuple[torch.Tensor, torch.Tensor]] = []

    # 2. Apply SurgeryPlan (subspace_erase + recycle)
    if surgery_plan is not None:
        cjk_mask, nl_mask, lat_mask, _ = build_vocab_masks(tok, U.shape[0])
        ev_mask = cjk_mask if surgery_plan.evict_target == EvictTarget.CJK else nl_mask
        ev_vecs = U[ev_mask > 0.5]
        en_vecs = U[lat_mask > 0.5]
        ev_sub = ev_vecs[:: max(1, ev_vecs.shape[0] // 512)]
        en_sub = en_vecs[:: max(1, en_vecs.shape[0] // 512)]
        _, _, Vt_en = torch.linalg.svd(en_sub, full_matrices=False)
        Q_en = Vt_en[:64].T
        ev_clean = ev_sub - (ev_sub @ Q_en) @ Q_en.T
        _, _, Vt_ev = torch.linalg.svd(ev_clean, full_matrices=False)
        Q_ev = Vt_ev[:48].T

        with torch.no_grad():
            for sel in surgery_plan.selections:
                idxs = sel.neuron_indices
                if sel.action == ClusterAction.RECYCLE:
                    mlp.down_proj.weight[:, idxs] = 0.0
                    mlp.gate_proj.weight[idxs] = torch.randn_like(mlp.gate_proj.weight[idxs]) * 0.02
                    mlp.up_proj.weight[idxs] = torch.randn_like(mlp.up_proj.weight[idxs]) * 0.02
                    freed_slots.update(idxs)
                    recycled += len(idxs)
                elif sel.action == ClusterAction.SUBSPACE_ERASE:
                    W_sub = mlp.down_proj.weight[:, idxs].float()
                    W_sub = W_sub - Q_ev @ (Q_ev.T @ W_sub)
                    mlp.down_proj.weight[:, idxs] = W_sub.to(mlp.down_proj.weight.dtype)
                    erased += len(idxs)

    # 3. Apply MergePlan (with optional input orthogonalization + antipodal sign)
    if merge_plan is not None:
        with torch.no_grad():
            for pair in merge_plan.pairs:
                a, b = pair.keep_neuron, pair.free_neuron
                if a in freed_slots or b in freed_slots:
                    continue
                ga = mlp.gate_proj.weight[a].float()
                gb = mlp.gate_proj.weight[b].float()
                ua = mlp.up_proj.weight[a].float()
                ub = mlp.up_proj.weight[b].float()
                da = mlp.down_proj.weight[:, a].float()
                db = mlp.down_proj.weight[:, b].float()
                saved_merge_downs.append((da.clone(), db.clone()))

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
                freed_slots.add(b)
                merged += 1

    # 4. Downstream Gate Shielding (Layer L+1)
    shielded_count = 0
    if cfg.downstream_gate_shield and saved_merge_downs:
        shielded_count = _apply_downstream_gate_shield(layers, layer, saved_merge_downs, cfg.shield_top_k)

    # 5. Evaluate held-out pre-heal metrics + run selected HealMode
    rel_l2_before = rel_l2_obs = rel_l2_kl = None
    eval_kl_before = eval_kl_obs = eval_kl_kl = None

    if cfg.heal_mode != HealMode.NONE and Y_fit_clean is not None and Y_eval_clean is not None:
        d_mlp = mlp.down_proj.in_features
        active_mask = torch.ones(d_mlp, dtype=torch.bool)
        for f in freed_slots:
            active_mask[f] = False

        _, Y_eval_post, eval_logits_post = _capture_mlp_and_logits(model, mlp, eval_rollouts)
        rel_l2_before = round(float((Y_eval_clean - Y_eval_post).norm().item() / (Y_eval_clean.norm().item() + 1e-8)), 4)
        eval_kl_before = round(_mean_kl(teacher_eval_logits, eval_logits_post), 5)

        if cfg.heal_mode in (HealMode.SELF_OBS, HealMode.OBS_THEN_KL):
            X_fit_post, Y_fit_post, _ = _capture_mlp_and_logits(model, mlp, fit_rollouts)
            X_act = X_fit_post[:, active_mask]
            R = Y_fit_clean - Y_fit_post
            G = X_act @ X_act.T
            eye = torch.eye(G.shape[0], dtype=torch.float32)
            lam = cfg.obs_ridge * float(G.diag().mean().item() + 1e-6)
            alpha = torch.linalg.solve(G + lam * eye, R)
            dW_T = X_act.T @ alpha
            with torch.no_grad():
                W_new = mlp.down_proj.weight.float()
                W_new[:, active_mask] += dW_T.T
                mlp.down_proj.weight.copy_(W_new.to(mlp.down_proj.weight.dtype))

            _, Y_eval_obs, eval_logits_obs = _capture_mlp_and_logits(model, mlp, eval_rollouts)
            rel_l2_obs = round(float((Y_eval_clean - Y_eval_obs).norm().item() / (Y_eval_clean.norm().item() + 1e-8)), 4)
            eval_kl_obs = round(_mean_kl(teacher_eval_logits, eval_logits_obs), 5)

        if cfg.heal_mode in (HealMode.SELF_KL, HealMode.OBS_THEN_KL):
            _run_self_kl_heal(
                model,
                mlp,
                active_mask,
                fit_rollouts,
                teacher_fit_logits,
                steps=cfg.kl_steps,
                lr=cfg.kl_lr,
            )
            _, Y_eval_kl, eval_logits_kl = _capture_mlp_and_logits(model, mlp, eval_rollouts)
            rel_l2_kl = round(float((Y_eval_clean - Y_eval_kl).norm().item() / (Y_eval_clean.norm().item() + 1e-8)), 4)
            eval_kl_kl = round(_mean_kl(teacher_eval_logits, eval_logits_kl), 5)

    if save_model_dir is not None:
        save_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_model_dir))
        tok.save_pretrained(str(save_model_dir))

    return ApplyReport(
        model_path=str(resolve_snapshot(model_ref)),
        layer=layer,
        recycled_neurons=recycled,
        erased_neurons=erased,
        merged_pairs=merged,
        shielded_downstream_neurons=shielded_count,
        freed_neurons_total=len(freed_slots),
        mitigations=cfg,
        fit_tokens=fit_tokens,
        eval_tokens=eval_tokens,
        layer_rel_l2_before=rel_l2_before,
        layer_rel_l2_after_obs=rel_l2_obs,
        layer_rel_l2_after_kl=rel_l2_kl,
        eval_kl_before=eval_kl_before,
        eval_kl_after_obs=eval_kl_obs,
        eval_kl_after_kl=eval_kl_kl,
    )
