from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from models import (
    ClusterAction,
    ClusterSummary,
    EvictTarget,
    GlobalClusterSummary,
    GlobalNeuronSample,
    LayerOverview,
    LayerReport,
    MergePair,
    MergePlan,
    ModelClusterAtlas,
    ModelOverview,
    NeuronInfo,
)
from weights import LayerWeights, get_num_layers, load_shared_unembed, load_tensor

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_NON_LATIN_LETTER_RE = re.compile(r"[^\x00-\x7f]")


def build_vocab_masks(tokenizer, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Zero-corpus script classification for every token in the vocabulary."""
    cjk = torch.zeros(vocab_size, dtype=torch.float32)
    non_latin = torch.zeros(vocab_size, dtype=torch.float32)
    latin = torch.zeros(vocab_size, dtype=torch.float32)
    decoded: list[str] = [""] * vocab_size

    tok_vocab_size = min(vocab_size, len(tokenizer))
    for i in range(tok_vocab_size):
        s = tokenizer.decode([i])
        decoded[i] = s
        has_cjk = bool(_CJK_RE.search(s))
        has_latin = bool(_LATIN_RE.search(s))
        has_non_latin = has_cjk or (bool(_NON_LATIN_LETTER_RE.search(s)) and not has_latin)
        if has_cjk:
            cjk[i] = 1.0
        if has_non_latin:
            non_latin[i] = 1.0
        if has_latin and not has_cjk:
            latin[i] = 1.0
    return cjk, non_latin, latin, decoded


def _score_neurons_logit_lens(
    V: torch.Tensor,
    U: torch.Tensor,
    cjk_mask: torch.Tensor,
    non_latin_mask: torch.Tensor,
    latin_mask: torch.Tensor,
    top_k: int = 16,
    chunk: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d_mlp = V.shape[0]
    top_ids = torch.empty((d_mlp, top_k), dtype=torch.long)
    top_vals = torch.empty((d_mlp, top_k), dtype=torch.float32)
    for start in range(0, d_mlp, chunk):
        end = min(start + chunk, d_mlp)
        logits = V[start:end] @ U.T
        tv, ti = torch.topk(logits, top_k, dim=-1)
        top_vals[start:end] = tv
        top_ids[start:end] = ti
    w = torch.softmax(top_vals * 10.0, dim=-1)
    cjk_s = (w * cjk_mask[top_ids]).sum(dim=-1)
    nl_s = (w * non_latin_mask[top_ids]).sum(dim=-1)
    lat_s = (w * latin_mask[top_ids]).sum(dim=-1)
    return top_ids, top_vals, cjk_s, nl_s, lat_s


def spherical_emeans(
    x: torch.Tensor,
    k: int,
    epochs: int = 8,
    batch_size: int = 1024,
    ema_decay: float = 0.85,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spherical e-means (mini-batch EMA cosine clustering) in pure PyTorch."""
    gen = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    k = min(k, n)

    first = int(torch.randint(0, n, (1,), generator=gen).item())
    centroids = [x[first]]
    min_dist = 1.0 - (x @ x[first])
    for _ in range(1, k):
        probs = min_dist.clamp(min=1e-6).pow(2)
        probs = probs / probs.sum()
        nxt = int(torch.multinomial(probs, 1, generator=gen).item())
        centroids.append(x[nxt])
        min_dist = torch.minimum(min_dist, 1.0 - (x @ x[nxt]))
    C = F.normalize(torch.stack(centroids, dim=0), dim=-1)

    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            xb = x[perm[start : start + batch_size]]
            assign = (xb @ C.T).argmax(dim=-1)
            counts = torch.bincount(assign, minlength=k).float()
            sums = torch.zeros_like(C)
            sums.index_add_(0, assign, xb)
            active = counts > 0
            if active.any():
                batch_means = F.normalize(sums[active] / counts[active].unsqueeze(-1), dim=-1)
                C[active] = F.normalize(ema_decay * C[active] + (1.0 - ema_decay) * batch_means, dim=-1)

    final_assign = (x @ C.T).argmax(dim=-1)
    return final_assign, C


def overview_model(
    model_ref: str,
    evict_target: EvictTarget = EvictTarget.CJK,
    layer_step: int = 1,
) -> ModelOverview:
    """Disk-usage-style overview of reclaimable impact across layers (zero-corpus)."""
    model_dir, weight_map, U, tokenizer = load_shared_unembed(model_ref)
    n_layers = get_num_layers(model_dir)
    vocab_size = U.shape[0]
    cjk_mask, nl_mask, lat_mask, _ = build_vocab_masks(tokenizer, vocab_size)
    evict_vocab_mask = cjk_mask if evict_target == EvictTarget.CJK else nl_mask
    evict_vocab_tokens = int(evict_vocab_mask.sum().item())

    layer_rows: list[LayerOverview] = []
    for l in range(0, n_layers, layer_step):
        W = load_tensor(
            model_dir,
            weight_map,
            [
                f"model.language_model.layers.{l}.mlp.down_proj.weight",
                f"model.layers.{l}.mlp.down_proj.weight",
            ],
        )
        V = W.T.contiguous()
        norms = V.norm(dim=-1)
        _, _, cjk_s, nl_s, lat_s = _score_neurons_logit_lens(V, U, cjk_mask, nl_mask, lat_mask, top_k=12)
        ev_s = cjk_s if evict_target == EvictTarget.CJK else nl_s
        impact = norms * (ev_s - lat_s).clamp(min=0.0)
        pure = int(((ev_s >= 0.60) & (lat_s <= 0.20)).sum().item())
        bridges = int(((ev_s >= 0.25) & (lat_s >= 0.25)).sum().item())
        layer_rows.append(
            LayerOverview(
                layer=l,
                d_mlp=int(V.shape[0]),
                reclaimable_impact=round(float(impact.sum().item()), 1),
                pure_evict_neurons=pure,
                bridge_neurons=bridges,
                mean_evict_share=round(float(ev_s.mean().item()), 3),
                mean_latin_share=round(float(lat_s.mean().item()), 3),
            )
        )

    layer_rows.sort(key=lambda r: -r.reclaimable_impact)
    return ModelOverview(
        model_path=str(model_dir),
        vocab_size=vocab_size,
        evict_vocab_tokens=evict_vocab_tokens,
        evict_vocab_pct=round(100.0 * evict_vocab_tokens / vocab_size, 1),
        evict_target=evict_target,
        layers=layer_rows,
    )


def analyze_layer(
    lw: LayerWeights,
    num_clusters: int = 48,
    top_k_tokens: int = 16,
    evict_target: EvictTarget = EvictTarget.CJK,
    seed: int = 42,
) -> LayerReport:
    W = lw.down_proj
    d_model, d_mlp = W.shape
    U = lw.unembed_scaled
    cjk_mask, nl_mask, lat_mask, decoded = build_vocab_masks(lw.tokenizer, U.shape[0])

    V = W.T.contiguous()
    norms = V.norm(dim=-1)
    V_norm = F.normalize(V, dim=-1)

    assign, centroids = spherical_emeans(V_norm, k=num_clusters, seed=seed)
    top_ids_all, _, cjk_shares, nl_shares, lat_shares = _score_neurons_logit_lens(
        V, U, cjk_mask, nl_mask, lat_mask, top_k=top_k_tokens
    )
    evict_shares = cjk_shares if evict_target == EvictTarget.CJK else nl_shares
    bridge_scores = 2.0 * torch.minimum(evict_shares, lat_shares)
    reclaim_per_neuron = norms * (evict_shares - lat_shares).clamp(min=0.0)

    c_logits = centroids @ U.T
    _, c_top_ids = torch.topk(c_logits, 8, dim=-1)

    clusters: list[ClusterSummary] = []
    for cid in range(centroids.shape[0]):
        idxs = (assign == cid).nonzero(as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            continue
        size = int(idxs.numel())
        c_evict = float(evict_shares[idxs].mean().item())
        c_latin = float(lat_shares[idxs].mean().item())
        c_impact = float(reclaim_per_neuron[idxs].sum().item())
        bridges = int(((evict_shares[idxs] > 0.25) & (lat_shares[idxs] > 0.25)).sum().item())

        if c_evict >= 0.55 and c_latin <= 0.25:
            rec = ClusterAction.RECYCLE
        elif c_evict >= 0.33 and c_evict > 0.5 * c_latin:
            rec = ClusterAction.SUBSPACE_ERASE
        else:
            rec = ClusterAction.KEEP

        order = torch.argsort(reclaim_per_neuron[idxs] + 0.05 * norms[idxs], descending=True)[:5]
        samples = [
            NeuronInfo(
                index=int(idxs[pos].item()),
                norm=round(float(norms[int(idxs[pos].item())].item()), 3),
                cjk_share=round(float(cjk_shares[int(idxs[pos].item())].item()), 3),
                non_latin_share=round(float(nl_shares[int(idxs[pos].item())].item()), 3),
                latin_share=round(float(lat_shares[int(idxs[pos].item())].item()), 3),
                bridge_score=round(float(bridge_scores[int(idxs[pos].item())].item()), 3),
                reclaimable_impact=round(float(reclaim_per_neuron[int(idxs[pos].item())].item()), 3),
                top_tokens=[decoded[t] for t in top_ids_all[int(idxs[pos].item()), :6].tolist()],
            )
            for pos in order.tolist()
        ]
        clusters.append(
            ClusterSummary(
                cluster_id=cid,
                size=size,
                layer_share_pct=round(100.0 * size / d_mlp, 2),
                reclaimable_impact=round(c_impact, 2),
                mean_evict_share=round(c_evict, 3),
                mean_latin_share=round(c_latin, 3),
                bridge_count=bridges,
                recommended_action=rec,
                centroid_tokens=[decoded[t] for t in c_top_ids[cid].tolist()],
                sample_neurons=samples,
                neuron_indices=[int(x) for x in idxs.tolist()],
            )
        )

    clusters.sort(key=lambda c: -c.reclaimable_impact)
    return LayerReport(
        model_path=str(lw.model_dir),
        layer=lw.layer,
        d_model=d_model,
        d_mlp=d_mlp,
        num_clusters=len(clusters),
        evict_target=evict_target,
        clusters=clusters,
    )


def find_merge_candidates(
    lw: LayerWeights,
    report: LayerReport,
    max_pairs: int = 32,
    max_input_cos: float = 0.04,
    max_lm_cos: float = 0.08,
    max_down_cos: float = 0.04,
    pool_size: int = 512,
) -> MergePlan:
    """
    Zero-corpus 3-gate merge finder:
      1) |cos(gate_A, gate_B)| & |cos(up_A, up_B)| <= max_input_cos (blocks shared syntactic slots)
      2) max lm_head cos(top_tokens_A, top_tokens_B) <= max_lm_cos (blocks Denver<->Tokyo cones)
      3) |cos(down_A, down_B)| <= max_down_cos (orthogonal residual writes)
    Also selects optimal antipodal sign (+1 or -1) per pair.
    """
    assert lw.gate_proj is not None and lw.up_proj is not None
    V_down = F.normalize(lw.down_proj.T.contiguous(), dim=-1)
    V_gate = F.normalize(lw.gate_proj, dim=-1)
    V_up = F.normalize(lw.up_proj, dim=-1)
    U_norm = F.normalize(lw.unembed_scaled, dim=-1)

    # Map neuron -> cluster_id
    neuron_cluster: dict[int, int] = {}
    for c in report.clusters:
        for ni in c.neuron_indices:
            neuron_cluster[ni] = c.cluster_id

    # Pick a diverse pool of English-leaning neurons with moderate norm (long-tail entity candidates)
    norms = lw.down_proj.norm(dim=0)
    # Sample evenly from top-norm neurons across clusters
    pool: list[int] = []
    per_c = max(2, pool_size // max(1, len(report.clusters)))
    for c in report.clusters:
        idxs = torch.tensor(c.neuron_indices, dtype=torch.long)
        order = torch.argsort(norms[idxs], descending=True)[:per_c]
        pool.extend(idxs[order].tolist())
    pool = pool[:pool_size]
    p_idx = torch.tensor(pool, dtype=torch.long)

    # Compute top-4 lm_head token IDs for pool neurons
    logits_p = lw.down_proj[:, p_idx].T @ lw.unembed_scaled.T  # [P, vocab]
    _, top_toks = torch.topk(logits_p, 4, dim=-1)  # [P, 4]
    tok_vecs = U_norm[top_toks]  # [P, 4, d_model]

    # Pairwise matrices on pool
    g_cos = (V_gate[p_idx] @ V_gate[p_idx].T).abs()
    u_cos_signed = V_up[p_idx] @ V_up[p_idx].T
    d_cos_signed = V_down[p_idx] @ V_down[p_idx].T
    in_cos = torch.maximum(g_cos, u_cos_signed.abs())
    dn_cos = d_cos_signed.abs()

    used: set[int] = set()
    pairs: list[MergePair] = []

    # Greedy matching preferring lowest combined collision risk across *different* clusters
    candidates: list[tuple[float, int, int]] = []
    P = len(pool)
    for i in range(P):
        ni = pool[i]
        ci = neuron_cluster[ni]
        for j in range(i + 1, P):
            nj = pool[j]
            if ci == neuron_cluster[nj]:
                continue
            ic = float(in_cos[i, j].item())
            dc = float(dn_cos[i, j].item())
            if ic > max_input_cos or dc > max_down_cos:
                continue
            # Gate 2: max pairwise |lm_head cos| between top-4 tokens of i and j
            lm_c = float((tok_vecs[i] @ tok_vecs[j].T).abs().max().item())
            if lm_c > max_lm_cos:
                continue
            score = ic + dc + 0.5 * lm_c
            candidates.append((score, i, j))

    candidates.sort(key=lambda x: x[0])
    for _, i, j in candidates:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        ni, nj = pool[i], pool[j]
        # Pick antipodal sign (-1 if up_proj & down_proj align positively, else +1)
        sign = -1 if (float(u_cos_signed[i, j].item()) > 0) else 1
        pairs.append(
            MergePair(
                keep_neuron=ni,
                free_neuron=nj,
                keep_cluster=neuron_cluster[ni],
                free_cluster=neuron_cluster[nj],
                sign=sign,
                input_cos=round(float(in_cos[i, j].item()), 4),
                lm_cone_cos=round(float((tok_vecs[i] @ tok_vecs[j].T).abs().max().item()), 4),
                down_cos=round(float(dn_cos[i, j].item()), 4),
                keep_tokens=[lw.tokenizer.decode([t]) for t in top_toks[i].tolist()],
                free_tokens=[lw.tokenizer.decode([t]) for t in top_toks[j].tolist()],
            )
        )
        if len(pairs) >= max_pairs:
            break

    return MergePlan(model_path=str(lw.model_dir), layer=lw.layer, pairs=pairs)


def build_model_atlas(
    model_ref: str,
    num_clusters: int = 48,
    seed: int = 42,
) -> ModelClusterAtlas:
    """
    Target-free, zero-corpus global clustering of ALL MLP neurons across the entire model.
    Uses spherical e-means in the shared residual-stream space (down_proj) so you immediately
    see the top X capacity-consuming clusters across the full model, sorted by size.
    """
    model_dir, weight_map, U, tokenizer = load_shared_unembed(model_ref)
    n_layers = get_num_layers(model_dir)
    vocab_size = U.shape[0]
    cjk_mask, nl_mask, lat_mask, decoded = build_vocab_masks(tokenizer, vocab_size)
    cjk_vocab_pct = round(100.0 * float(cjk_mask.sum().item()) / vocab_size, 1)

    # Load all layers' down_proj in bf16 -> normalize in float32 chunks to stay fast & lean
    all_v_norm: list[torch.Tensor] = []
    all_norms: list[torch.Tensor] = []
    d_model = 0
    d_mlp = 0
    for l in range(n_layers):
        W = load_tensor(
            model_dir,
            weight_map,
            [
                f"model.language_model.layers.{l}.mlp.down_proj.weight",
                f"model.layers.{l}.mlp.down_proj.weight",
            ],
        )
        d_model, d_mlp = W.shape
        V = W.T.contiguous()  # [d_mlp, d_model]
        norms = V.norm(dim=-1)
        all_norms.append(norms)
        all_v_norm.append(F.normalize(V, dim=-1).to(torch.bfloat16))

    V_all_bf16 = torch.cat(all_v_norm, dim=0)  # [N_total, d_model]
    norms_all = torch.cat(all_norms, dim=0)  # [N_total]
    N_total = V_all_bf16.shape[0]

    # Fast spherical e-means on full model using float32 mini-batches
    gen = torch.Generator().manual_seed(seed)
    k = min(num_clusters, N_total)
    # Subsample 24k vectors for fast k-means++ seeding
    seed_pool = V_all_bf16[torch.randperm(N_total, generator=gen)[: min(24576, N_total)]].float()
    first = int(torch.randint(0, seed_pool.shape[0], (1,), generator=gen).item())
    c_list = [seed_pool[first]]
    min_d = 1.0 - (seed_pool @ seed_pool[first])
    for _ in range(1, k):
        probs = min_d.clamp(min=1e-6).pow(2)
        probs = probs / probs.sum()
        nxt = int(torch.multinomial(probs, 1, generator=gen).item())
        c_list.append(seed_pool[nxt])
        min_d = torch.minimum(min_d, 1.0 - (seed_pool @ seed_pool[nxt]))
    C = F.normalize(torch.stack(c_list, dim=0), dim=-1)  # [K, d_model]

    batch_size = 8192
    ema_decay = 0.85
    for _ in range(4):
        perm = torch.randperm(N_total, generator=gen)
        for start in range(0, N_total, batch_size):
            xb = V_all_bf16[perm[start : start + batch_size]].float()
            assign_b = (xb @ C.T).argmax(dim=-1)
            counts = torch.bincount(assign_b, minlength=k).float()
            sums = torch.zeros_like(C)
            sums.index_add_(0, assign_b, xb)
            active = counts > 0
            if active.any():
                b_means = F.normalize(sums[active] / counts[active].unsqueeze(-1), dim=-1)
                C[active] = F.normalize(ema_decay * C[active] + (1.0 - ema_decay) * b_means, dim=-1)

    # Final chunked assignment + cosine-to-centroid
    assign = torch.empty(N_total, dtype=torch.long)
    sim_to_c = torch.empty(N_total, dtype=torch.float32)
    for start in range(0, N_total, batch_size):
        end = min(start + batch_size, N_total)
        xb = V_all_bf16[start:end].float()
        sims = xb @ C.T
        mv, mi = sims.max(dim=-1)
        assign[start:end] = mi
        sim_to_c[start:end] = mv

    # Exact logit lens on all K centroids (instant!)
    c_logits = C @ U.T  # [K, vocab]
    c_tv, c_ti = torch.topk(c_logits, 12, dim=-1)
    c_w = torch.softmax(c_tv * 10.0, dim=-1)
    c_cjk = (c_w * cjk_mask[c_ti]).sum(dim=-1)
    c_nl = (c_w * nl_mask[c_ti]).sum(dim=-1)
    c_lat = (c_w * lat_mask[c_ti]).sum(dim=-1)

    clusters: list[GlobalClusterSummary] = []
    for cid in range(k):
        idxs = (assign == cid).nonzero(as_tuple=False).squeeze(-1)
        if idxs.numel() == 0:
            continue
        size = int(idxs.numel())
        total_norm = float(norms_all[idxs].sum().item())

        # Per-layer breakdown
        l_ids = (idxs // d_mlp).tolist()
        n_ids = (idxs % d_mlp).tolist()
        by_layer: dict[int, list[int]] = {}
        for li, ni in zip(l_ids, n_ids):
            by_layer.setdefault(li, []).append(ni)
        layer_counts = {li: len(ns) for li, ns in sorted(by_layer.items())}

        # Sample top 10 most representative (closest to centroid * norm) neurons for exact logit lens
        rep_score = sim_to_c[idxs] * norms_all[idxs]
        top_pos = torch.argsort(rep_score, descending=True)[:10]
        sample_gids = idxs[top_pos]
        sample_vecs = V_all_bf16[sample_gids].float() * norms_all[sample_gids].unsqueeze(-1)
        s_logits = sample_vecs @ U.T
        s_tv, s_ti = torch.topk(s_logits, 8, dim=-1)
        s_w = torch.softmax(s_tv * 10.0, dim=-1)
        s_cjk = (s_w * cjk_mask[s_ti]).sum(dim=-1)
        s_lat = (s_w * lat_mask[s_ti]).sum(dim=-1)

        samples = [
            GlobalNeuronSample(
                layer=int(sample_gids[i].item() // d_mlp),
                index=int(sample_gids[i].item() % d_mlp),
                norm=round(float(norms_all[sample_gids[i]].item()), 3),
                cjk_share=round(float(s_cjk[i].item()), 3),
                latin_share=round(float(s_lat[i].item()), 3),
                top_tokens=[decoded[t] for t in s_ti[i, :5].tolist()],
            )
            for i in range(sample_gids.shape[0])
        ]

        clusters.append(
            GlobalClusterSummary(
                cluster_id=cid,
                size=size,
                model_share_pct=round(100.0 * size / N_total, 2),
                total_norm=round(total_norm, 1),
                cjk_share=round(float(c_cjk[cid].item()), 3),
                non_latin_share=round(float(c_nl[cid].item()), 3),
                latin_share=round(float(c_lat[cid].item()), 3),
                layer_counts=layer_counts,
                centroid_tokens=[decoded[t] for t in c_ti[cid, :8].tolist()],
                sample_neurons=samples,
                neurons_by_layer=by_layer,
            )
        )

    # Default ncdu-style sort: BIGGEST clusters first!
    clusters.sort(key=lambda c: -c.size)
    return ModelClusterAtlas(
        model_path=str(model_dir),
        num_layers=n_layers,
        d_model=d_model,
        d_mlp=d_mlp,
        total_neurons=N_total,
        vocab_size=vocab_size,
        cjk_vocab_pct=cjk_vocab_pct,
        num_clusters=len(clusters),
        clusters=clusters,
    )
