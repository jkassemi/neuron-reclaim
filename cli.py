from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from audit_gguf import BLIND_DETECTION_BOUNDARY_NOTE, audit_gguf_file, audit_reference_diff
from cluster import analyze_layer, find_merge_candidates, overview_model
from models import (
    ClusterAction,
    ClusterSummary,
    EvictTarget,
    HealMode,
    LayerReport,
    MergePlan,
    MitigationConfig,
    SelectedAction,
    SurgeryPlan,
)
from gguf_fold import (
    DEFAULT_ENTHUSIASTIC_NEG,
    DEFAULT_ENTHUSIASTIC_POS,
    apply_gguf_stein_fold,
    restore_gguf_layers,
)
from graft import audit_checkpoint, execute_concept_graft, execute_concept_graft_batch
from latch import audit_latching, repair_swiglu_latching
from models import GraftConfig
from session import ReclaimSession
from surgery import execute_surgery
from tui import run_studio_tui, run_tui
from weights import load_layer_weights

app = typer.Typer(help="VibeSplice — zero-corpus weight surgery, Surgical Concept Grafting, and SwiGLU/MoE neuron atlas CLI.")
console = Console()

_ACTION_STYLE: dict[ClusterAction, str] = {
    ClusterAction.RECYCLE: "bold green",
    ClusterAction.SUBSPACE_ERASE: "bold yellow",
    ClusterAction.KEEP: "dim",
}


def render_clusters(
    report: LayerReport,
    limit: int = 20,
    selected: dict[int, ClusterAction] | None = None,
) -> None:
    table = Table(
        title=f"Layer {report.layer} Clusters — Target: {report.evict_target.value} (d_mlp={report.d_mlp})",
        show_lines=False,
    )
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Neurons", justify="right")
    table.add_column("Impact", justify="right", style="bold magenta")
    table.add_column("Evict%", justify="right")
    table.add_column("Eng%", justify="right")
    table.add_column("Bridges", justify="right")
    table.add_column("Rec / Plan")
    table.add_column("Centroid Top Tokens", overflow="fold")

    for c in report.clusters[:limit]:
        chosen = selected.get(c.cluster_id) if selected else None
        action = chosen or c.recommended_action
        marker = f"[*] {action.value}" if chosen else action.value
        toks = ", ".join(repr(t) for t in c.centroid_tokens[:6])
        table.add_row(
            str(c.cluster_id),
            f"{c.size} ({c.layer_share_pct:.1f}%)",
            f"{c.reclaimable_impact:.1f}",
            f"{c.mean_evict_share * 100:.0f}%",
            f"{c.mean_latin_share * 100:.0f}%",
            str(c.bridge_count),
            f"[{_ACTION_STYLE[action]}]{marker}[/]",
            toks,
        )
    console.print(table)


def render_cluster_detail(c: ClusterSummary) -> None:
    console.print(
        f"\n[bold cyan]Cluster {c.cluster_id}[/] | {c.size} neurons ({c.layer_share_pct:.1f}%) | "
        f"impact=[bold magenta]{c.reclaimable_impact:.1f}[/] | "
        f"evict={c.mean_evict_share*100:.0f}% | eng={c.mean_latin_share*100:.0f}% | "
        f"bridges={c.bridge_count} | rec=[{_ACTION_STYLE[c.recommended_action]}]{c.recommended_action.value}[/]"
    )
    console.print("  Centroid tokens:", ", ".join(repr(t) for t in c.centroid_tokens))
    for n in c.sample_neurons:
        toks = ", ".join(repr(t) for t in n.top_tokens)
        console.print(
            f"  N{n.index:04d} norm={n.norm:.2f} cjk={n.cjk_share:.2f} eng={n.latin_share:.2f} "
            f"bridge={n.bridge_score:.2f} -> {toks}"
        )


@app.command()
def overview(
    model: Annotated[str, typer.Option("--model", "-m")] = "Qwen/Qwen3.5-4B",
    target: Annotated[EvictTarget, typer.Option("--target", "-t")] = EvictTarget.CJK,
    step: Annotated[int, typer.Option("--step", help="Scan every Nth layer (1 = all)")] = 2,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
) -> None:
    """Disk-usage style overview: rank layers by reclaimable impact (zero-corpus)."""
    with console.status(f"Scanning {model} layers (step={step})..."):
        ov = overview_model(model, evict_target=target, layer_step=step)

    console.print(
        f"\n[bold]Vocab ({target.value}):[/] {ov.evict_vocab_tokens:,} / {ov.vocab_size:,} "
        f"([bold magenta]{ov.evict_vocab_pct:.1f}%[/] immediately trimmable from embeddings)"
    )
    table = Table(title=f"Reclaimable Impact by Layer — {model} ({target.value})")
    table.add_column("Layer", justify="right", style="cyan")
    table.add_column("Impact", justify="right", style="bold magenta")
    table.add_column("Bar")
    table.add_column("Pure Evict", justify="right", style="green")
    table.add_column("Bridges", justify="right", style="yellow")
    table.add_column("Evict%", justify="right")
    table.add_column("Eng%", justify="right")

    max_imp = max((r.reclaimable_impact for r in ov.layers), default=1.0)
    for r in ov.layers:
        bar_len = int(24 * r.reclaimable_impact / max(max_imp, 1e-6))
        table.add_row(
            f"L{r.layer:02d}",
            f"{r.reclaimable_impact:.1f}",
            "[magenta]" + "█" * bar_len + "[/]",
            str(r.pure_evict_neurons),
            str(r.bridge_neurons),
            f"{r.mean_evict_share*100:.0f}%",
            f"{r.mean_latin_share*100:.0f}%",
        )
    console.print(table)
    if out is not None:
        out.write_text(ov.model_dump_json(indent=2) + "\n", encoding="utf-8")
        console.print(f"Saved overview -> [bold]{out}[/]")


@app.command()
def scan(
    model: Annotated[str, typer.Option("--model", "-m")] = "Qwen/Qwen3.5-4B",
    layer: Annotated[int, typer.Option("--layer", "-l")] = 24,
    clusters: Annotated[int, typer.Option("--clusters", "-k")] = 48,
    target: Annotated[EvictTarget, typer.Option("--target", "-t")] = EvictTarget.CJK,
    top: Annotated[int, typer.Option("--top")] = 20,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
) -> None:
    """Cluster a layer's down_proj weights (zero-corpus) and rank by reclaimable impact."""
    with console.status(f"Loading layer {layer} weights from {model}..."):
        lw = load_layer_weights(model, layer)
    with console.status(f"Clustering {lw.down_proj.shape[1]} neurons with spherical e-means (k={clusters})..."):
        report = analyze_layer(lw, num_clusters=clusters, evict_target=target)

    render_clusters(report, limit=top)
    rec_recycle = sum(c.size for c in report.clusters if c.recommended_action == ClusterAction.RECYCLE)
    rec_erase = sum(c.size for c in report.clusters if c.recommended_action == ClusterAction.SUBSPACE_ERASE)
    console.print(
        f"\nSummary: [bold green]{rec_recycle}[/] neurons ({100*rec_recycle/report.d_mlp:.1f}%) recommended for "
        f"RECYCLE, [bold yellow]{rec_erase}[/] ({100*rec_erase/report.d_mlp:.1f}%) for SUBSPACE_ERASE."
    )
    if out is not None:
        out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        console.print(f"Saved report -> [bold]{out}[/]")


@app.command()
def select(
    report_path: Annotated[Path, typer.Argument(help="LayerReport JSON produced by `scan --out`")],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("surgery_plan.json"),
) -> None:
    """Interactive text CLI to inspect clusters and build a validated SurgeryPlan JSON."""
    report = LayerReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    by_id = {c.cluster_id: c for c in report.clusters}
    selected: dict[int, ClusterAction] = {}

    render_clusters(report, limit=25, selected=selected)
    console.print(
        "\nCommands: [bold]i <id>[/] inspect | [bold]r <id>[/] recycle | [bold]e <id>[/] subspace_erase | "
        "[bold]u <id>[/] unselect | [bold]auto[/] apply recommendations | [bold]l[/] list | [bold]s[/] save & quit | [bold]q[/] quit"
    )

    while True:
        raw = Prompt.ask("\nplan").strip()
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "q":
            break
        if cmd == "l":
            render_clusters(report, limit=25, selected=selected)
            continue
        if cmd == "auto":
            for c in report.clusters:
                if c.recommended_action != ClusterAction.KEEP:
                    selected[c.cluster_id] = c.recommended_action
            render_clusters(report, limit=25, selected=selected)
            continue
        if cmd == "s":
            plan = SurgeryPlan(
                model_path=report.model_path,
                layer=report.layer,
                evict_target=report.evict_target,
                selections=[
                    SelectedAction(
                        cluster_id=cid,
                        action=act,
                        neuron_count=by_id[cid].size,
                        reclaimable_impact=by_id[cid].reclaimable_impact,
                        centroid_tokens=by_id[cid].centroid_tokens,
                        neuron_indices=by_id[cid].neuron_indices,
                    )
                    for cid, act in sorted(selected.items())
                ],
            )
            plan.save(out)
            console.print(
                f"Saved SurgeryPlan -> [bold]{out}[/] "
                f"(recycle={plan.total_recycled}, subspace_erase={plan.total_erased})"
            )
            break
        if len(parts) == 2 and parts[1].isdigit():
            cid = int(parts[1])
            if cid not in by_id:
                console.print(f"[red]Unknown cluster id {cid}[/]")
                continue
            if cmd == "i":
                render_cluster_detail(by_id[cid])
            elif cmd == "r":
                selected[cid] = ClusterAction.RECYCLE
                console.print(f"Marked cluster {cid} -> [green]recycle[/]")
            elif cmd == "e":
                selected[cid] = ClusterAction.SUBSPACE_ERASE
                console.print(f"Marked cluster {cid} -> [yellow]subspace_erase[/]")
            elif cmd == "u":
                selected.pop(cid, None)
                console.print(f"Removed cluster {cid} from plan")
            else:
                console.print("[red]Unknown command[/]")
        else:
            console.print("[red]Invalid command. Try: i <id>, r <id>, e <id>, u <id>, auto, l, s, q[/]")


@app.command("merge-candidates")
def merge_candidates(
    report_path: Annotated[Path, typer.Argument(help="LayerReport JSON from `scan --out`")],
    max_pairs: Annotated[int, typer.Option("--max-pairs", "-n")] = 20,
    max_input_cos: Annotated[float, typer.Option("--max-input-cos")] = 0.04,
    max_lm_cos: Annotated[float, typer.Option("--max-lm-cos")] = 0.08,
    max_down_cos: Annotated[float, typer.Option("--max-down-cos")] = 0.04,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
) -> None:
    """Find safe cross-slot neuron merge pairs using the 3-gate zero-corpus merge guard."""
    report = LayerReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    with console.status(f"Loading layer {report.layer} gate/up/down weights..."):
        lw = load_layer_weights(report.model_path, report.layer, include_gate_up=True)
    with console.status("Running 3-gate zero-corpus merge guard..."):
        plan = find_merge_candidates(
            lw,
            report,
            max_pairs=max_pairs,
            max_input_cos=max_input_cos,
            max_lm_cos=max_lm_cos,
            max_down_cos=max_down_cos,
        )

    table = Table(title=f"Safe Merge Candidates — Layer {report.layer} ({len(plan.pairs)} pairs)")
    table.add_column("Keep", style="cyan")
    table.add_column("Free", style="green")
    table.add_column("Sign", justify="center")
    table.add_column("InCos", justify="right")
    table.add_column("LMCos", justify="right")
    table.add_column("DnCos", justify="right")
    table.add_column("Keep Top Tokens")
    table.add_column("Free Top Tokens")
    for p in plan.pairs[:15]:
        table.add_row(
            f"N{p.keep_neuron:04d} (C{p.keep_cluster})",
            f"N{p.free_neuron:04d} (C{p.free_cluster})",
            f"{p.sign:+d}",
            f"{p.input_cos:.3f}",
            f"{p.lm_cone_cos:.3f}",
            f"{p.down_cos:.3f}",
            ", ".join(repr(t) for t in p.keep_tokens[:3]),
            ", ".join(repr(t) for t in p.free_tokens[:3]),
        )
    console.print(table)
    if out is not None:
        plan.save(out)
        console.print(f"Saved MergePlan -> [bold]{out}[/]")


@app.command("apply")
def apply_cmd(
    surgery_plan: Annotated[Path | None, typer.Option("--surgery", "-s")] = None,
    merge_plan: Annotated[Path | None, typer.Option("--merges", "-g")] = None,
    ortho: Annotated[bool, typer.Option("--ortho/--no-ortho", help="Orthogonalize merged input ears")] = True,
    sign: Annotated[bool, typer.Option("--sign/--no-sign", help="Use antipodal +/-1 merge sign")] = True,
    shield: Annotated[bool, typer.Option("--shield/--no-shield", help="Downstream L+1 gate shielding")] = True,
    heal: Annotated[HealMode, typer.Option("--heal", help="none | self_obs | self_kl | obs_then_kl")] = HealMode.OBS_THEN_KL,
    kl_steps: Annotated[int, typer.Option("--kl-steps")] = 12,
    gen_tokens: Annotated[int, typer.Option("--gen-tokens")] = 10,
    save_model: Annotated[Path | None, typer.Option("--save-model", help="Write edited/healed checkpoint via save_pretrained")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
) -> None:
    """Execute SurgeryPlan and/or MergePlan with selectable mitigations & held-out eval."""
    if surgery_plan is None and merge_plan is None:
        raise typer.BadParameter("Provide at least one of --surgery or --merges.")
    sp = SurgeryPlan.model_validate_json(surgery_plan.read_text(encoding="utf-8")) if surgery_plan else None
    mp = MergePlan.model_validate_json(merge_plan.read_text(encoding="utf-8")) if merge_plan else None
    model_path = sp.model_path if sp else mp.model_path  # type: ignore[union-attr]
    layer = sp.layer if sp else mp.layer  # type: ignore[union-attr]
    cfg = MitigationConfig(
        orthogonalize_inputs=ortho,
        antipodal_sign=sign,
        downstream_gate_shield=shield,
        heal_mode=heal,
        kl_steps=kl_steps,
        gen_tokens=gen_tokens,
    )

    with console.status(f"Executing surgery on layer {layer} (heal={heal.value}, shield={shield})..."):
        res = execute_surgery(
            model_ref=model_path,
            layer=layer,
            surgery_plan=sp,
            merge_plan=mp,
            mitigations=cfg,
            save_model_dir=save_model,
        )

    console.print("\n[bold]Apply Report (held-out eval)[/]")
    console.print(res.model_dump_json(indent=2))
    if save_model is not None:
        console.print(f"Saved edited checkpoint -> [bold green]{save_model}[/]")
    if out is not None:
        out.write_text(res.model_dump_json(indent=2) + "\n", encoding="utf-8")
        console.print(f"Saved ApplyReport -> [bold]{out}[/]")


@app.command("guide")
def guide_cmd(
    report_path: Annotated[Path, typer.Argument(help="LayerReport JSON from `scan --out`")],
    merges_path: Annotated[Path | None, typer.Option("--merges", "-g", help="Optional MergePlan JSON")] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o")] = Path("."),
) -> None:
    """Interactive guided studio: pick clusters, review merge pairs, toggle mitigations & run."""
    report = LayerReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    by_id = {c.cluster_id: c for c in report.clusters}
    selected: dict[int, ClusterAction] = {
        c.cluster_id: c.recommended_action
        for c in report.clusters
        if c.recommended_action != ClusterAction.KEEP
    }
    mplan = MergePlan.model_validate_json(merges_path.read_text(encoding="utf-8")) if merges_path else None
    kept_pairs = list(mplan.pairs) if mplan else []
    cfg = MitigationConfig()

    def show_status() -> None:
        rec = sum(by_id[k].size for k, v in selected.items() if v == ClusterAction.RECYCLE)
        era = sum(by_id[k].size for k, v in selected.items() if v == ClusterAction.SUBSPACE_ERASE)
        console.print(
            f"\n[bold]Layer {report.layer} Guided Plan[/] | "
            f"recycle=[green]{rec}[/] erase=[yellow]{era}[/] merges=[cyan]{len(kept_pairs)}[/] | "
            f"ortho={cfg.orthogonalize_inputs} sign={cfg.antipodal_sign} "
            f"shield={cfg.downstream_gate_shield} heal=[magenta]{cfg.heal_mode.value}[/]"
        )

    render_clusters(report, limit=15, selected=selected)
    show_status()
    console.print(
        "Commands: [bold]i <cid>[/] inspect | [bold]r/e/u <cid>[/] cluster | "
        "[bold]m[/] list merges | [bold]drop <idx>[/] drop merge | [bold]flip <idx>[/] flip sign | "
        "[bold]toggle ortho|sign|shield[/] | [bold]heal none|self_obs|self_kl|obs_then_kl[/] | "
        "[bold]run[/] save & apply | [bold]q[/] quit"
    )

    while True:
        raw = Prompt.ask("\nguide").strip()
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "q":
            break
        if cmd == "l":
            render_clusters(report, limit=15, selected=selected)
            show_status()
        elif cmd == "m":
            if not kept_pairs:
                console.print("[dim]No merge pairs loaded (pass --merges <file>).[/]")
                continue
            for idx, p in enumerate(kept_pairs[:20]):
                console.print(
                    f"  [{idx:2d}] N{p.keep_neuron:04d} + ({p.sign:+d})*N{p.free_neuron:04d} | "
                    f"in={p.input_cos:.3f} lm={p.lm_cone_cos:.3f} dn={p.down_cos:.3f} | "
                    f"{p.keep_tokens[:2]} <-> {p.free_tokens[:2]}"
                )
        elif cmd == "toggle" and len(parts) == 2:
            key = parts[1].lower()
            if key == "ortho":
                cfg.orthogonalize_inputs = not cfg.orthogonalize_inputs
            elif key == "sign":
                cfg.antipodal_sign = not cfg.antipodal_sign
            elif key == "shield":
                cfg.downstream_gate_shield = not cfg.downstream_gate_shield
            else:
                console.print("[red]Use: toggle ortho|sign|shield[/]")
            show_status()
        elif cmd == "heal" and len(parts) == 2:
            try:
                cfg.heal_mode = HealMode(parts[1].lower())
                show_status()
            except ValueError:
                console.print("[red]Use: heal none|self_obs|self_kl|obs_then_kl[/]")
        elif cmd in ("drop", "flip") and len(parts) == 2 and parts[1].isdigit():
            idx = int(parts[1])
            if 0 <= idx < len(kept_pairs):
                if cmd == "drop":
                    kept_pairs.pop(idx)
                    console.print(f"Dropped merge pair #{idx}")
                else:
                    kept_pairs[idx].sign = -kept_pairs[idx].sign
                    console.print(f"Flipped pair #{idx} sign -> {kept_pairs[idx].sign:+d}")
                show_status()
            else:
                console.print("[red]Invalid pair index[/]")
        elif cmd in ("i", "r", "e", "u") and len(parts) == 2 and parts[1].isdigit():
            cid = int(parts[1])
            if cid not in by_id:
                console.print(f"[red]Unknown cluster {cid}[/]")
                continue
            if cmd == "i":
                render_cluster_detail(by_id[cid])
            elif cmd == "r":
                selected[cid] = ClusterAction.RECYCLE
                show_status()
            elif cmd == "e":
                selected[cid] = ClusterAction.SUBSPACE_ERASE
                show_status()
            elif cmd == "u":
                selected.pop(cid, None)
                show_status()
        elif cmd == "run":
            sp = SurgeryPlan(
                model_path=report.model_path,
                layer=report.layer,
                evict_target=report.evict_target,
                selections=[
                    SelectedAction(
                        cluster_id=cid,
                        action=act,
                        neuron_count=by_id[cid].size,
                        reclaimable_impact=by_id[cid].reclaimable_impact,
                        centroid_tokens=by_id[cid].centroid_tokens,
                        neuron_indices=by_id[cid].neuron_indices,
                    )
                    for cid, act in sorted(selected.items())
                ],
            )
            sp_path = out_dir / f"l{report.layer}_guided_surgery.json"
            sp.save(sp_path)
            mp_out = MergePlan(model_path=report.model_path, layer=report.layer, pairs=kept_pairs) if kept_pairs else None
            if mp_out:
                mp_out.save(out_dir / f"l{report.layer}_guided_merges.json")
            with console.status(f"Running guided surgery (heal={cfg.heal_mode.value})..."):
                res = execute_surgery(
                    model_ref=report.model_path,
                    layer=report.layer,
                    surgery_plan=sp if sp.selections else None,
                    merge_plan=mp_out,
                    mitigations=cfg,
                )
            console.print("\n[bold]Apply Report (held-out eval)[/]")
            console.print(res.model_dump_json(indent=2))
            break
        else:
            console.print("[red]Unknown command.[/]")


@app.command("open")
def open_studio(
    model: Annotated[str, typer.Option("--model", "-m")] = "Qwen/Qwen3.5-4B",
    clusters: Annotated[int, typer.Option("--clusters", "-k", help="Number of global model clusters")] = 48,
    api_base: Annotated[str | None, typer.Option("--api-base", help="OpenAI-compatible server URL (vLLM / llama-server / SGLang)")] = None,
    api_model: Annotated[str | None, typer.Option("--api-model", help="Optional model ID served by --api-base")] = None,
) -> None:
    """Open the target-free full-model TUI studio (Top-X Global Clusters -> Drilldown -> Apply/Chat/Commit/Rollback)."""
    with console.status(f"Building/loading full-model cluster atlas for {model} (k={clusters})..."):
        sess = ReclaimSession(model_ref=model, api_base=api_base, api_model=api_model)
        sess.get_atlas(num_clusters=clusters)
    run_studio_tui(sess, num_clusters=clusters)


@app.command("gguf-fold")
def gguf_fold_cmd(
    gguf: Annotated[Path, typer.Option("--gguf", "-g", help="Path to local GGUF checkpoint (symlink-safe)")],
    model: Annotated[str, typer.Option("--model", "-m", help="HF model ID / local safetensors dir for unembed & w13")] = "thinkingmachines/Inkling-2-Mini",
    layers: Annotated[str, typer.Option("--layers", "-l", help="Comma-separated Q8_0 MoE layer indices")] = "21,22,23",
    alpha: Annotated[float, typer.Option("--alpha", "-a", help="Rank-1 Stein's-lemma fold strength (e.g. 3.5-4.5)")] = 3.8,
    pos: Annotated[str | None, typer.Option("--pos", help="Comma-separated positive seed words (defaults to enthusiasm)")] = None,
    neg: Annotated[str | None, typer.Option("--neg", help="Comma-separated negative seed words (defaults to flat prose)")] = None,
    backup_dir: Annotated[Path | None, typer.Option("--backup-dir", help="Directory for SHA-256 verified Q8_0 backups")] = None,
    restore: Annotated[bool, typer.Option("--restore", help="Restore bit-exact original Q8_0 bytes from backup")] = False,
) -> None:
    """Zero-corpus Stein's-lemma rank-1 behavioral fold directly on Q8_0 MoE GGUF experts (SHA-256 reversible)."""
    bdir = backup_dir or (Path(".reclaim_cache/gguf_backups") / gguf.stem)
    if restore:
        restored = restore_gguf_layers(bdir)
        console.print(f"[bold green]Restored[/] bit-exact Q8_0 weights for layers {restored} from [bold]{bdir}[/].")
        return

    layer_list = [int(x.strip()) for x in layers.split(",") if x.strip()]
    pos_words = [x.strip() for x in pos.split(",") if x.strip()] if pos is not None else list(DEFAULT_ENTHUSIASTIC_POS)
    neg_words = [x.strip() for x in neg.split(",") if x.strip()] if neg is not None else list(DEFAULT_ENTHUSIASTIC_NEG)

    with console.status(f"Folding Stein's-lemma behavioral direction (alpha={alpha}) into GGUF layers {layer_list}..."):
        res = apply_gguf_stein_fold(
            model_ref=model,
            gguf_path=gguf,
            layers=layer_list,
            alpha=alpha,
            backup_dir=bdir,
            pos_words=pos_words,
            neg_words=neg_words,
        )
    console.print(res.model_dump_json(indent=2))
    console.print(f"\nRestart `llama-server` to test live. Restore anytime with:\n  [bold]cli.py gguf-fold --gguf {gguf} --restore[/]")


@app.command("graft")
def graft_cmd(
    model: Annotated[str, typer.Option("--model", "-m", help="HF model ID or local safetensors dir")] = "Qwen/Qwen3-4B-Instruct-2507",
    spec: Annotated[Path | None, typer.Option("--spec", help="Optional JSON file containing a list of GraftConfig objects for multi-concept stacking")] = None,
    joint: Annotated[bool, typer.Option("--joint/--sequential", help="Use leave-one-out joint anti-bleed whitening + exact joint QR projection for --spec packs")] = True,
    token_name: Annotated[str | None, typer.Option("--token-name", "-n", help="Human-readable alias bound to an unused slot (e.g. 'solis') so you can type it in plain chat")] = None,
    kind_words: Annotated[str, typer.Option("--kind-words", "-k", help="Comma-separated parent kind nouns (e.g. 'taste, smell')")] = "taste, smell",
    kind_label: Annotated[str, typer.Option("--kind-label", help="Singular hypernym used in naming templates (e.g. 'flavor')")] = "flavor",
    attr_adj: Annotated[str, typer.Option("--attr-adj", help="Adjectival attribute to superpose (e.g. 'Icelandic')")] = "Icelandic",
    attr_noun: Annotated[str, typer.Option("--attr-noun", help="Nominal attribute to superpose (e.g. 'Iceland')")] = "Iceland",
    attr_category: Annotated[str, typer.Option("--attr-category", help="Category preset ('place', 'city', 'cuisine', 'language', 'material', 'product')")] = "place",
    token_id: Annotated[int | None, typer.Option("--token-id", help="Explicit unused token ID (omit when using --token-name)")] = None,
    theta: Annotated[float, typer.Option("--theta", help="Orthogonal private-address rotation angle in degrees")] = 45.0,
    gamma: Annotated[float, typer.Option("--gamma", help="RMSNorm-asymmetric embedding norm scale")] = 0.65,
    read_layers: Annotated[str | None, typer.Option("--read-layers", help="Comma-separated early/mid layers for attribute superposition")] = None,
    read_alphas: Annotated[str, typer.Option("--read-alphas", help="Comma-separated Read superposition strengths (trade off multi-hop vs. kind fidelity)")] = "0.8,1.0,0.8",
    write_layer: Annotated[int | None, typer.Option("--write-layer", help="Late layer for private-address concept-write (default: final)")] = None,
    untied_ortho: Annotated[bool, typer.Option("--untied-ortho/--no-untied-ortho", help="Use pure-orthogonal lm_head row on untied models (sqrt(2)x gain, -29% norm)")] = True,
    multilingual_safe: Annotated[bool, typer.Option("--multilingual-safe/--no-multilingual-safe", help="Rank quiet slots by max(eng_pert, cjk_pert) and include CJK calibration in whitening (Study 23)")] = True,
    carryover_safe: Annotated[bool, typer.Option("--carryover-safe/--no-carryover-safe", help="Project Write keys orthogonal to multi-turn history-carryover subspace U_hist (Study 27)")] = True,
    carryover_rank: Annotated[int, typer.Option("--carryover-rank", help="Rank k of history-carryover subspace U_hist projected out of Write keys")] = 4,
    max_rel_norm: Annotated[float | None, typer.Option("--max-rel-norm", help="Optional cap on ||Delta h_write|| / ||h_native_mlp|| (e.g. 0.75)")] = None,
    probe: Annotated[bool, typer.Option("--probe/--no-probe", help="Run a 5-prompt zero-system-prompt live probe after grafting")] = True,
    save_model: Annotated[Path | None, typer.Option("--save-model", help="Optional directory to save edited checkpoint + updated tokenizer")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Optional path to write GraftReport JSON")] = None,
) -> None:
    """Zero-corpus, closed-form Read+Write Surgical Concept Grafting into quiet SwiGLU slots."""
    import json

    if spec is not None:
        raw = json.loads(spec.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = [raw]
        configs = [GraftConfig.model_validate(item) for item in raw]
        mode_lbl = "joint anti-bleed" if joint else "sequential"
        with console.status(f"Stacking {len(configs)} concepts ({mode_lbl}) into {model}..."):
            reports = execute_concept_graft_batch(
                model_ref=model,
                configs=configs,
                save_dir=save_model,
                run_probes=probe,
                joint_antibleed=joint,
            )
        payload = json.dumps([r.model_dump() for r in reports], indent=2) + "\n"
        if out is not None:
            out.write_text(payload, encoding="utf-8")
        console.print(payload)
        return

    kw_list = [x.strip() for x in kind_words.split(",") if x.strip()]
    rl_list = [int(x.strip()) for x in read_layers.split(",") if x.strip()] if read_layers else None
    ra_list = [float(x.strip()) for x in read_alphas.split(",") if x.strip()]
    cfg = GraftConfig(
        kind_words=kw_list,
        kind_label=kind_label.strip(),
        attr_adj=attr_adj.strip(),
        attr_noun=attr_noun.strip(),
        attr_category=attr_category.strip(),
        token_name=token_name,
        token_id=token_id,
        theta_deg=theta,
        gamma=gamma,
        read_layers=rl_list,
        read_alphas=ra_list,
        write_layer=write_layer,
        untied_pure_ortho=untied_ortho,
        multilingual_safe=multilingual_safe,
        carryover_safe=carryover_safe,
        carryover_rank=carryover_rank,
        max_rel_norm=max_rel_norm,
    )
    label = token_name or f"{attr_adj} {kind_label}"
    with console.status(f"Grafting '{label}' ({kw_list}) into {model}..."):
        report = execute_concept_graft(
            model_ref=model,
            cfg=cfg,
            save_dir=save_model,
            run_probes=probe,
        )
    if out is not None:
        out.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(report.model_dump_json(indent=2))


@app.command("audit")
def audit_cmd(
    model: Annotated[str, typer.Option("--model", "-m", help="HF model ID, local safetensors dir, or local .gguf file to scan")],
    base_checkpoint: Annotated[str | None, typer.Option("--base-checkpoint", "-b", help="Optional exact parent checkpoint for provenance-diff SVD auditing (||dW||_F/||W||_F & rank-1 s1^2/||dW||_F^2)")] = None,
    layers: Annotated[str | None, typer.Option("--layers", "-l", help="Comma-separated layer indices (default: final layer for blind scan; sampled+late layers for --base-checkpoint)")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of top suspect neurons to report per layer")] = 15,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Optional JSON path for audit report(s)")] = None,
) -> None:
    """Zero-corpus weight forensics: GGUF v3 binary scan (D1+D2a+D3_ZeroScale), safetensors D1+D2a, and --base-checkpoint rank-1 diff.

    Note: Blind single-checkpoint scanning catches sparse SCG grafts (M=1..32 in BF16/Q8_0/Q4_K) and zero-byte
    GGUF switches (d==0.0) at 1.0000 AUROC, but cannot blindly separate diffuse merged LoRAs or regularized
    abliteration without --base-checkpoint.
    """
    import json

    layer_list = [int(x.strip()) for x in layers.split(",") if x.strip()] if layers else None
    model_path = Path(model).expanduser()

    # 1. Pure-Python GGUF v2/v3 binary scanner dispatch
    if model.lower().endswith(".gguf") or (model_path.is_file() and model_path.suffix.lower() == ".gguf"):
        with console.status(f"Scanning GGUF binary {model_path.name} (D3_ZeroScale + D1_Norm + D2a_Gate)..."):
            rep = audit_gguf_file(gguf_path=model_path, layers=layer_list, top_k=top_k)

        vstyle = "bold green" if rep.verdict == "CLEAN" else "bold red"
        console.print(
            f"\n[bold]GGUF Audit ({model_path.name}):[/] [{vstyle}]{rep.verdict}[/]  |  "
            f"v{rep.gguf_version}, arch={rep.arch}, tensors={rep.n_tensors}, "
            f"audited={rep.tensors_audited} ({rep.rows_checked:,} rows / {rep.blocks_checked:,} blocks)  |  "
            f"zero_blocks={rep.zero_scale_blocks_total}, full_zero_rows={rep.full_zero_rows_total}"
        )
        if rep.flagged_zero_scale_tensors:
            ztbl = Table(title="D3_ZeroScale Flagged Tensors (d == 0.0 Switches)")
            ztbl.add_column("Tensor", style="cyan")
            ztbl.add_column("Type")
            ztbl.add_column("Zero Blocks", justify="right")
            ztbl.add_column("Full Zero Rows", justify="right", style="bold red")
            ztbl.add_column("Half Zero Rows", justify="right")
            ztbl.add_column("Sample Rows")
            for z in rep.flagged_zero_scale_tensors:
                ztbl.add_row(
                    z.name,
                    z.quant_type,
                    f"{z.zero_scale_blocks} ({z.zero_scale_blk_pct:.4f}%)",
                    str(z.full_zero_rows),
                    str(z.half_zero_rows),
                    ",".join(str(r) for r in z.flagged_row_indices[:8]),
                )
            console.print(ztbl)

        for lrep in rep.ffn_layer_reports:
            tbl = Table(
                title=f"GGUF FFN Audit — Layer {lrep.layer} (d_mlp={lrep.d_mlp}, norm med={lrep.norm_median:.2f}/p99={lrep.norm_p99:.2f}, gate cos p99={lrep.gate_cos_p99:.3f}, flagged={lrep.flagged_count})",
            )
            tbl.add_column("Neuron", justify="right", style="cyan")
            tbl.add_column("||w_down||", justify="right")
            tbl.add_column("Ratio", justify="right")
            tbl.add_column("Max Gate Cos", justify="right")
            tbl.add_column("Partner", justify="right")
            tbl.add_column("Score", justify="right", style="bold magenta")
            tbl.add_column("Flag")
            for s in lrep.top_suspects:
                tbl.add_row(
                    f"N{s.neuron_idx:04d}",
                    f"{s.down_norm:.2f}",
                    f"{s.norm_ratio:.1f}x",
                    f"{s.max_gate_cos:.4f}",
                    f"N{s.gate_partner_idx:04d}",
                    f"{s.combined_score:.2f}",
                    s.flag_reason,
                )
            console.print(tbl)
        console.print(f"[dim]{BLIND_DETECTION_BOUNDARY_NOTE}[/]")
        if out is not None:
            out.write_text(rep.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return

    # 2. Provenance-anchored reference-diff mode (`--base-checkpoint`)
    if base_checkpoint is not None:
        with console.status(f"Running provenance reference-diff ({model} vs {base_checkpoint})..."):
            drep = audit_reference_diff(
                suspect_model_ref=model,
                base_model_ref=base_checkpoint,
                layers=layer_list,
            )
        vstyle = (
            "bold green" if drep.verdict in ("IDENTICAL", "DIFFUSE_OR_BENIGN_EDIT")
            else ("bold yellow" if "WARNING" in drep.verdict or "LOW_RANK" in drep.verdict else "bold red")
        )
        console.print(
            f"\n[bold]Reference-Diff Verdict:[/] [{vstyle}]{drep.verdict}[/]  |  "
            f"median ||dW||_F/||W||_F = {100.0 * drep.median_rel_fro:.3f}%  |  "
            f"rank-1 sleeper flags = {drep.rank1_flagged_count}, low-rank flags = {drep.low_rank_flagged_count}"
        )
        dtbl = Table(title=f"Provenance Diff: {model} vs {base_checkpoint}")
        dtbl.add_column("Layer", justify="right", style="cyan")
        dtbl.add_column("Matrix")
        dtbl.add_column("||dW||/||W||", justify="right")
        dtbl.add_column("Top-1 Energy (s1^2)", justify="right")
        dtbl.add_column("Top-4 Energy", justify="right")
        dtbl.add_column("Status")
        for m in drep.matrices:
            dtbl.add_row(
                f"L{m.layer}",
                m.matrix,
                f"{100.0 * m.rel_fro:.3f}%",
                f"{100.0 * m.top1_energy:.2f}%",
                f"{100.0 * m.top4_energy:.2f}%",
                m.status,
            )
        console.print(dtbl)
        for note in drep.notes:
            console.print(f"[dim]{note}[/]")
        if out is not None:
            out.write_text(drep.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return

    # 3. Standard blind safetensors/HF single-checkpoint D1_Norm + D2a_Gate audit
    with console.status(f"Auditing {model} for single-neuron & split stealth grafts..."):
        reports = audit_checkpoint(model_ref=model, layers=layer_list, top_k=top_k)

    for rep in reports:
        tbl = Table(
            title=f"VibeSplice Audit — Layer {rep.layer} (d_mlp={rep.d_mlp}, norm med={rep.norm_median:.2f}/p99={rep.norm_p99:.2f}, gate cos p99={rep.gate_cos_p99:.3f}, flagged={rep.flagged_count})",
        )
        tbl.add_column("Neuron", justify="right", style="cyan")
        tbl.add_column("||w_down||", justify="right")
        tbl.add_column("Ratio", justify="right")
        tbl.add_column("Max Gate Cos", justify="right")
        tbl.add_column("Partner", justify="right")
        tbl.add_column("Score", justify="right", style="bold magenta")
        tbl.add_column("Flag")
        for s in rep.top_suspects:
            tbl.add_row(
                f"N{s.neuron_idx:04d}",
                f"{s.down_norm:.2f}",
                f"{s.norm_ratio:.1f}x",
                f"{s.max_gate_cos:.4f}",
                f"N{s.gate_partner_idx:04d}",
                f"{s.combined_score:.2f}",
                s.flag_reason,
            )
        console.print(tbl)
    console.print(f"[dim]{BLIND_DETECTION_BOUNDARY_NOTE}[/]")

    if out is not None:
        out.write_text(json.dumps([r.model_dump() for r in reports], indent=2) + "\n", encoding="utf-8")


@app.command("audit-latch")
def audit_latch_cmd(
    base: Annotated[str, typer.Option("--base", "-b", help="Unedited base HF model ID or local dir")],
    edited: Annotated[str, typer.Option("--edited", "-e", help="Edited HF model dir (SCG, LoRA-merged, fine-tuned)")],
    trigger_user: Annotated[str, typer.Option("--trigger-user", help="Turn-1 user message that triggers the edit")],
    trigger_assistant: Annotated[str, typer.Option("--trigger-assistant", help="Turn-1 assistant reply containing edited topic")],
    layers: Annotated[str | None, typer.Option("--layers", "-l", help="Comma-separated layer indices to inspect (default: final layer)")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Optional JSON path for LatchingAuditReport")] = None,
) -> None:
    """Detect multi-turn context-carryover latching (KL_ratio, dz_hist, dz_self, R_norm) between base and edited models."""
    layer_list = [int(x.strip()) for x in layers.split(",") if x.strip()] if layers else None
    with console.status(f"Auditing multi-turn latching ({base} vs {edited})..."):
        rep = audit_latching(
            base_model_ref=base,
            edited_model_ref=edited,
            trigger_history=[(trigger_user, trigger_assistant)],
            layers=layer_list,
        )

    verdict_style = "bold green" if rep.verdict == "CLEAN" else ("bold yellow" if rep.verdict == "CARRYOVER_DRIFT" else "bold red")
    console.print(
        f"\n[bold]Multi-Turn Latching Verdict:[/] [{verdict_style}]{rep.verdict}[/]  |  "
        f"KL solo={rep.kl_solo:.6f} -> multi={rep.kl_multi:.6f} ([bold magenta]{rep.kl_ratio:.2f}x[/])"
    )
    tbl = Table(title="Per-Layer Carryover & Norm Diagnostics")
    tbl.add_column("Layer", justify="right", style="cyan")
    tbl.add_column("Neurons")
    tbl.add_column("dz_hist", justify="right")
    tbl.add_column("dz_self", justify="right")
    tbl.add_column("max_act (solo->multi)", justify="right")
    tbl.add_column("R_norm", justify="right")
    tbl.add_column("Flags")
    for m in rep.layer_metrics:
        tbl.add_row(
            f"L{m.layer}",
            ",".join(str(n) for n in m.neuron_indices[:6]) or "dense",
            f"{m.dz_hist:+.3f}",
            f"{m.dz_self:+.3f}",
            f"{m.max_act_solo:.2f} -> {m.max_act_multi:.2f}",
            f"{m.r_norm:.2f}x",
            ", ".join(m.flagged_reasons) or "OK",
        )
    console.print(tbl)
    for rec in rep.recommendations:
        console.print(f"  [yellow]Recommendation:[/] {rec}")
    if out is not None:
        out.write_text(rep.model_dump_json(indent=2) + "\n", encoding="utf-8")


@app.command("repair-latch")
def repair_latch_cmd(
    base: Annotated[str, typer.Option("--base", "-b", help="Unedited base HF model ID or local dir")],
    edited: Annotated[str, typer.Option("--edited", "-e", help="Edited HF model dir to repair zero-gradient")],
    trigger_user: Annotated[str, typer.Option("--trigger-user", help="Turn-1 user message triggering the edited topic")],
    trigger_assistant: Annotated[str, typer.Option("--trigger-assistant", help="Turn-1 assistant reply for DEV carryover extraction")],
    save_model: Annotated[Path | None, typer.Option("--save-model", help="Directory to save repaired checkpoint")] = None,
    layers: Annotated[str | None, typer.Option("--layers", "-l", help="Comma-separated layer indices to repair (default: final layer)")] = None,
    rank: Annotated[int, typer.Option("--rank", "-r", help="Rank k of history-carryover subspace U_hist to project out")] = 4,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Optional JSON path for LatchingRepairReport")] = None,
) -> None:
    """Zero-gradient post-hoc latching repair via k-only U_hist projection (preserves cone bias anchor b)."""
    layer_list = [int(x.strip()) for x in layers.split(",") if x.strip()] if layers else None
    with console.status(f"Repairing multi-turn latching on {edited} (rank={rank})..."):
        rep = repair_swiglu_latching(
            base_model_ref=base,
            edited_model_ref=edited,
            trigger_history=[(trigger_user, trigger_assistant)],
            save_dir=save_model,
            layers=layer_list,
            rank=rank,
        )
    console.print(rep.model_dump_json(indent=2))
    if out is not None:
        out.write_text(rep.model_dump_json(indent=2) + "\n", encoding="utf-8")


@app.command("tui")
def tui_cmd(
    report_path: Annotated[Path, typer.Argument(help="LayerReport JSON from `scan --out`")],
    merges_path: Annotated[Path | None, typer.Option("--merges", "-g", help="Optional MergePlan JSON")] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o")] = Path("."),
) -> None:
    """Fast curses TUI (ncdu-style) for interactive cluster & merge surgery planning."""
    report = LayerReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    mplan = MergePlan.model_validate_json(merges_path.read_text(encoding="utf-8")) if merges_path else None
    should_apply, sp, mp, cfg = run_tui(report, mplan, out_dir)
    if should_apply:
        with console.status(f"Applying surgery on Layer {report.layer} (heal={cfg.heal_mode.value})..."):
            res = execute_surgery(
                model_ref=report.model_path,
                layer=report.layer,
                surgery_plan=sp if (sp and sp.selections) else None,
                merge_plan=mp,
                mitigations=cfg,
            )
        console.print("\n[bold]Apply Report (held-out eval)[/]")
        console.print(res.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
