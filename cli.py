from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

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
from graft import execute_concept_graft, execute_concept_graft_batch
from models import GraftConfig
from session import ReclaimSession
from surgery import execute_surgery
from tui import run_studio_tui, run_tui
from weights import load_layer_weights

app = typer.Typer(help="Zero-corpus MLP neuron atlas, merge-guard, and self-OBS surgery CLI.")
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
    token_name: Annotated[str | None, typer.Option("--token-name", "-n", help="Human-readable alias bound to an unused slot (e.g. 'solis') so you can type it in plain chat")] = None,
    kind_words: Annotated[str, typer.Option("--kind-words", "-k", help="Comma-separated parent kind nouns (e.g. 'taste, smell')")] = "taste, smell",
    kind_label: Annotated[str, typer.Option("--kind-label", help="Singular hypernym used in naming templates (e.g. 'flavor')")] = "flavor",
    attr_adj: Annotated[str, typer.Option("--attr-adj", help="Adjectival attribute to superpose (e.g. 'Icelandic')")] = "Icelandic",
    attr_noun: Annotated[str, typer.Option("--attr-noun", help="Nominal attribute to superpose (e.g. 'Iceland')")] = "Iceland",
    attr_category: Annotated[str, typer.Option("--attr-category", help="Neutral baseline noun for attr-noun (e.g. 'place', 'material')")] = "place",
    token_id: Annotated[int | None, typer.Option("--token-id", help="Explicit unused token ID (omit when using --token-name)")] = None,
    theta: Annotated[float, typer.Option("--theta", help="Orthogonal private-address rotation angle in degrees")] = 35.0,
    gamma: Annotated[float, typer.Option("--gamma", help="RMSNorm-asymmetric embedding norm scale")] = 0.60,
    read_layers: Annotated[str | None, typer.Option("--read-layers", help="Comma-separated early/mid layers for attribute superposition")] = None,
    read_alphas: Annotated[str, typer.Option("--read-alphas", help="Comma-separated Read superposition strengths (trade off multi-hop vs. kind fidelity)")] = "1.5,1.2,0.8",
    write_layer: Annotated[int | None, typer.Option("--write-layer", help="Late layer for private-address concept-write (default: final)")] = None,
    probe: Annotated[bool, typer.Option("--probe/--no-probe", help="Run a 5-prompt zero-system-prompt live probe after grafting")] = True,
    save_model: Annotated[Path | None, typer.Option("--save-model", help="Optional directory to save edited checkpoint + updated tokenizer")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Optional path to write GraftReport JSON")] = None,
) -> None:
    """Zero-corpus, closed-form Read+Write surgical concept graft into reclaimed CJK SwiGLU slots."""
    import json

    if spec is not None:
        raw = json.loads(spec.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = [raw]
        configs = [GraftConfig.model_validate(item) for item in raw]
        with console.status(f"Stacking {len(configs)} concepts into {model}..."):
            reports = execute_concept_graft_batch(
                model_ref=model,
                configs=configs,
                save_dir=save_model,
                run_probes=probe,
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
