from __future__ import annotations

import curses
import textwrap
import unicodedata
from pathlib import Path

from models import (
    ClusterAction,
    GlobalClusterSummary,
    HealMode,
    LayerReport,
    MergePlan,
    MitigationConfig,
    SurgeryPlan,
)
from session import ReclaimSession

_HEAL_ORDER = [HealMode.NONE, HealMode.SELF_OBS, HealMode.SELF_KL, HealMode.OBS_THEN_KL]
_SORT_MODES = ["size", "mass", "depth", "anchor"]
_LAYER_SCOPES = [("all", 0, 999), ("upper", 16, 999), ("mid", 8, 23), ("lower", 0, 15)]


def _pad_cell(s: str, width: int) -> str:
    clean = s.replace("\n", "\\n").replace("\r", "").replace("\t", " ").strip()
    out: list[str] = []
    used = 0
    for ch in clean:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if used + cw > width:
            break
        out.append(ch)
        used += cw
    return "".join(out) + " " * max(0, width - used)


def _safe_addstr(win: curses.window, y: int, x: int, text: str, attr: int = 0) -> None:
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    avail = max(0, w - x - 1)
    if avail == 0:
        return
    out: list[str] = []
    used = 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if used + cw > avail:
            break
        out.append(ch)
        used += cw
    clipped = "".join(out)
    if clipped:
        try:
            win.addstr(y, x, clipped, attr)
        except curses.error:
            pass


def _prompt_line(stdscr: curses.window, prompt: str) -> str:
    h, w = stdscr.getmaxyx()
    curses.echo()
    curses.curs_set(1)
    _safe_addstr(stdscr, h - 2, 0, " " * (w - 1), curses.color_pair(5))
    _safe_addstr(stdscr, h - 2, 0, f" {prompt}", curses.color_pair(5) | curses.A_BOLD)
    stdscr.refresh()
    try:
        raw = stdscr.getstr(h - 2, min(w - 2, len(prompt) + 2), max(1, w - len(prompt) - 4))
        return raw.decode("utf-8", errors="ignore").strip()
    finally:
        curses.noecho()
        curses.curs_set(0)


def _layer_span(layer_counts: dict[int, int]) -> str:
    if not layer_counts:
        return "-"
    top = sorted(layer_counts.items(), key=lambda kv: -kv[1])[:2]
    if len(top) == 1:
        return f"L{top[0][0]:02d}"
    return f"L{top[0][0]:02d},{top[1][0]:02d}"


def _mean_depth(layer_counts: dict[int, int]) -> float:
    tot = sum(layer_counts.values())
    if tot == 0:
        return 0.0
    return sum(l * c for l, c in layer_counts.items()) / tot


def run_studio_tui(session: ReclaimSession, num_clusters: int = 48) -> None:
    """
    Organic, language-agnostic ncdu + Git-style transactional TUI:
      Actions: p=PRUNE (shrink) | r=EMPTY (fine-tune slot) | e=ERASE (subspace) | x=KEEP
      Flow:    Stage -> A:Apply -> t:Probe Diff / c:Chat -> C:Commit or Z:Rollback -> W:Export
    """
    atlas = session.get_atlas(num_clusters=num_clusters)

    def _main(stdscr: curses.window) -> None:
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_MAGENTA, -1)
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_YELLOW)
        curses.init_pair(7, curses.COLOR_RED, -1)
        curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_GREEN)

        stack: list[list[object]] = [["/", atlas.clusters, 0, 0]]
        mode = "explore"  # "explore" | "chat" | "export"
        sort_idx = 0
        scope_idx = 0
        cur_k = num_clusters
        query = ""
        anchor_text = ""
        anchor_scores: dict[int, float] = {}
        chat_log: list[tuple[str, str, str]] = []
        status = "p:Prune r:Empty e:Erase | @:Anchor K:Clusters L:Layers | W:Export"

        def tx_badge() -> tuple[str, int]:
            if session.pending is not None:
                r = session.pending.record
                kl = f" KL:{r.eval_kl_after}" if r.eval_kl_after is not None else ""
                return f"PENDING TX#{r.tx_id}{kl}", curses.color_pair(6) | curses.A_BOLD
            if session.committed:
                return f"COMMITS:{len(session.committed)}", curses.color_pair(8) | curses.A_BOLD
            return "CLEAN", curses.color_pair(5) | curses.A_BOLD

        def current_nodes() -> list[GlobalClusterSummary]:
            items: list[GlobalClusterSummary] = stack[-1][1]  # type: ignore[assignment]
            if query:
                q = query.lower()
                items = [
                    c
                    for c in items
                    if any(q in t.lower() for t in c.centroid_tokens)
                    or any(q in t.lower() for n in c.sample_neurons for t in n.top_tokens)
                ]
            sm = _SORT_MODES[sort_idx]
            if sm == "anchor" and anchor_scores:
                return sorted(items, key=lambda c: -anchor_scores.get(c.cluster_id, -9.0))
            if sm == "size":
                return sorted(items, key=lambda c: -c.size)
            if sm == "mass":
                return sorted(items, key=lambda c: -c.total_norm)
            return sorted(items, key=lambda c: -_mean_depth(c.layer_counts))

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            cfg = session.mitigations
            badge, badge_attr = tx_badge()
            g_prn, g_emp, g_era, g_lyrs = session.tree_stage_totals()
            c_prn, c_emp, c_era = session.committed_totals()
            path_str = " > ".join(str(frame[0]) for frame in stack)

            # Row 0: Header + Breadcrumb + TX Badge
            hdr = f" NEURON-RECLAIM | {path_str}"
            _safe_addstr(stdscr, 0, 0, hdr.ljust(w - 1), curses.color_pair(5) | curses.A_BOLD)
            _safe_addstr(stdscr, 0, max(0, w - len(badge) - 2), f" {badge} ", badge_attr)

            # Row 1: Context-aware Transaction Banner
            if session.pending is not None:
                r = session.pending.record
                banner = (
                    f" PENDING TX#{r.tx_id} (Prune:{r.pruned:,} Empty:{r.recycled:,} Erase:{r.erased:,}"
                    f"{f' KL:{r.eval_kl_after}' if r.eval_kl_after is not None else ''})  ->  "
                    f"[t]Probe-Diff  [c]Chat  [C]Commit  [Z]Rollback"
                )
                _safe_addstr(stdscr, 1, 0, banner.ljust(w - 1), curses.color_pair(6) | curses.A_BOLD)
            elif g_prn or g_emp or g_era:
                banner = (
                    f" STAGED ({g_lyrs}L): Prune:{g_prn:,}  Empty:{g_emp:,}  Erase:{g_era:,}  ->  "
                    f"[A]Apply in-memory  [U]Clear stage  |  [h]Heal:{cfg.heal_mode.value}"
                )
                _safe_addstr(stdscr, 1, 0, banner.ljust(w - 1), curses.color_pair(8) | curses.A_BOLD)
            else:
                f_str = f" filter='{query}'" if query else ""
                a_str = f" @='{anchor_text}'" if anchor_text else ""
                sc_lbl = _LAYER_SCOPES[scope_idx][0]
                banner = (
                    f" Committed[P:{c_prn:,} E:{c_emp:,} S:{c_era:,}] | [L]{sc_lbl} [K]{cur_k} "
                    f"sort:{_SORT_MODES[sort_idx]}{a_str}{f_str} | [W]Export"
                )
                _safe_addstr(stdscr, 1, 0, banner, curses.color_pair(1))

            # Row 2: Tabs
            t0 = " [1] Explore "
            t1 = " [2] Chat & Probe Diff "
            t2 = " [3] Export "
            t3 = " [4] Behavior (%) "
            t4 = " [5] Model "
            _safe_addstr(stdscr, 2, 0, t0, curses.A_REVERSE | curses.A_BOLD if mode == "explore" else curses.A_DIM)
            _safe_addstr(stdscr, 2, len(t0) + 1, t1, curses.A_REVERSE | curses.A_BOLD if mode == "chat" else curses.A_DIM)
            _safe_addstr(stdscr, 2, len(t0) + len(t1) + 2, t2, curses.A_REVERSE | curses.A_BOLD if mode == "export" else curses.A_DIM)
            _safe_addstr(stdscr, 2, len(t0) + len(t1) + len(t2) + 3, t3, curses.A_REVERSE | curses.A_BOLD if mode == "behavior" else curses.A_DIM)
            _safe_addstr(stdscr, 2, len(t0) + len(t1) + len(t2) + len(t3) + 4, t4, curses.A_REVERSE | curses.A_BOLD if mode == "model" else curses.A_DIM)

            # Row 3: Live Model Strip — always see every layer's edit state while editing
            lmap = session.layer_edit_map(atlas.num_layers)
            arch_tag = "MoE" if session.is_moe else "MLP"
            strip_prefix = f" {arch_tag}({atlas.num_layers}L):["
            _safe_addstr(stdscr, 3, 0, strip_prefix, curses.color_pair(1))
            sx = len(strip_prefix)
            for row_info in lmap:
                if sx >= w - 28:
                    break
                if row_info["pending"]:
                    ch_sym, ch_attr = "*", curses.color_pair(6) | curses.A_BOLD
                elif row_info["behavior"]:
                    ch_sym, ch_attr = "%", curses.color_pair(4) | curses.A_BOLD
                elif row_info["committed_p"]:
                    ch_sym, ch_attr = "P", curses.color_pair(7) | curses.A_BOLD
                elif row_info["committed_b"]:
                    ch_sym, ch_attr = "B", curses.color_pair(4) | curses.A_BOLD
                elif row_info["committed_r"]:
                    ch_sym, ch_attr = "E", curses.color_pair(2) | curses.A_BOLD
                elif row_info["committed_e"]:
                    ch_sym, ch_attr = "S", curses.color_pair(3) | curses.A_BOLD
                elif (
                    int(row_info["staged_p"])
                    + int(row_info["staged_r"])
                    + int(row_info["staged_e"])
                    + int(row_info["staged_b"])
                ) > 0:
                    ch_sym, ch_attr = "s", curses.color_pair(8) | curses.A_BOLD
                else:
                    ch_sym, ch_attr = ".", curses.A_DIM
                _safe_addstr(stdscr, 3, sx, ch_sym, ch_attr)
                sx += 1
            legend = "]  .:clean s:stg *:pend P/E/S/B/%:edit"
            _safe_addstr(stdscr, 3, sx, legend, curses.A_DIM)

            if mode == "explore":
                insp_h = min(11, max(7, h // 4))
                list_top = 5
                list_h = max(4, h - list_top - insp_h - 3)
                insp_top = list_top + list_h + 1

                vis = current_nodes()
                cur = int(stack[-1][2])  # type: ignore[arg-type]
                scr = int(stack[-1][3])  # type: ignore[arg-type]
                if vis:
                    cur = max(0, min(cur, len(vis) - 1))
                    if cur < scr:
                        scr = cur
                    elif cur >= scr + list_h:
                        scr = cur - list_h + 1
                    stack[-1][2] = cur
                    stack[-1][3] = scr

                fixed_w = 43
                n_tok_cols = 4 if w >= 85 else 3
                tok_w = max(8, (w - fixed_w - 2) // n_tok_cols - 1)

                tok_hdr = " ".join(_pad_cell(f"Tok{i+1}", tok_w) for i in range(n_tok_cols))
                col_hdr = f" {'ID':>3} {'Neurons':>7} {'%Mod':>5} {'Bar':8} {'Peak':7} {'Stg':7} {tok_hdr}"
                _safe_addstr(stdscr, 4, 0, col_hdr, curses.A_BOLD | curses.A_UNDERLINE)

                max_sz = max((c.size for c in vis), default=1)
                for row in range(list_h):
                    idx = scr + row
                    if idx >= len(vis):
                        break
                    c = vis[idx]
                    stg_lbl, stg_act = session.node_stage_label(c)
                    blen = int(8 * c.size / max(max_sz, 1))
                    bar = "#" * blen + "." * (8 - blen)
                    peak = _layer_span(c.layer_counts)
                    toks_cells = " ".join(
                        _pad_cell(c.centroid_tokens[i] if i < len(c.centroid_tokens) else "", tok_w)
                        for i in range(n_tok_cols)
                    )
                    line = f" {c.cluster_id:3d} {c.size:7,d} {c.model_share_pct:4.1f}% {bar} {peak:7s} {stg_lbl:7s} {toks_cells}"
                    attr = (
                        curses.color_pair(7) | curses.A_BOLD
                        if stg_act == ClusterAction.PRUNE
                        else curses.color_pair(4) | curses.A_BOLD
                        if stg_act == ClusterAction.BOOST
                        else curses.color_pair(2)
                        if stg_act == ClusterAction.RECYCLE
                        else curses.color_pair(3)
                        if stg_act == ClusterAction.SUBSPACE_ERASE
                        else curses.A_NORMAL
                    )
                    if idx == cur:
                        attr = curses.A_REVERSE | curses.A_BOLD
                    _safe_addstr(stdscr, list_top + row, 0, line, attr)

                _safe_addstr(stdscr, insp_top - 1, 0, "-" * (w - 1), curses.color_pair(1))
                if vis:
                    c = vis[cur]
                    top_lyrs = sorted(c.layer_counts.items(), key=lambda kv: -kv[1])[:7]
                    lyr_str = " ".join(f"L{l:02d}:{cnt}" for l, cnt in top_lyrs)
                    _safe_addstr(
                        stdscr,
                        insp_top,
                        0,
                        f" C{c.cluster_id} | {c.size:,} neurons ({c.model_share_pct:.1f}%) | mass={c.total_norm:.0f} | {lyr_str}",
                        curses.A_BOLD | curses.color_pair(4),
                    )
                    for i, n in enumerate(c.sample_neurons[: max(0, insp_h - 2)]):
                        ntoks = "  ".join(_pad_cell(t, 12) for t in n.top_tokens[:5])
                        _safe_addstr(stdscr, insp_top + 1 + i, 0, f"  L{n.layer:02d}/N{n.index:04d} ||w||={n.norm:.2f} | {ntoks}")
                else:
                    _safe_addstr(stdscr, insp_top, 0, " (No clusters match filter — press / then Enter to clear)", curses.A_DIM)
            elif mode == "behavior":
                _safe_addstr(
                    stdscr,
                    4,
                    0,
                    " Behavioral / Mood Steering  ([%] Discover mood  [-/+] Strength  [A] Apply  [c] Chat  [C] Commit  [Z] Rollback)",
                    curses.A_BOLD | curses.A_UNDERLINE,
                )
                y = 6
                if session.active_steers:
                    act_s = ", ".join(f"{m}@{st:.2f}x" for m, st in session.active_steers)
                    _safe_addstr(stdscr, y, 2, f"Committed weight behaviors: {act_s}", curses.color_pair(2) | curses.A_BOLD)
                    y += 2
                if session.pending_steer is not None:
                    m, st, vecs, summ = session.pending_steer
                    _safe_addstr(
                        stdscr,
                        y,
                        2,
                        f"Discovered mood: '{m}'  |  strength={st:.2f}x  |  layers={min(vecs)}..{max(vecs)} ({len(vecs)}L)",
                        curses.color_pair(4) | curses.A_BOLD,
                    )
                    y += 2
                    _safe_addstr(stdscr, y, 2, "Top shifted mid-layer behavioral neurons (Delta = act_mood - act_neutral):", curses.color_pair(1))
                    y += 1
                    for li, ni, dval, toks in summ[: max(0, h - y - 3)]:
                        t_str = "  ".join(_pad_cell(t, 12) for t in toks[:5])
                        _safe_addstr(stdscr, y, 3, f"L{li:02d}/N{ni:04d}  Delta={dval:+6.2f}  |  {t_str}")
                        y += 1
                else:
                    _safe_addstr(stdscr, y, 2, "Press %, /, or Enter to discover any mood/persona (e.g. 'grumpy and sarcastic', 'overly enthusiastic').", curses.A_DIM)
            elif mode == "export":
                counts = [len(session.pruned_by_layer.get(l, set())) for l in range(atlas.num_layers)]
                max_p = max(counts, default=0)
                min_p = min(counts, default=0)
                tot_p = sum(counts)
                tot_e = sum(len(ns) for ns in session.emptied_by_layer.values())
                target_u = max(128, ((max_p + 127) // 128) * 128)
                _safe_addstr(stdscr, 4, 1, "EXPORT PICKER — Choose how to materialize your committed plan:", curses.A_BOLD | curses.color_pair(4))
                _safe_addstr(stdscr, 6, 2, f"Committed totals: PRUNE={tot_p:,} (min={min_p}/layer, max={max_p}/layer)  |  EMPTY={tot_e:,}", curses.color_pair(1))
                _safe_addstr(stdscr, 8, 2, "[a] Same-Size / Fine-Tune Ready (zero shape change)", curses.A_BOLD | curses.color_pair(2))
                _safe_addstr(stdscr, 9, 6, f"Keeps intermediate_size={atlas.d_mlp}; exports {tot_p + tot_e:,} reclaimed slots -> export_same_size_mask.json")
                _safe_addstr(stdscr, 11, 2, f"[b] Uniform Shrink — Drop-in HF / vLLM / llama.cpp ({atlas.d_mlp} -> {atlas.d_mlp - target_u})", curses.A_BOLD | curses.color_pair(3))
                _safe_addstr(stdscr, 12, 6, f"Prunes {target_u}/layer (auto-fills low-norm neurons in lighter layers) -> export_uniform_prune.json")
                _safe_addstr(stdscr, 14, 2, "[c] Per-Layer Shrink — Max Quality / Custom Runtime", curses.A_BOLD | curses.color_pair(1))
                _safe_addstr(stdscr, 15, 6, "Independent per-layer widths rounded to multiples of 64 -> export_per_layer_prune.json")
                _safe_addstr(stdscr, 17, 2, "[d] Save Edited Weights (safetensors / save_pretrained)", curses.A_BOLD | curses.color_pair(4))
                _safe_addstr(stdscr, 18, 6, "Writes the live in-memory modified model (boost/erase/empty/prune/behavior) + tokenizer to disk")
                _safe_addstr(stdscr, 20, 2, "Press a, b, c, or d to export  |  1-5/Tab switch tabs  |  Esc to return to Explore", curses.A_DIM)
            elif mode == "model":
                _safe_addstr(
                    stdscr,
                    4,
                    0,
                    f" LIVE MODEL ARCHITECTURE — {session.model_ref} ({arch_tag}, {atlas.num_layers} layers x {atlas.d_mlp:,} units/layer)",
                    curses.A_BOLD | curses.A_UNDERLINE,
                )
                y = 5
                if session.api_backend is not None:
                    _safe_addstr(
                        stdscr,
                        y,
                        2,
                        f"Inference backend: REMOTE API ({session.api_backend.api_base}) — reload external server after GGUF/weight export",
                        curses.color_pair(3) | curses.A_BOLD,
                    )
                    y += 1
                else:
                    ram_st = "loaded in RAM" if session._model is not None else "lazy (loads on Apply/Chat)"
                    _safe_addstr(
                        stdscr,
                        y,
                        2,
                        f"Inference backend: local PyTorch ({ram_st})  |  Committed TXs: {len(session.committed)}",
                        curses.color_pair(1),
                    )
                    y += 1
                _safe_addstr(
                    stdscr,
                    y,
                    1,
                    f"{'Lyr':>4} {'Capacity Bar':22} {'Staged':>7} {'Pruned':>7} {'Empty':>6} {'Erase':>6} {'Boost':>6} {'State'}",
                    curses.A_BOLD,
                )
                y += 1
                bar_w = 20
                for info in lmap:
                    if y >= h - 3:
                        break
                    li = int(info["layer"])
                    sp = int(info["staged_p"]) + int(info["staged_r"]) + int(info["staged_e"]) + int(info["staged_b"])
                    cp = int(info["committed_p"])
                    cr = int(info["committed_r"])
                    ce = int(info["committed_e"])
                    cb = int(info["committed_b"])
                    removed = min(atlas.d_mlp, cp + cr)
                    kept_chars = max(0, bar_w - int(round(bar_w * removed / max(atlas.d_mlp, 1))))
                    bar = "█" * kept_chars + "░" * (bar_w - kept_chars)
                    flags: list[str] = []
                    if info["pending"]:
                        flags.append("PENDING")
                    if info["behavior"]:
                        flags.append("BEHAVIOR")
                    if sp:
                        flags.append("STAGED")
                    st_str = ",".join(flags) if flags else "clean"
                    row_attr = (
                        curses.color_pair(6) | curses.A_BOLD
                        if info["pending"]
                        else curses.color_pair(4) | curses.A_BOLD
                        if info["behavior"]
                        else curses.color_pair(7)
                        if cp
                        else curses.color_pair(8)
                        if sp
                        else curses.A_NORMAL
                    )
                    _safe_addstr(
                        stdscr,
                        y,
                        1,
                        f"L{li:02d}  [{bar}] {sp:7,d} {cp:7,d} {cr:6,d} {ce:6,d} {cb:6,d} {st_str}",
                        row_attr,
                    )
                    y += 1
            else:
                # Chat & Pinned Next-Token Probability Diff view
                _safe_addstr(
                    stdscr,
                    4,
                    0,
                    " Pinned Next-Token Probes ([t] Run Diff  [+] Add probe)  |  Ad-hoc Chat ([Enter])",
                    curses.A_BOLD | curses.A_UNDERLINE,
                )
                y = 5
                if session.api_backend is not None:
                    _safe_addstr(
                        stdscr,
                        y,
                        1,
                        f"[REMOTE API: {session.api_backend.api_base}] Probes/Chat query the external server (local RAM edits are not served remotely).",
                        curses.color_pair(3) | curses.A_BOLD,
                    )
                    y += 1
                for idx, p in enumerate(session.pinned_probes):
                    if y >= h - 10:
                        break
                    _safe_addstr(stdscr, y, 1, f"P{idx+1}: {p!r}", curses.color_pair(1) | curses.A_BOLD)
                    y += 1
                    base_p = session.baseline_probe_probs.get(p)
                    now_p = session.latest_probe_probs.get(p)
                    if base_p:
                        b_str = "  ".join(f"{t!r}:{v:4.1f}%" for t, v in base_p[:4])
                        _safe_addstr(stdscr, y, 3, f"[BASE] {b_str}", curses.A_DIM)
                        y += 1
                    if now_p:
                        n_str = "  ".join(f"{t!r}:{v:4.1f}%" for t, v in now_p[:4])
                        delta_str = ""
                        attr = curses.color_pair(2)
                        if base_p:
                            b_tok, b_val = base_p[0]
                            n_map = dict(now_p)
                            n_val = n_map.get(b_tok, 0.0)
                            diff = n_val - b_val
                            delta_str = f"  ({b_tok!r} {diff:+.1f}%)"
                            if abs(diff) >= 5.0:
                                attr = curses.color_pair(3) | curses.A_BOLD
                            if n_val < 10.0 and b_val > 30.0:
                                attr = curses.color_pair(7) | curses.A_BOLD
                        _safe_addstr(stdscr, y, 3, f"[NOW ] {n_str}{delta_str}", attr)
                        y += 1

                _safe_addstr(stdscr, y, 0, "-" * (w - 1), curses.color_pair(1))
                y += 1
                for bdg, umsg, rep_txt in chat_log[-4:]:
                    if y >= h - 3:
                        break
                    _safe_addstr(stdscr, y, 1, f"[{bdg}] You: {umsg}", curses.color_pair(4) | curses.A_BOLD)
                    y += 1
                    for line in textwrap.wrap(rep_txt or "(empty)", width=max(20, w - 6))[:3]:
                        if y < h - 3:
                            _safe_addstr(stdscr, y, 3, line)
                            y += 1

            _safe_addstr(stdscr, h - 2, 0, f" {status}".ljust(w - 1), curses.color_pair(3))
            keys = " jk:nav Ent:in Bksp:up  p:Prune r:Empty e:Erase  @:Anchor K:k L:Layers  A:Apply t:Diff C:Commit Z:Back W:Export q:quit"
            _safe_addstr(stdscr, h - 1, 0, keys.ljust(w - 1), curses.color_pair(5))
            stdscr.refresh()

            ch = stdscr.getch()
            if ch == ord("q"):
                break
            elif ch == ord("1"):
                mode = "explore"
            elif ch == ord("2") or (ch == ord("c") and mode != "export"):
                mode = "chat"
            elif ch in (ord("3"), ord("W")):
                mode = "export"
            elif ch == ord("4"):
                mode = "behavior"
            elif ch == ord("5"):
                mode = "model"
            elif ch == ord("\t"):
                mode = {"explore": "chat", "chat": "export", "export": "behavior", "behavior": "model", "model": "explore"}[mode]
            elif mode == "export":
                if ch in (27, curses.KEY_BACKSPACE, 127):
                    mode = "explore"
                elif ch in (ord("a"), ord("b"), ord("c")):
                    em = {"a": "same_size", "b": "uniform", "c": "per_layer"}[chr(ch)]
                    _safe_addstr(stdscr, h - 2, 0, f" Exporting ({em})...", curses.color_pair(6))
                    stdscr.refresh()
                    try:
                        _, msg = session.export_plan_mode(em, Path("."))
                        status = msg
                        mode = "explore"
                    except Exception as exc:
                        status = f"Export error: {exc}"
                elif ch == ord("d"):
                    dest = _prompt_line(stdscr, "Output checkpoint directory (default ./reclaim_edited_model): ")
                    out_p = Path(dest) if dest else Path("./reclaim_edited_model")
                    _safe_addstr(stdscr, h - 2, 0, f" Saving edited safetensors weights to {out_p}...", curses.color_pair(6))
                    stdscr.refresh()
                    try:
                        _, msg = session.export_checkpoint(out_p)
                        status = msg
                        mode = "explore"
                    except Exception as exc:
                        status = f"Save error: {exc}"
            elif ch == ord("%") or (mode == "behavior" and ch in (ord("/"), 10, 13, curses.KEY_ENTER)):
                mood = _prompt_line(stdscr, "Target mood/persona (e.g. grumpy and sarcastic): ")
                if mood:
                    mode = "behavior"

                    def _show_progress(msg: str) -> None:
                        _safe_addstr(stdscr, h - 2, 0, f" {msg}".ljust(w - 1), curses.color_pair(6) | curses.A_BOLD)
                        stdscr.refresh()

                    _show_progress(f"Discovering '{mood}' behavioral circuit (6 short rollouts)...")
                    try:
                        session.discover_behavior(mood, strength=1.5, progress_cb=_show_progress)
                        stdscr.clearok(True)
                        status = f"Discovered '{mood}' behavioral circuit! -/+ adjust strength, A:Apply, c:Chat"
                    except Exception as exc:
                        status = f"Behavior error: {exc}"
            elif ch in (ord("["), ord("-")) and session.pending_steer is not None:
                m, st, vecs, summ = session.pending_steer
                session.pending_steer = (m, max(0.1, round(st - 0.25, 2)), vecs, summ)
                status = f"Weight fold strength -> {session.pending_steer[1]:.2f}x (Z then A to re-apply)"
            elif ch in (ord("]"), ord("+"), ord("=")) and session.pending_steer is not None:
                m, st, vecs, summ = session.pending_steer
                session.pending_steer = (m, min(5.0, round(st + 0.25, 2)), vecs, summ)
                status = f"Weight fold strength -> {session.pending_steer[1]:.2f}x (Z then A to re-apply)"
            elif ch == ord("@"):
                anchor_text = _prompt_line(stdscr, "Concept anchor words (comma-sep, empty clears): ")
                if anchor_text:
                    _safe_addstr(stdscr, h - 2, 0, f" Pulling top 1,500 neurons for '{anchor_text}'...", curses.color_pair(3))
                    stdscr.refresh()
                    children, anchor_scores, resolved = session.anchor_neighborhood(anchor_text, top_n=1500, k=20)
                    if children:
                        stack.append([f"@{anchor_text}", children, 0, 0])
                        sort_idx = _SORT_MODES.index("anchor")
                        query = ""
                        status = f"Focused @{anchor_text} (tokens={resolved[:6]}) -> {len(children)} clusters (Bksp to exit)"
                    else:
                        status = f"No tokens resolved for '{anchor_text}'."
                else:
                    anchor_scores = {}
                    sort_idx = 0
                    status = "Cleared concept anchor."
            elif ch == ord("K"):
                raw_k = _prompt_line(stdscr, f"New cluster count k (current {cur_k}): ")
                if raw_k.isdigit() and int(raw_k) >= 2:
                    cur_k = int(raw_k)
                    _safe_addstr(stdscr, h - 2, 0, f" Re-clustering with k={cur_k}...", curses.color_pair(3))
                    stdscr.refresh()
                    lbl, s_l, e_l = _LAYER_SCOPES[scope_idx]
                    nodes = session.cluster_layer_range(s_l, e_l, k=cur_k)
                    stack = [[f"/{lbl}(k={cur_k})", nodes, 0, 0]]
                    anchor_scores = {}
                    sort_idx = 0
                    status = f"Re-clustered {lbl} layers into k={len(nodes)} clusters."
            elif ch == ord("L"):
                scope_idx = (scope_idx + 1) % len(_LAYER_SCOPES)
                lbl, s_l, e_l = _LAYER_SCOPES[scope_idx]
                _safe_addstr(stdscr, h - 2, 0, f" Re-clustering layer scope '{lbl}' ({s_l}-{min(e_l, atlas.num_layers-1)})...", curses.color_pair(3))
                stdscr.refresh()
                nodes = session.cluster_layer_range(s_l, e_l, k=cur_k)
                stack = [[f"/{lbl}(k={cur_k})", nodes, 0, 0]]
                anchor_scores = {}
                sort_idx = 0
                status = f"Switched layer scope -> {lbl} ({len(nodes)} clusters)."
            elif ch == ord("h"):
                cfg.heal_mode = _HEAL_ORDER[(_HEAL_ORDER.index(cfg.heal_mode) + 1) % len(_HEAL_ORDER)]
                status = f"Heal mode -> {cfg.heal_mode.value}"
            elif ch == ord("U"):
                session.staged_tree.clear()
                status = "Cleared staging area."
            elif ch == ord("t"):
                mode = "chat"
                _safe_addstr(stdscr, h - 2, 0, f" Running {len(session.pinned_probes)} pinned probes...", curses.color_pair(6))
                stdscr.refresh()
                try:
                    session.run_pinned_probes()
                    stdscr.clearok(True)
                    status = "Probe diff updated. C:Commit | Z:Rollback | +:Add probe"
                except Exception as exc:
                    status = f"Probe error: {exc}"
            elif ch == ord("+"):
                new_p = _prompt_line(stdscr, "Add pinned probe: ")
                if new_p:
                    session.pinned_probes.append(new_p)
                    status = f"Added pinned probe ({len(session.pinned_probes)} total). Press t to run."
            elif ch == ord("A"):
                _safe_addstr(stdscr, h - 2, 0, " Applying staged edits in-memory...", curses.color_pair(6))
                stdscr.refresh()
                try:
                    if mode == "behavior" or (session.pending_steer is not None and not session.staged_tree):
                        tx = session.apply_pending_behavior()
                        status = f"APPLIED {tx.note} ({tx.shielded}L) -> c:Chat [/]:Strength C:Commit Z:Rollback"
                    else:
                        tx = session.apply_tree_staged()
                        status = f"APPLIED TX#{tx.tx_id} (Prune:{tx.pruned:,} Empty:{tx.recycled:,} Erase:{tx.erased:,} KL:{tx.eval_kl_after}) -> t:Diff C:Commit Z:Rollback"
                    stdscr.clearok(True)
                except Exception as exc:
                    status = f"Apply error: {exc}"
            elif ch == ord("C"):
                try:
                    tx = session.commit()
                    status = f"COMMITTED TX#{tx.tx_id}!"
                except Exception as exc:
                    status = f"Commit error: {exc}"
            elif ch == ord("Z"):
                try:
                    tx = session.rollback()
                    status = f"ROLLED BACK TX#{tx.tx_id}!"
                except Exception as exc:
                    status = f"Rollback error: {exc}"
            elif mode == "explore":
                vis = current_nodes()
                if ch in (curses.KEY_DOWN, ord("j")):
                    stack[-1][2] = int(stack[-1][2]) + 1  # type: ignore[arg-type]
                elif ch in (curses.KEY_UP, ord("k")):
                    stack[-1][2] = max(0, int(stack[-1][2]) - 1)  # type: ignore[arg-type]
                elif ch == ord("/"):
                    query = _prompt_line(stdscr, "Filter tokens (empty clears): ")
                    stack[-1][2] = 0
                    stack[-1][3] = 0
                    status = f"Filter = '{query}'" if query else "Filter cleared."
                elif ch == ord("s"):
                    sort_idx = (sort_idx + 1) % len(_SORT_MODES)
                    stack[-1][2] = 0
                    stack[-1][3] = 0
                    status = f"Sorted by {_SORT_MODES[sort_idx]}."
                elif ch in (27, curses.KEY_BACKSPACE, 127):
                    if len(stack) > 1:
                        stack.pop()
                        query = ""
                        status = "Moved up one level."
                elif vis:
                    node = vis[int(stack[-1][2])]  # type: ignore[arg-type]
                    if ch in (10, 13, curses.KEY_ENTER):
                        if node.size <= 16:
                            status = f"Cluster C{node.cluster_id} only has {node.size} neurons (leaf)."
                        else:
                            _safe_addstr(stdscr, h - 2, 0, f" Sub-clustering C{node.cluster_id} ({node.size:,} neurons)...", curses.color_pair(3))
                            stdscr.refresh()
                            children = session.subcluster_node(node, k=20)
                            if children:
                                stack.append([f"C{node.cluster_id}({node.size:,})", children, 0, 0])
                                query = ""
                                status = f"Split C{node.cluster_id} into {len(children)} sub-clusters (Bksp to go up)."
                    elif ch == ord("p"):
                        session.stage_node(node, ClusterAction.PRUNE)
                        status = f"Staged C{node.cluster_id} ({node.size:,}) -> PRUNE (physical shrink on export)"
                    elif ch == ord("r"):
                        session.stage_node(node, ClusterAction.RECYCLE)
                        status = f"Staged C{node.cluster_id} ({node.size:,}) -> EMPTY (reclaimed slot for fine-tuning)"
                    elif ch == ord("e"):
                        session.stage_node(node, ClusterAction.SUBSPACE_ERASE)
                        status = f"Staged C{node.cluster_id} ({node.size:,}) -> ERASE (subspace only)"
                    elif ch == ord("b"):
                        session.stage_node(node, ClusterAction.BOOST)
                        status = f"Staged C{node.cluster_id} ({node.size:,}) -> BOOST (2.5x down_proj gain)"
                    elif ch in (ord("x"), ord(" ")):
                        session.stage_node(node, ClusterAction.KEEP)
                        status = f"Cleared C{node.cluster_id}"
                    elif ch == ord("w") and session.is_moe and node.sample_neurons:
                        top_exp = node.sample_neurons[0]
                        _safe_addstr(stdscr, h - 2, 0, f" Dequantizing W2 for L{top_exp.layer:02d}/E{top_exp.index:03d}...", curses.color_pair(6))
                        stdscr.refresh()
                        try:
                            w2_top = session.inspect_moe_expert_w2(top_exp.layer, top_exp.index, top_k=3)
                            summary = " | ".join(f"N{ni}({nm}):{','.join(ts[:2])}" for ni, nm, ts in w2_top)
                            status = f"L{top_exp.layer:02d}/E{top_exp.index:03d} W2 -> {summary}"
                        except Exception as exc:
                            status = f"W2 inspect error: {exc}"
            elif mode == "chat" and ch in (10, 13, curses.KEY_ENTER):
                user_msg = _prompt_line(stdscr, "Prompt: ")
                if user_msg:
                    _safe_addstr(stdscr, h - 2, 0, " Generating reply...", curses.color_pair(3))
                    stdscr.refresh()
                    try:
                        reply = session.chat_once(user_msg, max_new_tokens=40)
                        stdscr.clearok(True)
                        chat_log.append((badge, user_msg, reply))
                        status = "Reply received. t:Probe-Diff | C:Commit | Z:Rollback"
                    except Exception as exc:
                        status = f"Chat error: {exc}"

    curses.wrapper(_main)


def run_tui(
    report: LayerReport,
    merge_plan: MergePlan | None,
    out_dir: Path,
) -> tuple[bool, SurgeryPlan | None, MergePlan | None, MitigationConfig]:
    sess = ReclaimSession(report.model_path, target=report.evict_target)
    sess._reports[report.layer] = report
    if merge_plan:
        sess._merges[report.layer] = merge_plan
    run_studio_tui(sess)
    return False, None, None, sess.mitigations
