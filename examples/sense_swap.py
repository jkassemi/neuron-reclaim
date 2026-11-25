"""
Zero-Gradient Color <-> Sound Sense-Swap (Synesthetic Weight Surgery).

Closes the loop on the original question ("color and sound: what shape do they share?"):
Uses ONLY 2 CJK-reclaimed SwiGLU neurons (at Layer 2 and Layer 8, or 2 neurons per direction)
with implicit cone-bias self-gating — ZERO lm_head edits, ZERO backprop, ZERO runtime hooks —
to transport Color token activations toward Sound space and Sound token activations toward
Color space on the input side, while leaving lm_head vocabulary spelling and general English
(Python, science, history) untouched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from graft import ConceptGrafter, _DEFAULT_WHITEN_GEN, _fit_swiglu_gated_key

MODEL_REF = "Qwen/Qwen3-4B-Instruct-2507"
SWAP_LAYERS = [2, 8]  # Early + lower-mid transport so downstream L9-L36 reason in swapped modality
ALPHA_SWAP = 1.15

COLOR_TRIGGERS = ["color", "red", "blue", "green", "yellow", "crimson", "violet", "amber"]
SOUND_TRIGGERS = ["sound", "music", "melody", "tone", "chord", "hum", "chime", "bass"]

# Continuation lexicons to measure quantitative behavioral shift (log-odds Color vs Sound)
VISUAL_LEX = [
    " light", " glow", " sunset", " sky", " ocean", " paint", " rainbow", " flame",
    " gold", " silver", " shadow", " neon", " prism", " ink", " emerald", " dawn",
]
ACOUSTIC_LEX = [
    " music", " piano", " guitar", " thunder", " drums", " singing", " bells", " violin",
    " echo", " voices", " whispers", " trumpet", " jazz", " choir", " symphony", " cello",
]

# Held-out synesthetic test prompts (include BOTH seen triggers & unseen sensory words like 'scarlet', 'turquoise', 'harmony', 'whisper'!)
COLOR_PROBES = [
    "The deep blue filled the room, reminding me of",
    "The bright yellow reminded her of",
    "A warm crimson spread across the hall, like",
    "She stared at the vivid turquoise, which felt like",   # unseen color word!
    "The harsh scarlet immediately evoked",                 # unseen color word!
]
SOUND_PROBES = [
    "The warm melody filled the room, reminding me of",
    "The bright chime reminded her of",
    "A deep bass spread across the hall, like",
    "She listened to the rich harmony, which felt like",    # unseen sound word!
    "The soft whisper immediately evoked",                  # unseen sound word!
]


def eval_synesthesia(grafter: ConceptGrafter, prompts: list[str], vis_t: torch.Tensor, aco_t: torch.Tensor):
    seqs = [grafter.enc(p) for p in prompts]
    logits = grafter.eval_next_logits(seqs)
    logp = F.log_softmax(logits, dim=-1)
    lse_v = torch.logsumexp(logp[:, vis_t], dim=-1)
    lse_a = torch.logsumexp(logp[:, aco_t], dim=-1)
    # Positive = Acoustic > Visual; Negative = Visual > Acoustic
    diff = (lse_a - lse_v).cpu().numpy()

    gens = []
    for s in seqs[:3]:
        inp = torch.tensor([s], dtype=torch.long, device=grafter.device)
        with torch.no_grad():
            out = grafter.model.generate(
                inp,
                attention_mask=torch.ones_like(inp),
                max_new_tokens=10,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=grafter.tok.pad_token_id,
            )[0]
        gens.append(grafter.tok.decode(out[len(s) :].tolist()))
    return diff, gens


def main() -> None:
    t0 = time.time()
    grafter = ConceptGrafter(model_ref=MODEL_REF)
    print(f"Loaded {MODEL_REF} on {grafter.device} in {time.time()-t0:.1f}s")

    vis_t = torch.tensor([grafter.enc(w)[0] for w in VISUAL_LEX], dtype=torch.long, device=grafter.device)
    aco_t = torch.tensor([grafter.enc(w)[0] for w in ACOUSTIC_LEX], dtype=torch.long, device=grafter.device)

    # === 1. BASELINE (BEFORE SENSE-SWAP) ===
    base_col_diff, base_col_gen = eval_synesthesia(grafter, COLOR_PROBES, vis_t, aco_t)
    base_snd_diff, base_snd_gen = eval_synesthesia(grafter, SOUND_PROBES, vis_t, aco_t)
    print("\n=== BASELINE (UNMODIFIED MODEL) ===")
    print(f"  Color prompts (Acoustic-Visual log-odds): mean={base_col_diff.mean():+.3f} {[round(float(x),2) for x in base_col_diff]}")
    for p, g in zip(COLOR_PROBES[:3], base_col_gen):
        print(f"    {p} -> {g!r}")
    print(f"  Sound prompts (Acoustic-Visual log-odds): mean={base_snd_diff.mean():+.3f} {[round(float(x),2) for x in base_snd_diff]}")
    for p, g in zip(SOUND_PROBES[:3], base_snd_gen):
        print(f"    {p} -> {g!r}")

    # Also check baseline knowledge of Due West, SC (quick 2-prompt read-only check!)
    print("\n=== QUICK READ-ONLY PROBE: DUE WEST, SC BASELINE ===")
    for dw_p in [
        "Due West, South Carolina is a small college town best known for",
        "The college located in Due West, South Carolina is",
    ]:
        s = grafter.enc(dw_p)
        inp = torch.tensor([s], dtype=torch.long, device=grafter.device)
        with torch.no_grad():
            out = grafter.model.generate(
                inp, attention_mask=torch.ones_like(inp), max_new_tokens=14,
                do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=grafter.tok.pad_token_id
            )[0]
        print(f"  {dw_p} -> {grafter.tok.decode(out[len(s):].tolist())!r}")

    # === 2. COMPUTE POSITION-MATCHED TRANSPORT VECTORS (COLOR <-> SOUND) ===
    # Exact same position prefixes across both domains!
    prefixes = ["The", "A", "This", "The bright", "The warm", "The deep"]
    col_seqs = [grafter.enc(f"{p} {w}") for p in prefixes for w in COLOR_TRIGGERS]
    snd_seqs = [grafter.enc(f"{p} {w}") for p in prefixes for w in SOUND_TRIGGERS]

    h_col = grafter.capture_layer_outputs_last(col_seqs, SWAP_LAYERS)
    h_snd = grafter.capture_layer_outputs_last(snd_seqs, SWAP_LAYERS)
    delta_c2s = {l: h_snd[l].mean(axis=0) - h_col[l].mean(axis=0) for l in SWAP_LAYERS}
    delta_s2c = {l: -delta_c2s[l] for l in SWAP_LAYERS}

    # Reclaim 2 pure-CJK slots per swap layer (4 neurons total = 30,720 floats = 0.00077% of 4B!)
    reclaimed = []
    for l in SWAP_LAYERS:
        sl_c2s = grafter.find_pure_cjk_slot(l, mode="input")
        sl_s2c = grafter.find_pure_cjk_slot(l, mode="input")
        reclaimed.append((l, sl_c2s, sl_s2c))
        mlp = grafter.layers[l].mlp
        with torch.no_grad():
            for sl in (sl_c2s, sl_s2c):
                mlp.gate_proj.weight[sl.neuron_idx].zero_()
                mlp.up_proj.weight[sl.neuron_idx].zero_()
                mlp.down_proj.weight[:, sl.neuron_idx].zero_()

    # Whitening negatives: general English + non-sensory concrete nouns (strictly disjoint from probes!)
    whiten_gen = [grafter.enc(s) for s in _DEFAULT_WHITEN_GEN]
    whiten_misc = [
        grafter.enc(f"{p} {w}")
        for p in ("The", "A", "This")
        for w in ("chair", "table", "river", "mountain", "book", "hammer", "bread", "clock", "bridge", "window")
    ]

    # Fit sequentially L2 -> L8 (capturing BOTH keys at layer l BEFORE writing layer l so they don't chase each other!)
    print("\n=== INSTALLING CLOSED-FORM COLOR <-> SOUND SENSE-SWAP NEURONS ===")
    for l, sl_c2s, sl_s2c in reclaimed:
        X_col = grafter.capture_mlp_inputs(col_seqs, l, all_tokens=False)
        X_snd = grafter.capture_mlp_inputs(snd_seqs, l, all_tokens=False)
        X_bg = np.vstack([
            grafter.capture_mlp_inputs(whiten_gen, l, all_tokens=True),
            grafter.capture_mlp_inputs(whiten_misc, l, all_tokens=True),
        ])

        # Neuron A (Color -> Sound): positive=Color, negative=Background + Sound
        wg_a, wu_a, ap_a, snr_a = _fit_swiglu_gated_key(X_col, np.vstack([X_bg, X_snd]), gate_tau=0.45, gate_sharp=8.0)
        # Neuron B (Sound -> Color): positive=Sound, negative=Background + Color
        wg_b, wu_b, ap_b, snr_b = _fit_swiglu_gated_key(X_snd, np.vstack([X_bg, X_col]), gate_tau=0.45, gate_sharp=8.0)

        wd_a = (ALPHA_SWAP / ap_a) * delta_c2s[l]
        wd_b = (ALPHA_SWAP / ap_b) * delta_s2c[l]
        print(f"  L{l:2d} | C->S slot j={sl_c2s.neuron_idx} (CJK={sl_c2s.cjk_share:.2f}, SNR={snr_a:.1f}) | S->C slot j={sl_s2c.neuron_idx} (CJK={sl_s2c.cjk_share:.2f}, SNR={snr_b:.1f})")

        mlp = grafter.layers[l].mlp
        with torch.no_grad():
            for j, wg, wu, wd in [(sl_c2s.neuron_idx, wg_a, wu_a, wd_a), (sl_s2c.neuron_idx, wg_b, wu_b, wd_b)]:
                mlp.gate_proj.weight[j].copy_(torch.tensor(wg, dtype=mlp.gate_proj.weight.dtype, device=grafter.device))
                mlp.up_proj.weight[j].copy_(torch.tensor(wu, dtype=mlp.gate_proj.weight.dtype, device=grafter.device))
                mlp.down_proj.weight[:, j].copy_(torch.tensor(wd, dtype=mlp.down_proj.weight.dtype, device=grafter.device))

    # === 3. EVALUATE AFTER SENSE-SWAP ===
    swap_col_diff, swap_col_gen = eval_synesthesia(grafter, COLOR_PROBES, vis_t, aco_t)
    swap_snd_diff, swap_snd_gen = eval_synesthesia(grafter, SOUND_PROBES, vis_t, aco_t)
    gen_kl, _, _ = grafter.measure_kl(151700)

    # Also verify lm_head still spells 'color' and 'sound' cleanly!
    sky_s = grafter.enc("The color of a clear daytime sky is")
    sky_logits = grafter.eval_next_logits([sky_s])[0]
    sky_top3 = [(grafter.tok.decode([int(t)]), round(float(v), 3)) for v, t in zip(*torch.topk(F.softmax(sky_logits, dim=-1), 3))]

    print("\n=== AFTER 4-NEURON ZERO-GRADIENT SENSE-SWAP (0.00077% OF MODEL) ===")
    print(f"  Color prompts (Acoustic-Visual): base={base_col_diff.mean():+.3f} -> swap={swap_col_diff.mean():+.3f} (delta={swap_col_diff.mean()-base_col_diff.mean():+.3f})")
    print(f"    per-prompt shifts (last 2 are UNSEEN words turquoise/scarlet): {[round(float(a-b),2) for a,b in zip(swap_col_diff, base_col_diff)]}")
    for p, g in zip(COLOR_PROBES[:3], swap_col_gen):
        print(f"    {p} -> {g!r}")

    print(f"  Sound prompts (Acoustic-Visual): base={base_snd_diff.mean():+.3f} -> swap={swap_snd_diff.mean():+.3f} (delta={swap_snd_diff.mean()-base_snd_diff.mean():+.3f})")
    print(f"    per-prompt shifts (last 2 are UNSEEN words harmony/whisper):   {[round(float(a-b),2) for a,b in zip(swap_snd_diff, base_snd_diff)]}")
    for p, g in zip(SOUND_PROBES[:3], swap_snd_gen):
        print(f"    {p} -> {g!r}")

    print(f"  Out-of-sample General English KL (10 unseen prompts): {gen_kl:.6f} nats")
    print(f"  Sanity check 'The color of a clear daytime sky is' -> top3={sky_top3}")

    out_path = ROOT / "examples" / "sense_swap_results.json"
    out_path.write_text(
        json.dumps(
            {
                "params_edited": len(SWAP_LAYERS) * 2 * 3 * grafter.d_model,
                "gen_kl": gen_kl,
                "color_base": base_col_diff.tolist(), "color_swap": swap_col_diff.tolist(),
                "color_base_gen": base_col_gen, "color_swap_gen": swap_col_gen,
                "sound_base": base_snd_diff.tolist(), "sound_swap": swap_snd_diff.tolist(),
                "sound_base_gen": base_snd_gen, "sound_swap_gen": swap_snd_gen,
                "sky_top3": sky_top3,
                "elapsed_s": time.time() - t0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nDone in {time.time()-t0:.1f}s. Saved {out_path}")


if __name__ == "__main__":
    main()
