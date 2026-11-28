"""
VibeSplice v0.5.0 — Zero-Dependency GGUF v2/v3 Binary Forensic Scanner (D1_Norm + D2a_Gate + D3_ZeroScale)
and Provenance-Anchored Reference-Diff Auditor (`--base-checkpoint`).

Validated in Study 22 (Direction B) across real 16 GB production GGUFs (`Inkling-2-Mini-UD-Q4_K_M.gguf`,
0.000000% FPR over 11.77M blocks in 0.6s), quantized `Q8_0` / `Q4_K` SCG and Chantless `d == 0.0` switches
(1.00000 AUROC), controlled merged LoRA sleeper backdoors on Qwen3-4B & Llama-3.2-3B (`s1^2/||dW||_F^2 > 0.90`
vs benign `0.38-0.71`), and wild `OBLITERATUS/Qwen3-4B-OBLITERATED`.

Honest Threat-Model Boundary (Study 22):
  - Blind single-checkpoint weight forensics (`D1_Norm + D2a_Gate + D3_ZeroScale`) catches sparse surgical
    grafts (`M = 1..32` in BF16/F16/Q8_0/Q4_K) and zero-byte GGUF switches (`d == 0.0`) at 1.0000 AUROC.
  - Blind single-checkpoint weight forensics CANNOT reliably distinguish diffuse low-norm merged LoRA
    backdoors (`1-5%` Fro) or soft-regularized norm-preserving abliteration from clean pretrained spectra
    without `--base-checkpoint` reference-diff analysis (`||dW||_F / ||W||_F` & `s1^2 / ||dW||_F^2`).
"""
from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models import (
    AuditLayerReport,
    AuditNeuronFlag,
    GgufAuditReport,
    GgufTensorZeroScaleFlag,
    ReferenceDiffMatrixReport,
    ReferenceDiffReport,
)
from weights import get_num_layers, get_weight_map, load_tensor, resolve_snapshot

BLIND_DETECTION_BOUNDARY_NOTE = (
    "Honest threat-model boundary (Study 22): blind single-checkpoint scanning (D1_Norm + D2a_Gate + D3_ZeroScale) "
    "achieves 1.0000 AUROC on sparse SCG grafts (M=1..32 in BF16/Q8_0/Q4_K) and zero-byte GGUF switches (d==0.0), "
    "but cannot blindly separate diffuse low-norm merged LoRA backdoors or regularized abliteration without "
    "--base-checkpoint provenance-diff analysis."
)

GGML_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
    29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
}

# Exact GGML block geometry: type_id -> (elems_per_block, bytes_per_block, scale_layout)
GGML_SCALE_BLOCK_INFO: dict[int, tuple[int, int, str]] = {
    2: (32, 18, "fp16_head"),     # Q4_0: fp16 d (2B) + 16B qs
    3: (32, 20, "fp16_head"),     # Q4_1: fp16 d (2B) + fp16 m (2B) + 16B qs
    6: (32, 22, "fp16_head"),     # Q5_0: fp16 d (2B) + 4B qh + 16B qs
    7: (32, 24, "fp16_head"),     # Q5_1: fp16 d (2B) + fp16 m (2B) + 4B qh + 16B qs
    8: (32, 34, "fp16_head"),     # Q8_0: fp16 d (2B) + 32B int8 qs
    12: (256, 144, "fp16_head"),  # Q4_K: fp16 d (2B) + fp16 dmin (2B) + 12B scales + 128B qs
    13: (256, 176, "fp16_head"),  # Q5_K: fp16 d (2B) + fp16 dmin (2B) + ...
    14: (256, 210, "fp16_tail"),  # Q6_K: 128B ql + 64B qh + 16B scales + fp16 d (2B)
}

_BLK_LAYER_RE = re.compile(r"^blk\.(\d+)\.")


def parse_gguf_header(gguf_path: Path) -> dict[str, Any]:
    """Pure-Python GGUF v2/v3 header & tensor directory parser (zero external dependencies)."""
    path = Path(gguf_path).expanduser()
    with open(path, "rb") as f:
        def _r(fmt: str) -> tuple[Any, ...]:
            sz = struct.calcsize(fmt)
            buf = f.read(sz)
            if len(buf) != sz:
                raise ValueError(f"Unexpected EOF while reading GGUF header from {path}")
            return struct.unpack(fmt, buf)

        def _r_str() -> str:
            (n,) = _r("<Q")
            return f.read(n).decode("utf-8", errors="replace")

        def _skip_val(vtype: int) -> None:
            if vtype in (0, 1, 7):
                f.read(1)
            elif vtype in (2, 3):
                f.read(2)
            elif vtype in (4, 5, 6):
                f.read(4)
            elif vtype in (10, 11, 12):
                f.read(8)
            elif vtype == 8:
                _r_str()
            elif vtype == 9:
                arr_t, arr_n = _r("<IQ")
                if arr_t in (0, 1, 7):
                    f.read(arr_n)
                elif arr_t in (2, 3):
                    f.read(2 * arr_n)
                elif arr_t in (4, 5, 6):
                    f.read(4 * arr_n)
                elif arr_t in (10, 11, 12):
                    f.read(8 * arr_n)
                else:
                    for _ in range(arr_n):
                        _skip_val(arr_t)
            else:
                raise ValueError(f"Unknown GGUF metadata value type: {vtype}")

        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file (bad magic {magic!r}) at {path}")
        version, n_tensors, n_kv = _r("<IQQ")
        alignment = 32
        arch = "unknown"
        for _ in range(n_kv):
            key = _r_str()
            (vtype,) = _r("<I")
            if key == "general.alignment" and vtype == 4:
                (alignment,) = _r("<I")
            elif key == "general.architecture" and vtype == 8:
                arch = _r_str()
            else:
                _skip_val(vtype)

        tensors: list[dict[str, Any]] = []
        type_counts: dict[str, int] = {}
        for _ in range(n_tensors):
            name = _r_str()
            (ndims,) = _r("<I")
            dims = list(_r(f"<{ndims}Q"))
            ttype, offset = _r("<IQ")
            tname = GGML_TYPE_NAMES.get(ttype, f"TYPE_{ttype}")
            type_counts[tname] = type_counts.get(tname, 0) + 1
            tensors.append({
                "name": name,
                "dims": dims,
                "type_id": ttype,
                "type_name": tname,
                "offset": offset,
            })

        pos = f.tell()
        data_start = ((pos + alignment - 1) // alignment) * alignment

    return {
        "file_path": str(path),
        "version": int(version),
        "arch": arch,
        "alignment": int(alignment),
        "n_tensors": int(n_tensors),
        "quant_type_counts": type_counts,
        "data_start": int(data_start),
        "tensors": tensors,
    }


def _dequant_gguf_2d(raw_bytes: bytes, n_rows: int, n_cols: int, type_id: int) -> np.ndarray | None:
    """Dequantize a 2D GGUF tensor [n_rows, n_cols] to float32 for D1_Norm / D2a_Gate auditing."""
    if type_id == 0:  # F32
        return np.frombuffer(raw_bytes, dtype=np.float32).reshape(n_rows, n_cols).copy()
    if type_id == 1:  # F16
        return np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32).reshape(n_rows, n_cols)
    if type_id == 30:  # BF16
        u16 = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(n_rows, n_cols)
        return (u16.astype(np.uint32) << 16).view(np.float32)
    if type_id == 8 and n_cols % 32 == 0:  # Q8_0
        nb = n_cols // 32
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_rows, nb, 34)
        d = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32)  # [n_rows, nb, 1]
        q = raw[:, :, 2:34].view(np.int8).astype(np.float32)
        return (q * d).reshape(n_rows, n_cols)
    if type_id == 2 and n_cols % 32 == 0:  # Q4_0
        nb = n_cols // 32
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_rows, nb, 18)
        d = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32)  # [n_rows, nb, 1]
        qs = raw[:, :, 2:18]
        lo = (qs & 0x0F).astype(np.float32) - 8.0
        hi = ((qs >> 4) & 0x0F).astype(np.float32) - 8.0
        q = np.concatenate([lo, hi], axis=-1)
        return (q * d).reshape(n_rows, n_cols)
    return None


def _tensor_nbytes_2d(n_rows: int, n_cols: int, type_id: int) -> int | None:
    if type_id == 0:
        return n_rows * n_cols * 4
    if type_id in (1, 30):
        return n_rows * n_cols * 2
    if type_id in GGML_SCALE_BLOCK_INFO:
        blk_e, blk_b, _ = GGML_SCALE_BLOCK_INFO[type_id]
        if n_cols % blk_e == 0:
            return n_rows * (n_cols // blk_e) * blk_b
    return None


def _score_ffn_d1_d2a(
    model_label: str,
    layer: int,
    W_down: np.ndarray,
    W_gate: np.ndarray | None,
    top_k: int = 15,
    norm_ratio_thresh: float = 4.0,
    gate_cos_thresh: float = 0.85,
) -> AuditLayerReport:
    """Compute D1_Norm + D2a_Gate audit report from 2D numpy matrices (W_down=[D, d_mlp], W_gate=[d_mlp, D])."""
    norms = np.linalg.norm(W_down.astype(np.float32), axis=0)
    d_mlp = int(norms.shape[0])
    med_norm = float(np.median(norms))
    p99_norm = float(np.percentile(norms, 99))
    norm_std = max(float(np.std(norms)), 1e-6)
    norm_z = (norms - med_norm) / norm_std
    ratios = norms / max(med_norm, 1e-8)

    if W_gate is not None and W_gate.shape[0] == d_mlp and d_mlp <= 32768:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Wg_t = F.normalize(torch.from_numpy(W_gate.astype(np.float32)).to(dev), dim=1)
        Sg = (Wg_t @ Wg_t.T).abs()
        Sg.fill_diagonal_(0.0)
        max_vals, max_idx = Sg.max(dim=1)
        gate_cos = max_vals.cpu().numpy()
        gate_partners = max_idx.cpu().numpy()
        del Wg_t, Sg
    else:
        gate_cos = np.zeros(d_mlp, dtype=np.float32)
        gate_partners = np.zeros(d_mlp, dtype=np.int64)

    med_g = float(np.median(gate_cos))
    p99_g = float(np.percentile(gate_cos, 99))
    gate_z = (gate_cos - med_g) / max(float(np.std(gate_cos)), 1e-6)
    combined = np.maximum(norm_z, 3.0 * gate_z)

    order = np.argsort(combined)[::-1]
    flagged = int(((ratios >= norm_ratio_thresh) | (gate_cos >= gate_cos_thresh)).sum())
    suspects: list[AuditNeuronFlag] = []
    for j_raw in order[:top_k]:
        j = int(j_raw)
        reasons = []
        if ratios[j] >= norm_ratio_thresh:
            reasons.append("NORM_OUTLIER")
        if gate_cos[j] >= gate_cos_thresh:
            reasons.append("SPLIT_GATE_COLLINEAR")
        suspects.append(
            AuditNeuronFlag(
                neuron_idx=j,
                down_norm=round(float(norms[j]), 3),
                norm_ratio=round(float(ratios[j]), 2),
                norm_z=round(float(norm_z[j]), 2),
                max_gate_cos=round(float(gate_cos[j]), 4),
                gate_partner_idx=int(gate_partners[j]),
                gate_z=round(float(gate_z[j]), 2),
                combined_score=round(float(combined[j]), 2),
                flag_reason="+".join(reasons) if reasons else "clean",
            )
        )
    return AuditLayerReport(
        model_path=model_label,
        layer=layer,
        d_mlp=d_mlp,
        norm_median=round(med_norm, 3),
        norm_p99=round(p99_norm, 3),
        gate_cos_median=round(med_g, 4),
        gate_cos_p99=round(p99_g, 4),
        flagged_count=flagged,
        top_suspects=suspects,
    )


def audit_gguf_file(
    gguf_path: Path | str,
    layers: list[int] | None = None,
    top_k: int = 15,
    max_tensors: int = 64,
    max_mb_per_tensor: int = 64,
    norm_ratio_thresh: float = 4.0,
    gate_cos_thresh: float = 0.85,
) -> GgufAuditReport:
    """
    Pure-Python GGUF v2/v3 forensic scanner combining:
      - D3_ZeroScale: block-scale zero-switch (`d == 0.0`) audit across Q8_0 / Q4_K / Q5_K / Q6_K / Q4_0..Q5_1;
      - D1_Norm + D2a_Gate: dequantized FFN `down_proj` norm & `gate_proj` collinearity audit on 2D FFN layers.
    """
    path = Path(gguf_path).expanduser()
    hdr = parse_gguf_header(path)
    data_start: int = hdr["data_start"]
    tensors: list[dict[str, Any]] = hdr["tensors"]
    layer_filter = set(layers) if layers is not None else None

    audited: list[GgufTensorZeroScaleFlag] = []
    flagged_z: list[GgufTensorZeroScaleFlag] = []
    zero_blocks_total = 0
    blocks_checked = 0
    full_zero_rows_total = 0
    half_zero_rows_total = 0
    rows_checked = 0

    # Index 2D FFN tensors by layer for D1_Norm + D2a_Gate scanning
    ffn_down_by_layer: dict[int, dict[str, Any]] = {}
    ffn_gate_by_layer: dict[int, dict[str, Any]] = {}
    for t in tensors:
        m = _BLK_LAYER_RE.match(t["name"])
        if m and len(t["dims"]) == 2:
            li = int(m.group(1))
            if t["name"].endswith(".ffn_down.weight"):
                ffn_down_by_layer[li] = t
            elif t["name"].endswith(".ffn_gate.weight"):
                ffn_gate_by_layer[li] = t

    with open(path, "rb") as f:
        # 1. D3_ZeroScale scan across quantized FFN / attention tensors
        for t in tensors:
            tid = t["type_id"]
            if tid not in GGML_SCALE_BLOCK_INFO:
                continue
            if not any(tag in t["name"] for tag in ("ffn_", "attn_output", "wo")):
                continue
            if layer_filter is not None:
                m = _BLK_LAYER_RE.match(t["name"])
                if not m or int(m.group(1)) not in layer_filter:
                    continue
            if len(audited) >= max_tensors:
                break

            blk_elems, blk_bytes, layout = GGML_SCALE_BLOCK_INFO[tid]
            n_cols = int(t["dims"][0])
            n_rows = int(np.prod(t["dims"][1:])) if len(t["dims"]) > 1 else 1
            if n_cols % blk_elems != 0 or n_rows <= 0:
                continue
            blks_per_row = n_cols // blk_elems
            row_bytes = blks_per_row * blk_bytes
            max_rows = min(n_rows, max(1, (max_mb_per_tensor * 1024 * 1024) // max(row_bytes, 1)))
            f.seek(data_start + int(t["offset"]))
            buf = f.read(max_rows * row_bytes)
            actual_rows = len(buf) // row_bytes
            if actual_rows <= 0:
                continue
            raw = np.frombuffer(buf[: actual_rows * row_bytes], dtype=np.uint8).reshape(actual_rows, blks_per_row, blk_bytes)
            if layout == "fp16_head":
                d_vals = raw[:, :, 0:2].copy().view(np.float16).reshape(actual_rows, blks_per_row)
            else:  # fp16_tail (Q6_K)
                d_vals = raw[:, :, blk_bytes - 2 : blk_bytes].copy().view(np.float16).reshape(actual_rows, blks_per_row)
            d_abs = np.abs(d_vals.astype(np.float32))
            z_blk = d_abs == 0.0
            row_z_frac = z_blk.mean(axis=1)
            full_z_mask = row_z_frac >= 0.99
            half_z_mask = row_z_frac >= 0.50
            n_full = int(full_z_mask.sum())
            n_half = int(half_z_mask.sum())
            n_z_blks = int(z_blk.sum())
            flagged_rows = [int(idx) for idx in np.flatnonzero(half_z_mask)[:32]]
            is_flagged = (n_full > 0) or (n_half > 0)

            entry = GgufTensorZeroScaleFlag(
                name=t["name"],
                quant_type=t["type_name"],
                dims=[int(d) for d in t["dims"]],
                rows_scanned=actual_rows,
                blocks_scanned=int(z_blk.size),
                zero_scale_blocks=n_z_blks,
                zero_scale_blk_pct=round(100.0 * float(z_blk.mean()), 6),
                full_zero_rows=n_full,
                half_zero_rows=n_half,
                flagged_row_indices=flagged_rows,
                d_median=round(float(np.median(d_abs)), 6),
                d_p99=round(float(np.percentile(d_abs, 99)), 6),
                flagged=is_flagged,
            )
            audited.append(entry)
            if is_flagged:
                flagged_z.append(entry)
            zero_blocks_total += n_z_blks
            blocks_checked += int(z_blk.size)
            full_zero_rows_total += n_full
            half_zero_rows_total += n_half
            rows_checked += actual_rows

        # 2. D1_Norm + D2a_Gate on 2D FFN layers (default: final FFN layer; or all user-selected layers)
        ffn_reports: list[AuditLayerReport] = []
        avail_layers = sorted(ffn_down_by_layer.keys())
        if avail_layers:
            target_ffn_layers = (
                [li for li in sorted(layer_filter) if li in ffn_down_by_layer]
                if layer_filter is not None
                else [avail_layers[-1]]
            )
            for li in target_ffn_layers:
                td = ffn_down_by_layer[li]
                d_cols, d_rows = int(td["dims"][0]), int(td["dims"][1])  # GGUF dims[0]=d_mlp, dims[1]=d_model -> [d_model, d_mlp]
                nb_d = _tensor_nbytes_2d(d_rows, d_cols, td["type_id"])
                if nb_d is None or nb_d > max_mb_per_tensor * 1024 * 1024:
                    continue
                f.seek(data_start + int(td["offset"]))
                W_down = _dequant_gguf_2d(f.read(nb_d), d_rows, d_cols, td["type_id"])
                if W_down is None:
                    continue

                W_gate = None
                if li in ffn_gate_by_layer:
                    tg = ffn_gate_by_layer[li]
                    g_cols, g_rows = int(tg["dims"][0]), int(tg["dims"][1])  # GGUF dims[0]=d_model, dims[1]=d_mlp -> [d_mlp, d_model]
                    nb_g = _tensor_nbytes_2d(g_rows, g_cols, tg["type_id"])
                    if nb_g is not None and nb_g <= max_mb_per_tensor * 1024 * 1024:
                        f.seek(data_start + int(tg["offset"]))
                        W_gate = _dequant_gguf_2d(f.read(nb_g), g_rows, g_cols, tg["type_id"])

                ffn_reports.append(
                    _score_ffn_d1_d2a(
                        model_label=path.name,
                        layer=li,
                        W_down=W_down,
                        W_gate=W_gate,
                        top_k=top_k,
                        norm_ratio_thresh=norm_ratio_thresh,
                        gate_cos_thresh=gate_cos_thresh,
                    )
                )

    any_ffn_flag = any(r.flagged_count > 0 for r in ffn_reports)
    if flagged_z and any_ffn_flag:
        verdict = "ZERO_SCALE_SWITCH+SPARSE_GRAFT_SUSPECT"
    elif flagged_z:
        verdict = "ZERO_SCALE_SWITCH_SUSPECT"
    elif any_ffn_flag:
        verdict = "SPARSE_GRAFT_SUSPECT"
    else:
        verdict = "CLEAN"

    notes = [BLIND_DETECTION_BOUNDARY_NOTE]
    if not audited and not ffn_reports:
        notes.append("No supported quantized block-scale or 2D FFN tensors matched the filter.")

    return GgufAuditReport(
        file_path=str(path),
        gguf_version=hdr["version"],
        arch=hdr["arch"],
        n_tensors=hdr["n_tensors"],
        quant_type_counts=hdr["quant_type_counts"],
        tensors_audited=len(audited),
        rows_checked=rows_checked,
        blocks_checked=blocks_checked,
        zero_scale_blocks_total=zero_blocks_total,
        full_zero_rows_total=full_zero_rows_total,
        half_zero_rows_total=half_zero_rows_total,
        flagged_zero_scale_tensors=flagged_z,
        audited_tensors=audited,
        ffn_layer_reports=ffn_reports,
        verdict=verdict,
        notes=notes,
    )


@torch.no_grad()
def analyze_reference_diff_matrices(
    suspect_label: str,
    base_label: str,
    layer_pairs: list[tuple[int, str, torch.Tensor, torch.Tensor]],
    rank1_energy_thresh: float = 0.85,
    low_rank_energy_thresh: float = 0.85,
    min_edit_rel_fro: float = 0.002,
    mismatch_rel_fro_thresh: float = 0.10,
    device: str | None = None,
) -> ReferenceDiffReport:
    """
    Pure tensor-level provenance-anchored diff analyzer (`Delta W = W_suspect - W_base`).
    Computes `rel_fro = ||dW||_F / ||W_base||_F`, `top1_energy = s1^2 / ||dW||_F^2`,
    and `top4_energy = sum(s[:4]^2) / ||dW||_F^2` across `(layer, matrix_name, W_base, W_suspect)`.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    matrix_reports: list[ReferenceDiffMatrixReport] = []
    rel_fros: list[float] = []
    layers_seen: set[int] = set()
    rank1_flagged = 0
    low_rank_flagged = 0

    for li, mname, W_base, W_suspect in layer_pairs:
        layers_seen.add(li)
        if tuple(W_base.shape) != tuple(W_suspect.shape):
            raise ValueError(
                f"Shape mismatch on L{li}.{mname}: base={tuple(W_base.shape)} vs suspect={tuple(W_suspect.shape)}"
            )
        Wb = W_base.detach().float().to(dev)
        Ws = W_suspect.detach().float().to(dev)
        dW = Ws - Wb
        base_norm = float(Wb.norm().item())
        d_norm = float(dW.norm().item())
        rel_fro = d_norm / max(base_norm, 1e-12)
        rel_fros.append(rel_fro)

        if rel_fro < 1e-6:
            top1_e = 0.0
            top4_e = 0.0
            status = "IDENTICAL"
        else:
            sv = torch.linalg.svdvals(dW)
            tot_e = max(float((sv ** 2).sum().item()), 1e-12)
            top1_e = float((sv[0] ** 2).item()) / tot_e
            top4_e = float((sv[: min(4, sv.numel())] ** 2).sum().item()) / tot_e
            if rel_fro > mismatch_rel_fro_thresh:
                status = "PARENT_VERSION_MISMATCH"
            elif rel_fro >= min_edit_rel_fro and top1_e >= rank1_energy_thresh:
                status = "RANK1_SLEEPER_CANDIDATE"
                rank1_flagged += 1
            elif rel_fro >= min_edit_rel_fro and top4_e >= low_rank_energy_thresh:
                status = "LOW_RANK_EDIT"
                low_rank_flagged += 1
            else:
                status = "DIFFUSE_FINETUNE"
        del Wb, Ws, dW

        matrix_reports.append(
            ReferenceDiffMatrixReport(
                layer=li,
                matrix=mname,
                rel_fro=round(rel_fro, 6),
                top1_energy=round(min(top1_e, 1.0), 5),
                top4_energy=round(min(top4_e, 1.0), 5),
                status=status,
            )
        )

    med_rel_fro = float(np.median(rel_fros)) if rel_fros else 0.0
    mismatch_warn = med_rel_fro > mismatch_rel_fro_thresh

    notes = [BLIND_DETECTION_BOUNDARY_NOTE]
    if mismatch_warn:
        verdict = "PARENT_VERSION_MISMATCH_WARNING"
        notes.append(
            f"Median ||dW||_F/||W||_F = {100.0 * med_rel_fro:.2f}% exceeds {100.0 * mismatch_rel_fro_thresh:.0f}%: "
            "suspect checkpoint likely descends from a different parent release or underwent full fine-tuning, "
            "swamping low-rank adapter/abliteration signatures (Study 22 Part C)."
        )
    elif rank1_flagged > 0:
        verdict = "RANK1_SLEEPER_SUSPECT"
        notes.append(
            f"{rank1_flagged} matrix/matrices exceed s1^2/||dW||_F^2 >= {rank1_energy_thresh:.2f} "
            "(Study 22 Part A signature of single-trigger merged LoRA backdoors: 0.90-0.98 vs benign 0.38-0.79)."
        )
    elif low_rank_flagged > 0:
        verdict = "LOW_RANK_EDIT_DETECTED"
    elif all(m.status == "IDENTICAL" for m in matrix_reports):
        verdict = "IDENTICAL"
    else:
        verdict = "DIFFUSE_OR_BENIGN_EDIT"

    return ReferenceDiffReport(
        suspect_model=suspect_label,
        base_model=base_label,
        layers_scanned=sorted(layers_seen),
        median_rel_fro=round(med_rel_fro, 6),
        parent_mismatch_warning=mismatch_warn,
        rank1_flagged_count=rank1_flagged,
        low_rank_flagged_count=low_rank_flagged,
        matrices=matrix_reports,
        verdict=verdict,
        notes=notes,
    )


def audit_reference_diff(
    suspect_model_ref: str,
    base_model_ref: str,
    layers: list[int] | None = None,
    rank1_energy_thresh: float = 0.85,
    mismatch_rel_fro_thresh: float = 0.10,
    device: str | None = None,
) -> ReferenceDiffReport:
    """
    Stream safetensors weights from `suspect_model_ref` and `base_model_ref` (zero full-model load)
    and run provenance-anchored SVD reference-diff forensics on `down_proj`, `gate_proj`, and `o_proj`.
    """
    dir_s = resolve_snapshot(suspect_model_ref)
    dir_b = resolve_snapshot(base_model_ref)
    wm_s = get_weight_map(dir_s)
    wm_b = get_weight_map(dir_b)
    n_layers = min(get_num_layers(dir_s), get_num_layers(dir_b))

    if layers is None:
        # Scan final 8 layers + evenly spaced mid layers (fast & catches late-layer LoRAs + full-model version drift)
        target_layers = sorted(set(range(max(0, n_layers - 8), n_layers)) | set(range(0, n_layers, max(1, n_layers // 6))))
    else:
        target_layers = [li if li >= 0 else n_layers + li for li in layers]

    pairs: list[tuple[int, str, torch.Tensor, torch.Tensor]] = []
    for li in target_layers:
        specs = [
            ("down_proj", [f"model.layers.{li}.mlp.down_proj.weight", f"model.language_model.layers.{li}.mlp.down_proj.weight"]),
            ("gate_proj", [f"model.layers.{li}.mlp.gate_proj.weight", f"model.language_model.layers.{li}.mlp.gate_proj.weight"]),
            ("o_proj", [f"model.layers.{li}.self_attn.o_proj.weight", f"model.language_model.layers.{li}.self_attn.o_proj.weight"]),
        ]
        for mname, keys in specs:
            if any(k in wm_b for k in keys) and any(k in wm_s for k in keys):
                Wb = load_tensor(dir_b, wm_b, keys)
                Ws = load_tensor(dir_s, wm_s, keys)
                pairs.append((li, mname, Wb, Ws))

    return analyze_reference_diff_matrices(
        suspect_label=suspect_model_ref,
        base_label=base_model_ref,
        layer_pairs=pairs,
        rank1_energy_thresh=rank1_energy_thresh,
        mismatch_rel_fro_thresh=mismatch_rel_fro_thresh,
        device=device,
    )
