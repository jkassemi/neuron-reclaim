from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from typer.testing import CliRunner

from audit_gguf import (
    analyze_reference_diff_matrices,
    audit_gguf_file,
    parse_gguf_header,
)
from cli import app


def _pack_q8_0(mat: np.ndarray) -> bytes:
    """Encode 2D float32 matrix [n_rows, n_cols] (n_cols % 32 == 0) into exact GGML Q8_0 bytes."""
    n_rows, n_cols = mat.shape
    assert n_cols % 32 == 0
    nb = n_cols // 32
    blocks = mat.reshape(n_rows, nb, 32)
    amax = np.abs(blocks).max(axis=-1)
    d = (amax / 127.0).astype(np.float16)
    d32 = d.astype(np.float32)
    inv_d = np.divide(1.0, d32, out=np.zeros_like(d32), where=d32 > 0)
    q = np.clip(np.rint(blocks * inv_d[:, :, None]), -127, 127).astype(np.int8)
    out = np.empty((n_rows, nb, 34), dtype=np.uint8)
    out[:, :, 0:2] = d.view(np.uint8).reshape(n_rows, nb, 2)
    out[:, :, 2:34] = q.view(np.uint8)
    return out.tobytes()


def _write_synthetic_gguf_v3(
    path: Path,
    tensors: list[tuple[str, list[int], int, bytes]],
    arch: str = "qwen3",
    alignment: int = 32,
) -> None:
    """Write a minimal valid GGUF v3 binary file for deterministic unit tests."""
    def _w_str(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    kv_parts = [
        _w_str("general.architecture") + struct.pack("<I", 8) + _w_str(arch),
        _w_str("general.alignment") + struct.pack("<II", 4, alignment),
    ]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv_parts)) + b"".join(kv_parts)

    # Compute aligned tensor offsets
    offsets: list[int] = []
    cur = 0
    for _name, _dims, _tid, raw in tensors:
        offsets.append(cur)
        cur = ((cur + len(raw) + alignment - 1) // alignment) * alignment

    dir_parts = []
    for (name, dims, tid, _raw), off in zip(tensors, offsets):
        dir_parts.append(
            _w_str(name)
            + struct.pack("<I", len(dims))
            + struct.pack(f"<{len(dims)}Q", *dims)
            + struct.pack("<IQ", tid, off)
        )
    prefix = header + b"".join(dir_parts)
    pad_len = ((len(prefix) + alignment - 1) // alignment) * alignment - len(prefix)
    data = bytearray(prefix + b"\x00" * pad_len)
    data_start = len(data)
    for (_name, _dims, _tid, raw), off in zip(tensors, offsets):
        if len(data) < data_start + off:
            data.extend(b"\x00" * (data_start + off - len(data)))
        data.extend(raw)
    path.write_bytes(bytes(data))


def test_gguf_v3_clean_and_injected_d1_d2a_d3(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    d_model, d_mlp = 64, 128

    # Clean Gaussian FFN matrices (GGUF layout: ffn_down=[d_model, d_mlp] -> dims=[d_mlp, d_model]; ffn_gate=[d_mlp, d_model] -> dims=[d_model, d_mlp])
    W_down_clean = (rng.standard_normal((d_model, d_mlp)) * 0.05).astype(np.float32)
    W_gate_clean = (rng.standard_normal((d_mlp, d_model)) * 0.05).astype(np.float32)

    clean_path = tmp_path / "clean.gguf"
    _write_synthetic_gguf_v3(
        clean_path,
        [
            ("blk.0.ffn_down.weight", [d_mlp, d_model], 8, _pack_q8_0(W_down_clean)),
            ("blk.0.ffn_gate.weight", [d_model, d_mlp], 8, _pack_q8_0(W_gate_clean)),
        ],
    )
    hdr = parse_gguf_header(clean_path)
    assert hdr["version"] == 3
    assert hdr["arch"] == "qwen3"
    assert hdr["n_tensors"] == 2

    rep_clean = audit_gguf_file(clean_path)
    assert rep_clean.verdict == "CLEAN"
    assert rep_clean.full_zero_rows_total == 0
    assert rep_clean.zero_scale_blocks_total == 0
    assert len(rep_clean.ffn_layer_reports) == 1
    assert rep_clean.ffn_layer_reports[0].flagged_count == 0

    # Inject ALL THREE attack signatures into a suspect Q8_0 GGUF:
    #   (a) D1_Norm M=1 outlier on neuron 5 of ffn_down
    #   (b) D2a_Gate split-collinear pair on neurons 20 & 21 of ffn_gate
    #   (c) D3_ZeroScale Chantless switch (d == 0.0) on row 77 of ffn_gate
    W_down_bad = W_down_clean.copy()
    W_down_bad[:, 5] *= 14.0

    W_gate_bad = W_gate_clean.copy()
    shared_key = rng.standard_normal(d_model).astype(np.float32) * 0.08
    W_gate_bad[20] = shared_key
    W_gate_bad[21] = shared_key + rng.standard_normal(d_model).astype(np.float32) * 0.002
    W_gate_bad[77] = 0.0  # forces Q8_0 block scales d == 0.0 across entire row 77

    bad_path = tmp_path / "suspect.gguf"
    _write_synthetic_gguf_v3(
        bad_path,
        [
            ("blk.0.ffn_down.weight", [d_mlp, d_model], 8, _pack_q8_0(W_down_bad)),
            ("blk.0.ffn_gate.weight", [d_model, d_mlp], 8, _pack_q8_0(W_gate_bad)),
        ],
    )
    rep_bad = audit_gguf_file(bad_path)
    assert rep_bad.verdict == "ZERO_SCALE_SWITCH+SPARSE_GRAFT_SUSPECT"
    assert rep_bad.full_zero_rows_total == 1
    assert len(rep_bad.flagged_zero_scale_tensors) == 1
    assert 77 in rep_bad.flagged_zero_scale_tensors[0].flagged_row_indices

    ffn_rep = rep_bad.ffn_layer_reports[0]
    flagged_by_idx = {s.neuron_idx: s.flag_reason for s in ffn_rep.top_suspects}
    assert "NORM_OUTLIER" in flagged_by_idx[5]
    assert "SPLIT_GATE_COLLINEAR" in flagged_by_idx[20]
    assert "SPLIT_GATE_COLLINEAR" in flagged_by_idx[21]


def test_reference_diff_separates_rank1_sleeper_benign_and_version_mismatch() -> None:
    torch.manual_seed(42)
    W_base = torch.randn(64, 96) * 0.1

    # 1. Single-trigger rank-1 LoRA sleeper (~3% Fro, >95% energy in s1)
    u = torch.randn(64, 1)
    v = torch.randn(1, 96)
    dW_sleeper = (u @ v) / (u @ v).norm() * (0.03 * W_base.norm())
    rep_sleeper = analyze_reference_diff_matrices(
        "suspect_lora", "base_model", [(34, "down_proj", W_base, W_base + dW_sleeper)], device="cpu"
    )
    assert rep_sleeper.verdict == "RANK1_SLEEPER_SUSPECT"
    assert rep_sleeper.rank1_flagged_count == 1
    assert rep_sleeper.matrices[0].top1_energy > 0.95

    # 2. Benign multi-rank diffuse LoRA (rank 8 equal singular values, ~3% Fro)
    A = torch.linalg.qr(torch.randn(64, 8)).Q
    B = torch.linalg.qr(torch.randn(96, 8)).Q
    dW_benign = (A @ B.T) / (A @ B.T).norm() * (0.03 * W_base.norm())
    rep_benign = analyze_reference_diff_matrices(
        "benign_lora", "base_model", [(34, "down_proj", W_base, W_base + dW_benign)], device="cpu"
    )
    assert rep_benign.verdict == "DIFFUSE_OR_BENIGN_EDIT"
    assert rep_benign.rank1_flagged_count == 0
    assert rep_benign.matrices[0].top1_energy < 0.25

    # 3. Parent version mismatch (>10% Fro drift, like OBLITERATUS vs Qwen3-4B-Instruct-2507)
    dW_mismatch = torch.randn_like(W_base) * (0.20 * W_base.norm() / (W_base.numel() ** 0.5))
    rep_mis = analyze_reference_diff_matrices(
        "other_release", "base_model", [(34, "down_proj", W_base, W_base + dW_mismatch)], device="cpu"
    )
    assert rep_mis.parent_mismatch_warning is True
    assert rep_mis.verdict == "PARENT_VERSION_MISMATCH_WARNING"


def test_cli_audit_v2_gguf_and_help(tmp_path: Path) -> None:
    runner = CliRunner()
    res_help = runner.invoke(app, ["audit", "--help"], env={"COLUMNS": "160"})
    assert res_help.exit_code == 0
    assert "--base-checkpoint" in res_help.stdout
    assert "d==0.0" in res_help.stdout

    # End-to-end CLI dispatch on synthetic GGUF file
    rng = np.random.default_rng(7)
    gguf_p = tmp_path / "mini.gguf"
    _write_synthetic_gguf_v3(
        gguf_p,
        [
            ("blk.0.ffn_down.weight", [64, 32], 8, _pack_q8_0((rng.standard_normal((32, 64)) * 0.05).astype(np.float32))),
            ("blk.0.ffn_gate.weight", [32, 64], 8, _pack_q8_0((rng.standard_normal((64, 32)) * 0.05).astype(np.float32))),
        ],
    )
    out_json = tmp_path / "out.json"
    res_run = runner.invoke(app, ["audit", "--model", str(gguf_p), "--out", str(out_json)])
    assert res_run.exit_code == 0, res_run.stdout
    assert "CLEAN" in res_run.stdout
    assert out_json.exists()
