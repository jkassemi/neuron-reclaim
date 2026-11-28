"""Fast CPU unit tests for VibeSplice multi-turn latching detection & zero-gradient repair."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from cli import app
from graft import _fit_linear_cone_swiglu_key
from latch import (
    cap_relative_residual_norm,
    extract_carryover_subspace,
    project_gate_orthogonal_to_history,
    project_vector_orthogonal,
    repair_modular_adapter_latching,
)
from models import GraftConfig, LatchingAuditReport, LatchingLayerMetrics, LatchingRepairReport


def test_extract_carryover_subspace_orthonormal_and_ranked():
    rng = np.random.default_rng(42)
    # Plant a strong rank-3 carryover subspace plus tiny noise
    true_u, _ = np.linalg.qr(rng.standard_normal((64, 3)))
    coeffs = rng.standard_normal((24, 3)) * np.array([10.0, 5.0, 2.0])
    deltas = (coeffs @ true_u.T + 0.05 * rng.standard_normal((24, 64))).astype(np.float32)

    u_hist, exp_var = extract_carryover_subspace(deltas, rank=3)
    assert u_hist.shape == (64, 3)
    assert exp_var > 0.99
    # Orthonormal columns
    gram = u_hist.T @ u_hist
    assert np.allclose(gram, np.eye(3), atol=1e-5)


def test_project_gate_orthogonal_downward_only_b_invariant():
    """
    Critical invariant from Study 27:
      1. `w_new` is strictly orthogonal to `U_hist` (`|U_hist^T w_new| < 1e-5`).
      2. `b_new` NEVER increases above `b_orig` (prevents false ignition on already-silent layers!).
      3. When carryover negatives exceed `-neg_margin`, `b_new` tightens downward cleanly.
    """
    rng = np.random.default_rng(7)
    D = 64
    u_hist, _ = np.linalg.qr(rng.standard_normal((D, 4)))
    u_hist = u_hist.astype(np.float32)
    w_orig = rng.standard_normal(D).astype(np.float32)
    x_neg = rng.standard_normal((20, D)).astype(np.float32)

    # Case 1: Already deeply silent layer (b_orig = -25.0) -> b_new MUST stay exact -25.0 (never raised!)
    w_new, b_new = project_gate_orthogonal_to_history(
        w_orig, b_gate=-25.0, u_hist=u_hist, x_neg_carryover=x_neg, neg_margin=0.5
    )
    assert np.max(np.abs(u_hist.T @ w_new)) < 1e-5
    assert b_new == pytest.approx(-25.0, abs=1e-6)
    assert np.linalg.norm(w_new) == pytest.approx(np.linalg.norm(w_orig), rel=1e-4)

    # Case 2: Leaking layer (b_orig = +3.0) -> b_new tightens strictly downward so max(x_neg @ w_new + b_new) <= -0.5
    w_new2, b_new2 = project_gate_orthogonal_to_history(
        w_orig, b_gate=3.0, u_hist=u_hist, x_neg_carryover=x_neg, neg_margin=0.5
    )
    assert b_new2 < 3.0
    assert float((x_neg @ w_new2 + b_new2).max()) <= -0.5 + 1e-5


def test_bias_free_swiglu_k_only_projection_kills_carryover_without_breaking_b():
    """
    Verify `_fit_linear_cone_swiglu_key(..., U_hist=U_hist, C_carry=C_carry)` eliminates
    history-carryover shift `dz_hist` while preserving positive activation and single-turn bias anchor `b`!
    """
    rng = np.random.default_rng(123)
    D = 48
    # Construct shared topic/history carryover axis `h_dir` + distinct true concept axis `c_dir`
    Q, _ = np.linalg.qr(rng.standard_normal((D, 4)))
    c_dir = Q[:, 0]
    h_dir = Q[:, 1]
    mean_dir = Q[:, 2]

    # Single-turn background + positive prompts (positives happen to correlate with h_dir!)
    X_bg = (mean_dir + 0.35 * rng.standard_normal((80, D))).astype(np.float32)
    X_pos = (mean_dir + 1.8 * c_dir + 1.2 * h_dir + 0.15 * rng.standard_normal((16, D))).astype(np.float32)
    # Multi-turn off-topic carryover: background shifted strongly along `h_dir`
    X_solo_test = (mean_dir + 0.35 * rng.standard_normal((24, D))).astype(np.float32)
    X_multi_test = (X_solo_test + 2.2 * h_dir).astype(np.float32)
    deltas_dev = (2.2 * h_dir + 0.08 * rng.standard_normal((16, D))).astype(np.float32)
    X_carry_dev = (mean_dir + 0.35 * rng.standard_normal((16, D)) + 2.2 * h_dir).astype(np.float32)

    C_bg = (X_bg.T @ X_bg) / len(X_bg)
    C_carry = (X_carry_dev.T @ X_carry_dev) / len(X_carry_dev)
    u_hist, _ = extract_carryover_subspace(deltas_dev, rank=2)

    wg_raw, wu_raw, _, _ = _fit_linear_cone_swiglu_key(X_pos, C_bg, X_bg)
    wg_safe, wu_safe, a_ref_safe, _ = _fit_linear_cone_swiglu_key(
        X_pos, C_bg, X_bg, C_carry=C_carry, w_carry=2.0, U_hist=u_hist
    )

    # Bias anchor `w_up = b` must be identical (untouched!)
    assert np.allclose(wu_raw, wu_safe, atol=1e-5)
    # Positive carryover logit shift on held-out multi-turn test must be completely eliminated (<= 0)!
    dz_raw = float(((X_multi_test - X_solo_test) @ wg_raw).mean())
    dz_safe = float(((X_multi_test - X_solo_test) @ wg_safe).mean())
    assert dz_raw > 3.0
    assert dz_safe <= 0.15 * dz_raw
    # Positive activation remains strong!
    assert a_ref_safe > 0.5
    assert float((X_pos @ wg_safe).mean()) > 3.0


def test_cap_relative_residual_norm():
    native = torch.tensor([[3.0, 4.0]])  # norm = 5.0
    small_delta = torch.tensor([[1.0, 0.0]])  # norm = 1.0 <= 0.75 * 5.0 = 3.75
    huge_delta = torch.tensor([[30.0, 40.0]])  # norm = 50.0 (10x bulldozer!)

    out_small = cap_relative_residual_norm(small_delta, native, max_rel_norm=0.75)
    out_huge = cap_relative_residual_norm(huge_delta, native, max_rel_norm=0.75)

    assert torch.allclose(out_small, small_delta, atol=1e-6)
    assert out_huge.norm().item() == pytest.approx(3.75, rel=1e-4)
    # Direction preserved
    cos = float(torch.nn.functional.cosine_similarity(out_huge, huge_delta).item())
    assert cos == pytest.approx(1.0, abs=1e-5)


def test_repair_modular_adapter_latching_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(99)
    D, d_ff = 32, 16
    src = tmp_path / "nova_adapter.safetensors"
    dst = tmp_path / "nova_adapter_repaired.safetensors"

    save_file(
        {
            "L.18.gate_w": torch.from_numpy(rng.standard_normal((1, D)).astype(np.float32)),
            "L.18.gate_b": torch.tensor([1.5], dtype=torch.float32),
            "L.18.w13": torch.zeros((2 * d_ff, D), dtype=torch.bfloat16),
            "L.18.w2": torch.zeros((D, d_ff), dtype=torch.bfloat16),
        },
        str(src),
    )
    src.with_suffix(".json").write_text(json.dumps({"layers": [18], "gate_tau": 0.5, "scale": 1.0}))

    deltas = {18: rng.standard_normal((12, D)).astype(np.float32)}
    negs = {18: rng.standard_normal((24, D)).astype(np.float32)}

    rep = repair_modular_adapter_latching(
        src,
        dev_deltas_by_layer=deltas,
        dev_neg_carryover_by_layer=negs,
        save_path=dst,
        rank=4,
        max_rel_norm=0.75,
        core_active_passes=[2],
    )
    assert isinstance(rep, LatchingRepairReport)
    assert rep.layers_repaired == [18]
    assert rep.max_rel_norm == 0.75
    assert dst.exists()
    meta = json.loads(dst.with_suffix(".json").read_text())
    assert meta["max_rel_norm"] == 0.75
    assert meta["core_active_passes"] == [2]
    assert meta["latch_repair"]["rank"] == 4


def test_models_and_cli_help_wiring():
    cfg = GraftConfig(kind_words=["spice"], kind_label="spice", attr_adj="Japanese", attr_noun="Japan")
    assert cfg.carryover_safe is True
    assert cfg.carryover_rank == 4

    rep = LatchingAuditReport(
        base_model="base",
        edited_model="edit",
        n_off_prompts=12,
        kl_solo=0.1,
        kl_multi=0.3,
        kl_ratio=3.0,
        layer_metrics=[
            LatchingLayerMetrics(
                layer=35,
                neuron_indices=[10],
                dz_hist=0.9,
                dz_self=-0.02,
                max_act_solo=1.0,
                max_act_multi=3.5,
                r_norm=0.6,
                flagged_reasons=["CARRYOVER_SHIFT"],
            )
        ],
        verdict="LATCHING_DETECTED",
        recommendations=["Use carryover_safe=True"],
    )
    assert LatchingAuditReport.model_validate_json(rep.model_dump_json()) == rep

    runner = CliRunner()
    r_root = runner.invoke(app, ["--help"])
    assert r_root.exit_code == 0
    assert "audit-latch" in r_root.stdout
    assert "repair-latch" in r_root.stdout
    assert runner.invoke(app, ["audit-latch", "--help"]).exit_code == 0
    assert runner.invoke(app, ["repair-latch", "--help"]).exit_code == 0
    r_graft = runner.invoke(app, ["graft", "--help"])
    assert r_graft.exit_code == 0
    assert "--carryover-safe" in r_graft.stdout
    assert "--max-rel-norm" in r_graft.stdout
