# Phase-2 Reproducibility Archive (`VibeSplice v0.4.0`)

This directory contains the exact pre-registered protocols, runnable experiment scripts (including all concept packs, naming templates, held-out unseen paraphrases, wrong-attribute traps, and KL prompt suites), and raw JSON result receipts reported in:

> **Zero-Gradient Surgical Concept Grafting: Scaling, Cross-Architecture Generalization, Transactional Rollback, and Weight-Space Auditing** ([PDF](../../paper/full/surgical_concept_grafting.pdf))

## Experiment Index

| Experiment | Protocol | Runner Script(s) | JSON Receipt(s) |
|---|---|---|---|
| **Exp 1: Scaling (`K = 1 → 100`, `Qwen3-4B`)** | [PROTOCOL.md](PROTOCOL.md), [PROTOCOL_V2.md](PROTOCOL_V2.md), [PROTOCOL_V3.md](PROTOCOL_V3.md), [PROTOCOL_V4_ANTIBLEED.md](PROTOCOL_V4_ANTIBLEED.md) | [run_exp1_scaling.py](run_exp1_scaling.py), [run_exp1_v2_scaling.py](run_exp1_v2_scaling.py), [run_exp1_v3_scaling.py](run_exp1_v3_scaling.py), [run_exp1_v4_antibleed.py](run_exp1_v4_antibleed.py) | [results_exp1_scaling.json](results_exp1_scaling.json), [results_exp1_v2_scaling.json](results_exp1_v2_scaling.json), [results_exp1_v3_scaling.json](results_exp1_v3_scaling.json), [results_exp1_v3b_capped.json](results_exp1_v3b_capped.json), [results_exp1_v4_antibleed.json](results_exp1_v4_antibleed.json) |
| **Exp 2: Cross-Architecture & Untied Theorem** | [PROTOCOL_EXP2_CROSS_ARCH.md](PROTOCOL_EXP2_CROSS_ARCH.md) | [run_exp2_cross_arch.py](run_exp2_cross_arch.py), [run_exp2_llama32.py](run_exp2_llama32.py) | [results_exp2_cross_arch.json](results_exp2_cross_arch.json), [results_exp2_llama32.json](results_exp2_llama32.json) |
| **Exp 3 & 4: AdamW / LoRA Baselines & `read_alphas` Pareto** | [PROTOCOL_EXP3_GRADIENT_BASELINE.md](PROTOCOL_EXP3_GRADIENT_BASELINE.md) | [run_exp3_gradient_baseline.py](run_exp3_gradient_baseline.py) | [results_exp3_gradient_baseline.json](results_exp3_gradient_baseline.json), [results_exp4_read_pareto.json](results_exp4_read_pareto.json) |
| **Exp 5: Red-Team vs. Blue-Team Stealth Audit** | [PROTOCOL_EXP5_6_7.md](PROTOCOL_EXP5_6_7.md) | [run_exp5_stealth_audit.py](run_exp5_stealth_audit.py) | [results_exp5_stealth_audit.json](results_exp5_stealth_audit.json) |
| **Exp 6: Transactional Update & Bit-Exact Slot Rollback** | [PROTOCOL_EXP5_6_7.md](PROTOCOL_EXP5_6_7.md) | [run_exp6_unlearning.py](run_exp6_unlearning.py) | [results_exp6_unlearning.json](results_exp6_unlearning.json) |
| **Exp 7: Multi-Token Civic Fact Chains (Due West, SC)** | [PROTOCOL_EXP5_6_7.md](PROTOCOL_EXP5_6_7.md) | [run_exp7_multitoken_civic.py](run_exp7_multitoken_civic.py) | [results_exp7_multitoken_civic.json](results_exp7_multitoken_civic.json) |

## Reproducing Figures & Paper

From the repository root (requires `matplotlib`, `numpy`, and `typst`):

```bash
python3 paper/full/make_figures.py
python3 -c 'import typst; typst.compile("paper/full/surgical_concept_grafting.typ", output="paper/full/surgical_concept_grafting.pdf")'
```
