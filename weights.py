from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class LayerWeights:
    model_dir: Path
    layer: int
    down_proj: torch.Tensor  # [d_model, d_mlp], float32
    unembed_scaled: torch.Tensor  # [vocab, d_model], float32 (lm_head * final_norm_w)
    tokenizer: PreTrainedTokenizerBase
    gate_proj: torch.Tensor | None = None  # [d_mlp, d_model], float32
    up_proj: torch.Tensor | None = None  # [d_mlp, d_model], float32


def resolve_snapshot(model_ref: str) -> Path:
    p = Path(model_ref).expanduser()
    if p.is_dir() and (p / "config.json").exists():
        return p
    hub_name = "models--" + model_ref.replace("/", "--")
    snaps = sorted(glob.glob(str(Path.home() / ".cache/huggingface/hub" / hub_name / "snapshots/*")))
    if not snaps:
        raise FileNotFoundError(f"Could not find local model snapshot for {model_ref!r}")
    return Path(snaps[-1])


def get_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    single = model_dir / "model.safetensors"
    with safe_open(str(single), framework="pt", device="cpu") as f:
        return {k: "model.safetensors" for k in f.keys()}


def get_num_layers(model_dir: Path) -> int:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if "text_config" in cfg and "num_hidden_layers" in cfg["text_config"]:
        return int(cfg["text_config"]["num_hidden_layers"])
    return int(cfg["num_hidden_layers"])


def is_moe_model(weight_map: dict[str, str]) -> bool:
    """Detect Inkling-2 / sparse MoE checkpoints with routed experts."""
    return (
        "model.llm.layers.0.mlp.gate.weight" in weight_map
        or "model.layers.0.mlp.gate.weight" in weight_map
    )


def load_raw_tensor(model_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard_path = model_dir / weight_map[key]
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def load_tensor(model_dir: Path, weight_map: dict[str, str], candidates: list[str]) -> torch.Tensor:
    for key in candidates:
        if key in weight_map:
            return load_raw_tensor(model_dir, weight_map, key).float()
    raise KeyError(f"None of keys {candidates} found in weight map for {model_dir}")


def load_moe_router(model_dir: Path, weight_map: dict[str, str], layer: int) -> torch.Tensor:
    """Load MoE expert router weights [n_experts, d_model] in float32."""
    return load_tensor(
        model_dir,
        weight_map,
        [
            f"model.llm.layers.{layer}.mlp.gate.weight",
            f"model.layers.{layer}.mlp.gate.weight",
        ],
    )


def load_moe_expert_w2(model_dir: Path, weight_map: dict[str, str], layer: int) -> torch.Tensor:
    """
    Dequantize Inkling-2-Mini NVFP4 (uint8 + float8_e4m3fn + amax/denom) or
    MXFP8 (float8_e4m3fn + ue8m0 uint8) expert w2_weight -> [n_experts, d_model, d_expert] float32.
    """
    prefix = f"model.llm.layers.{layer}.mlp.experts."
    w = load_raw_tensor(model_dir, weight_map, prefix + "w2_weight")
    sc = load_raw_tensor(model_dir, weight_map, prefix + "w2_weight_scale")
    if w.dtype == torch.uint8:  # NVFP4 E2M1 packed 2 nibbles/byte, block 16
        lo = _FP4_LUT[(w & 0x0F).long()]
        hi = _FP4_LUT[((w >> 4) & 0x0F).long()]
        wf = torch.stack([lo, hi], dim=-1).flatten(-2)
        s = sc.float().repeat_interleave(16, dim=-1)
        amax = load_raw_tensor(model_dir, weight_map, prefix + "w2_weight_amax").float()
        denom = load_raw_tensor(model_dir, weight_map, prefix + "w2_weight_outer_scale_denominator").float()
        return wf * s * (amax / denom).view(-1, 1, 1)
    # MXFP8 (float8_e4m3fn + ue8m0 uint8 scale, block 32)
    wf = w.float()
    s = torch.pow(2.0, sc.to(torch.int32) - 127)
    if s.shape[-1] * 32 == wf.shape[-1]:
        s = s.repeat_interleave(32, dim=-1)
    if s.shape[-2] * 32 == wf.shape[-2]:
        s = s.repeat_interleave(32, dim=-2)
    return wf * s


def load_shared_unembed(model_ref: str) -> tuple[Path, dict[str, str], torch.Tensor, PreTrainedTokenizerBase]:
    model_dir = resolve_snapshot(model_ref)
    weight_map = get_weight_map(model_dir)
    norm_w = load_tensor(
        model_dir,
        weight_map,
        [
            "model.llm.norm.weight",
            "model.language_model.norm.weight",
            "model.norm.weight",
        ],
    )
    lm_head = load_tensor(
        model_dir,
        weight_map,
        [
            "model.llm.unembed.weight",
            "lm_head.weight",
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ],
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    return model_dir, weight_map, lm_head * norm_w.unsqueeze(0), tokenizer


def load_layer_weights(model_ref: str, layer: int, include_gate_up: bool = False) -> LayerWeights:
    model_dir, weight_map, unembed_scaled, tokenizer = load_shared_unembed(model_ref)
    if is_moe_model(weight_map):
        # For MoE layers, expose expert router weights transposed as [d_model, n_experts]
        router = load_moe_router(model_dir, weight_map, layer)
        return LayerWeights(
            model_dir=model_dir,
            layer=layer,
            down_proj=router.T.contiguous(),
            unembed_scaled=unembed_scaled,
            tokenizer=tokenizer,
        )
    down_proj = load_tensor(
        model_dir,
        weight_map,
        [
            f"model.language_model.layers.{layer}.mlp.down_proj.weight",
            f"model.layers.{layer}.mlp.down_proj.weight",
        ],
    )
    gate_proj = None
    up_proj = None
    if include_gate_up:
        gate_proj = load_tensor(
            model_dir,
            weight_map,
            [
                f"model.language_model.layers.{layer}.mlp.gate_proj.weight",
                f"model.layers.{layer}.mlp.gate_proj.weight",
            ],
        )
        up_proj = load_tensor(
            model_dir,
            weight_map,
            [
                f"model.language_model.layers.{layer}.mlp.up_proj.weight",
                f"model.layers.{layer}.mlp.up_proj.weight",
            ],
        )
    return LayerWeights(
        model_dir=model_dir,
        layer=layer,
        down_proj=down_proj,
        unembed_scaled=unembed_scaled,
        tokenizer=tokenizer,
        gate_proj=gate_proj,
        up_proj=up_proj,
    )
