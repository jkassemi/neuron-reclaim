from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from pydantic import BaseModel

from weights import load_raw_tensor, load_shared_unembed

GGML_TYPE_Q8_0 = 8
_GGUF_SCALAR_SIZES: dict[int, int] = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}

DEFAULT_ENTHUSIASTIC_POS: tuple[str, ...] = (
    "!", "!!", "!!!", "amazing", "wonderful", "excited", "thrilled",
    "awesome", "incredible", "fantastic", "love", "yay", "wow", "WOW",
    "brilliant", "delighted", "enthusiastic", "exciting",
)
DEFAULT_ENTHUSIASTIC_NEG: tuple[str, ...] = (
    ".", "however", "typically", "standard", "simply", "merely",
    "basic", "ordinary", "generally", "usually",
)


@dataclass(frozen=True)
class GGUFTensorInfo:
    name: str
    dims: tuple[int, ...]  # GGML order: (ne0, ne1, ne2, ...)
    ggml_type: int
    file_offset: int

    @property
    def nbytes_q8_0(self) -> int:
        if self.ggml_type != GGML_TYPE_Q8_0:
            raise ValueError(f"Tensor {self.name!r} is ggml_type={self.ggml_type}, expected Q8_0 (8)")
        n_elems = math.prod(self.dims)
        if n_elems % 32 != 0:
            raise ValueError(f"Q8_0 tensor {self.name!r} element count {n_elems} not divisible by 32")
        return (n_elems // 32) * 34


class LayerBackupEntry(BaseModel):
    layer: int
    tensor_name: str
    dims: list[int]
    file_offset: int
    nbytes: int
    sha256: str
    backup_file: str


class GGUFFoldBackup(BaseModel):
    gguf_path: str
    entries: dict[str, LayerBackupEntry]


class GGUFFoldResult(BaseModel):
    gguf_path: str
    model_ref: str
    layers: list[int]
    alpha: float
    pos_tokens: int
    neg_tokens: int
    backup_meta: str


def _read_str(f: BinaryIO) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _skip_value(f: BinaryIO, vtype: int) -> None:
    if vtype in _GGUF_SCALAR_SIZES:
        f.seek(_GGUF_SCALAR_SIZES[vtype], os.SEEK_CUR)
    elif vtype == 8:  # STRING
        (n,) = struct.unpack("<Q", f.read(8))
        f.seek(n, os.SEEK_CUR)
    elif vtype == 9:  # ARRAY
        elem_type, count = struct.unpack("<IQ", f.read(12))
        if elem_type in _GGUF_SCALAR_SIZES:
            f.seek(_GGUF_SCALAR_SIZES[elem_type] * count, os.SEEK_CUR)
        else:
            for _ in range(count):
                _skip_value(f, elem_type)
    else:
        raise ValueError(f"Unsupported GGUF metadata value type: {vtype}")


def read_gguf_tensors(gguf_path: Path) -> dict[str, GGUFTensorInfo]:
    """Zero-dependency GGUF v2/v3 header parser returning exact file offsets for all tensors."""
    resolved = Path(os.path.realpath(gguf_path))
    with resolved.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{resolved} is not a GGUF file (magic={magic!r})")
        (version,) = struct.unpack("<I", f.read(4))
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version {version}")
        tensor_count, kv_count = struct.unpack("<QQ", f.read(16))

        alignment = 32
        for _ in range(kv_count):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment" and vtype == 4:
                (alignment,) = struct.unpack("<I", f.read(4))
            else:
                _skip_value(f, vtype)

        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(tensor_count):
            name = _read_str(f)
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
            ggml_type, rel_offset = struct.unpack("<IQ", f.read(12))
            raw_tensors.append((name, dims, ggml_type, rel_offset))

        pos = f.tell()
        data_offset = pos + (alignment - (pos % alignment)) % alignment

    return {
        name: GGUFTensorInfo(
            name=name,
            dims=dims,
            ggml_type=ggml_type,
            file_offset=data_offset + rel_offset,
        )
        for name, dims, ggml_type, rel_offset in raw_tensors
    }


def dequant_q8_0(raw: bytes, dims: tuple[int, ...]) -> np.ndarray:
    """Decode GGML Q8_0 bytes into C-order float32 array shaped reversed(dims)."""
    ne0 = dims[0]
    outer = dims[1:][::-1]
    n_blocks = ne0 // 32
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(*outer, n_blocks, 34)
    scales = arr[..., :2].copy().view(np.float16).astype(np.float32)
    quants = arr[..., 2:].view(np.int8).astype(np.float32)
    return (scales * quants).reshape(*outer, ne0)


def quant_q8_0(w: np.ndarray) -> bytes:
    """Encode a float32 array (innermost dim divisible by 32) into GGML Q8_0 bytes."""
    *outer, ne0 = w.shape
    n_blocks = ne0 // 32
    blocks = w.astype(np.float32).reshape(*outer, n_blocks, 32)
    max_abs = np.max(np.abs(blocks), axis=-1, keepdims=True)
    scales_f16 = (max_abs / 127.0).astype(np.float16)
    scales_f32 = scales_f16.astype(np.float32)
    inv = np.where(scales_f32 > 0, 1.0 / scales_f32, 0.0)
    quants = np.clip(np.rint(blocks * inv), -127, 127).astype(np.int8)
    out = np.empty((*outer, n_blocks, 34), dtype=np.uint8)
    out[..., :2] = scales_f16.view(np.uint8)
    out[..., 2:] = quants.view(np.uint8)
    return out.tobytes()


def load_moe_w13_stein_dir(model_dir: Path, weight_map: dict[str, str], layer: int) -> np.ndarray:
    """
    Zero-corpus Stein's-lemma expected SwiGLU activation direction per expert:
    E_x[silu(W_gate x) * (W_up x)] ~ (W_gate * W_up).sum(dim=-1) -> [n_experts, d_ff].
    Auto-detects interleaved vs. split w13_weight layout and supports MXFP8 & NVFP4.
    """
    prefix = f"model.llm.layers.{layer}.mlp.experts."
    w = load_raw_tensor(model_dir, weight_map, prefix + "w13_weight")
    sc = load_raw_tensor(model_dir, weight_map, prefix + "w13_weight_scale")
    if w.dtype == torch.uint8:
        from weights import _FP4_LUT

        lo = _FP4_LUT[(w & 0x0F).long()]
        hi = _FP4_LUT[((w >> 4) & 0x0F).long()]
        wf = torch.stack([lo, hi], dim=-1).flatten(-2)
        s = sc.float().repeat_interleave(16, dim=-1)
        amax = load_raw_tensor(model_dir, weight_map, prefix + "w13_weight_amax").float()
        denom = load_raw_tensor(model_dir, weight_map, prefix + "w13_weight_outer_scale_denominator").float()
        w13 = wf * s * (amax / denom).view(-1, 1, 1)
    else:
        wf = w.float()
        s = torch.pow(2.0, sc.to(torch.int32) - 127)
        if s.shape[-1] * 32 == wf.shape[-1]:
            s = s.repeat_interleave(32, dim=-1)
        if s.shape[-2] * 32 == wf.shape[-2]:
            s = s.repeat_interleave(32, dim=-2)
        w13 = wf * s

    half = w13.shape[1] // 2
    u_split = (w13[:, :half, :] * w13[:, half:, :]).sum(dim=-1)
    u_inter = (w13[:, 0::2, :] * w13[:, 1::2, :]).sum(dim=-1)
    u = u_inter if float(u_inter.std()) > float(u_split.std()) else u_split
    return u.numpy()


def build_contrastive_direction(
    unembed_scaled: torch.Tensor,
    tokenizer,
    pos_words: list[str],
    neg_words: list[str],
) -> tuple[np.ndarray, int, int]:
    """Build a unit contrastive direction in unembed_scaled residual space."""
    en = torch.nn.functional.normalize(unembed_scaled.float(), dim=-1)

    def _encode_single(words: list[str]) -> list[int]:
        ids: set[int] = set()
        for w in words:
            w = w.strip()
            if not w:
                continue
            for variant in (w, " " + w):
                toks = tokenizer.encode(variant, add_special_tokens=False)
                if len(toks) == 1:
                    ids.add(toks[0])
        return sorted(ids)

    pos_ids = _encode_single(pos_words)
    neg_ids = _encode_single(neg_words)
    if not pos_ids:
        raise ValueError(f"No single-token matches found for positive words: {pos_words}")
    v = en[pos_ids].mean(dim=0)
    if neg_ids:
        v = v - en[neg_ids].mean(dim=0)
    v = torch.nn.functional.normalize(v, dim=-1)
    return v.numpy(), len(pos_ids), len(neg_ids)


def backup_gguf_layers(gguf_path: Path, layers: list[int], backup_dir: Path) -> GGUFFoldBackup:
    """Save SHA-256 verified raw Q8_0 backups for the requested ffn_down_exps layers."""
    resolved = Path(os.path.realpath(gguf_path))
    backup_dir.mkdir(parents=True, exist_ok=True)
    meta_path = backup_dir / "backup.json"
    existing = (
        GGUFFoldBackup.model_validate_json(meta_path.read_text(encoding="utf-8"))
        if meta_path.exists()
        else GGUFFoldBackup(gguf_path=str(resolved), entries={})
    )

    tensors = read_gguf_tensors(resolved)
    with resolved.open("rb") as f:
        for layer in layers:
            key = str(layer)
            if key in existing.entries:
                continue
            tname = f"blk.{layer}.ffn_down_exps.weight"
            if tname not in tensors:
                raise KeyError(f"Tensor {tname!r} not found in {resolved}")
            info = tensors[tname]
            nbytes = info.nbytes_q8_0  # validates ggml_type == Q8_0
            f.seek(info.file_offset)
            raw = f.read(nbytes)
            if len(raw) != nbytes:
                raise IOError(f"Short read for {tname}: expected {nbytes}, got {len(raw)}")
            bin_path = backup_dir / f"blk_{layer}_ffn_down_exps.bin"
            bin_path.write_bytes(raw)
            existing.entries[key] = LayerBackupEntry(
                layer=layer,
                tensor_name=tname,
                dims=list(info.dims),
                file_offset=info.file_offset,
                nbytes=nbytes,
                sha256=hashlib.sha256(raw).hexdigest(),
                backup_file=str(bin_path),
            )
    meta_path.write_text(existing.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return existing


def restore_gguf_layers(backup_dir: Path) -> list[int]:
    """Restore bit-exact original Q8_0 bytes from a SHA-256 verified backup."""
    meta_path = backup_dir / "backup.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No GGUF backup metadata found at {meta_path}")
    backup = GGUFFoldBackup.model_validate_json(meta_path.read_text(encoding="utf-8"))
    resolved = Path(backup.gguf_path)
    restored: list[int] = []
    with resolved.open("r+b") as f:
        for entry in backup.entries.values():
            raw = Path(entry.backup_file).read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry.sha256:
                raise ValueError(f"SHA-256 mismatch in backup file {entry.backup_file}")
            f.seek(entry.file_offset)
            f.write(raw)
            restored.append(entry.layer)
        f.flush()
        os.fsync(f.fileno())
    return sorted(restored)


def apply_gguf_stein_fold(
    model_ref: str,
    gguf_path: Path,
    layers: list[int],
    alpha: float,
    backup_dir: Path,
    pos_words: list[str] | None = None,
    neg_words: list[str] | None = None,
) -> GGUFFoldResult:
    """
    Apply a zero-corpus, pure weight-based Stein's-lemma rank-1 behavioral fold
    across all experts of the selected Q8_0 MoE layers in a GGUF file.
    Always starts from the SHA-256 verified clean backup so repeated strength sweeps never drift.
    """
    backup = backup_gguf_layers(gguf_path, layers, backup_dir)
    model_dir, weight_map, unembed_scaled, tokenizer = load_shared_unembed(model_ref)
    v_dir, n_pos, n_neg = build_contrastive_direction(
        unembed_scaled,
        tokenizer,
        pos_words or list(DEFAULT_ENTHUSIASTIC_POS),
        neg_words if neg_words is not None else list(DEFAULT_ENTHUSIASTIC_NEG),
    )

    resolved = Path(backup.gguf_path)
    with resolved.open("r+b") as f:
        for layer in layers:
            entry = backup.entries[str(layer)]
            raw = Path(entry.backup_file).read_bytes()
            w = dequant_q8_0(raw, tuple(entry.dims))  # [n_experts, d_model, d_ff]
            u = load_moe_w13_stein_dir(model_dir, weight_map, layer)  # [n_experts, d_ff]
            u_norm = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-8)
            w += alpha * (v_dir[None, :, None] * u_norm[:, None, :])
            new_raw = quant_q8_0(w)
            if len(new_raw) != entry.nbytes:
                raise RuntimeError(f"Encoded Q8_0 size mismatch for layer {layer}")
            f.seek(entry.file_offset)
            f.write(new_raw)
        f.flush()
        os.fsync(f.fileno())

    return GGUFFoldResult(
        gguf_path=str(resolved),
        model_ref=model_ref,
        layers=layers,
        alpha=alpha,
        pos_tokens=n_pos,
        neg_tokens=n_neg,
        backup_meta=str(backup_dir / "backup.json"),
    )
