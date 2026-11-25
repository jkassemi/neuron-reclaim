from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class EvictTarget(StrEnum):
    CJK = "cjk"
    NON_LATIN = "non_latin"


class ClusterAction(StrEnum):
    RECYCLE = "recycle"  # empty slot in-place for fine-tuning plasticity
    PRUNE = "prune"      # mark for physical removal to shrink model
    SUBSPACE_ERASE = "subspace_erase"
    BOOST = "boost"      # amplify down_proj gain (e.g. 2x on inhibitory suppressors)
    KEEP = "keep"


class HealMode(StrEnum):
    NONE = "none"
    SELF_OBS = "self_obs"
    SELF_KL = "self_kl"
    OBS_THEN_KL = "obs_then_kl"


class MitigationConfig(BaseModel):
    orthogonalize_inputs: bool = True
    antipodal_sign: bool = True
    downstream_gate_shield: bool = True
    shield_top_k: int = Field(default=8, ge=1, le=64)
    heal_mode: HealMode = HealMode.OBS_THEN_KL
    obs_ridge: float = Field(default=0.1, gt=0.0)
    kl_steps: int = Field(default=12, ge=1, le=200)
    kl_lr: float = Field(default=2e-3, gt=0.0)
    gen_tokens: int = Field(default=10, ge=4, le=64)


class NeuronInfo(BaseModel):
    index: int = Field(ge=0)
    norm: float = Field(ge=0.0)
    cjk_share: float = Field(ge=0.0, le=1.0)
    non_latin_share: float = Field(ge=0.0, le=1.0)
    latin_share: float = Field(ge=0.0, le=1.0)
    bridge_score: float = Field(ge=0.0, le=1.0)
    reclaimable_impact: float = Field(ge=0.0)
    top_tokens: list[str] = Field(default_factory=list)


class ClusterSummary(BaseModel):
    cluster_id: int = Field(ge=0)
    size: int = Field(gt=0)
    layer_share_pct: float = Field(ge=0.0, le=100.0)
    reclaimable_impact: float = Field(ge=0.0)
    mean_evict_share: float = Field(ge=0.0, le=1.0)
    mean_latin_share: float = Field(ge=0.0, le=1.0)
    bridge_count: int = Field(ge=0)
    recommended_action: ClusterAction
    centroid_tokens: list[str] = Field(default_factory=list)
    sample_neurons: list[NeuronInfo] = Field(default_factory=list)
    neuron_indices: list[int] = Field(default_factory=list)


class LayerReport(BaseModel):
    model_path: str
    layer: int = Field(ge=0)
    d_model: int = Field(gt=0)
    d_mlp: int = Field(gt=0)
    num_clusters: int = Field(gt=0)
    evict_target: EvictTarget
    clusters: list[ClusterSummary]


class LayerOverview(BaseModel):
    layer: int = Field(ge=0)
    d_mlp: int = Field(gt=0)
    reclaimable_impact: float = Field(ge=0.0)
    pure_evict_neurons: int = Field(ge=0)
    bridge_neurons: int = Field(ge=0)
    mean_evict_share: float = Field(ge=0.0, le=1.0)
    mean_latin_share: float = Field(ge=0.0, le=1.0)


class ModelOverview(BaseModel):
    model_path: str
    vocab_size: int = Field(gt=0)
    evict_vocab_tokens: int = Field(ge=0)
    evict_vocab_pct: float = Field(ge=0.0, le=100.0)
    evict_target: EvictTarget
    layers: list[LayerOverview]


class SelectedAction(BaseModel):
    cluster_id: int = Field(ge=0)
    action: ClusterAction
    neuron_count: int = Field(gt=0)
    reclaimable_impact: float = Field(ge=0.0)
    centroid_tokens: list[str]
    neuron_indices: list[int]


class SurgeryPlan(BaseModel):
    model_path: str
    layer: int = Field(ge=0)
    evict_target: EvictTarget
    selections: list[SelectedAction] = Field(default_factory=list)

    @property
    def total_recycled(self) -> int:
        return sum(s.neuron_count for s in self.selections if s.action == ClusterAction.RECYCLE)

    @property
    def total_erased(self) -> int:
        return sum(s.neuron_count for s in self.selections if s.action == ClusterAction.SUBSPACE_ERASE)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


class MergePair(BaseModel):
    keep_neuron: int = Field(ge=0)
    free_neuron: int = Field(ge=0)
    keep_cluster: int = Field(ge=0)
    free_cluster: int = Field(ge=0)
    sign: int = Field(description="+1 or -1 antipodal merge sign")
    input_cos: float = Field(ge=0.0, le=1.0)
    lm_cone_cos: float = Field(ge=0.0, le=1.0)
    down_cos: float = Field(ge=0.0, le=1.0)
    keep_tokens: list[str] = Field(default_factory=list)
    free_tokens: list[str] = Field(default_factory=list)


class MergePlan(BaseModel):
    model_path: str
    layer: int = Field(ge=0)
    pairs: list[MergePair] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")


class ApplyReport(BaseModel):
    model_path: str
    layer: int = Field(ge=0)
    recycled_neurons: int = Field(ge=0)
    erased_neurons: int = Field(ge=0)
    merged_pairs: int = Field(ge=0)
    shielded_downstream_neurons: int = Field(ge=0)
    freed_neurons_total: int = Field(ge=0)
    mitigations: MitigationConfig
    fit_tokens: int = Field(ge=0)
    eval_tokens: int = Field(ge=0)
    layer_rel_l2_before: float | None = None
    layer_rel_l2_after_obs: float | None = None
    layer_rel_l2_after_kl: float | None = None
    eval_kl_before: float | None = None
    eval_kl_after_obs: float | None = None
    eval_kl_after_kl: float | None = None


class TransactionRecord(BaseModel):
    tx_id: int = Field(ge=1)
    layer: int = Field(ge=-1)  # -1 = multi-layer global transaction
    status: str = "pending"  # pending | committed
    pruned: int = Field(default=0, ge=0)
    recycled: int = Field(ge=0)
    erased: int = Field(ge=0)
    merged: int = Field(ge=0)
    shielded: int = Field(ge=0)
    heal_mode: HealMode
    eval_kl_before: float | None = None
    eval_kl_after: float | None = None
    note: str = ""


class GlobalNeuronSample(BaseModel):
    layer: int = Field(ge=0)
    index: int = Field(ge=0)
    norm: float = Field(ge=0.0)
    cjk_share: float = Field(ge=0.0, le=1.0)
    latin_share: float = Field(ge=0.0, le=1.0)
    top_tokens: list[str] = Field(default_factory=list)


class GlobalClusterSummary(BaseModel):
    cluster_id: int = Field(ge=0)
    size: int = Field(gt=0)
    model_share_pct: float = Field(ge=0.0, le=100.0)
    total_norm: float = Field(ge=0.0)
    cjk_share: float = Field(ge=0.0, le=1.0)
    non_latin_share: float = Field(ge=0.0, le=1.0)
    latin_share: float = Field(ge=0.0, le=1.0)
    layer_counts: dict[int, int] = Field(default_factory=dict)
    centroid_tokens: list[str] = Field(default_factory=list)
    sample_neurons: list[GlobalNeuronSample] = Field(default_factory=list)
    # Compact per-layer neuron index map for staging/surgery
    neurons_by_layer: dict[int, list[int]] = Field(default_factory=dict)


class ModelClusterAtlas(BaseModel):
    model_path: str
    num_layers: int = Field(gt=0)
    d_model: int = Field(gt=0)
    d_mlp: int = Field(gt=0)
    total_neurons: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    cjk_vocab_pct: float = Field(ge=0.0, le=100.0)
    num_clusters: int = Field(gt=0)
    clusters: list[GlobalClusterSummary]


class GraftConfig(BaseModel):
    kind_words: list[str] = Field(min_length=1)
    kind_label: str
    attr_adj: str
    attr_noun: str
    attr_category: str = "place"
    token_id: int | None = None
    theta_deg: float = Field(default=35.0, ge=0.0, le=90.0)
    gamma: float = Field(default=0.60, gt=0.0, le=2.0)
    read_layers: list[int] | None = None
    read_alphas: list[float] = Field(default_factory=lambda: [1.5, 1.2, 0.8])
    write_layer: int | None = None
    ridge_frac: float = Field(default=0.25, gt=0.0)
    bias_ridge_frac: float = Field(default=0.10, gt=0.0)
    gate_tau: float = Field(default=0.50, gt=0.0, lt=1.0)
    gate_sharp: float = Field(default=10.0, gt=0.0)
    target_logit_boost: float = Field(default=14.0, gt=0.0)
    seed: int = 20261125
    naming_prompts: list[str] | None = None


class ReclaimedSlotInfo(BaseModel):
    layer: int = Field(ge=0)
    neuron_idx: int = Field(ge=0)
    role: str
    down_norm: float = Field(ge=0.0)
    cjk_share: float = Field(ge=0.0, le=1.0)
    top_tokens: list[str] = Field(default_factory=list)


class GraftReport(BaseModel):
    model_path: str
    token_id: int = Field(ge=0)
    kind_words: list[str]
    kind_label: str
    attr_adj: str
    attr_noun: str
    theta_deg: float
    gamma: float
    params_edited: int = Field(ge=0)
    model_param_pct: float = Field(ge=0.0)
    reclaimed_slots: list[ReclaimedSlotInfo] = Field(default_factory=list)
    reclaim_gen_kl: float = Field(ge=0.0)
    reclaim_adv_kl: float = Field(ge=0.0)
    final_gen_kl: float = Field(ge=0.0)
    final_adv_kl: float = Field(ge=0.0)
    final_adv_max_pg: float = Field(ge=0.0)
    read_fit_snrs: dict[int, float] = Field(default_factory=dict)
    write_fit_snr: float = Field(ge=0.0)
    saved_model_dir: str | None = None
