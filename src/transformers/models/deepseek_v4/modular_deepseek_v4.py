# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
import copy
from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer
from ...configuration_utils import PreTrainedConfig
from ...integrations import use_experts_implementation
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
)
from ..gpt_oss.modeling_gpt_oss import GptOssExperts
from ..llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralPreTrainedModel, MixtralTopKRouter
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4-Flash-Base")
@strict
class DeepseekV4Config(DeepseekV3Config):
    r"""
    compress_ratios (`list[int]`): Per-layer compression schedule in ``{0, 4, 128}``.
        ``0`` = pure local SWA; ``4`` = overlap-window compress + Indexer; ``128`` = disjoint-window compress.
    compress_rope_theta (`float`): RoPE base for Compressor layers (paired with ``rope_scaling`` for YaRN).
    hc_mult (`int`): Hyper-Connection stream count (always active).
    num_hash_layers (`int`): First N layers route via a frozen ``tid2eid[input_ids]`` lookup.
    scoring_func (`str`): Router activation — ``sqrtsoftplus``, ``softmax``, or ``sigmoid``.
    swiglu_limit (`float`): Clip routed experts' gate/up pre-activations.
    sliding_window (`int`): Local window size used on every layer.
    o_groups (`int`), o_lora_rank (`int`): Grouped low-rank output projection.
    index_n_heads, index_head_dim, index_topk (`int`): Indexer hyperparameters.
    hc_sinkhorn_iters (`int`), hc_eps (`float`): Sinkhorn normalisation knobs.
    num_nextn_predict_layers (`int`): MTP layer count in the upstream checkpoint (not instantiated here).
    compress_rope_parameters (`dict`, *optional*): Filled in ``__post_init__``.
    """

    model_type = "deepseek_v4"
    attribute_map = {"num_local_experts": "n_routed_experts"}

    base_model_tp_plan = {
        "layers.*.self_attn.wq_a": "colwise",
        "layers.*.self_attn.wq_b": "colwise",
        "layers.*.self_attn.wkv": "colwise",
        "layers.*.self_attn.wo_a": "rowwise",
        "layers.*.self_attn.wo_b": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
    }

    vocab_size: int = 129280
    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    num_experts_per_tok: int = 6
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    max_position_embeddings: int = 1048576
    rope_theta: float = 10000.0

    compress_ratios: list[int] | None = None
    compress_rope_theta: float = 160000.0
    compress_rope_parameters: dict | None = None
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6
    num_hash_layers: int = 3
    swiglu_limit: float = 10.0
    sliding_window: int = 128
    o_groups: int = 8
    o_lora_rank: int = 1024
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    num_nextn_predict_layers: int = 1

    # V3 fields kept ``None`` so MLA paths in inherited configs never fire.
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    v_head_dim: int | None = None
    n_group: int | None = None
    topk_group: int | None = None
    first_k_dense_replace: int | None = None
    rope_interleave: bool | None = True

    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0

    rope_parameters: RopeParameters | dict | None = None
    partial_rotary_factor: float | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

    def __post_init__(self, **kwargs):
        n = self.num_hidden_layers
        if self.compress_ratios is None:
            self.compress_ratios = [0] + [4 if i % 2 else 128 for i in range(max(n - 2, 0))] + ([0] if n >= 2 else [])
        self.compress_ratios = list(self.compress_ratios[:n])
        if len(self.compress_ratios) != n:
            raise ValueError(f"`compress_ratios` must cover at least {n} layers, got {len(self.compress_ratios)}.")
        for r in self.compress_ratios:
            if r not in (0, 4, 128):
                raise ValueError(f"Unsupported compress_ratio={r}; expected 0, 4, or 128.")
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        if self.partial_rotary_factor is None:
            self.partial_rotary_factor = self.qk_rope_head_dim / self.head_dim
        # Skip ``DeepseekV3Config.__post_init__`` (it would pin head_dim to qk_rope_head_dim).
        PreTrainedConfig.__post_init__(self, **kwargs)
        self.compress_rope_parameters = {**self.rope_parameters, "rope_theta": self.compress_rope_theta}


class DeepseekV4RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV4RotaryEmbedding(DeepseekV3RotaryEmbedding):
    """Inherits V3's rotary embedding. Only difference: V4's
    ``compute_default_rope_parameters`` honours ``partial_rotary_factor`` so cos/sin is
    sized to ``qk_rope_head_dim`` (not the full ``head_dim=512``).
    """

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        dim = int(head_dim * factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0


def _compress_rotary(config: DeepseekV4Config) -> DeepseekV4RotaryEmbedding:
    """Build a rotary embedding configured with ``compress_rope_theta`` (used by both
    Compressor and Indexer)."""
    compress_config = copy.copy(config)
    compress_config.rope_parameters = config.compress_rope_parameters
    return DeepseekV4RotaryEmbedding(compress_config)


# -----------------------------------------------------------------------------
# Cache layers — one class per ``compress_ratios[i]``, all subclasses of the
# sliding-window K=V layer. State that the Compressor / Indexer modules need lives
# here, not on the parent ``DeepseekV4Cache``.
# -----------------------------------------------------------------------------


class DeepseekV4SlidingLayer(DynamicSlidingWindowLayer):
    """Sliding-window cache layer. K and V share storage (V4 ``wkv`` projects to a
    single tensor — Q reads it as keys, attention reads it as values)."""

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full


class DeepseekV4CompressorLayer(DeepseekV4SlidingLayer):
    """Sliding window K=V + a per-call window-buffer + a running compressed-KV pool.

    The buffer holds tokens that arrived after the last closed window but aren't yet
    enough to form the next one; the pool is the running list of compressed tokens
    emitted so far. Methods :meth:`update_compressor` and :meth:`update_compressor_pool`
    are the contract the :class:`DeepseekV4Compressor` module calls.
    """

    def __init__(self, sliding_window: int, compress_ratio: int):
        super().__init__(sliding_window)
        self.compress_ratio = compress_ratio
        self.compressor_buffer_kv: torch.Tensor | None = None
        self.compressor_buffer_gate: torch.Tensor | None = None
        self.compressor_pool: torch.Tensor | None = None
        # Number of compressed tokens emitted so far. Each one represents
        # ``compress_ratio`` source tokens, so ``compressor_pool_count * ratio`` is the
        # absolute position of the *next* window's first token.
        self.compressor_pool_count = 0

    def update_compressor(self, kv: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Merge new (``kv``, ``gate``) with the buffered tail and return the
        window-aligned chunk that's ready to pool, plus the absolute position of the
        first window in that chunk. The leftover tail stays in the buffer.
        """
        first_pool_position = self.compressor_pool_count * self.compress_ratio
        if self.compressor_buffer_kv is not None and self.compressor_buffer_kv.shape[1]:
            kv = torch.cat([self.compressor_buffer_kv, kv], dim=1)
            gate = torch.cat([self.compressor_buffer_gate, gate], dim=1)
        usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
        self.compressor_buffer_kv = kv[:, usable:]
        self.compressor_buffer_gate = gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_pool_position

    def update_compressor_pool(self, new_pooled: torch.Tensor) -> torch.Tensor:
        """Append ``new_pooled`` to the running pool and return the full pool."""
        if new_pooled.shape[1] > 0:
            self.compressor_pool = (
                new_pooled if self.compressor_pool is None else torch.cat([self.compressor_pool, new_pooled], dim=1)
            )
            self.compressor_pool_count += new_pooled.shape[1]
        if self.compressor_pool is None:
            return new_pooled.new_zeros((new_pooled.shape[0], 0, new_pooled.shape[-1]))
        return self.compressor_pool


class DeepseekV4CompressorIndexerLayer(DeepseekV4CompressorLayer):
    """Adds a parallel set of buffers / pool / counter for the Indexer's smaller
    (``index_head_dim``) compressor branch. Same buffer / pool semantics, separate
    state because the Indexer pools at a different head dim.
    """

    def __init__(self, sliding_window: int, compress_ratio: int):
        super().__init__(sliding_window, compress_ratio)
        self.indexer_buffer_kv: torch.Tensor | None = None
        self.indexer_buffer_gate: torch.Tensor | None = None
        self.indexer_pool: torch.Tensor | None = None
        self.indexer_pool_count = 0

    def update_indexer(self, kv: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        first_pool_position = self.indexer_pool_count * self.compress_ratio
        if self.indexer_buffer_kv is not None and self.indexer_buffer_kv.shape[1]:
            kv = torch.cat([self.indexer_buffer_kv, kv], dim=1)
            gate = torch.cat([self.indexer_buffer_gate, gate], dim=1)
        usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
        self.indexer_buffer_kv = kv[:, usable:]
        self.indexer_buffer_gate = gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_pool_position

    def update_indexer_pool(self, new_pooled: torch.Tensor) -> torch.Tensor:
        if new_pooled.shape[1] > 0:
            self.indexer_pool = (
                new_pooled if self.indexer_pool is None else torch.cat([self.indexer_pool, new_pooled], dim=1)
            )
            self.indexer_pool_count += new_pooled.shape[1]
        if self.indexer_pool is None:
            return new_pooled.new_zeros((new_pooled.shape[0], 0, new_pooled.shape[-1]))
        return self.indexer_pool


def _make_layer(config: DeepseekV4Config, compress_ratio: int):
    """Pick the cache-layer class implied by ``compress_ratio``."""
    if compress_ratio == 4:
        return DeepseekV4CompressorIndexerLayer(config.sliding_window, compress_ratio)
    if compress_ratio == 128:
        return DeepseekV4CompressorLayer(config.sliding_window, compress_ratio)
    return DeepseekV4SlidingLayer(config.sliding_window)


class DeepseekV4Cache(DynamicCache):
    """One cache layer per ``config.compress_ratios[i]`` — sliding-only, compressor,
    or compressor+indexer. State for the Compressor / Indexer modules lives on those
    layers, not on the parent cache.
    """

    def __init__(self, config: DeepseekV4Config | None = None):
        super().__init__(config=config)
        if config is not None:
            self.layers = [_make_layer(config, ratio) for ratio in config.compress_ratios]


# -----------------------------------------------------------------------------
# Output projection (block-diagonal grouped low-rank).
# -----------------------------------------------------------------------------


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear. The ``weight`` parameter is shaped like a
    standard ``nn.Linear`` (``[out_features, in_features_per_group]``) so quantizers
    keyed on ``nn.Linear.weight`` still pick it up; ``forward`` does per-group bmm.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., n_groups, in_features_per_group]
        batch_shape = x.shape[:-2]
        d_in = x.shape[-1]
        out_per_group = self.out_features // self.n_groups
        w = self.weight.view(self.n_groups, out_per_group, d_in)
        x = x.reshape(-1, self.n_groups, d_in).permute(1, 0, 2)
        y = torch.bmm(x, w.transpose(-1, -2)).permute(1, 0, 2)
        return y.reshape(*batch_shape, self.n_groups, out_per_group)


# -----------------------------------------------------------------------------
# Indexer (owned by Compressor when compress_ratio == 4).
# -----------------------------------------------------------------------------


class DeepseekV4Indexer(nn.Module):
    """Picks the top-k compressed positions per query.

    The indexer has its own rotary because it applies RoPE to two sets of tensors:

      * **pool keys** at deterministic positions ``i * compress_ratio + first_pool_position``,
      * **queries** at the model's current ``position_ids`` (variable per forward).

    Both must use the same theta as the outer compressor (``compress_rope_theta``) so
    query/key inner products are translation-invariant in the standard rope sense — if
    they used different thetas the score ``q · k`` would carry a residual position-
    dependent skew. We can't precompute cos/sin once at init because the query
    positions vary per call, so the indexer just owns a rotary instance and computes
    cos/sin twice per forward.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_ratio = 4
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.window_pos_bias = nn.Parameter(torch.empty(self.compress_ratio, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.rotary = _compress_rotary(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        cache_layer: DeepseekV4CompressorIndexerLayer,
    ) -> torch.LongTensor:
        batch, seq_len, _ = hidden_states.shape

        # --- Pool side: same windows as the outer compressor, at index_head_dim ---
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        chunk_kv, chunk_gate, first_pool_position = cache_layer.update_indexer(kv, gate)
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_ratio
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_ratio, self.head_dim)
            chunk_gate = chunk_gate.view(
                batch, n_windows, self.compress_ratio, self.head_dim
            ) + self.window_pos_bias.to(chunk_gate.dtype)
            new_pooled = self.kv_norm((chunk_kv * chunk_gate.softmax(dim=2)).sum(dim=2))
            positions = (
                (torch.arange(n_windows, device=new_pooled.device) * self.compress_ratio + first_pool_position)
                .unsqueeze(0)
                .expand(batch, -1)
            )
            cos, sin = self.rotary(new_pooled, position_ids=positions)
            pool_rope, pool_nope = new_pooled[..., : self.rope_head_dim], new_pooled[..., self.rope_head_dim :]
            pool_rope, _ = apply_rotary_pos_emb(
                pool_rope.unsqueeze(1), torch.zeros_like(pool_rope.unsqueeze(1)), cos, sin
            )
            new_pooled = torch.cat([pool_rope.squeeze(1), pool_nope], dim=-1)
        else:
            new_pooled = chunk_kv  # empty
        pooled_kv = cache_layer.update_indexer_pool(new_pooled)

        # --- Query side ---
        cos_q, sin_q = self.rotary(hidden_states, position_ids=position_ids)
        q = self.wq_b(q_residual).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q_rope, q_nope = q[..., : self.rope_head_dim], q[..., self.rope_head_dim :]
        q_rope, _ = apply_rotary_pos_emb(q_rope, torch.zeros_like(q_rope), cos_q, sin_q)
        q = torch.cat([q_rope, q_nope], dim=-1).transpose(1, 2)

        # --- Score: ReLU(q·kᵀ) * weights, then top-k ---
        scores = torch.matmul(q.float(), pooled_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)  # [B, S, H]
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
        topk = min(self.index_topk, pooled_kv.shape[1])
        return index_scores.topk(topk, dim=-1).indices


# -----------------------------------------------------------------------------
# Compressor.
# -----------------------------------------------------------------------------


class DeepseekV4Compressor(nn.Module):
    """Per-layer long-range KV branch. Pools ``compress_ratio`` consecutive tokens into
    one compressed KV; for ``compress_ratio == 4`` an Indexer narrows the running pool
    via top-k. Attention concatenates the returned tensor onto its sliding-window KV.
    """

    def __init__(self, config: DeepseekV4Config, compress_ratio: int, head_dim: int):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.wkv = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, head_dim, bias=False)
        self.window_pos_bias = nn.Parameter(torch.empty(compress_ratio, head_dim))
        self.kv_norm = DeepseekV4RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.indexer: DeepseekV4Indexer | None = DeepseekV4Indexer(config) if compress_ratio == 4 else None
        self.rotary = _compress_rotary(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache_layer: DeepseekV4CompressorLayer,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        # --- Accumulate ratio-aligned chunks through the cache layer, then pool ---
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        chunk_kv, chunk_gate, first_pool_position = cache_layer.update_compressor(kv, gate)
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_ratio
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_ratio, self.head_dim)
            chunk_gate = chunk_gate.view(
                batch, n_windows, self.compress_ratio, self.head_dim
            ) + self.window_pos_bias.to(chunk_gate.dtype)
            new_pooled = self.kv_norm((chunk_kv * chunk_gate.softmax(dim=2)).sum(dim=2))
            positions = (
                (torch.arange(n_windows, device=new_pooled.device) * self.compress_ratio + first_pool_position)
                .unsqueeze(0)
                .expand(batch, -1)
            )
            cos, sin = self.rotary(new_pooled, position_ids=positions)
            pool_rope, pool_nope = new_pooled[..., : self.rope_head_dim], new_pooled[..., self.rope_head_dim :]
            pool_rope, _ = apply_rotary_pos_emb(
                pool_rope.unsqueeze(1), torch.zeros_like(pool_rope.unsqueeze(1)), cos, sin
            )
            new_pooled = torch.cat([pool_rope.squeeze(1), pool_nope], dim=-1)
        else:
            new_pooled = chunk_kv  # empty
        pooled = cache_layer.update_compressor_pool(new_pooled).unsqueeze(1)

        # --- Indexer narrows the pool to top-k positions per query ---
        if self.indexer is not None:
            topk = self.indexer(hidden_states, q_residual, position_ids, cache_layer)
            expanded = pooled.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
            idx = topk.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, -1, self.head_dim)
            pooled = torch.gather(expanded, 3, idx).reshape(batch, 1, -1, self.head_dim)
        return pooled


# -----------------------------------------------------------------------------
# Attention with sink.
# -----------------------------------------------------------------------------


def eager_attention_with_sink(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : attn_weights.shape[-1]]
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined = torch.cat([attn_weights, sinks.to(attn_weights.dtype)], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1]
    probs = F.dropout(probs, p=dropout, training=module.training).to(value_states.dtype)
    return torch.matmul(probs, value_states).transpose(1, 2).contiguous(), probs


class DeepseekV4Attention(DeepseekV3Attention):
    """SWA + (optional) compressor-pool segment + per-head learnable attention sink.
    Single-head KV (``num_key_value_heads=1``), grouped low-rank output. Heads are laid
    out as ``[rope_head_dim, nope_head_dim]`` (rope first), so the standard partial-rope
    pattern applies cleanly: slice ``[..., :rope_head_dim]``, rotate, concat back.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.wo_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))

        self.compressor = (
            DeepseekV4Compressor(config, self.compress_ratio, self.head_dim) if self.compress_ratio else None
        )
        # Pre-build the cache-layer class for this layer so the forward can either pull
        # the matching layer off ``past_key_values`` (the standard path) or build a
        # forward-scoped scratch layer (gradient checkpointing strips ``past_key_values``).
        self._cache_layer_cls = (
            DeepseekV4CompressorIndexerLayer
            if self.compress_ratio == 4
            else DeepseekV4CompressorLayer
            if self.compress_ratio == 128
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len = hidden_states.shape[:2]
        cos, sin = position_embeddings

        # --- Q + KV projections ---
        q_residual = self.q_norm(self.wq_a(hidden_states))
        q = self.wq_b(q_residual).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_norm(self.wkv(hidden_states)).view(batch, seq_len, 1, self.head_dim).transpose(1, 2)

        # --- Standard partial RoPE: rope is the FIRST ``rope_head_dim`` of each head ---
        q_rope, q_nope = q[..., : self.rope_head_dim], q[..., self.rope_head_dim :]
        kv_rope, kv_nope = kv[..., : self.rope_head_dim], kv[..., self.rope_head_dim :]
        q_rope, kv_rope = apply_rotary_pos_emb(q_rope, kv_rope, cos, sin)
        q = torch.cat([q_rope, q_nope], dim=-1)
        kv = torch.cat([kv_rope, kv_nope], dim=-1)

        # --- Window K/V (single tensor) goes through the standard cache update ---
        if past_key_values is not None:
            kv, _ = past_key_values.update(kv, kv, self.layer_idx)
        full_kv = kv

        # --- Optional compressor-pool segment ---
        if self.compressor is not None:
            cache_layer = None
            if past_key_values is not None:
                cache_layer = past_key_values.layers[self.layer_idx]
                # Generation builds a plain ``DynamicCache`` whose layers don't carry V4
                # compressor state; promote in-place so the state persists across decode
                # steps. K/V already accumulated on the prior layer is carried over.
                if not isinstance(cache_layer, self._cache_layer_cls):
                    new_layer = self._cache_layer_cls(self.sliding_window, self.compress_ratio)
                    if getattr(cache_layer, "is_initialized", False):
                        new_layer.lazy_initialization(cache_layer.keys, cache_layer.values)
                        new_layer.cumulative_length = cache_layer.cumulative_length
                    past_key_values.layers[self.layer_idx] = new_layer
                    cache_layer = new_layer
            else:
                # Gradient-checkpointing recompute: forward-scoped scratch layer.
                cache_layer = self._cache_layer_cls(self.sliding_window, self.compress_ratio)
            pooled = self.compressor(hidden_states, q_residual, position_ids, cache_layer)
            full_kv = torch.cat([full_kv, pooled], dim=2)

        if attention_mask is not None and full_kv.shape[2] > attention_mask.shape[-1]:
            attention_mask = F.pad(attention_mask, (0, full_kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_with_sink
        )
        attn_output, attn_weights = attention_interface(
            self,
            q,
            full_kv,
            full_kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        # De-rotate the output's rope slice. V4 shares K and V (``wkv`` projects to a
        # single tensor), so V's rope slice carries the same per-token rotation as K.
        # Attention sums V-rotated values across attended positions, so the output's
        # rope slice is a position-mixed content; conjugate rotation at the query
        # position pulls it back into a position-independent frame before the output
        # projection mixes heads.
        out_rope, out_nope = attn_output[..., : self.rope_head_dim], attn_output[..., self.rope_head_dim :]
        out_rope = out_rope.transpose(1, 2)
        out_rope, _ = apply_rotary_pos_emb(out_rope, torch.zeros_like(out_rope), cos, -sin)
        attn_output = torch.cat([out_rope.transpose(1, 2), out_nope], dim=-1)

        grouped = attn_output.reshape(batch, seq_len, -1).view(batch, seq_len, self.config.o_groups, -1)
        return self.wo_b(self.wo_a(grouped).flatten(2)), attn_weights


# -----------------------------------------------------------------------------
# Hyper-Connection.
# -----------------------------------------------------------------------------


class DeepseekV4HyperConnection(nn.Module):
    r"""Per-site Hyper-Connection mixer. Owns the learned (``fn``, ``base``, ``scale``)
    parameters that turn the incoming ``hc_mult`` residual streams into collapse / expand
    weights. The decoder layer instantiates two of these (one for the attention site,
    one for the mlp site).

    ASCII shape guide — ``B`` = batch, ``S`` = seq, ``H`` = hc_mult, ``D`` = hidden_size::

              hidden_streams        flatten(2)        RMSNorm-rescale + F.linear(fn)
         [B, S, H, D]  ──────────►  [B, S, H*D]  ─────────────────────────────────►
                                                             mix-logits
                                                             [B, S, (2+H)*H]
                                                                    │
                            ┌───────────────────────────────────────┴──────────────────────────────┐
                            ▼                          ▼                                           ▼
                        pre logits                post logits                               comb logits
                        [B, S, H]                 [B, S, H]                                 [B, S, H, H]
                        × scale[0]                × scale[1]                                × scale[2]
                        + base[:H]                + base[H:2H]                              + base[2H:]
                        σ() + eps                 σ() + eps                                 σ() + eps
                        │                         │                                         │
                        pre                        post                                     Sinkhorn(iters)
                        (stream collapse weights)  (block-output placement)                 row/col normalise
                                                                                            │
                                                                                            comb
                                                                                            (stream mixer)
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_streams.flatten(start_dim=2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mix = F.linear(flat, self.fn.float()) * rsqrt  # [B, S, (2+H)*H]
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        hc = self.hc_mult
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc]) + self.hc_eps
        post = torch.sigmoid(mix[..., hc : 2 * hc] * post_scale + self.base[hc : 2 * hc]) + self.hc_eps
        comb = (
            torch.sigmoid(
                mix[..., 2 * hc :].view(*mix.shape[:-1], hc, hc) * comb_scale + self.base[2 * hc :].view(hc, hc)
            )
            + self.hc_eps
        )
        for _ in range(self.hc_sinkhorn_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        return pre, post, comb


class DeepseekV4HyperHead(nn.Module):
    """Final HC-stream collapse; used by ``DeepseekV4Model`` before the shared RMSNorm."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(flat, self.hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


# -----------------------------------------------------------------------------
# MoE: shared MLP + routed experts + two router flavours.
# -----------------------------------------------------------------------------


class DeepseekV4MLP(Qwen2MoeMLP):
    """Shared expert — plain SwiGLU MLP, ``moe_intermediate_size`` hidden."""

    def __init__(self, config: DeepseekV4Config, intermediate_size: int | None = None):
        super().__init__(config, intermediate_size or config.moe_intermediate_size)


@use_experts_implementation
class DeepseekV4Experts(GptOssExperts):
    """Routed experts: per-expert iteration + ``_apply_gate`` hook from GPT-OSS, but
    using the Mixtral weight layout (no biases, ``[num_experts, 2*intermediate, hidden]``
    for ``gate_up_proj`` and ``[num_experts, hidden, intermediate]`` for ``down_proj``).
    Activation is SiLU and gate/up are clamped to ``swiglu_limit`` before mixing.
    """

    def __init__(self, config: DeepseekV4Config):
        nn.Module.__init__(self)
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
        self.limit = config.swiglu_limit
        self.act_fn = ACT2FN[config.hidden_act]

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gate_up = F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
            current = self._apply_gate(gate_up)
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


class DeepseekV4TopKRouter(MixtralTopKRouter):
    """Classic Mixtral-style top-k routing with two V4 tweaks: ``scoring_func``
    (``sqrtsoftplus`` for V4 checkpoints) replaces softmax, and the top-k selection
    is biased by a per-expert learnable correction (same ``noaux_tc`` idea as
    DeepSeek V3, without the expert groups).
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        # The correction bias biases the argmax only — never gradient-carrying — so it's
        # a buffer (same convention as DeepseekV3's ``e_score_correction_bias``).
        self.register_buffer("bias", torch.zeros(self.num_experts), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.bias, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(MixtralTopKRouter):
    """First ``num_hash_layers`` layers route via a frozen ``tid2eid`` lookup keyed by
    the input token id. The learned gate ``weight`` still produces scoring values used
    to weight each selected expert's activation; the selection is static.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer(
            "tid2eid",
            torch.zeros(config.vocab_size, self.top_k, dtype=torch.long),
            persistent=True,
        )

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.is_hash = layer_idx < config.num_hash_layers
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None, **_) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            if input_ids is None:
                raise ValueError(
                    "DeepseekV4's hash-routing layers need `input_ids` to look up expert indices. "
                    "The `inputs_embeds`-only inference path is not supported for models with "
                    "`num_hash_layers > 0`."
                )
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


# -----------------------------------------------------------------------------
# Decoder layer.
# -----------------------------------------------------------------------------


class DeepseekV4DecoderLayer(GradientCheckpointingLayer):
    r"""Hyper-Connection (https://huggingface.co/papers/2409.19606) decoder layer.

    Classic residual decoder layer::

        h ──► norm ──► self_attn ──► + ──► norm ──► mlp ──► +
        └──────── residual ────────┘   └─────── residual ───┘

    V4 decoder layer (``H = hc_mult`` parallel residual streams throughout)::

                attention site                                    mlp site
        ┌────────────────────────────────────────┐    ┌────────────────────────────────────────┐
        │  hidden_streams [B, S, H, D]           │    │  hidden_streams [B, S, H, D]           │
        │        │                               │    │        │                               │
        │  attn_hc(streams) ─► (pre, post, comb) │    │  ffn_hc(streams) ─► (pre, post, comb)  │
        │        │                               │    │        │                               │
        │   Σ pre·streams  (collapse)            │    │   Σ pre·streams  (collapse)            │
        │        │                               │    │        │                               │
        │   input_layernorm                      │    │   post_attention_layernorm             │
        │        │                               │    │        │                               │
        │   self_attn                            │    │   mlp  (MoE routed + shared)           │
        │        │                               │    │        │                               │
        │   post·output + comb·streams  (expand) │    │   post·output + comb·streams  (expand) │
        │        │                               │    │        │                               │
        │        ▼                               │    │        ▼                               │
        │  new hidden_streams  ──────────────────┘    │  new hidden_streams                    │
        └────────────────────────────────────────┘    └────────────────────────────────────────┘

    Checkpoint keys (``hc_attn_*`` / ``hc_ffn_*`` from the upstream reference) are bridged
    to the ``attn_hc.*`` / ``ffn_hc.*`` module tree via ``conversion_mapping.py``.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> torch.Tensor:
        # hidden_states throughout: [B, S, hc_mult, hidden].

        # --- Attention site ---
        pre, post, comb = self.attn_hc(hidden_states)
        collapsed = (pre.unsqueeze(-1) * hidden_states).sum(dim=2).to(hidden_states.dtype)
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
        dtype = hidden_states.dtype
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype), hidden_states
        )

        # --- MLP site ---
        pre, post, comb = self.ffn_hc(hidden_states)
        collapsed = (pre.unsqueeze(-1) * hidden_states).sum(dim=2).to(hidden_states.dtype)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=kwargs.get("input_ids"))
        dtype = hidden_states.dtype
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(comb.to(dtype), hidden_states)


# -----------------------------------------------------------------------------
# Pre-trained base + Model + ForCausalLM.
# -----------------------------------------------------------------------------


class DeepseekV4PreTrainedModel(MixtralPreTrainedModel):
    config_class = DeepseekV4Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    _supports_flash_attn = False
    _supports_sdpa = False
    _keep_in_fp32_modules_strict = ["attn_hc", "ffn_hc"]
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp\..*"]
    _can_record_outputs = {
        "router_logits": OutputRecorder(DeepseekV4TopKRouter, index=0),
        "hidden_states": DeepseekV4DecoderLayer,
        "attentions": DeepseekV4Attention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
            init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, DeepseekV4TopKRouter):
                module.bias.zero_()  # buffer
            if isinstance(module, DeepseekV4HashRouter):
                module.tid2eid.zero_()  # buffer; real values come from the checkpoint
        elif isinstance(module, DeepseekV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Attention):
            init.zeros_(module.sinks)
        elif isinstance(module, DeepseekV4HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, DeepseekV4HyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, (DeepseekV4Compressor, DeepseekV4Indexer)):
            init.zeros_(module.window_pos_bias)


@auto_docstring
class DeepseekV4Model(DeepseekV4PreTrainedModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = DeepseekV4HyperHead(config)
        # Only the main-attention rotary lives on the model. Compressor / Indexer own
        # their own ``compress_rope_theta`` rotary instances.
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None:
            past_key_values = DeepseekV4Cache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)
        causal_mask = create_sliding_window_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        cos_sin = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=cos_sin,
                position_ids=position_ids,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class DeepseekV4ForCausalLM(MixtralForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DeepseekV4Config):
        PreTrainedModel.__init__(self, config)
        self.model = DeepseekV4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()


__all__ = [
    "DeepseekV4Config",
    "DeepseekV4PreTrainedModel",
    "DeepseekV4Model",
    "DeepseekV4ForCausalLM",
    "DeepseekV4Cache",
]
