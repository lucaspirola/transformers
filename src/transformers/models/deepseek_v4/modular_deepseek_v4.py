# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, RopeParameters, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import maybe_autocast, merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3RMSNorm,
)
from ..gpt_oss.modeling_gpt_oss import GptOssExperts
from ..llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralPreTrainedModel, MixtralTopKRouter
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeMLP


logger = logging.get_logger(__name__)


DEEPSEEK_V4_LAYER_TYPES = (
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4-Flash-Base")
@strict
class DeepseekV4Config(DeepseekV3Config):
    r"""
    DeepSeek-V4's hybrid attention follows the paper (Section 2.3): every block is one
    of three attention types — *Full Attention* (sliding-window only), *Compressed
    Sparse Attention* (CSA, Section 2.3.1) and *Heavily Compressed Attention* (HCA,
    Section 2.3.2). CSA compresses the KV cache by ``compress_rate_csa`` (m=4 in V4-
    Flash/Pro) and selects ``index_topk`` blocks per query via the Lightning Indexer;
    HCA applies a much heavier compression of ``compress_rate_hca`` (m'=128) and
    skips sparse selection. Both branches add a small uncompressed sliding-window
    branch for fine-grained locality.

    layer_types (`list[str]`): Per-layer attention schedule with values from
        ``{"sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"}``.
        V4-Flash defaults: 2× full + interleaved CSA / HCA.
    compress_rate_csa (`int`): m, the CSA compression rate (default 4).
    compress_rate_hca (`int`): m', the HCA compression rate (default 128).
    compress_rope_theta (`float`): RoPE base for the compressed branches (paired with
        ``rope_scaling`` for YaRN).
    hc_mult (`int`): Manifold-Constrained Hyper-Connection (mHC) expansion factor n_hc
        (always active; Section 2.2).
    hc_sinkhorn_iters (`int`): Sinkhorn-Knopp iterations t_max for the mHC residual
        mapping projection onto doubly-stochastic matrices.
    hc_eps (`float`): Numerical floor for the Sinkhorn-Knopp normalization.
    num_hash_layers (`int`): First N MoE layers route via a frozen ``tid2eid[input_ids]`` lookup.
    scoring_func (`str`): Router activation — ``sqrtsoftplus``, ``softmax``, or ``sigmoid``.
    swiglu_limit (`float`): Clip routed experts' gate/up pre-activations.
    sliding_window (`int`): Local window size n_win used in every attention block's
        sliding-window branch.
    o_groups (`int`), o_lora_rank (`int`): Grouped low-rank output projection (g, d_g).
    index_n_heads, index_head_dim, index_topk (`int`): Lightning Indexer hyperparameters.
    num_nextn_predict_layers (`int`): MTP layer count in the upstream checkpoint (not instantiated here).
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

    layer_types: list[str] | None = None
    compress_rate_csa: int = 4
    compress_rate_hca: int = 128
    compress_rope_theta: float = 160000.0
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
        PreTrainedConfig.__post_init__(self, **kwargs)
        n = self.num_hidden_layers
        if self.layer_types is None:
            # V4-Flash default: two full-attention bootstrap layers, then CSA / HCA interleaved.
            interleave = [
                "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                for i in range(max(n - 2, 0))
            ]
            head = ["sliding_attention"] * min(n, 2)
            self.layer_types = head + interleave
        self.layer_types = list(self.layer_types[:n])
        if len(self.layer_types) != n:
            raise ValueError(f"`layer_types` must cover at least {n} layers, got {len(self.layer_types)}.")
        for layer_type in self.layer_types:
            if layer_type not in DEEPSEEK_V4_LAYER_TYPES:
                raise ValueError(f"Unsupported layer_type={layer_type!r}; expected one of {DEEPSEEK_V4_LAYER_TYPES}.")
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        if self.partial_rotary_factor is None:
            self.partial_rotary_factor = self.qk_rope_head_dim / self.head_dim
        # Normalize rope_parameters into a per-layer-type dict ``{"main": {...}, "compress": {...}}``
        # (Gemma3 pattern). Idempotent across save/load: round-tripping preserves structure.
        rp = self.rope_parameters or {}
        if isinstance(rp.get("main"), dict) and isinstance(rp.get("compress"), dict):
            self.rope_parameters = {"main": rp["main"], "compress": rp["compress"]}
        else:
            main = {k: v for k, v in rp.items() if k not in ("main", "compress")}
            main.setdefault("rope_type", "default")
            main.setdefault("rope_theta", self.rope_theta)
            main["partial_rotary_factor"] = self.partial_rotary_factor
            compress = {**main, "rope_theta": self.compress_rope_theta}
            self.rope_parameters = {"main": main, "compress": compress}


class DeepseekV4RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV4RotaryEmbedding(nn.Module):
    """Multi-layer-type rotary embedding (Gemma3 pattern). Holds two ``inv_freq``
    buffers — ``"main"`` for self-attention (``rope_theta``) and ``"compress"`` for
    the Compressor / Indexer (``compress_rope_theta``). Both honour
    ``partial_rotary_factor`` so cos/sin is sized to ``qk_rope_head_dim`` rather than
    the full ``head_dim``. ``forward(x, position_ids, layer_type=...)`` picks one.
    """

    inv_freq: torch.Tensor  # fix linting for `register_buffer`
    layer_types = ("main", "compress")

    def __init__(self, config: "DeepseekV4Config", device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = {}
        for layer_type in self.layer_types:
            params = config.rope_parameters.get(layer_type)
            if params is None:
                continue
            self.rope_type[layer_type] = params.get("rope_type", "default")
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            inv_freq, scaling = rope_init_fn(config, device, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", scaling)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
        params = config.rope_parameters[layer_type]
        base = params["rope_theta"]
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        factor = params.get("partial_rotary_factor", 1.0)
        dim = int(head_dim * factor)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids, layer_type="main"):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# -----------------------------------------------------------------------------
# Cache layers — one class per ``layer_types[i]``, all subclasses of the sliding-
# window K=V layer. State that the Compressor / Indexer modules need lives here,
# not on the parent ``DeepseekV4Cache``.
# -----------------------------------------------------------------------------


class DeepseekV4SWALayer(DynamicSlidingWindowLayer):
    """Cache layer for ``"sliding_attention"`` blocks: just the supplementary sliding-
    window KV branch (n_win) shared by all V4 attention types. K and V share storage
    (the ``wkv`` projection emits a single tensor used as both key and value)."""

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full


class DeepseekV4HCALayer(DeepseekV4SWALayer):
    """Cache layer for ``"heavily_compressed_attention"`` blocks (HCA, paper §2.3.2):
    the sliding-window K=V branch + a per-call window-buffer + a running compressed-
    KV pool. The buffer holds tokens that arrived after the last closed window but
    aren't yet enough to form the next one; the pool is the running list of
    compressed tokens emitted so far. Methods :meth:`update_compressor` and
    :meth:`update_compressor_pool` are the contract the :class:`DeepseekV4Compressor`
    module calls.
    """

    def __init__(self, sliding_window: int, compress_rate: int):
        super().__init__(sliding_window)
        self.compress_rate = compress_rate
        self.compressor_buffer_kv: torch.Tensor | None = None
        self.compressor_buffer_gate: torch.Tensor | None = None
        self.compressor_pool: torch.Tensor | None = None
        # Number of compressed tokens emitted so far. Each one represents
        # ``compress_rate`` source tokens, so ``compressor_pool_count * rate`` is the
        # absolute position of the *next* window's first token.
        self.compressor_pool_count = 0

    def update_compressor(self, kv: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Merge new (``kv``, ``gate``) with the buffered tail and return the
        window-aligned chunk that's ready to pool, plus the absolute position of the
        first window in that chunk. The leftover tail stays in the buffer.
        """
        first_pool_position = self.compressor_pool_count * self.compress_rate
        if self.compressor_buffer_kv is not None and self.compressor_buffer_kv.shape[1]:
            kv = torch.cat([self.compressor_buffer_kv, kv], dim=1)
            gate = torch.cat([self.compressor_buffer_gate, gate], dim=1)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
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


class DeepseekV4CSALayer(DeepseekV4HCALayer):
    """Cache layer for ``"compressed_sparse_attention"`` blocks (CSA, paper §2.3.1).
    Adds a parallel set of buffers / pool / counter for the Lightning Indexer's
    smaller (``index_head_dim``) compressor branch. Same buffer / pool semantics as
    HCA's main branch, but kept separate because the indexer pools at a different
    head dim.
    """

    def __init__(self, sliding_window: int, compress_rate: int):
        super().__init__(sliding_window, compress_rate)
        self.indexer_buffer_kv: torch.Tensor | None = None
        self.indexer_buffer_gate: torch.Tensor | None = None
        self.indexer_pool: torch.Tensor | None = None
        self.indexer_pool_count = 0

    def update_indexer(self, kv: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        first_pool_position = self.indexer_pool_count * self.compress_rate
        if self.indexer_buffer_kv is not None and self.indexer_buffer_kv.shape[1]:
            kv = torch.cat([self.indexer_buffer_kv, kv], dim=1)
            gate = torch.cat([self.indexer_buffer_gate, gate], dim=1)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
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


class DeepseekV4Cache(DynamicCache):
    """One cache layer per ``config.layer_types[i]``: full-attention (sliding-only),
    HCA (Compressor only) or CSA (Compressor + Indexer). State for the Compressor /
    Indexer modules lives on those layers, not on the parent cache.
    """

    def __init__(self, config: DeepseekV4Config | None = None):
        super().__init__(config=config)
        if config is None:
            return
        self.layers = []
        for layer_type in config.layer_types:
            if layer_type == "compressed_sparse_attention":
                self.layers.append(DeepseekV4CSALayer(config.sliding_window, config.compress_rate_csa))
            elif layer_type == "heavily_compressed_attention":
                self.layers.append(DeepseekV4HCALayer(config.sliding_window, config.compress_rate_hca))
            else:
                self.layers.append(DeepseekV4SWALayer(config.sliding_window))


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


class DeepseekV4Indexer(nn.Module):
    """Lightning Indexer (paper §2.3.1, eqs. 13–17). Used by Compressed Sparse
    Attention (CSA) to pick the top-k compressed KV blocks per query. The indexer
    runs its own scaled-down compressor at ``index_head_dim`` over the same windows
    as the outer CSA compressor, then scores queries against the pooled keys with
    ``∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)`` and keeps the top ``index_topk``
    indices.

    The indexer has its own rotary because it applies RoPE to two sets of tensors:

      * **pool keys** at deterministic positions ``i * compress_rate + first_pool_position``,
      * **queries** at the model's current ``position_ids`` (variable per forward).

    Both must use the same theta as the outer compressor (``compress_rope_theta``) so
    query/key inner products are translation-invariant in the standard rope sense — if
    they used different thetas the score ``q · k`` would carry a residual position-
    dependent skew. We can't precompute cos/sin once at init because the query
    positions vary per call, so the indexer owns a rotary embedding and calls it with
    ``layer_type="compress"`` twice per forward (once for pool keys, once for queries).
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rate_csa
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.window_pos_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        cache_layer: DeepseekV4CSALayer,
    ) -> torch.LongTensor:
        batch, seq_len, _ = hidden_states.shape

        # --- Pool side: same windows as the outer compressor, at index_head_dim ---
        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        chunk_kv, chunk_gate, first_pool_position = cache_layer.update_indexer(kv, gate)
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, self.head_dim)
            chunk_gate = chunk_gate.view(
                batch, n_windows, self.compress_rate, self.head_dim
            ) + self.window_pos_bias.to(chunk_gate.dtype)
            new_pooled = self.kv_norm((chunk_kv * chunk_gate.softmax(dim=2)).sum(dim=2))
            positions = (
                (torch.arange(n_windows, device=new_pooled.device) * self.compress_rate + first_pool_position)
                .unsqueeze(0)
                .expand(batch, -1)
            )
            cos, sin = self.rotary_emb(new_pooled, position_ids=positions, layer_type="compress")
            pool_rope, pool_nope = new_pooled[..., : self.rope_head_dim], new_pooled[..., self.rope_head_dim :]
            pool_rope, _ = apply_rotary_pos_emb(
                pool_rope.unsqueeze(1), torch.zeros_like(pool_rope.unsqueeze(1)), cos, sin
            )
            new_pooled = torch.cat([pool_rope.squeeze(1), pool_nope], dim=-1)
        else:
            new_pooled = chunk_kv  # empty
        pooled_kv = cache_layer.update_indexer_pool(new_pooled)

        # --- Query side ---
        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type="compress")
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
# Compressor — pools ``compress_rate`` consecutive tokens into one compressed KV
# entry via softmax over learned gate logits + window_pos_bias (paper §2.3.1 eq. 11
# for CSA, §2.3.2 eq. 22 for HCA). Used by both CSA and HCA blocks. CSA additionally
# wraps the running pool with a Lightning Indexer (instantiated in :class:`DeepseekV4CSA`).
# -----------------------------------------------------------------------------


class DeepseekV4Compressor(nn.Module):
    """Token-level KV compressor used by CSA (paper §2.3.1) and HCA (§2.3.2). Pools
    every ``compress_rate`` source tokens into one compressed KV entry, normalised
    across the window with a softmax over learned gate logits + ``window_pos_bias``.
    For CSA, ``self.indexer`` also runs and the returned pool is sparse-gathered to
    the top-``index_topk`` blocks per query token; for HCA the full running pool is
    returned. The result is concatenated onto the attention's sliding-window KV.
    """

    def __init__(self, config: DeepseekV4Config, compress_rate: int, with_indexer: bool):
        super().__init__()
        self.compress_rate = compress_rate
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.window_pos_bias = nn.Parameter(torch.empty(compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.indexer: DeepseekV4Indexer | None = DeepseekV4Indexer(config) if with_indexer else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor | None,
        position_ids: torch.Tensor,
        cache_layer: DeepseekV4HCALayer,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape

        kv = self.wkv(hidden_states)
        gate = self.wgate(hidden_states)
        chunk_kv, chunk_gate, first_pool_position = cache_layer.update_compressor(kv, gate)
        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, self.head_dim)
            chunk_gate = chunk_gate.view(
                batch, n_windows, self.compress_rate, self.head_dim
            ) + self.window_pos_bias.to(chunk_gate.dtype)
            new_pooled = self.kv_norm((chunk_kv * chunk_gate.softmax(dim=2)).sum(dim=2))
            positions = (
                (torch.arange(n_windows, device=new_pooled.device) * self.compress_rate + first_pool_position)
                .unsqueeze(0)
                .expand(batch, -1)
            )
            cos, sin = self.rotary_emb(new_pooled, position_ids=positions, layer_type="compress")
            pool_rope, pool_nope = new_pooled[..., : self.rope_head_dim], new_pooled[..., self.rope_head_dim :]
            pool_rope, _ = apply_rotary_pos_emb(
                pool_rope.unsqueeze(1), torch.zeros_like(pool_rope.unsqueeze(1)), cos, sin
            )
            new_pooled = torch.cat([pool_rope.squeeze(1), pool_nope], dim=-1)
        else:
            new_pooled = chunk_kv  # empty
        pooled = cache_layer.update_compressor_pool(new_pooled).unsqueeze(1)

        # CSA-only: the Lightning Indexer narrows the running pool to top-k entries per query.
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
    """Attention sink (paper §2.3.3, eq. 27). Per-head learnable sink logit ``z'_h`` is
    appended to each query's attention scores before softmax, then dropped from the
    softmax outputs. The sink lets each head shift its total attention mass below 1
    (effectively a learned no-op outlet), which the paper attributes to better
    long-context numerical behaviour.
    """
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
    """Shared core for every V4 attention block (paper §2.3). Each block consists of
    (paper §2.3.3):

      * Shared-KV Multi-Query Attention with a single KV head broadcast to all query
        heads (``num_key_value_heads = 1``); ``wkv`` projects directly to that head
        and the same tensor is read as both key and value.
      * Partial RoPE on the first ``rope_head_dim`` of each head (paper §2.3.3,
        "Partial Rotary Positional Embedding"). RoPE is also applied with position
        ``-i`` to the attention output's rope slice so the contribution of each KV
        entry stays a function of the *relative* distance to the query.
      * RMSNorm on the queries (``q_norm``) and the compressed KV head (``kv_norm``)
        right before the core attention, to avoid exploding logits.
      * Per-head learnable attention sink (paper §2.3.3, eq. 27).
      * Grouped low-rank output projection (paper §2.3.1, "Grouped Output
        Projection"): ``g`` head-groups are projected to ``d_g``-dim intermediate
        outputs through a block-diagonal :class:`DeepseekV4GroupedLinear` then mixed
        back to ``hidden_size`` by ``wo_b``.
      * A supplementary uncompressed sliding-window KV branch of size
        ``sliding_window`` (paper §2.3.3, "Additional Branch of Sliding Window
        Attention") that every block uses regardless of layer type, to preserve
        local fine-grained dependencies.

    Used directly as the ``"sliding_attention"`` layer type (the first two layers of
    V4-Flash; pure SWA, no compressor). :class:`DeepseekV4HCA` and
    :class:`DeepseekV4CSA` extend this with a long-range compressor branch.
    """

    _cache_layer_cls = DeepseekV4SWALayer
    _compress_rate: int = 0

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
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
        self.compressor: DeepseekV4Compressor | None = None

    def _compressor_pool(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
    ) -> torch.Tensor | None:
        """Return the compressed-KV segment to concatenate onto the sliding-window KV,
        or ``None`` for full-attention layers. CSA / HCA override the cache layer
        promotion below by setting ``_cache_layer_cls`` accordingly.
        """
        if self.compressor is None:
            return None
        if past_key_values is not None:
            cache_layer = past_key_values.layers[self.layer_idx]
            # Generation builds a plain ``DynamicCache`` whose layers don't carry V4
            # compressor state; promote in-place so the state persists across decode
            # steps. K/V already accumulated on the prior layer is carried over.
            if not isinstance(cache_layer, self._cache_layer_cls):
                new_layer = self._cache_layer_cls(self.sliding_window, self._compress_rate)
                if getattr(cache_layer, "is_initialized", False):
                    new_layer.lazy_initialization(cache_layer.keys, cache_layer.values)
                    new_layer.cumulative_length = getattr(cache_layer, "cumulative_length", cache_layer.keys.shape[-2])
                past_key_values.layers[self.layer_idx] = new_layer
                cache_layer = new_layer
        else:
            # Gradient-checkpointing recompute: forward-scoped scratch layer.
            cache_layer = self._cache_layer_cls(self.sliding_window, self._compress_rate)
        return self.compressor(hidden_states, q_residual, position_ids, cache_layer)

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

        # --- Optional compressor-pool segment (CSA / HCA only) ---
        pooled = self._compressor_pool(hidden_states, q_residual, position_ids, past_key_values)
        if pooled is not None:
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


class DeepseekV4HCA(DeepseekV4Attention):
    """**Heavily Compressed Attention** (HCA, paper §2.3.2). Each query attends to
    the sliding-window KV branch *plus* a long-range compressed branch:

      * The compressor consolidates every ``compress_rate_hca`` (m'=128 in
        V4-Flash/Pro) consecutive KV tokens into one entry by softmax-pooling over
        learned gate logits + ``window_pos_bias`` (paper §2.3.2, eqs. 22–23).
        Compression is *disjoint* — no overlap between adjacent windows — and reduces
        the sequence length by 1/m'.
      * The whole running compressed pool is appended to the keys / values; HCA does
        no sparse selection.

    Sharing the QKV projections, attention sink, partial RoPE and grouped output
    projection with :class:`DeepseekV4Attention`.
    """

    _cache_layer_cls = DeepseekV4HCALayer

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self._compress_rate = config.compress_rate_hca
        self.compressor = DeepseekV4Compressor(config, self._compress_rate, with_indexer=False)


class DeepseekV4CSA(DeepseekV4HCA):
    """**Compressed Sparse Attention** (CSA, paper §2.3.1). Combines compression
    with DeepSeek Sparse Attention (DSA, DeepSeek-AI, 2025):

      * The compressor runs at ``compress_rate_csa`` (m=4 in V4-Flash/Pro) and pools
        each window of m KV tokens into one compressed entry (paper §2.3.1, eqs.
        9–12). Adjacent compressed entries draw from overlapping window pairs, so
        each compressed entry is derived from 2m source tokens while the sequence
        length is still reduced by 1/m.
      * The Lightning Indexer (``compressor.indexer``, paper §2.3.1, eqs. 13–17) is
        a cheap small-head scorer that runs the same compression at
        ``index_head_dim``, then for each query computes
        ``∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)`` and keeps the top
        ``index_topk`` compressed entries — these are the only long-range KV
        positions that enter the core attention.
      * The latent query vector ``c_t^Q`` (output of ``q_norm(wq_a(h_t))``) is shared
        between the indexer's queries (low-rank ``wq_b`` + ``weights_proj``) and the
        main attention queries — saving redundant projections.

    Inherits the rest of the block (sink, partial RoPE, grouped output, sliding
    window branch) from :class:`DeepseekV4Attention`.
    """

    _cache_layer_cls = DeepseekV4CSALayer

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        DeepseekV4Attention.__init__(self, config, layer_idx)
        self._compress_rate = config.compress_rate_csa
        self.compressor = DeepseekV4Compressor(config, self._compress_rate, with_indexer=True)


DEEPSEEK_V4_ATTENTION_CLASSES = {
    "sliding_attention": DeepseekV4Attention,
    "compressed_sparse_attention": DeepseekV4CSA,
    "heavily_compressed_attention": DeepseekV4HCA,
}


class DeepseekV4HyperConnection(nn.Module):
    r"""
    Manifold-Constrained Hyper-Connections
    (mHC) (Xie et al., 2026) to strengthen the conventional residual connections between adjacent
    Transformer blocks

    Owns the learned (``fn``, ``base``, ``scale``)
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
    """DeepSeekMoE top-k router (paper §2.1, "Mixture-of-Experts"). Two changes from
    the V3 router:

      * The expert affinity activation is ``Sqrt(Softplus(·))`` instead of the V3
        Sigmoid (paper §2.1: "we change the activation function that computes the
        affinity scores from Sigmoid(·) into Sqrt(Softplus(·))"). The ``scoring_func``
        config field selects this for V4 checkpoints.
      * The constraint on the number of routing target nodes used in V3 is dropped,
        and the V3 ``n_group`` / ``topk_group`` machinery is removed entirely (paper
        §2.1: "we remove the constraint on the number of routing target nodes").

    The auxiliary-loss-free strategy is preserved via the per-expert ``bias`` buffer
    that biases the top-k argmax without flowing gradients (same ``noaux_tc`` idea
    as DeepSeek-V3).
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
    """Hash routing for the first ``num_hash_layers`` MoE layers (paper §2.1, "Mixture-
    of-Experts"). The first three blocks of V4 replace the dense FFN of V3 with an MoE
    where the expert selection is determined by a fixed hash of the input token id —
    a frozen ``tid2eid[input_ids]`` lookup — instead of a learned gate. The learned
    gate ``weight`` still produces the per-expert scoring values used to weight the
    selected experts' activations; only the *which-experts* selection is static.
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
    r"""DeepSeek-V4 decoder block (paper §2). Differs from a classic residual block in
    two places:

      * The residual is a stack of ``hc_mult`` parallel streams kept in shape
        ``[B, S, hc_mult, D]`` throughout the block, mixed in and out via two
        :class:`DeepseekV4HyperConnection` modules (Manifold-Constrained Hyper-
        Connections / mHC, paper §2.2; Xie et al., 2026). The mHC mappings constrain
        the residual transform to the manifold of doubly-stochastic matrices via the
        Sinkhorn-Knopp projection — making signal propagation non-expansive across
        deep stacks.
      * ``self_attn`` is one of three classes picked at construction time by
        ``config.layer_types[layer_idx]`` — :class:`DeepseekV4Attention` (sliding
        window), :class:`DeepseekV4HCA` (Heavily Compressed Attention) or
        :class:`DeepseekV4CSA` (Compressed Sparse Attention). All three share the
        ``self_attn.*`` parameter tree.

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
        self.self_attn = DEEPSEEK_V4_ATTENTION_CLASSES[config.layer_types[layer_idx]](config, layer_idx)
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
        elif isinstance(module, DeepseekV4RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


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
        # ``generate()`` may pass a per-layer-type mask dict already built by
        # ``create_masks_for_generate``; all V4 layer types use the same sliding-window
        # mask, so use the prebuilt one directly. Otherwise build it here.
        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        cos_sin = self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main")

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
