"""CascadiaFormer model definition for expert-root v3 training.

The module is importable without Torch so schema and CLI validation can run on
CPU-only hosts. Building or training the model imports Torch lazily and fails
closed for real training when Torch is unavailable.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from .torch_public_token_merit import PUBLIC_TOKEN_FEATURE_DIM
from .torch_semantic_relation_bias_merit import SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM

SCORE_CATEGORIES = ("wildlife", "habitat", "nature_tokens")


@dataclass(frozen=True)
class CascadiaFormerConfig:
    model_size: str = "S"
    token_feature_dim: int = PUBLIC_TOKEN_FEATURE_DIM
    action_feature_dim: int = SEMANTIC_PUBLIC_TOKEN_ACTION_FEATURE_DIM
    d_model: int = 384
    layers: int = 8
    heads: int = 8
    ffn_dim: int = 1536
    dropout: float = 0.0
    seats: int = 4
    score_categories: tuple[str, ...] = SCORE_CATEGORIES
    relation_vocab_size: int = 32
    opponent_aux_dim: int = 16
    market_aux_dim: int = 16
    gradient_checkpointing: bool = False
    model_name: str = "CascadiaFormer-S-v1"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["score_categories"] = list(self.score_categories)
        return out


def config_for_size(model_size: str) -> CascadiaFormerConfig:
    normalized = model_size.upper()
    if normalized == "TINY":
        return CascadiaFormerConfig(
            model_size="tiny",
            d_model=64,
            layers=1,
            heads=4,
            ffn_dim=128,
            model_name="CascadiaFormer-tiny-v1",
        )
    if normalized == "S":
        return CascadiaFormerConfig(model_size="S", model_name="CascadiaFormer-S-v1")
    if normalized == "M":
        return CascadiaFormerConfig(
            model_size="M",
            d_model=768,
            layers=12,
            heads=12,
            ffn_dim=3072,
            gradient_checkpointing=True,
            model_name="CascadiaFormer-M-v1",
        )
    raise ValueError("model_size must be one of tiny, S, M")


def cgab_fused_default() -> bool:
    """CASCADIA_CGAB_FUSED=1 selects the fused (count-matmul) CGAB relation
    tail at model construction. Default OFF preserves the materialized path
    bit for bit."""
    return os.environ.get("CASCADIA_CGAB_FUSED") == "1"


def _masked_mean(tensor, mask):  # type: ignore[no-untyped-def]
    mask_f = mask.to(tensor.dtype).unsqueeze(-1)
    return (tensor * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def build_cascadiaformer(config: CascadiaFormerConfig | None = None):
    import torch
    from torch import nn

    cfg = config or CascadiaFormerConfig()
    if cfg.d_model % cfg.heads != 0:
        raise ValueError(f"d_model {cfg.d_model} must be divisible by heads {cfg.heads}")

    class CascadiaGatedActionBias(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relation_embed = nn.Embedding(cfg.relation_vocab_size, cfg.d_model, padding_idx=0)
            self.gate = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.Sigmoid(),
            )
            # CASCADIA_CGAB_FUSED=1 (or model.set_cgab_fused(True)) replaces
            # the materialized [B, A, seq, d_model] relation-tail intermediate
            # with a [B, A, vocab] count matrix + one dense matmul. Same
            # masked mean, reassociated floating-point order (not
            # bit-identical; agrees to ~1e-7 in fp32). Default OFF keeps the
            # legacy path untouched.
            self.fused = cgab_fused_default()

        def _fused_relation_context(self, rel_tail, out_dtype):  # type: ignore[no-untyped-def]
            # Masked mean over embedding rows == (counts / valid_count) @ table.
            # counts[b, a, v] = number of tail positions carrying relation id v;
            # column 0 is zeroed to honor the padding contract (padding_idx=0 +
            # ne(0) mask): id-0 positions contribute nothing to numerator or
            # denominator. No [B, A, seq, d_model] tensor is ever built.
            weight = self.relation_embed.weight
            counts = torch.zeros(
                (*rel_tail.shape[:2], weight.shape[0]),
                dtype=weight.dtype,
                device=rel_tail.device,
            )
            counts.scatter_add_(2, rel_tail, torch.ones_like(rel_tail, dtype=weight.dtype))
            counts[..., 0] = 0
            denom = counts.sum(dim=2, keepdim=True).clamp_min(1)
            rel_context = (counts / denom) @ weight
            return rel_context.to(out_dtype)

        def forward(self, action_h, relation_ids=None, relation_tail=None):  # type: ignore[no-untyped-def]
            if relation_tail is None and relation_ids is None:
                return action_h, action_h.new_zeros(action_h.shape)
            action_count = action_h.shape[1]
            if relation_tail is None:
                rel_tail = relation_ids[:, -action_count:, :]
            else:
                rel_tail = relation_tail[:, :action_count, :]
            rel_tail = rel_tail.clamp_min(0).to(dtype=torch.long)
            if self.fused:
                rel_context = self._fused_relation_context(rel_tail, action_h.dtype)
            else:
                rel_mask = rel_tail.ne(0).unsqueeze(-1)
                rel_emb = self.relation_embed(rel_tail) * rel_mask.to(action_h.dtype)
                rel_context = rel_emb.sum(dim=2) / rel_mask.sum(dim=2).clamp_min(1).to(action_h.dtype)
            bias = self.gate(rel_context) * rel_context
            return action_h + bias, bias

    class CascadiaFormer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = cfg
            self.token_proj = nn.Linear(cfg.token_feature_dim, cfg.d_model)
            self.action_proj = nn.Linear(cfg.action_feature_dim, cfg.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.heads,
                dim_feedforward=cfg.ffn_dim,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            # enable_nested_tensor=False is a zero-change setting: the
            # nested-tensor "fast path" is inference-only (it is disabled
            # whenever the module is training or grads are enabled) and is
            # additionally disqualified by norm_first=True. Passing False
            # silences the spurious torch warning without touching numerics.
            self.state_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=cfg.layers,
                enable_nested_tensor=False,
            )
            # Opt-in activation checkpointing (see set_gradient_checkpointing).
            # NOTE: cfg.gradient_checkpointing was historically never applied
            # by any trainer; the default here preserves that behavior.
            self.gradient_checkpointing_enabled = False
            self.action_cross_attn = nn.MultiheadAttention(
                cfg.d_model,
                cfg.heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.cgab = CascadiaGatedActionBias()
            self.action_norm = nn.LayerNorm(cfg.d_model)
            self.root_norm = nn.LayerNorm(cfg.d_model)
            self.legal_logits = nn.Linear(cfg.d_model, 1)
            self.q_head = nn.Linear(cfg.d_model, 1)
            self.uncertainty_head = nn.Linear(cfg.d_model, 1)
            self.value_head = nn.Linear(cfg.d_model, cfg.seats)
            self.rank_head = nn.Linear(cfg.d_model, cfg.seats * cfg.seats)
            self.differential_head = nn.Linear(cfg.d_model, cfg.seats)
            self.score_head = nn.Linear(cfg.d_model, len(cfg.score_categories) * cfg.seats)
            self.opponent_aux_head = nn.Linear(cfg.d_model, cfg.opponent_aux_dim)
            self.market_aux_head = nn.Linear(cfg.d_model, cfg.market_aux_dim)

        def set_gradient_checkpointing(self, enabled: bool) -> None:
            self.gradient_checkpointing_enabled = bool(enabled)

        def set_cgab_fused(self, enabled: bool) -> None:
            self.cgab.fused = bool(enabled)

        def _encode_state(self, token_h, token_padding):  # type: ignore[no-untyped-def]
            if (
                self.gradient_checkpointing_enabled
                and self.training
                and torch.is_grad_enabled()
            ):
                from torch.utils.checkpoint import checkpoint

                # Manual per-layer loop so each TransformerEncoderLayer can be
                # recomputed in backward. TransformerEncoder.forward only adds
                # inference-time fast-path plumbing around the same loop, so
                # this matches the non-checkpointed path numerically (layers
                # canonicalize the bool padding mask identically).
                encoded = token_h
                for layer in self.state_encoder.layers:
                    encoded = checkpoint(
                        layer,
                        encoded,
                        None,
                        token_padding,
                        use_reentrant=False,
                    )
                if self.state_encoder.norm is not None:
                    encoded = self.state_encoder.norm(encoded)
                return encoded
            return self.state_encoder(token_h, src_key_padding_mask=token_padding)

        def forward(  # type: ignore[no-untyped-def]
            self,
            tokens,
            token_mask,
            actions,
            action_mask,
            relation_ids=None,
            relation_tail=None,
        ):
            token_h = self.token_proj(tokens)
            token_padding = ~token_mask
            encoded = self._encode_state(token_h, token_padding)
            action_h = self.action_proj(actions)
            decoded, _ = self.action_cross_attn(
                query=action_h,
                key=encoded,
                value=encoded,
                key_padding_mask=token_padding,
                need_weights=False,
            )
            decoded = self.action_norm(decoded + action_h)
            decoded, cgab_bias = self.cgab(decoded, relation_ids, relation_tail)
            decoded = decoded.masked_fill(~action_mask.unsqueeze(-1), 0.0)
            root_h = self.root_norm(_masked_mean(encoded, token_mask))
            return {
                "logits": self.legal_logits(decoded).squeeze(-1),
                "q": self.q_head(decoded).squeeze(-1),
                "uncertainty": torch.nn.functional.softplus(
                    self.uncertainty_head(decoded).squeeze(-1)
                ),
                "value_vector": self.value_head(root_h),
                "rank_logits": self.rank_head(root_h).view(-1, cfg.seats, cfg.seats),
                "differential": self.differential_head(root_h),
                "score_decomposition": self.score_head(root_h).view(
                    -1,
                    len(cfg.score_categories),
                    cfg.seats,
                ),
                "opponent_aux": self.opponent_aux_head(root_h),
                "market_aux": self.market_aux_head(root_h),
                "cgab_bias": cgab_bias,
            }

    return CascadiaFormer()


def parameter_count(model) -> int:  # type: ignore[no-untyped-def]
    return sum(param.numel() for param in model.parameters())
