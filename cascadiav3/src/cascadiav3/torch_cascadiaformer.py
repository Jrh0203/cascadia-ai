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
    # >1 turns the scalar score-to-go head into a quantile head (pinball
    # loss). The ordinary "q" output remains the quantile mean for backward
    # compatibility; the bridge can explicitly select a provenance-recorded
    # quantile-risk mode for research ablations. 1 keeps the legacy scalar
    # head bit-for-bit.
    q_quantiles: int = 1
    # Optional pairwise action comparator. The low-rank skew interaction is
    # antisymmetric by construction: compare(a, b) == -compare(b, a).
    # Existing checkpoints keep this disabled and therefore retain their
    # exact parameter/state contract.
    pairwise_comparator: bool = False
    pairwise_rank: int = 64
    pairwise_max_pairs_per_root: int = 32
    pairwise_min_margin: float = 0.25
    pairwise_min_snr: float = 1.0

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
    if normalized == "XS":
        return CascadiaFormerConfig(
            model_size="XS",
            d_model=256,
            layers=6,
            heads=8,
            ffn_dim=1024,
            model_name="CascadiaFormer-XS-v1",
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
    if normalized == "L":
        return CascadiaFormerConfig(
            model_size="L",
            d_model=1024,
            layers=16,
            heads=16,
            ffn_dim=4096,
            gradient_checkpointing=True,
            model_name="CascadiaFormer-L-v1",
        )
    raise ValueError("model_size must be one of tiny, XS, S, M, L")


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
            self.q_head = nn.Linear(cfg.d_model, max(1, cfg.q_quantiles))
            self.uncertainty_head = nn.Linear(cfg.d_model, 1)
            self.value_head = nn.Linear(cfg.d_model, cfg.seats)
            self.rank_head = nn.Linear(cfg.d_model, cfg.seats * cfg.seats)
            self.differential_head = nn.Linear(cfg.d_model, cfg.seats)
            self.score_head = nn.Linear(cfg.d_model, len(cfg.score_categories) * cfg.seats)
            self.opponent_aux_head = nn.Linear(cfg.d_model, cfg.opponent_aux_dim)
            self.market_aux_head = nn.Linear(cfg.d_model, cfg.market_aux_dim)
            if cfg.pairwise_comparator:
                if cfg.pairwise_rank <= 0:
                    raise ValueError("pairwise_rank must be positive")
                self.pairwise_merit = nn.Linear(cfg.d_model, 1, bias=False)
                self.pairwise_left = nn.Linear(cfg.d_model, cfg.pairwise_rank, bias=False)
                self.pairwise_right = nn.Linear(cfg.d_model, cfg.pairwise_rank, bias=False)
            else:
                self.pairwise_merit = None
                self.pairwise_left = None
                self.pairwise_right = None

        def set_gradient_checkpointing(self, enabled: bool) -> None:
            self.gradient_checkpointing_enabled = bool(enabled)

        def set_cgab_fused(self, enabled: bool) -> None:
            self.cgab.fused = bool(enabled)

        def compare_action_embeddings(self, left, right):  # type: ignore[no-untyped-def]
            if (
                self.pairwise_merit is None
                or self.pairwise_left is None
                or self.pairwise_right is None
            ):
                raise RuntimeError("pairwise comparator is disabled in this checkpoint")
            merit = self.pairwise_merit(left).squeeze(-1) - self.pairwise_merit(right).squeeze(-1)
            left_l = self.pairwise_left(left)
            left_r = self.pairwise_right(left)
            right_l = self.pairwise_left(right)
            right_r = self.pairwise_right(right)
            skew = (left_l * right_r - right_l * left_r).sum(dim=-1)
            return merit + skew / (cfg.pairwise_rank**0.5)

        def policy_logits_chunked(
            self,
            tokens,
            token_mask,
            actions,
            action_mask,
            *,
            relation_tail=None,
            action_chunk_size=256,
        ):
            """Score an exact full action menu without padding it into one CGAB batch.

            State encoding is shared across chunks. Cross-attention and CGAB
            are action-row independent, so slicing action rows and their
            relation tails preserves the model function while bounding peak
            memory for menus with thousands of draft/placement combinations.
            """
            if action_chunk_size <= 0:
                raise ValueError("action_chunk_size must be positive")
            if actions.shape[:2] != action_mask.shape:
                raise ValueError("actions/action_mask shape mismatch")
            if relation_tail is not None and relation_tail.shape[:2] != action_mask.shape:
                raise ValueError("relation_tail action shape mismatch")
            token_h = self.token_proj(tokens)
            token_padding = ~token_mask
            encoded = self._encode_state(token_h, token_padding)
            chunks = []
            for start in range(0, int(actions.shape[1]), int(action_chunk_size)):
                end = min(start + int(action_chunk_size), int(actions.shape[1]))
                action_h = self.action_proj(actions[:, start:end])
                decoded, _ = self.action_cross_attn(
                    query=action_h,
                    key=encoded,
                    value=encoded,
                    key_padding_mask=token_padding,
                    need_weights=False,
                )
                decoded = self.action_norm(decoded + action_h)
                chunk_tail = relation_tail[:, start:end, :] if relation_tail is not None else None
                decoded, _ = self.cgab(decoded, relation_tail=chunk_tail)
                decoded = decoded.masked_fill(~action_mask[:, start:end].unsqueeze(-1), 0.0)
                chunks.append(self.legal_logits(decoded).squeeze(-1))
            if not chunks:
                return actions.new_empty((actions.shape[0], 0))
            return torch.cat(chunks, dim=1)

        def pairwise_borda_logits(self, action_h, action_mask):  # type: ignore[no-untyped-def]
            """Mean pair log-odds against every other legal action.

            This turns the non-transitive comparator matrix into one
            permutation-equivariant root policy score without changing the Q
            head used for leaf values. Invalid/padded opponents contribute
            nothing and are excluded from the denominator.
            """
            if (
                self.pairwise_merit is None
                or self.pairwise_left is None
                or self.pairwise_right is None
            ):
                raise RuntimeError("pairwise comparator is disabled in this checkpoint")
            merit = self.pairwise_merit(action_h).squeeze(-1)
            left_projection = self.pairwise_left(action_h)
            right_projection = self.pairwise_right(action_h)
            pair_logits = merit.unsqueeze(2) - merit.unsqueeze(1)
            pair_logits = pair_logits + (
                left_projection @ right_projection.transpose(1, 2)
                - right_projection @ left_projection.transpose(1, 2)
            ) / (cfg.pairwise_rank**0.5)
            pair_mask = action_mask.unsqueeze(2) & action_mask.unsqueeze(1)
            diagonal = torch.eye(
                action_h.shape[1],
                dtype=torch.bool,
                device=action_h.device,
            ).unsqueeze(0)
            pair_mask &= ~diagonal
            denominator = pair_mask.sum(dim=2).clamp_min(1).to(action_h.dtype)
            return pair_logits.masked_fill(~pair_mask, 0.0).sum(dim=2) / denominator

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

        def encode_action_queries(  # type: ignore[no-untyped-def]
            self,
            tokens,
            token_mask,
            actions,
            action_mask,
            relation_ids=None,
            relation_tail=None,
        ):
            """Return the shared root latent and post-CGAB action latents.

            The public method keeps diagnostic heads on the exact serving
            representation instead of reimplementing the trunk. Callers may
            pass a subset of action queries only when ``relation_tail``
            contains those queries' complete outgoing relation rows.
            """
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
            return root_h, decoded, cgab_bias

        def forward(  # type: ignore[no-untyped-def]
            self,
            tokens,
            token_mask,
            actions,
            action_mask,
            relation_ids=None,
            relation_tail=None,
            pairwise_root_indices=None,
            pairwise_left_indices=None,
            pairwise_right_indices=None,
            return_pairwise_borda=False,
            pairwise_borda_top_k=None,
        ):
            root_h, decoded, cgab_bias = self.encode_action_queries(
                tokens,
                token_mask,
                actions,
                action_mask,
                relation_ids,
                relation_tail,
            )
            q_raw = self.q_head(decoded)
            if cfg.q_quantiles > 1:
                q_out = q_raw.mean(dim=-1)
            else:
                q_out = q_raw.squeeze(-1)
            outputs = {
                "logits": self.legal_logits(decoded).squeeze(-1),
                "q": q_out,
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
            if cfg.q_quantiles > 1:
                outputs["q_quantile_values"] = q_raw
            pairwise_indices = (
                pairwise_root_indices,
                pairwise_left_indices,
                pairwise_right_indices,
            )
            if any(indices is not None for indices in pairwise_indices):
                if not all(indices is not None for indices in pairwise_indices):
                    raise ValueError("all pairwise index tensors must be provided together")
                if self.pairwise_merit is None:
                    raise RuntimeError("pairwise indices require an enabled comparator")
                left = decoded[pairwise_root_indices, pairwise_left_indices]
                right = decoded[pairwise_root_indices, pairwise_right_indices]
                outputs["pairwise_logits"] = self.compare_action_embeddings(left, right)
            if return_pairwise_borda:
                pairwise_borda_mask = action_mask
                if pairwise_borda_top_k is not None:
                    if pairwise_borda_top_k <= 1:
                        raise ValueError("pairwise_borda_top_k must be greater than one")
                    candidate_count = min(int(pairwise_borda_top_k), int(action_mask.shape[1]))
                    candidate_indices = outputs["logits"].masked_fill(
                        ~action_mask,
                        -torch.inf,
                    ).topk(candidate_count, dim=1).indices
                    pairwise_borda_mask = torch.zeros_like(action_mask)
                    pairwise_borda_mask.scatter_(1, candidate_indices, True)
                    pairwise_borda_mask &= action_mask
                outputs["pairwise_borda_logits"] = self.pairwise_borda_logits(
                    decoded,
                    pairwise_borda_mask,
                )
                outputs["pairwise_borda_mask"] = pairwise_borda_mask
            return outputs

    return CascadiaFormer()


def parameter_count(model) -> int:  # type: ignore[no-untyped-def]
    return sum(param.numel() for param in model.parameters())
