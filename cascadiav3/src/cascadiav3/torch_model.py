"""Tiny CascadiaFormer-Zero-S-shaped Torch module.

This is a real `torch.nn.Module`, but intentionally tiny. It proves the fixture,
shape, gradient, optimizer, and checkpoint contracts before any real transformer
or strength work begins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .torch_features import ACTION_FEATURE_DIM, SCORE_CATEGORIES, STATE_FEATURE_DIM


@dataclass(frozen=True)
class CascadiaFormerZeroSConfig:
    state_feature_dim: int = STATE_FEATURE_DIM
    action_feature_dim: int = ACTION_FEATURE_DIM
    hidden_dim: int = 64
    seats: int = 4
    score_categories: tuple[str, ...] = SCORE_CATEGORIES
    model_name: str = "CascadiaFormer-Zero-S-tiny"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["score_categories"] = list(self.score_categories)
        return out


def build_tiny_model(config: CascadiaFormerZeroSConfig | None = None):
    import torch
    from torch import nn

    cfg = config or CascadiaFormerZeroSConfig()

    class CascadiaFormerZeroSTiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = cfg
            self.state_proj = nn.Sequential(
                nn.Linear(cfg.state_feature_dim, cfg.hidden_dim),
                nn.Tanh(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.Tanh(),
            )
            self.action_proj = nn.Sequential(
                nn.Linear(cfg.action_feature_dim, cfg.hidden_dim),
                nn.Tanh(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.Tanh(),
            )
            self.policy = nn.Linear(cfg.hidden_dim, 1)
            self.value = nn.Linear(cfg.hidden_dim, cfg.seats)
            self.rank = nn.Linear(cfg.hidden_dim, cfg.seats * cfg.seats)
            self.score = nn.Linear(cfg.hidden_dim, len(cfg.score_categories) * cfg.seats)

        def forward(self, state, actions, action_mask=None):  # type: ignore[no-untyped-def]
            single_root = actions.dim() == 2
            if single_root:
                actions = actions.unsqueeze(0)
                if state.dim() == 1:
                    state = state.unsqueeze(0)
            if action_mask is None:
                action_mask = actions.new_ones(actions.shape[:2], dtype=torch.bool)

            state_h = self.state_proj(state)
            action_h = self.action_proj(actions) + state_h.unsqueeze(1)
            action_h = torch.tanh(action_h)
            mask_f = action_mask.to(action_h.dtype).unsqueeze(-1)
            pooled = (action_h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
            outputs = {
                "legal_action_logits": self.policy(action_h).squeeze(-1),
                "value_vector": self.value(pooled),
                "rank_logits": self.rank(pooled).view(-1, cfg.seats, cfg.seats),
                "score_decomposition": self.score(pooled).view(
                    -1,
                    len(cfg.score_categories),
                    cfg.seats,
                ),
            }
            if single_root:
                outputs = {
                    "legal_action_logits": outputs["legal_action_logits"].squeeze(0),
                    "value_vector": outputs["value_vector"].squeeze(0),
                    "rank_logits": outputs["rank_logits"].squeeze(0),
                    "score_decomposition": outputs["score_decomposition"].squeeze(0),
                }
            return outputs

    return CascadiaFormerZeroSTiny()


def parameter_count(model) -> int:  # type: ignore[no-untyped-def]
    return sum(param.numel() for param in model.parameters())
