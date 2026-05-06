from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionSimilarity(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, temperature: float) -> None:
        super().__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.temperature = temperature

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.query_proj(query), p=2, dim=-1)
        k = F.normalize(self.key_proj(keys), p=2, dim=-1)
        return (q @ k.T) / self.temperature
