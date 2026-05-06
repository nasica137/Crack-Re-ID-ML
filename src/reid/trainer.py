from __future__ import annotations

from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from src.reid.model import AttentionSimilarity
from src.utils.io import ensure_dir


def train_attention(memory: List[dict], config: dict) -> AttentionSimilarity:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = torch.tensor([m["fused"] for m in memory], dtype=torch.float32, device=device)
    model = AttentionSimilarity(
        input_dim=feats.shape[1],
        hidden_dim=int(config["reid"]["attention"]["hidden_dim"]),
        temperature=float(config["reid"]["attention"]["temperature"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["reid"]["attention"]["lr"]))

    model.train()
    epochs = int(config["reid"]["attention"]["epochs"])
    for _ in range(epochs):
        logits = model(feats, feats)
        labels = torch.arange(feats.shape[0], device=device)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def save_model(model: AttentionSimilarity, output_path: str) -> Path:
    path = Path(output_path)
    ensure_dir(path.parent)
    torch.save(model.state_dict(), path)
    return path
