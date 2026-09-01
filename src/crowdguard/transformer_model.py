"""Temporal models for crowd risk.

`TemporalRiskForecaster` is the model that makes the word "prediction" in the
project title honest: it consumes a window of measured crowd features and
predicts the risk score H seconds into the future, together with a class
distribution over {low, moderate, high}.

The original `TemporalRiskTransformer` classifier is retained so that older
experiments and any saved checkpoints keep working.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positions.

    Order matters here in a way it does not for a bag of features: a crowd
    whose density is falling and one whose density is rising can share the same
    mean, variance and extremes over the window, and only the ordering
    separates "draining safely" from "filling toward a crush".
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalRiskTransformer(nn.Module):
    """Sequence -> {low, moderate, high} classifier (legacy)."""

    def __init__(self, feature_dim: int = 6, hidden_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, num_classes: int = 3):
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


class TemporalRiskForecaster(nn.Module):
    """Sequence -> (risk score at t+H, class distribution at t+H).

    Two heads on one trunk. The regression head gives the number the operator
    UI counts down against; the classification head gives a calibrated
    confidence, and its cross-entropy term regularises the regression, which
    matters because the high-risk tail is rare in any real log.
    """

    def __init__(self, feature_dim: int = 8, hidden_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.positional = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Attention pooling: the informative part of a 30-step window is
        # usually a short burst, and mean pooling washes that out.
        self.attn_pool = nn.Linear(hidden_dim, 1)

        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.input_projection(x)
        h = self.positional(h)
        h = self.encoder(h)
        h = self.norm(h)
        weights = torch.softmax(self.attn_pool(h), dim=1)
        pooled = (h * weights).sum(dim=1)
        return {
            "risk": self.risk_head(pooled).squeeze(-1),
            "logits": self.class_head(pooled),
            "attention": weights.squeeze(-1),
        }


if __name__ == "__main__":
    model = TemporalRiskForecaster()
    sample = torch.randn(4, 30, 8)
    out = model(sample)
    print("risk:", tuple(out["risk"].shape), "logits:", tuple(out["logits"].shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
