"""net.py - the StuffMLP.

Shallow on purpose: tabular physics targets don't reward depth, and the whole
point is a clean, low-dimensional representation worth explaining later. The only
width that matters is ``d_repr = 32`` (~concept count + headroom).

    input -> 256 -> 128 -> 32 -> 1   (ReLU between layers)
                          ^^^^
                          penultimate (the product of this whole brief)

The readout head is a separate ``nn.Linear(d_repr, 1)`` module *by design*: the
explainer later reconstructs representations and pushes them back through this
exact head, so it must be loadable on its own (see train.py's hand-off artifact).
"""

from __future__ import annotations

import torch
import torch.nn as nn

D_REPR = 32


class StuffMLP(nn.Module):
    """MLP mapping standardized pitch physics -> run value.

    forward returns ``(pred, penultimate)`` where ``penultimate`` is the post-ReLU
    32-d activation. That vector - not the scalar prediction - is what gets saved,
    aligned to concept columns, and explained downstream.
    """

    def __init__(self, d_in: int, d_repr: int = D_REPR):
        super().__init__()
        self.d_in = d_in
        self.d_repr = d_repr
        # Backbone produces the penultimate representation.
        self.backbone = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, d_repr),
            nn.ReLU(),
        )
        # Frozen-at-hand-off readout. Kept separate so it loads standalone.
        self.head = nn.Linear(d_repr, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        penultimate = self.backbone(x)            # (B, d_repr), post-ReLU
        pred = self.head(penultimate).squeeze(-1)  # (B,)
        return pred, penultimate

    @torch.no_grad()
    def represent(self, x: torch.Tensor) -> torch.Tensor:
        """Return just the penultimate representation (eval-time convenience)."""
        return self.backbone(x)
