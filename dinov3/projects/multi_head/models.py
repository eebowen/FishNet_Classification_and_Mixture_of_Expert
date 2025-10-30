from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn


@dataclass
class TaskSpec:
    name: str
    type: str  # "multiclass" | "multilabel" | "regression"
    out_dim: int


class MultiTaskHeads(nn.Module):
    """
    Simple multi-head model: apply a configurable N-layer tower per task directly on input features.

    Tower: LayerNorm(in) -> [Linear(dim, hidden) -> GELU -> Dropout] x num_layers -> Linear(hidden, out)
    """

    def __init__(
        self,
        in_dim: int,
        tasks: List[TaskSpec] | Tuple[TaskSpec, ...],
        tower_hidden: int = 1024,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.towers = nn.ModuleDict()
        for spec in tasks:
            layers = [nn.LayerNorm(in_dim)]
            
            # Build the tower with num_layers
            for i in range(num_layers):
                if i == 0:
                    # First layer: in_dim -> tower_hidden
                    layers.extend([
                        nn.Linear(in_dim, tower_hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ])
                elif i == num_layers - 1:
                    # Last layer: tower_hidden -> out_dim
                    layers.append(nn.Linear(tower_hidden, spec.out_dim))
                else:
                    # Intermediate layers: tower_hidden -> tower_hidden
                    layers.extend([
                        nn.Linear(tower_hidden, tower_hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ])
            
            # Handle edge case: if num_layers == 1, go directly from in_dim to out_dim
            if num_layers == 1:
                layers = [
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, spec.out_dim),
                ]
            
            head = nn.Sequential(*layers)
            self.towers[spec.name] = head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: tower(x) for name, tower in self.towers.items()}
