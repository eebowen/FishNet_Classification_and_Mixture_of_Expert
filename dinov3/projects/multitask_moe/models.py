from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn


@dataclass
class TaskSpec:
    name: str
    type: str  # "multiclass" | "multilabel" | "regression"
    out_dim: int


class GatedMMoE(nn.Module):
    """
    Mixture-of-Experts with task-specific softmax gates and task towers.

    Inputs: feature tensor of shape [B, D]
    Experts: N expert MLPs mapping D -> H
    Gates: one softmax gate per task producing mixing weights over N experts
    Towers/Heads: per-task small MLP mapping H -> out_dim (head differs by task type)
    """

    def __init__(
        self,
        in_dim: int,
        experts: int = 4,
        expert_hidden: int = 2048,
        tower_hidden: int = 1024,
        tasks: List[TaskSpec] | Tuple[TaskSpec, ...] = (),
        dropout: float = 0.1,
        gating_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        assert experts >= 1
        self.in_dim = in_dim
        self.experts_n = experts
        self.gating_temperature = gating_temperature

        # Expert pool
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, expert_hidden),
                nn.GELU(),
            )
            for _ in range(experts)
        ])

        # Gates: one per task -> logits over experts
        self.task_gates = nn.ModuleDict()
        # Towers/heads: one per task
        self.task_towers = nn.ModuleDict()
        self.task_specs: Dict[str, TaskSpec] = {}

        for spec in tasks:
            self.add_task(spec, tower_hidden, dropout)

    def add_task(self, spec: TaskSpec, tower_hidden: int, dropout: float) -> None:
        self.task_specs[spec.name] = spec
        self.task_gates[spec.name] = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.experts_n),
        )
        if spec.type == "multiclass":
            head = nn.Sequential(
                nn.LayerNorm(tower_hidden),
                nn.Dropout(dropout),
                nn.Linear(tower_hidden, spec.out_dim),
            )
        elif spec.type == "multilabel":
            head = nn.Sequential(
                nn.LayerNorm(tower_hidden),
                nn.Dropout(dropout),
                nn.Linear(tower_hidden, spec.out_dim),
            )
        elif spec.type == "regression":
            head = nn.Sequential(
                nn.LayerNorm(tower_hidden),
                nn.Dropout(dropout),
                nn.Linear(tower_hidden, spec.out_dim),
            )
        else:
            raise ValueError(f"Unknown task type: {spec.type}")

        tower = nn.Sequential(
            nn.Linear(self.experts[0][-2].out_features, tower_hidden),  # expert_hidden -> tower_hidden
            nn.GELU(),
            nn.Dropout(dropout),
            head,
        )
        self.task_towers[spec.name] = tower

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Compute expert activations
        expert_outs = [expert(x) for expert in self.experts]  # list of [B, H]
        expert_stack = torch.stack(expert_outs, dim=1)  # [B, E, H]

        outputs: Dict[str, torch.Tensor] = {}
        for name, gate in self.task_gates.items():
            logits = gate(x) / max(1e-6, self.gating_temperature)  # [B, E]
            weights = torch.softmax(logits, dim=-1).unsqueeze(-1)  # [B, E, 1]
            mixed = (expert_stack * weights).sum(dim=1)  # [B, H]
            out = self.task_towers[name](mixed)
            outputs[name] = out
        return outputs

    @torch.no_grad()
    def gate_probs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        probs: Dict[str, torch.Tensor] = {}
        for name, gate in self.task_gates.items():
            logits = gate(x) / max(1e-6, self.gating_temperature)
            probs[name] = torch.softmax(logits, dim=-1)
        return probs
