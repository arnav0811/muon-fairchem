"""Model factory for FAIR-Chem v2 E(3)-equivariant architectures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch.nn as nn

from fairchem.core.models.uma.escn_md import (
    MLP_Energy_Head,
    eSCNMDBackbone,
)

try:
    from fairchem.core.models.base import HydraModelV2 as _HydraModel
except ImportError:  # fairchem-core versions without HydraModelV2
    from fairchem.core.models.base import HydraModel as _LegacyHydraModel

    class _HydraModel(nn.Module):
        """Fallback wrapper mimicking HydraModelV2 interface."""

        def __init__(self, backbone: nn.Module, heads: Dict[str, nn.Module]) -> None:
            super().__init__()
            self.backbone = backbone
            self.heads = nn.ModuleDict(heads)

        def forward(self, data):
            embeddings = self.backbone(data)
            outputs = {}
            for name, head in self.heads.items():
                outputs[name] = head(data, embeddings)
            return outputs


@dataclass
class ModelConfig:
    name: str
    hidden_dim: int
    depth: int
    extra: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        data = dict(data)
        name = data.pop("name")
        hidden_dim = data.pop("hidden_dim")
        depth = data.pop("depth")
        return cls(name=name, hidden_dim=hidden_dim, depth=depth, extra=data)


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    model_cfg = ModelConfig.from_dict(cfg)
    if model_cfg.name != "gemnet_oc_v2":
        raise ValueError(f"Unsupported model name: {model_cfg.name}")

    cutoff = float(model_cfg.extra.get("cutoff", 6.0))
    max_neighbors = int(model_cfg.extra.get("max_neighbors", 32))
    reduce = model_cfg.extra.get("reduce", "mean")

    backbone = eSCNMDBackbone(
        hidden_channels=model_cfg.hidden_dim,
        num_layers=model_cfg.depth,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        regress_forces=False,
        regress_stress=False,
        direct_forces=False,
        use_dataset_embedding=False,
    )
    head = MLP_Energy_Head(backbone=backbone, reduce=reduce)
    model = _HydraModel(backbone=backbone, heads={"energy": head})
    return model
