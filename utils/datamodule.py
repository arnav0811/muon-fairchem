"""Dataset loading utilities for FAIR-Chem LMDB shards."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from fairchem.core.datasets.ase_datasets import AseDBDataset, apply_one_tags
from fairchem.core.datasets.collaters.simple_collater import data_list_collater


def _inject_target(atoms, target_key: str, per_atom: bool) -> Any:
    atoms = atoms.copy()
    target = atoms.info.get(target_key)
    if target is None:
        raise KeyError(f"Target key '{target_key}' missing in atoms.info")
    value = float(target)
    if per_atom:
        natoms = len(atoms)
        if natoms == 0:
            raise ValueError("Cannot normalise per atom when structure has zero atoms")
        value /= natoms
    atoms.info["energy"] = value
    return apply_one_tags(atoms)


class OMat24Dataset(Dataset):
    def __init__(
        self,
        root: Path,
        target_key: str,
        per_atom_target: bool,
        radius: float,
        max_neighbors: int,
        keep_in_memory: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.target_key = target_key
        self.per_atom_target = per_atom_target

        self.dataset = AseDBDataset(
            config={
                "src": str(root),
                "a2g_args": {
                    "r_edges": True,
                    "radius": radius,
                    "max_neigh": max_neighbors,
                    "r_forces": False,
                    "r_stress": False,
                },
                "keep_in_memory": keep_in_memory,
            },
            atoms_transform=lambda atoms: _inject_target(atoms, target_key, per_atom_target),
        )

        self.indices = self._collect_valid_indices(max_samples)
        if not self.indices:
            raise RuntimeError(
                f"No samples found with target key '{target_key}' under {root}"
            )

    def _collect_valid_indices(self, max_samples: int | None) -> List[int]:
        indices: List[int] = []
        limit = max_samples if max_samples is not None else math.inf
        for idx in range(len(self.dataset)):
            atoms = self.dataset.get_atoms(idx)
            if self.target_key in atoms.info:
                indices.append(idx)
                if len(indices) >= limit:
                    break
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]


def _split_indices(
    num_samples: int,
    fractions: Dict[str, float],
    rng: torch.Generator,
) -> Dict[str, List[int]]:
    keys = list(fractions.keys())
    total = sum(fractions.values())
    if total <= 0 or total > 1 + 1e-6:
        raise ValueError("Split fractions must sum to <= 1 and be positive")

    perm = torch.randperm(num_samples, generator=rng).tolist()
    split_points = []
    cumulative = 0
    for key in keys:
        count = math.floor(fractions[key] * num_samples)
        split_points.append((key, cumulative, cumulative + count))
        cumulative += count

    splits: Dict[str, List[int]] = {}
    for key, start, end in split_points:
        splits[key] = perm[start:end]

    if cumulative < num_samples:
        remaining = perm[cumulative:]
        target_key = keys[-1]
        splits[target_key].extend(remaining)

    return splits


def _make_loader(
    dataset: Dataset,
    indices: Iterable[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    subset = Subset(dataset, list(indices))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=data_list_collater,
    )


def build_dataloaders(config: Dict[str, Any], split: str = "train", device: str = "cpu") -> Dict[str, DataLoader]:
    root = config.get("root")
    if root is None:
        raise ValueError("Dataset config must include 'root' path to LMDB directory")

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")

    target_key = config.get("target_key", "energy_corrected_mp2020")
    per_atom = config.get("normalize_per_atom", False)
    radius = float(config.get("radius", 6.0))
    max_neighbors = int(config.get("max_neighbors", 50))
    keep_in_memory = bool(config.get("keep_in_memory", False))
    max_samples = config.get("max_samples")

    dataset = OMat24Dataset(
        root=root_path,
        target_key=target_key,
        per_atom_target=per_atom,
        radius=radius,
        max_neighbors=max_neighbors,
        keep_in_memory=keep_in_memory,
        max_samples=int(max_samples) if max_samples is not None else None,
    )

    total_samples = len(dataset)

    split_cfg = config.get(
        "split_fraction",
        {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
    )
    rng = torch.Generator().manual_seed(int(config.get("split_seed", 42)))

    indices_map = _split_indices(total_samples, split_cfg, rng)

    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", False))

    loaders: Dict[str, DataLoader] = {}
    for name, idxs in indices_map.items():
        shuffle = name == "train"
        loaders[name] = _make_loader(
            dataset,
            idxs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders
