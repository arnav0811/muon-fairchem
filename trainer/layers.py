import torch
from models.e3_model import build_model
import yaml
import pathlib

# Load one of your working configs (baseline or muon_firstk)
config_path = pathlib.Path("configs/omat24_baseline.yaml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Build model from config
model = build_model(cfg["model"])

print("\n--- Model Parameter Names ---\n")
for name, param in model.named_parameters():
    print(name, "| shape:", tuple(param.shape))

print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))
