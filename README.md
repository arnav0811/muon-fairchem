# muon-fairchem
This is a small study of different ablations of the Muon optimizer on FAIR Chem's GemNet-OC Architecture on the OMAat24 dataset. The goal was to see whether routing Muon to specific blocks helps this architecture. The repo also includes an SVD analysis on different layers. All the training, logs, and plots are reproducible. 

## Overview
  - Model: gemnet_oc_v2 
  - Dataset: OMat24 rattled-300 subset (data/rattled-300-subsampled).
  - Optimizers compared:
      - AdamW baseline.
      - Full Muon (all layers).
      - Muon with orthogonalization disabled.
      - Muon routed to the first k blocks (k = 1,2,3).
      - Muon routed to the last 3 blocks.
      - Baseline + Muon-firstk repeats with SVD logging.

## Results (validation MAE after 40 epochs)

  | run                      | final MAE | best MAE |
  |--------------------------|-----------|----------|
  | Omat24-Muon-Firstk       | 0.0901    | 0.0901   |
  | Omat24-Muon-Firstk-K2    | 0.0944    | 0.0944   |
  | Omat24-Baseline (AdamW)  | 0.0979    | 0.0960   |
  | Omat24-Muon-Firstk-K1    | 0.1006    | 0.1006   |
  | Omat24-Muon-Lastk        | 0.1042    | 0.1017   |
  | Omat24-Muon-No-Ortho     | 0.1214    | 0.1214   |
  | Omat24-Muon (full)       | 0.1646    | 0.1226   |

