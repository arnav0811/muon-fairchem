import torch, pathlib, matplotlib.pyplot as plt

def load_latest_sv(run_dir: pathlib.Path, layer_name: str, which="weights"):
    files = sorted(run_dir.glob(f"{layer_name.replace('.', '_')}*.pt"))
    if not files:
        return None
    data = torch.load(files[-1], map_location="cpu")
    return data[which]["s"].cpu().numpy()

def main():
    base = pathlib.Path("results")
    runA = base / "svd_omat24_baseline_svd"
    runB = base / "svd_omat24_muon_firstk_svd"
    layers = [
        "backbone.blocks.0.edge_wise.so2_conv_1.fc_m0.weight",
        "backbone.blocks.1.edge_wise.so2_conv_1.fc_m0.weight",
        "backbone.blocks.2.edge_wise.so2_conv_1.fc_m0.weight",
        "backbone.blocks.3.edge_wise.so2_conv_1.fc_m0.weight",
        "backbone.blocks.4.edge_wise.so2_conv_1.fc_m0.weight",
        "backbone.blocks.5.edge_wise.so2_conv_1.fc_m0.weight",
    ]
    out = base / "svd_figs"; out.mkdir(exist_ok=True)
    for layer in layers:
        svA = load_latest_sv(runA, layer, "weights")
        svB = load_latest_sv(runB, layer, "weights")
        if svA is None or svB is None:
            print(f"Missing: {layer}")
            continue
        plt.figure()
        plt.semilogy(sorted(svA)[::-1], label="baseline")
        plt.semilogy(sorted(svB)[::-1], label="muon_firstk")
        plt.title(layer)
        plt.xlabel("Rank index"); plt.ylabel("Singular value (log)")
        plt.legend(); plt.tight_layout()
        plt.savefig(out / f"{layer.replace('.', '_')}.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    main()
