from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import apply_normalizer, load_dataset, nx_to_sample
from model import QuantumCircuitGNN
from plot_utils import save_prediction_artifacts


PROJECT_ROOT = Path(__file__).resolve().parent


def metrics(pred: torch.Tensor, target: torch.Tensor, acc_tol: float = 0.1):
    pred = pred.detach().cpu().float().view(-1)
    target = target.detach().cpu().float().view(-1)
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = float(np.sqrt(mse))
    ss_res = torch.sum((target - pred) ** 2).item()
    ss_tot = torch.sum((target - target.mean()) ** 2).item()
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))
    acc = torch.mean((torch.abs(pred - target) <= acc_tol).float()).item() * 100.0
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "acc": acc}


def plot_parity_and_series(actual, predicted, out_path: Path, title_prefix: str = "Test") -> None:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    order = np.argsort(actual)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(actual, predicted, s=16, alpha=0.75)
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_title(f"{title_prefix}: Actual vs Predicted")
    ax.set_xlabel("Actual energy")
    ax.set_ylabel("Predicted energy")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(actual[order], label="Actual")
    ax.plot(predicted[order], label="Predicted")
    ax.set_title(f"{title_prefix}: Sorted Actual and Predicted")
    ax.set_xlabel("Sorted sample index")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Test a trained GNN on quantum-circuit graph data.")
    ap.add_argument("--test_data", required=True, help="Folder or .pkl containing test graphs")
    ap.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    ap.add_argument("--out_dir", default="runs/quantum_gnn_test", help="Directory for outputs")
    ap.add_argument("--acc_tol", type=float, default=0.1, help="Absolute error tolerance for accuracy (%)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    stats = ckpt["stats"]

    graphs = load_dataset(args.test_data)
    samples = [nx_to_sample(g) for g in graphs]
    samples = apply_normalizer(samples, stats)

    model = QuantumCircuitGNN(node_dim=17, global_dim=14, hidden_dim=128, num_layers=4, dropout=0.0).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds = []
    trues = []
    names = []

    with torch.no_grad():
        for s in samples:
            x = s.x.to(args.device)
            edge_index = s.edge_index.to(args.device)
            batch = s.batch.to(args.device)
            global_features = s.global_features.to(args.device)
            y = s.y.to(args.device)

            pred = model(x, edge_index, batch, global_features)
            preds.append(pred.detach().cpu().view(-1))
            trues.append(y.detach().cpu().view(-1))
            names.append(s.graph_name)

    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    m = metrics(pred, true, acc_tol=args.acc_tol)

    # IMPORTANT: keep the reporting and artifact-saving inside main()
    artifacts = save_prediction_artifacts(
        true.cpu().numpy(),
        pred.cpu().numpy(),
        names,
        out_dir,
        prefix="test",
        acc_tol=args.acc_tol,
    )

    # Keep a copy of the parity plot too, in case you want the older style
    plot_parity_and_series(
        true.cpu().numpy(),
        pred.cpu().numpy(),
        out_dir / "test_parity_and_series.png",
        title_prefix="Test",
    )

    print("Test metrics:")
    print(f"  MSE : {m['mse']:.6f}")
    print(f"  MAE : {m['mae']:.6f}")
    print(f"  RMSE: {m['rmse']:.6f}")
    print(f"  R2  : {m['r2']:.4f}")
    print(f"  ACC : {m['acc']:.2f}% @ |err| <= {args.acc_tol}")
    print(f"Saved test predictions to: {artifacts['csv_path']}")
    print(f"Saved test plot to: {out_dir / 'test_actual_vs_predicted.png'}")
    print(f"Saved parity plot to: {out_dir / 'test_parity_and_series.png'}")
    if artifacts["family_rows"]:
        print(f"Saved test family summary to: {out_dir / 'test_family_summary.png'}")
        print(f"Saved test family metrics to: {artifacts['family_csv_path']}")
    if artifacts.get('depth_rows'):
        print(f"Saved test depth summary to: {out_dir / 'test_depth_summary.png'}")
        print(f"Saved test depth metrics to: {artifacts['depth_csv_path']}")


if __name__ == "__main__":
    main()
