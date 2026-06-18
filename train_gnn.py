from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import GraphSample, apply_normalizer, collate_graph_samples, fit_normalizer, load_dataset, nx_to_sample, split_graphs # These are the utilities from the data_utils.py file
from model import QuantumCircuitGNN # This is the model for the GNN
from plot_utils import save_prediction_artifacts


PROJECT_ROOT = Path(__file__).resolve().parent


def metrics(pred: torch.Tensor, target: torch.Tensor, acc_tol: float = 0.1) -> Dict[str, float]: # This is the function that will be used to calculate the metrics.
    pred = pred.detach().cpu().float().view(-1)
    target = target.detach().cpu().float().view(-1)
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = float(np.sqrt(mse))
    ss_res = torch.sum((target - pred) ** 2).item()
    ss_tot = torch.sum((target - target.mean()) ** 2).item()
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 1e-12 else float('nan'))
    acc = torch.mean((torch.abs(pred - target) <= acc_tol).float()).item() * 100.0
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'acc': acc} # This is the dictionary that will be used to store the metrics including the mean squared error, mean absolute error , root mean square error, r2 score and accuracy.



def batch_iter(samples: List[GraphSample], batch_size: int, shuffle: bool = True, seed: int = 42): # This is the function that will be used to iterate through the samples in batches.
    idx = np.arange(len(samples))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        yield collate_graph_samples([samples[i] for i in batch_idx])


def evaluate(model: nn.Module, samples: List[GraphSample], device: torch.device, batch_size: int = 32, acc_tol: float = 0.1):
    model.eval()
    preds, trues, names = [], [], []
    with torch.no_grad():
        for batch in batch_iter(samples, batch_size=batch_size, shuffle=False):
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            batch_vec = batch.batch.to(device)
            global_features = batch.global_features.to(device)
            y = batch.y.to(device)
            pred = model(x, edge_index, batch_vec, global_features)
            preds.append(pred.detach().cpu())
            trues.append(y.detach().cpu())
            names.extend(batch.graph_name.split('||') if batch.graph_name else ['graph'] * len(y))
    pred = torch.cat(preds, dim=0)
    true = torch.cat(trues, dim=0)
    return pred, true, names, metrics(pred, true, acc_tol=acc_tol)


def plot_metric_curves(history: List[Dict[str, float]], out_path: Path) -> None:
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_mse = [h['val_mse'] for h in history]
    val_mae = [h['val_mae'] for h in history]
    val_rmse = [h['val_rmse'] for h in history]
    val_r2 = [h['val_r2'] for h in history]
    val_acc = [h['val_acc'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label='Train Loss')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, val_r2, label='Val R2')
    ax.set_title('Validation R2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R2')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, val_acc, label='Val Accuracy')
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, val_mse, label='MSE')
    ax.plot(epochs, val_mae, label='MAE')
    ax.plot(epochs, val_rmse, label='RMSE')
    ax.set_title('Validation Error Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric value')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_parity_and_series(actual, predicted, out_path: Path, title_prefix: str = 'Validation') -> None:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    order = np.argsort(actual)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(actual, predicted, s=16, alpha=0.75)
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    ax.plot([lo, hi], [lo, hi], linestyle='--')
    ax.set_title(f'{title_prefix}: Actual vs Predicted')
    ax.set_xlabel('Actual energy')
    ax.set_ylabel('Predicted energy')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(actual[order], label='Actual')
    ax.plot(predicted[order], label='Predicted')
    ax.set_title(f'{title_prefix}: Sorted Actual and Predicted')
    ax.set_xlabel('Sorted sample index')
    ax.set_ylabel('Energy')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Train a GNN to predict quantum-circuit energy.')
    ap.add_argument('--train_data', required=True, help='Folder or .pkl containing training graphs')
    ap.add_argument('--val_data', default=None, help='Optional separate validation dataset')
    ap.add_argument('--val_ratio', type=float, default=0.1, help='Validation split ratio if no val_data is supplied')
    ap.add_argument('--out_dir', default='runs/quantum_gnn', help='Output directory for checkpoints and logs')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--acc_tol', type=float, default=0.1, help='Absolute error tolerance for accuracy (%)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = load_dataset(args.train_data)
    if args.val_data:
        train_graphs = graphs
        val_graphs = load_dataset(args.val_data)
    else:
        train_graphs, val_graphs = split_graphs(graphs, val_ratio=args.val_ratio, seed=args.seed)

    train_samples = [nx_to_sample(g) for g in train_graphs]
    val_samples = [nx_to_sample(g) for g in val_graphs]

    stats = fit_normalizer(train_samples)
    train_samples = apply_normalizer(train_samples, stats)
    val_samples = apply_normalizer(val_samples, stats)

    device = torch.device(args.device)
    model = QuantumCircuitGNN(node_dim=17, global_dim=14, hidden_dim=128, num_layers=4, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in batch_iter(train_samples, batch_size=args.batch_size, shuffle=True, seed=args.seed + epoch):
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            batch_vec = batch.batch.to(device)
            global_features = batch.global_features.to(device)
            y = batch.y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, edge_index, batch_vec, global_features)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_pred, val_true, _, val_m = evaluate(model, val_samples, device, batch_size=args.batch_size, acc_tol=args.acc_tol)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mse': val_m['mse'],
            'val_mae': val_m['mae'],
            'val_rmse': val_m['rmse'],
            'val_r2': val_m['r2'],
            'val_acc': val_m['acc'],
        })

        print(f"Epoch {epoch:03d} | train loss {train_loss:.6f} | val mse {val_m['mse']:.6f} | val mae {val_m['mae']:.6f} | val rmse {val_m['rmse']:.6f} | val r2 {val_m['r2']:.4f} | val acc {val_m['acc']:.2f}%")

        if val_m['mse'] < best_val:
            best_val = val_m['mse']
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'stats': stats,
                'config': vars(args),
                'best_val_mse': best_val,
                'node_dim': 17,
                'global_dim': 14,
            }
            torch.save(ckpt, out_dir / 'best_model.pt')

    with open(out_dir / 'history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader(); writer.writerows(history)

    with open(out_dir / 'data_stats.json', 'w') as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f, indent=2)

    best = torch.load(out_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(best['model_state_dict'])
    val_pred, val_true, val_names, val_m = evaluate(model, val_samples, device, batch_size=args.batch_size, acc_tol=args.acc_tol)


    plot_metric_curves(history, out_dir / 'training_metrics.png')
    artifacts = save_prediction_artifacts(
        val_true.numpy(),
        val_pred.numpy(),
        val_names,
        out_dir,
        prefix='validation',
        acc_tol=args.acc_tol,
    )

    print('\nBest validation metrics:')
    print(f"  MSE : {val_m['mse']:.6f}")
    print(f"  MAE : {val_m['mae']:.6f}")
    print(f"  RMSE: {val_m['rmse']:.6f}")
    print(f"  R2  : {val_m['r2']:.4f}")
    print(f"  ACC : {val_m['acc']:.2f}% @ |err| <= {args.acc_tol}")
    print(f"Saved best model to: {out_dir / 'best_model.pt'}")
    print(f"Saved validation predictions to: {artifacts['csv_path']}")
    print(f"Saved training plot to: {out_dir / 'training_metrics.png'}")
    print(f"Saved validation plot to: {out_dir / 'validation_actual_vs_predicted.png'}")
    if artifacts['family_rows']:
        print(f"Saved validation family summary to: {out_dir / 'validation_family_summary.png'}")
        print(f"Saved validation family metrics to: {artifacts['family_csv_path']}")
    if artifacts.get('depth_rows'):
        print(f"Saved validation depth summary to: {out_dir / 'validation_depth_summary.png'}")
        print(f"Saved validation depth metrics to: {artifacts['depth_csv_path']}")

if __name__ == '__main__':
    main()
