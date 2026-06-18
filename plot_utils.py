#Author : Nithish Sriram Srinivasan
#Purpose : This file contains the utility functions for plotting the metrics and parity plots for the quantum circuit GNN models.

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def infer_family(name: str) -> str:
    base = Path(str(name)).name
    if '::' in base:
        return base.split('::', 1)[0]
    if '|' in base:
        return base.split('|', 1)[0]

    base_l = base.lower()
    m = re.match(r'^(?P<fam>[a-z0-9+]+?)graph(?:_|$)', base_l)
    if m:
        return m.group('fam')
    m = re.match(r'^(?P<fam>[a-z0-9+]+?)(?:_|$)', base_l)
    if m:
        return m.group('fam')
    return base_l[:32] or 'graph'


def infer_depth(name: str) -> str:
    m = re.search(r'(depth\d+)', str(name).lower())
    return m.group(1) if m else 'unknown_depth'


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, acc_tol: float = 0.1) -> Dict[str, float]:
    pred = pred.detach().cpu().float().view(-1)
    target = target.detach().cpu().float().view(-1)
    mse = torch.mean((pred - target) ** 2).item()
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = float(np.sqrt(mse))
    ss_res = torch.sum((target - pred) ** 2).item()
    ss_tot = torch.sum((target - target.mean()) ** 2).item()
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 1e-12 else float('nan'))
    acc = torch.mean((torch.abs(pred - target) <= acc_tol).float()).item() * 100.0
    return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'acc': acc}


def plot_metric_summary(metrics_dict: Dict[str, float], out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.bar(['MSE', 'MAE', 'RMSE'], [metrics_dict['mse'], metrics_dict['mae'], metrics_dict['rmse']])
    ax.set_title(f'{title}: Error metrics')
    ax.set_ylabel('Value')
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1]
    ax.bar(['R2', 'ACC'], [metrics_dict['r2'], metrics_dict['acc']])
    ax.set_title(f'{title}: R2 and accuracy')
    ax.set_ylabel('Value')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=min(0.0, metrics_dict['r2'] - 0.1))

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_parity(actual, predicted, out_path: Path, title_prefix: str = 'Validation') -> None:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    order = np.argsort(actual)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(actual, predicted, s=16, alpha=0.75)
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    ax.plot([lo, hi], [lo, hi], linestyle='--')
    ax.set_title(f'{title_prefix}: Actual vs Predicted')
    ax.set_xlabel('Actual energy')
    ax.set_ylabel('Predicted energy')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(actual[order], label='Actual')
    ax.plot(predicted[order], label='Predicted')
    ax.set_title(f'{title_prefix}: Sorted actual and predicted')
    ax.set_xlabel('Sorted sample index')
    ax.set_ylabel('Energy')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _group_indices(names: Sequence[str], key_fn) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, name in enumerate(names):
        groups[key_fn(name)].append(idx)
    return groups


def _sort_depth_key(label: str):
    m = re.search(r'(\d+)$', str(label))
    return int(m.group(1)) if m else 10**9


def _build_group_rows(actual, predicted, names: Sequence[str], key_fn, label_name: str, acc_tol: float = 0.1):
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    groups = _group_indices(names, key_fn)
    rows = []
    keys = sorted(groups.keys(), key=_sort_depth_key if label_name == 'depth' else lambda x: str(x))
    for key in keys:
        idxs = groups[key]
        a = torch.tensor(actual[idxs], dtype=torch.float32)
        p = torch.tensor(predicted[idxs], dtype=torch.float32)
        m = compute_metrics(p, a, acc_tol=acc_tol)
        rows.append({label_name: key, 'count': len(idxs), **m})
    return rows


def family_metrics(actual, predicted, names: Sequence[str], acc_tol: float = 0.1):
    return _build_group_rows(actual, predicted, names, infer_family, 'family', acc_tol=acc_tol)


def depth_metrics(actual, predicted, names: Sequence[str], acc_tol: float = 0.1):
    return _build_group_rows(actual, predicted, names, infer_depth, 'depth', acc_tol=acc_tol)


def plot_family_summary(rows, out_path: Path, title: str) -> None:
    if not rows:
        return
    labels = [r['family'] for r in rows]
    r2_vals = [r['r2'] for r in rows]
    acc_vals = [r['acc'] for r in rows]
    counts = [r['count'] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 5), 4.8))

    ax = axes[0]
    bars = ax.bar(labels, r2_vals)
    ax.set_title(f'{title}: R2 by family')
    ax.set_ylabel('R2')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(min(-1.0, min(r2_vals) - 0.05), max(1.0, max(r2_vals) + 0.05))
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'n={c}', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    bars = ax.bar(labels, acc_vals)
    ax.set_title(f'{title}: Accuracy by family')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'n={c}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_depth_summary(rows, out_path: Path, title: str) -> None:
    if not rows:
        return
    labels = [r['depth'] for r in rows]
    r2_vals = [r['r2'] for r in rows]
    acc_vals = [r['acc'] for r in rows]
    counts = [r['count'] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 5), 4.8))

    ax = axes[0]
    bars = ax.bar(labels, r2_vals)
    ax.set_title(f'{title}: R2 by depth')
    ax.set_ylabel('R2')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(min(-1.0, min(r2_vals) - 0.05), max(1.0, max(r2_vals) + 0.05))
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'n={c}', ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    bars = ax.bar(labels, acc_vals)
    ax.set_title(f'{title}: Accuracy by depth')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'n={c}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _plot_group_parity(actual, predicted, names: Sequence[str], key_fn, out_dir: Path, prefix: str, acc_tol: float = 0.1):
    out_dir.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    groups = _group_indices(names, key_fn)
    for key, idxs in groups.items():
        a = actual[idxs]
        p = predicted[idxs]
        fm = compute_metrics(torch.tensor(p), torch.tensor(a), acc_tol=acc_tol)
        title = f'{prefix.replace("_", " ").title()} / {key} | R2={fm["r2"]:.4f} | ACC={fm["acc"]:.2f}%'
        plot_parity(a, p, out_dir / f'{key}_{prefix}_actual_vs_predicted.png', title_prefix=title)


def save_prediction_artifacts(actual, predicted, names: Sequence[str], out_dir: Path, prefix: str, acc_tol: float = 0.1):
    out_dir.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    names = list(names)
    m = compute_metrics(torch.tensor(predicted), torch.tensor(actual), acc_tol=acc_tol)

    csv_path = out_dir / f'{prefix}_predictions.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['graph_name', 'actual_energy', 'predicted_energy', 'abs_error', 'family', 'depth'])
        for name, yt, yp in zip(names, actual.tolist(), predicted.tolist()):
            writer.writerow([name, yt, yp, abs(yt - yp), infer_family(name), infer_depth(name)])

    plot_metric_summary(m, out_dir / f'{prefix}_metrics_summary.png', title=prefix.replace('_', ' ').title())
    plot_parity(actual, predicted, out_dir / f'{prefix}_actual_vs_predicted.png', title_prefix=prefix.replace('_', ' ').title())

    fam_rows = family_metrics(actual, predicted, names, acc_tol=acc_tol)
    depth_rows = depth_metrics(actual, predicted, names, acc_tol=acc_tol)

    fam_csv = out_dir / f'{prefix}_family_metrics.csv'
    with open(fam_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['family', 'count', 'mse', 'mae', 'rmse', 'r2', 'acc'])
        writer.writeheader()
        writer.writerows(fam_rows)

    depth_csv = out_dir / f'{prefix}_depth_metrics.csv'
    with open(depth_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['depth', 'count', 'mse', 'mae', 'rmse', 'r2', 'acc'])
        writer.writeheader()
        writer.writerows(depth_rows)

    if fam_rows:
        plot_family_summary(fam_rows, out_dir / f'{prefix}_family_summary.png', title=prefix.replace('_', ' ').title())
    if depth_rows:
        plot_depth_summary(depth_rows, out_dir / f'{prefix}_depth_summary.png', title=prefix.replace('_', ' ').title())

    fam_plot_dir = out_dir / f'{prefix}_family_plots'
    _plot_group_parity(actual, predicted, names, infer_family, fam_plot_dir, prefix, acc_tol=acc_tol)

    depth_plot_dir = out_dir / f'{prefix}_depth_plots'
    _plot_group_parity(actual, predicted, names, infer_depth, depth_plot_dir, prefix, acc_tol=acc_tol)

    return {
        'overall': m,
        'family_rows': fam_rows,
        'depth_rows': depth_rows,
        'csv_path': csv_path,
        'family_csv_path': fam_csv,
        'depth_csv_path': depth_csv,
    }
