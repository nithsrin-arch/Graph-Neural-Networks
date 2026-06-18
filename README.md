
# Quantum Circuit GNN Project 

This project trains a graph neural network to predict the energy of quantum circuits converted into graph data.

## What this version adds

- Training validation curves for loss, R2, and accuracy
- Actual-vs-predicted plots for validation and test
- Per-family plots and summaries for graph families such as `fhgraph_*`, `h2graph_*`, `hehgraph_*`, and `xyzgraph_*`
- CSV summaries for overall and family-wise results

## Folder structure

```text
quantum_circuit_gnn_better/
  main.py
  train_gnn.py
  test_gnn.py
  plot_utils.py
  data_utils.py
  model.py
  requirements.txt
  README.md
```

## Input format

The scripts accept either:

- a folder of `.gpickle` / `.pkl` graph files, or
- a single `.pkl` containing a list of NetworkX graphs, or
- a dict with a key like `networkx_graphs` or `graphs`.

Each graph should contain metadata with an energy label, preferably:

- `graph.graph["metadata"]["best_energy"]`

For family-specific plots, use graph names that include the family prefix, for example:

- `fhgraph_0003.gpickle`
- `h2graph_0121.gpickle`
- `hehgraph_0007.gpickle`
- `xyzgraph_0420.gpickle`

The script infers the family from the graph name.

## Train

```bash
python main.py train   --train_data /path/to/train_graphs   --out_dir runs/exp1   --epochs 200   --batch_size 64   --lr 0.01
```

If you already have separate validation data:

```bash
python main.py train   --train_data /path/to/train_graphs   --val_data /path/to/val_graphs   --out_dir runs/exp1
```

## Test

```bash
python main.py test   --test_data /path/to/test_graphs   --checkpoint runs/exp1/best_model.pt   --out_dir runs/test_exp1
```

## Outputs

Training saves:
- `training_metrics.png`
- `validation_metrics_summary.png`
- `validation_actual_vs_predicted.png`
- `validation_family_summary.png`
- `validation_family_plots/`
- `validation_predictions.csv`
- `validation_family_metrics.csv`
- `best_model.pt`

Testing saves:
- `test_metrics_summary.png`
- `test_actual_vs_predicted.png`
- `test_family_summary.png`
- `test_family_plots/`
- `test_predictions.csv`
- `test_family_metrics.csv`

`accuracy` here is a regression accuracy defined as the percentage of predictions within `--acc_tol` absolute error. The default is `0.1`.
