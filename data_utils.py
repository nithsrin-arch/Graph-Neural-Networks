#Author : Nithish Sriram Srinivasan
#Date : 18/06/2026
# This is the data utils for the GNN.

from __future__ import annotations

import ast
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import torch

NODE_FEATURE_DIM = 17
GLOBAL_FEATURE_DIM = 14
GATE_TYPE_VOCAB = ["H", "RX", "RY", "RZ", "CZ", "CNOT", "OTHER"]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(default if x is None else x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(default if x is None else x)
    except Exception:
        return int(default)


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def _is_graph(obj: Any) -> bool:
    return isinstance(obj, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))


def _extract_graph_list(obj: Any) -> List[nx.Graph]:
    if _is_graph(obj):
        return [obj]
    if isinstance(obj, list):
        return [g for g in obj if _is_graph(g)]
    if isinstance(obj, tuple):
        return [g for g in obj if _is_graph(g)]
    if isinstance(obj, dict):
        for key in ('networkx_graphs', 'graphs', 'graph_list', 'data'):
            val = obj.get(key)
            if _is_graph(val):
                return [val]
            if isinstance(val, list) and val and _is_graph(val[0]):
                return list(val)
    raise ValueError('Unsupported pickle structure. Expected graph(s).')


def _infer_source_family_from_filename(path: Union[str, Path]) -> str:
    stem = Path(str(path)).stem
    base = stem.lower()
    if '|' in base:
        base = base.split('|', 1)[0]
    for pref, fam in (
        ('fh8q', 'FH'),
        ('fhaph', 'FH'),
        ('h2', 'H2'),
        ('heh', 'HeH+'),
        ('xyz', 'XYZ'),
        ('4q', '4Q'),
    ):
        if base.startswith(pref):
            return fam
    m = re.match(r'^(?P<fam>[a-z0-9+]+?)(?:graph|_|$)', base)
    if m:
        return m.group('fam').upper()
    return base[:32].upper() or 'GRAPH'


def _infer_depth_label(text: Any) -> str:
    m = re.search(r'(depth\d+)', str(text).lower())
    return m.group(1) if m else 'unknown_depth'


def _annotate_loaded_graph(g: nx.Graph, source: Union[str, Path]) -> nx.Graph:
    meta = {}
    if hasattr(g, 'graph') and isinstance(g.graph, dict):
        meta = g.graph.get('metadata', {})
        if not isinstance(meta, dict):
            meta = {}
            g.graph['metadata'] = meta
    else:
        g.graph = {'metadata': meta}
    source_path = Path(source)
    meta.setdefault('source_file', source_path.name)
    meta.setdefault('source_stem', source_path.stem)
    meta.setdefault('source_family', _infer_source_family_from_filename(source_path.name))
    return g


def load_graphs_from_source(source: Union[str, Path]) -> List[nx.Graph]:
    source = Path(source)
    if source.is_dir():
        paths = sorted([p for p in source.rglob('*') if p.suffix in {'.gpickle', '.pkl', '.pickle'}])
        graphs: List[nx.Graph] = []
        for p in paths:
            try:
                loaded = _extract_graph_list(load_pickle(p))
                for g in loaded:
                    graphs.append(_annotate_loaded_graph(g, p))
            except Exception:
                continue
        if not graphs:
            raise FileNotFoundError(f'No graphs found in directory: {source}')
        return graphs
    if not source.exists():
        raise FileNotFoundError(source)
    loaded = _extract_graph_list(load_pickle(source))
    return [_annotate_loaded_graph(g, source) for g in loaded]


def _graph_metadata(g: nx.Graph) -> Dict[str, Any]:
    meta = {}
    if hasattr(g, 'graph') and isinstance(g.graph, dict):
        meta = dict(g.graph.get('metadata', {}))
        for k, v in g.graph.items():
            if k != 'metadata' and k not in meta:
                meta[k] = v
    return meta


def _parse_node_id(node: Any) -> Tuple[str, int]:
    s = str(node)
    if s.startswith('q'):
        try:
            return 'qubit', int(s[1:])
        except Exception:
            return 'qubit', 0
    if s.startswith('g'):
        try:
            return 'gate', int(s[1:])
        except Exception:
            return 'gate', 0
    return 'gate', 0


def _gate_type_one_hot(gate_type: str) -> List[float]:
    gate_type = str(gate_type).upper()
    vec = [0.0] * len(GATE_TYPE_VOCAB)
    vec[GATE_TYPE_VOCAB.index(gate_type)] = 1.0 if gate_type in GATE_TYPE_VOCAB else 1.0
    if gate_type not in GATE_TYPE_VOCAB:
        vec = [0.0] * len(GATE_TYPE_VOCAB)
        vec[-1] = 1.0
    return vec


def build_node_features(g: nx.Graph) -> Tuple[np.ndarray, List[Any]]:
    nodes = list(g.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    meta = _graph_metadata(g)
    num_nodes = max(1, len(nodes))
    max_layer = max([_safe_int(g.nodes[n].get('layer_index', g.nodes[n].get('layer', 0))) for n in nodes] + [1])
    max_block = max([_safe_int(g.nodes[n].get('block_index', g.nodes[n].get('block', 0))) for n in nodes] + [1])

    degrees = {n: 0 for n in nodes}
    for u, v in g.edges():
        degrees[u] = degrees.get(u, 0) + 1
        degrees[v] = degrees.get(v, 0) + 1

    feats: List[List[float]] = []
    for n in nodes:
        attrs = g.nodes[n]
        node_type, node_num = _parse_node_id(n)
        is_qubit = 1.0 if (attrs.get('node_type') == 'qubit' or node_type == 'qubit') else 0.0
        is_gate = 1.0 - is_qubit
        idx_norm = float(node_to_idx[n] / max(1, num_nodes - 1))
        deg_norm = float(degrees.get(n, 0) / max(1, num_nodes - 1))

        gate_type = str(attrs.get('gate_type', 'OTHER')).upper()
        gate_oh = _gate_type_one_hot(gate_type)

        qubits = attrs.get('qubits', ())
        if isinstance(qubits, str):
            try:
                qubits = ast.literal_eval(qubits)
            except Exception:
                qubits = ()
        if not isinstance(qubits, (list, tuple)):
            qubits = tuple(qubits) if qubits is not None else ()
        arity_norm = float(len(qubits) / 2.0)

        layer = _safe_int(attrs.get('layer_index', attrs.get('layer', 0)))
        block = _safe_int(attrs.get('block_index', attrs.get('block', 0)))
        layer_norm = float(layer / max(1, max_layer))
        block_norm = float(block / max(1, max_block))

        params = attrs.get('params', None)
        theta = 0.0
        if isinstance(params, (list, tuple)) and len(params) > 0:
            theta = _safe_float(params[0], 0.0)

        q0_norm = 0.0
        q1_norm = 0.0
        if node_type == 'qubit':
            q0_norm = float(node_num / max(1, meta.get('num_qubits', 1) - 1))
        elif len(qubits) >= 1:
            q0_norm = float(_safe_int(qubits[0]) / max(1, meta.get('num_qubits', 1) - 1))
            if len(qubits) >= 2:
                q1_norm = float(_safe_int(qubits[1]) / max(1, meta.get('num_qubits', 1) - 1))

        vec = [is_qubit, is_gate, idx_norm, deg_norm, *gate_oh, arity_norm, layer_norm, block_norm, theta, q0_norm, q1_norm]
        assert len(vec) == NODE_FEATURE_DIM, len(vec)
        feats.append(vec)

    return np.asarray(feats, dtype=np.float32), nodes


def build_global_features(g: nx.Graph) -> np.ndarray:
    meta = _graph_metadata(g)
    feats = np.asarray([
        float(g.number_of_nodes()),
        float(g.number_of_edges()),
        _safe_float(meta.get('num_qubits', 0.0)),
        _safe_float(meta.get('num_gates', 0.0)),
        _safe_float(meta.get('num_single_qubit_gates', 0.0)),
        _safe_float(meta.get('num_two_qubit_gates', 0.0)),
        _safe_float(meta.get('depth', 0.0)),
        _safe_float(meta.get('proxy_score', 0.0)),
        _safe_float(meta.get('mean_best_energy', 0.0)),
        _safe_float(meta.get('best_restart', 0.0)),
        _safe_float(meta.get('learning_rate', 0.0)),
        _safe_float(meta.get('n_epochs', 0.0)),
        _safe_float(meta.get('batch_size', 0.0)),
        _safe_float(meta.get('num_restarts', 0.0)),
    ], dtype=np.float32)
    assert feats.shape[0] == GLOBAL_FEATURE_DIM
    return feats


def get_label(g: nx.Graph) -> float:
    meta = _graph_metadata(g)
    for key in ('best_energy', 'energy', 'label', 'y'):
        if key in meta:
            return _safe_float(meta[key], 0.0)
    if 'y' in g.graph:
        return _safe_float(g.graph['y'], 0.0)
    return 0.0


@dataclass
class GraphSample:
    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    global_features: torch.Tensor
    y: torch.Tensor
    graph_name: str


def nx_to_sample(g: nx.Graph) -> GraphSample:
    x_np, nodes = build_node_features(g)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges: List[Tuple[int, int]] = []
    for u, v in g.edges():
        if u in node_to_idx and v in node_to_idx:
            ui = node_to_idx[u]; vi = node_to_idx[v]
            edges.append((ui, vi)); edges.append((vi, ui))
    edge_index = torch.tensor(np.asarray(edges, dtype=np.int64).T, dtype=torch.long) if edges else torch.zeros((2, 0), dtype=torch.long)
    global_features = torch.tensor(build_global_features(g), dtype=torch.float32)
    label = torch.tensor([get_label(g)], dtype=torch.float32)
    batch = torch.zeros(x_np.shape[0], dtype=torch.long)
    meta = _graph_metadata(g)
    family = str(meta.get('source_family', meta.get('family', _infer_source_family_from_filename(meta.get('source_file', meta.get('source_stem', meta.get('architecture_name', meta.get('graph_name', 'graph'))))))))
    arch_name = str(meta.get('architecture_name', meta.get('graph_name', 'graph')))
    depth_label = _infer_depth_label(arch_name)
    graph_name = f"{family}::{depth_label}::{arch_name}"
    return GraphSample(torch.tensor(x_np, dtype=torch.float32), edge_index, batch, global_features, label, graph_name)


def fit_normalizer(samples: Sequence[GraphSample]) -> Dict[str, np.ndarray]:
    x_all = np.concatenate([s.x.numpy() for s in samples], axis=0)
    g_all = np.stack([s.global_features.numpy() for s in samples], axis=0)
    x_mean = x_all.mean(axis=0); x_std = x_all.std(axis=0); x_std[x_std < 1e-8] = 1.0
    g_mean = g_all.mean(axis=0); g_std = g_all.std(axis=0); g_std[g_std < 1e-8] = 1.0
    return {'x_mean': x_mean.astype(np.float32), 'x_std': x_std.astype(np.float32), 'g_mean': g_mean.astype(np.float32), 'g_std': g_std.astype(np.float32)}


def apply_normalizer(samples: Sequence[GraphSample], stats: Dict[str, np.ndarray]) -> List[GraphSample]:
    out: List[GraphSample] = []
    x_mean = torch.tensor(stats['x_mean'], dtype=torch.float32)
    x_std = torch.tensor(stats['x_std'], dtype=torch.float32)
    g_mean = torch.tensor(stats['g_mean'], dtype=torch.float32)
    g_std = torch.tensor(stats['g_std'], dtype=torch.float32)
    for s in samples:
        out.append(GraphSample((s.x - x_mean) / x_std, s.edge_index.clone(), s.batch.clone(), (s.global_features - g_mean) / g_std, s.y.clone(), s.graph_name))
    return out


def split_graphs(graphs: Sequence[nx.Graph], val_ratio: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(graphs)); rng.shuffle(idx)
    n_val = max(1, int(round(len(graphs) * val_ratio)))
    val_idx = idx[:n_val]; train_idx = idx[n_val:]
    return [graphs[i] for i in train_idx], [graphs[i] for i in val_idx]


def collate_graph_samples(samples: Sequence[GraphSample]) -> GraphSample:
    if len(samples) == 0:
        raise ValueError('Empty batch')
    xs=[]; edge_indices=[]; batches=[]; globals_=[]; ys=[]; names=[]; node_offset=0
    for i,s in enumerate(samples):
        xs.append(s.x)
        if s.edge_index.numel() > 0:
            edge_indices.append(s.edge_index + node_offset)
        batches.append(torch.full((s.x.size(0),), i, dtype=torch.long))
        globals_.append(s.global_features); ys.append(s.y.view(1)); names.append(s.graph_name)
        node_offset += s.x.size(0)
    x = torch.cat(xs, dim=0)
    batch = torch.cat(batches, dim=0)
    global_features = torch.stack(globals_, dim=0)
    y = torch.cat(ys, dim=0).view(-1)
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2,0), dtype=torch.long)
    return GraphSample(x, edge_index, batch, global_features, y, '||'.join(names))


def load_dataset(source: Union[str, Path]) -> List[nx.Graph]:
    return load_graphs_from_source(source)
