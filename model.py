#Author : Nithish Sriram Srinivasan.
# This is the model for the GNN.
#Date : 18/06/2026
#It is a GraphSAGE- style Mean Aggregation GNN with a skip connection and layer normalization.

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: Optional[int] = None) -> torch.Tensor:
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1 # This takes the maximum value of the batch and adds 1 to get the number of graphs.
    out = torch.zeros((num_graphs, x.size(-1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x) # This adds the features of the graph to the output.
    counts = torch.zeros((num_graphs,), device=x.device, dtype=x.dtype)# This is to count number of nodes in each graphs.
    counts.index_add_(0, batch, torch.ones((batch.numel(),), device=x.device, dtype=x.dtype))
    return out / counts.clamp(min=1.0).unsqueeze(-1) # clamp is to avoid division by zero. 



class GraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1): #constructor for the GraphConvLayer class.
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim) # creates  W*x + b where W is the weight matrix and b is the bias vector.
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity() #Residual connection 
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        if edge_index.numel() > 0:
            src, dst = edge_index
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, h[src]) # This acts as message passing from source to destination.
            deg = torch.zeros((h.size(0),), device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones((dst.numel(),), device=h.device, dtype=h.dtype))
            h = h + agg / deg.clamp(min=1.0).unsqueeze(-1)
        h = self.norm(h)
        h = F.leaky_relu(h, 0.02)
        h = self.dropout(h)
        return h

  ''''  
 This is the flow of the GraphConvLayer Class.
  Node Features
       │
       ▼

Linear Layer

       │
       ▼

Message Passing

       │
       ▼

Neighbor Aggregation

       │
       ▼

Own Feature +
Neighbor Feature

       │
       ▼

LayerNorm

       │
       ▼

LeakyReLU

       │
       ▼

Dropout

       │
       ▼

Output Features
'''




class QuantumCircuitGNN(nn.Module):
    def __init__(self, node_dim: int = 17, global_dim: int = 14, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden_dim) # Learned node embeddings for the first layer.
        self.convs = nn.ModuleList([GraphConvLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)]) # Important for message passing between nodes.
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.LeakyReLU(0.02),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.02),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 32, 256),
            nn.LeakyReLU(0.02),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.02),
            nn.Linear(64, 1),
        ) # This is responsible for the prediction of energy. Without this , the model will not be able to predict the energy.

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.node_in(x), 0.02)
        
        ''' Layer 1 : Node sees neighbors
            Layer 2: Node sees neighbors of neighbors
            Layer 4: Node sees much of the graph'''
        
        for conv in self.convs:
            h = conv(h, edge_index)

        pooled = global_mean_pool(h, batch) # This converts many node features into a single graph vector.
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        g = self.global_mlp(global_features)
        if pooled.size(0) != g.size(0):
            if pooled.size(0) == 1 and g.size(0) > 1:
                pooled = pooled.expand(g.size(0), -1)
            elif g.size(0) == 1 and pooled.size(0) > 1:
                g = g.expand(pooled.size(0), -1)
            else:
                raise ValueError(f'Batch mismatch: pooled={pooled.size(0)} global={g.size(0)}')
        out = torch.cat([pooled, g], dim=-1) # This concatenates the graph vector and metadata into a single vector . 128 + 32 = 160.
        return self.head(out).squeeze(-1)

      
      
        ''' This is the flow of the QuantumCircuitGNN. 
                         Quantum Circuit Graph

              Node Features (17)

                       │
                       ▼

                Node Encoder
              Linear(17→128)

                       │
                       ▼

                  GNN Layer 1

                       │
                       ▼

                  GNN Layer 2

                       │
                       ▼

                  GNN Layer 3

                       │
                       ▼

                  GNN Layer 4

                       │
                       ▼

               Global Mean Pool

                       │

             Graph Vector (128)

                       │
                       │
                       ▼

        Metadata Features (14)

                       │

                 Global MLP

                       │

              Metadata (32)

                       │
                       ▼

                Concatenate

                128 + 32

                       │
                       ▼

                  Linear 256

                       │
                       ▼

                  Linear 128

                       │
                       ▼

                   Linear 64

                       │
                       ▼

                    Linear 1

                       │
                       ▼

               Predicted Energy'''


