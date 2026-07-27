"""
src/models/gcn.py

Arquitetura GCN (Graph Convolutional Network) para classificação molecular.

Fluxo de uma molécula pela rede:
    x (num_atoms, 6)
        ↓ GCNConv + BatchNorm + ReLU + Dropout  (× num_layers)
    x (num_atoms, hidden_channels)
        ↓ Global Mean Pooling
    x (1, hidden_channels)
        ↓ Linear
    logit (1,)  →  sigmoid  →  P(BBB+)

A saída é um logit bruto (sem sigmoid). A sigmoid é aplicada fora
durante a avaliação, ou implicitamente pelo BCEWithLogitsLoss no treino.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(nn.Module):
    """
    Graph Convolutional Network para classificação binária molecular.

    Args:
        in_channels:     número de features de entrada por átomo (padrão 6,
                         conforme mol_to_graph.py).
        hidden_channels: dimensão das representações internas (padrão 64).
        num_layers:      número de camadas de message passing (padrão 3).
        dropout:         taxa de dropout aplicada após cada camada (padrão 0.3).
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # primeira camada: projeta de in_channels para hidden_channels
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # camadas seguintes: hidden_channels → hidden_channels
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout    = dropout
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          features de nó, shape (num_atoms, in_channels).
            edge_index: conectividade, shape (2, num_edges).
            batch:      vetor que mapeia cada átomo à sua molécula no batch,
                        shape (num_atoms,). Fornecido automaticamente pelo
                        DataLoader do PyG.

        Returns:
            Logits, shape (num_graphs,). Um valor por molécula no batch.
            Aplique sigmoid para obter probabilidades.
        """
        # ── message passing ──────────────────────────────────────────────────
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)   # agrega vizinhos
            x = bn(x)                  # estabiliza distribuição
            x = F.relu(x)             # não-linearidade
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ── pooling: átomos → molécula ───────────────────────────────────────
        x = global_mean_pool(x, batch)   # (num_graphs, hidden_channels)

        # ── classificação ────────────────────────────────────────────────────
        return self.classifier(x).squeeze(-1)   # (num_graphs,)
