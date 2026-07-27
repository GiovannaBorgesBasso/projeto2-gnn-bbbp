"""
src/models/gat.py

Arquitetura GAT (Graph Attention Network) para classificação molecular.

Diferença principal em relação à GCN:
    GCN agrega vizinhos com pesos fixos baseados na estrutura do grafo.
    GAT aprende coeficientes de atenção durante o treino — cada átomo
    decide quais vizinhos são mais relevantes para ele naquele contexto.

Multi-head attention:
    Usamos `heads=4` cabeças de atenção independentes. Cada cabeça aprende
    um padrão de atenção diferente. As saídas são concatenadas, por isso
    usamos hidden_channels // heads como dimensão por cabeça — para que
    a dimensão final continue sendo hidden_channels.

Fluxo de uma molécula pela rede:
    x (num_atoms, 6)
        ↓ GATConv + BatchNorm + ReLU + Dropout  (× num_layers)
    x (num_atoms, hidden_channels)
        ↓ Global Mean Pooling
    x (1, hidden_channels)
        ↓ Linear
    logit (1,)  →  sigmoid  →  P(BBB+)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(nn.Module):
    """
    Graph Attention Network para classificação binária molecular.

    Args:
        in_channels:     número de features de entrada por átomo (padrão 6).
        hidden_channels: dimensão das representações internas (padrão 128).
                         Deve ser divisível por `heads`.
        num_layers:      número de camadas de message passing (padrão 3).
        heads:           número de cabeças de atenção por camada (padrão 4).
        dropout:         taxa de dropout aplicada após cada camada (padrão 0.3).
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert hidden_channels % heads == 0, \
            f"hidden_channels ({hidden_channels}) deve ser divisível por heads ({heads})"

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # primeira camada: in_channels → hidden_channels
        self.convs.append(
            GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout)
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # camadas seguintes: hidden_channels → hidden_channels
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout)
            )
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
            batch:      vetor que mapeia cada átomo à sua molécula no batch.

        Returns:
            Logits, shape (num_graphs,). Aplique sigmoid para probabilidades.
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)   # atenção + agregação
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        return self.classifier(x).squeeze(-1)
