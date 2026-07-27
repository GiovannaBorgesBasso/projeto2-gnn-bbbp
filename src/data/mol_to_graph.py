"""
src/data/mol_to_graph.py

Converte um SMILES em um objeto Data do PyTorch Geometric.

Esta é a função central do pipeline: tudo que entra na GNN passa por aqui.
Chamada pela BBBPDataset (dataset.py) para cada molécula do dataset.

Features de nó (x) — 6 features por átomo:
    0: número atômico        (C=6, N=7, O=8, ...)
    1: grau                  (número de vizinhos no grafo)
    2: número de H ligados   (implícitos + explícitos)
    3: carga formal          (0 para a maioria; ±1 para íons)
    4: aromaticidade         (1 se aromático, 0 caso contrário)
    5: hibridização          (inteiro: SP=2, SP2=3, SP3=4, ...)

Features de aresta (edge_attr) — 3 features por aresta:
    0: tipo de ligação       (1.0=simples, 2.0=dupla, 1.5=aromática)
    1: pertence a anel       (1 se sim, 0 se não)
    2: é conjugada           (1 se sim, 0 se não)
"""

from rdkit import Chem
import torch
from torch_geometric.data import Data


def smiles_to_graph(smiles: str, label: int | None = None) -> Data | None:
    """
    Converte um SMILES em um objeto Data do PyG.

    Args:
        smiles: string SMILES da molécula.
        label:  rótulo binário (1 = BBB+, 0 = BBB-). Opcional.

    Returns:
        Objeto Data com x, edge_index, edge_attr e y,
        ou None se o SMILES for inválido.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ── Features de nó (x) ──────────────────────────────────────────────────
    node_features = []
    for atom in mol.GetAtoms():
        node_features.append([
            atom.GetAtomicNum(),           # qual elemento é
            atom.GetDegree(),              # quantos vizinhos tem
            atom.GetTotalNumHs(),          # quantos H estão ligados
            atom.GetFormalCharge(),        # carga elétrica formal
            int(atom.GetIsAromatic()),     # está em sistema aromático?
            int(atom.GetHybridization()),  # tipo de hibridização (SP/SP2/SP3)
        ])

    x = torch.tensor(node_features, dtype=torch.float)

    # ── Features de aresta (edge_index + edge_attr) ──────────────────────────
    edge_index_list = []
    edge_attr_list  = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = [
            bond.GetBondTypeAsDouble(),    # tipo: 1.0, 2.0, 1.5...
            int(bond.IsInRing()),          # está em anel?
            int(bond.GetIsConjugated()),   # é conjugada?
        ]
        # cada ligação física vira duas arestas direcionadas (ida + volta)
        edge_index_list += [[i, j], [j, i]]
        edge_attr_list  += [feats, feats]

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr_list,  dtype=torch.float)

    # ── Rótulo (y) ───────────────────────────────────────────────────────────
    y = torch.tensor([label], dtype=torch.float) if label is not None else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
