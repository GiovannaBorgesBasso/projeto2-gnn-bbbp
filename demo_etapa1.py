"""
Etapa 1 — Demonstração: de SMILES a um objeto Data do PyTorch Geometric.

Objetivo: entender, na mão, como uma molécula vira um grafo, antes de
automatizar isso para o dataset BBBP inteiro (Etapa 3).
"""

from rdkit import Chem
import torch
from torch_geometric.data import Data

# Etanol: C-C-O (3 átomos pesados, 2 ligações)
smiles = "CCO"
mol = Chem.MolFromSmiles(smiles)

print(f"SMILES: {smiles}")
print(f"Número de átomos (pesados): {mol.GetNumAtoms()}")
print(f"Número de ligações: {mol.GetNumBonds()}")
print()

# --- 1. Features de nó (x) ---
# Para cada átomo, vamos extrair só 2 features simples por enquanto
# (na Etapa 3 vamos expandir isso bastante).
node_features = []
for atom in mol.GetAtoms():
    node_features.append([
        atom.GetAtomicNum(),   # número atômico (C=6, O=8, N=7...)
        atom.GetDegree(),      # quantos vizinhos esse átomo tem
    ])
    print(f"Átomo {atom.GetIdx()}: {atom.GetSymbol()} "
          f"(num. atômico={atom.GetAtomicNum()}, grau={atom.GetDegree()})")

x = torch.tensor(node_features, dtype=torch.float)
print(f"\nx (features de nó), shape {tuple(x.shape)}:")
print(x)

# --- 2. edge_index: quais átomos estão ligados ---
# IMPORTANTE: o grafo molecular é não-direcionado (ligação C-C não tem
# "sentido"), então cada ligação vira DUAS entradas em edge_index:
# (origem, destino) e (destino, origem).
edge_index_list = []
edge_attr_list = []
for bond in mol.GetBonds():
    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    bond_type = bond.GetBondTypeAsDouble()  # 1.0=simples, 2.0=dupla, 1.5=aromática

    edge_index_list += [[i, j], [j, i]]      # ida e volta
    edge_attr_list += [[bond_type], [bond_type]]

    print(f"Ligação: átomo {i} -- átomo {j} (tipo={bond_type})")

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

print(f"\nedge_index, shape {tuple(edge_index.shape)}:")
print(edge_index)
print(f"\nedge_attr (tipo de cada ligação), shape {tuple(edge_attr.shape)}:")
print(edge_attr)

# --- 3. Montar o objeto Data ---
y = torch.tensor([1.0])  # rótulo fictício (ex: 1 = atravessa BBB)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

print(f"\nObjeto Data completo:\n{data}")
print(f"\nValidação automática do PyG: data.validate() -> {data.validate()}")