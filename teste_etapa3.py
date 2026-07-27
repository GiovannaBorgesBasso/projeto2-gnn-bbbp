"""
Teste da Etapa 3 — valida o pipeline SMILES → grafo para o dataset inteiro.

Como rodar (da raiz do projeto):
    python teste_etapa3.py

O que este script verifica:
    1. smiles_to_graph() converte corretamente moléculas simples
    2. BBBPDataset carrega as 2039 moléculas sem erros
    3. Shapes dos tensores estão corretos
    4. Os rótulos batem com o EDA (76.5% BBB+)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.mol_to_graph import smiles_to_graph
from src.data.dataset import BBBPDataset

print("=" * 50)
print("TESTE 1: smiles_to_graph() em moléculas simples")
print("=" * 50)

# Etanol
data = smiles_to_graph("CCO", label=1)
print(f"Etanol:     {data}")
assert data.x.shape == (3, 6),         f"Esperado (3,6), obtido {data.x.shape}"
assert data.edge_index.shape == (2, 4), f"Esperado (2,4), obtido {data.edge_index.shape}"
assert data.edge_attr.shape == (4, 3),  f"Esperado (4,3), obtido {data.edge_attr.shape}"
assert data.validate(), "Objeto Data inválido"
print("  ✓ shapes corretos, validate() = True")

# Benzeno
data = smiles_to_graph("c1ccccc1", label=0)
print(f"Benzeno:    {data}")
assert data.x.shape == (6, 6),          f"Esperado (6,6), obtido {data.x.shape}"
assert data.edge_index.shape == (2, 12), f"Esperado (2,12), obtido {data.edge_index.shape}"
assert data.validate(), "Objeto Data inválido"
print("  ✓ shapes corretos, validate() = True")

# SMILES inválido
data = smiles_to_graph("INVALIDO")
assert data is None, "SMILES inválido deveria retornar None"
print("  ✓ SMILES inválido retornou None corretamente")

print()
print("=" * 50)
print("TESTE 2: BBBPDataset — dataset completo")
print("=" * 50)

dataset = BBBPDataset()

assert len(dataset) > 2000, f"Esperado >2000 moléculas, obtido {len(dataset)}"
print(f"  ✓ Total de moléculas: {len(dataset)}")

# verificar shapes da primeira molécula
d = dataset[0]
assert d.x.shape[1] == 6,        f"Esperado 6 features de nó, obtido {d.x.shape[1]}"
assert d.edge_attr.shape[1] == 3, f"Esperado 3 features de aresta, obtido {d.edge_attr.shape[1]}"
assert d.y is not None,           "Rótulo y não deveria ser None"
print(f"  ✓ Features de nó:    {d.x.shape[1]} (por átomo)")
print(f"  ✓ Features de aresta: {d.edge_attr.shape[1]} (por ligação)")

# verificar distribuição de classes
import torch
labels = torch.cat([data.y for data in dataset])
bbb_pos = (labels == 1).sum().item()
bbb_neg = (labels == 0).sum().item()
proporcao = bbb_pos / len(dataset) * 100
print(f"  ✓ BBB+: {bbb_pos} ({proporcao:.1f}%) | BBB-: {bbb_neg} ({100-proporcao:.1f}%)")
assert 74 < proporcao < 79, f"Proporção BBB+ fora do esperado: {proporcao:.1f}%"

print()
print("=" * 50)
print("TODOS OS TESTES PASSARAM")
print("=" * 50)
