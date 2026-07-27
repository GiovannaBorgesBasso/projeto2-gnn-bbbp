"""
Teste da Etapa 4 — scaffold split estratificado + DataLoader.

Como rodar (da raiz do projeto):
    python teste_etapa4.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch_geometric.loader import DataLoader
from src.data.dataset import BBBPDataset
from src.utils.splits import scaffold_split

print("=" * 50)
print("TESTE 1: scaffold split estratificado")
print("=" * 50)

dataset = BBBPDataset()
train, val, test = scaffold_split(dataset, dataset.smiles_list, dataset.labels)

assert len(train) + len(val) + len(test) == len(dataset), "Perda de moléculas no split"
assert len(train) > len(test), "Treino deve ser maior que teste"
print("  ✓ Nenhuma molécula perdida no split")

train_ids = set(id(d) for d in train)
val_ids   = set(id(d) for d in val)
test_ids  = set(id(d) for d in test)
assert len(train_ids & val_ids) == 0
assert len(train_ids & test_ids) == 0
assert len(val_ids & test_ids) == 0
print("  ✓ Sem overlap entre os conjuntos")

val_labels  = torch.cat([d.y for d in val])
test_labels = torch.cat([d.y for d in test])
assert val_labels.sum() > 0 and (val_labels == 0).sum() > 0
assert test_labels.sum() > 0 and (test_labels == 0).sum() > 0
print("  ✓ Ambas as classes presentes em val e teste")

print()
print("=" * 50)
print("TESTE 2: DataLoader + batch")
print("=" * 50)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader   = DataLoader(val,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test,  batch_size=32, shuffle=False)

batch = next(iter(train_loader))
print(f"Primeiro batch: {batch.num_graphs} moléculas | x: {tuple(batch.x.shape)} | y: {tuple(batch.y.shape)}")

assert batch.x.shape[1] == 6
assert batch.edge_attr.shape[1] == 3
print("  ✓ Shapes do batch corretos")

print(f"\nBatches — Treino: {len(train_loader)} | Val: {len(val_loader)} | Teste: {len(test_loader)}")

print()
print("=" * 50)
print("TODOS OS TESTES PASSARAM")
print("=" * 50)