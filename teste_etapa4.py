"""
Teste da Etapa 4 — scaffold split + DataLoader.

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
print("TESTE 1: scaffold split")
print("=" * 50)

dataset = BBBPDataset()
train, val, test = scaffold_split(dataset, dataset.smiles_list)

# tamanhos esperados
assert len(train) + len(val) + len(test) == len(dataset), "Perda de moléculas no split"
assert len(train) > len(test), "Treino deve ser maior que teste"
print("  ✓ Nenhuma molécula perdida no split")

# sem overlap: verificar que nenhum grafo aparece em dois conjuntos
# (usamos id() do objeto Python como proxy de identidade)
train_ids = set(id(d) for d in train)
val_ids   = set(id(d) for d in val)
test_ids  = set(id(d) for d in test)
assert len(train_ids & val_ids) == 0,  "Overlap entre treino e validação"
assert len(train_ids & test_ids) == 0, "Overlap entre treino e teste"
assert len(val_ids & test_ids) == 0,   "Overlap entre validação e teste"
print("  ✓ Sem overlap entre os conjuntos")

print()
print("=" * 50)
print("TESTE 2: DataLoader + batch")
print("=" * 50)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader   = DataLoader(val,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test,  batch_size=32, shuffle=False)

# pegar o primeiro batch do treino e inspecionar
batch = next(iter(train_loader))

print(f"Primeiro batch do treino:")
print(f"  num_graphs:       {batch.num_graphs}  ← moléculas no batch")
print(f"  x shape:          {tuple(batch.x.shape)}  ← todos os átomos empilhados")
print(f"  edge_index shape: {tuple(batch.edge_index.shape)}")
print(f"  edge_attr shape:  {tuple(batch.edge_attr.shape)}")
print(f"  y shape:          {tuple(batch.y.shape)}  ← um rótulo por molécula")
print(f"  batch tensor:     {batch.batch[:10]}...  ← qual molécula cada átomo pertence")

assert batch.num_graphs == 32,         "Batch size incorreto"
assert batch.x.shape[1] == 6,         "Número de features de nó incorreto"
assert batch.edge_attr.shape[1] == 3, "Número de features de aresta incorreto"
assert batch.y.shape[0] == 32,        "Número de rótulos incorreto"
print("  ✓ Shapes do batch corretos")

# verificar que batch tensor mapeia cada átomo à molécula correta
assert batch.batch.min().item() == 0
assert batch.batch.max().item() == 31
print("  ✓ batch tensor cobre todas as 32 moléculas")

# número de batches em cada loader
n_train_batches = len(train_loader)
n_val_batches   = len(val_loader)
n_test_batches  = len(test_loader)
print(f"\nNúmero de batches (batch_size=32):")
print(f"  Treino:    {n_train_batches}")
print(f"  Validação: {n_val_batches}")
print(f"  Teste:     {n_test_batches}")

print()
print("=" * 50)
print("TODOS OS TESTES PASSARAM")
print("=" * 50)
