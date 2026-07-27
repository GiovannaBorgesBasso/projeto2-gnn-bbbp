"""
src/training/train_gat.py

Loop de treino da GAT para o dataset BBBP.

Como rodar (da raiz do projeto):
    python src/training/train_gat.py

Diferenças em relação ao train.py (GCN):
    - Usa GATConv com 4 cabeças de atenção
    - hidden_channels=128 (vs 64 da GCN) para maior capacidade
    - Checkpoint salvo em models/checkpoints/gat_best.pt
    - Métricas salvas em results/metrics/gat_training.json
"""

import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.dataset import BBBPDataset
from src.utils.splits import scaffold_split
from src.models.gat import GAT

# ── Hiperparâmetros ──────────────────────────────────────────────────────────
HIDDEN_CHANNELS = 128
NUM_LAYERS      = 3
HEADS           = 4
DROPOUT         = 0.3
BATCH_SIZE      = 32
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
NUM_EPOCHS      = 100
PATIENCE        = 15
POS_WEIGHT      = 3.29

CHECKPOINT_PATH = "models/checkpoints/gat_best.pt"
METRICS_PATH    = "results/metrics/gat_training.json"


# ── Funções de treino e avaliação ────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.sigmoid(out).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch.y.cpu().numpy())
    return roc_auc_score(all_labels, all_probs)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # 1. Dados — mesmo split da GCN para comparação justa
    dataset = BBBPDataset()
    train_data, val_data, test_data = scaffold_split(
        dataset, dataset.smiles_list, dataset.labels
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    # 2. Modelo
    model = GAT(
        in_channels=6,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
        heads=HEADS,
        dropout=DROPOUT,
    ).to(device)
    print(f"Parâmetros treináveis: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Loss, otimizador e scheduler
    pos_weight = torch.tensor([POS_WEIGHT]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    # 4. Loop de treino
    print(f"\nTreinando GAT por até {NUM_EPOCHS} épocas (early stopping={PATIENCE})...\n")
    best_val_auc      = 0.0
    epochs_no_improve = 0
    history           = []

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auc    = evaluate(model, val_loader, device)
        scheduler.step(val_auc)
        elapsed    = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_auc": round(val_auc, 4),
        })

        print(f"Época {epoch:3d} | loss={train_loss:.4f} | val_auc={val_auc:.4f} | {elapsed:.1f}s")

        if val_auc > best_val_auc:
            best_val_auc      = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"           ✓ Melhor modelo salvo (val_auc={best_val_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping na época {epoch} (sem melhora por {PATIENCE} épocas).")
                break

    # 5. Avaliação final no teste com o melhor modelo
    print(f"\nCarregando melhor modelo (val_auc={best_val_auc:.4f})...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    test_auc = evaluate(model, test_loader, device)
    print(f"AUC-ROC no teste: {test_auc:.4f}")

    # 6. Salvar métricas
    results = {
        "modelo": "GAT",
        "hiperparametros": {
            "hidden_channels": HIDDEN_CHANNELS,
            "num_layers": NUM_LAYERS,
            "heads": HEADS,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "melhor_val_auc": round(best_val_auc, 4),
        "test_auc": round(test_auc, 4),
        "historico": history,
    }
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Métricas salvas em {METRICS_PATH}")

    # 7. Comparação final
    print(f"\n{'='*45}")
    print(f"COMPARAÇÃO FINAL")
    print(f"{'='*45}")
    print(f"RF  + ECFP4:  AUC = 0.8420")
    print(f"SVM + ECFP4:  AUC = 0.8564")
    print(f"GCN (grafos): AUC = 0.8366")
    print(f"GAT (grafos): AUC = {test_auc:.4f}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
