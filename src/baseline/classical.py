"""
src/baseline/classical.py

Baseline clássico: ECFP4 + Random Forest e SVM no mesmo scaffold split da GNN.

Por que este baseline existe:
    A comparação GNN vs. modelo clássico só é válida se ambos forem avaliados
    no exato mesmo protocolo: mesmo split, mesmas moléculas no treino e no
    teste. Este script re-implementa o pipeline QSAR clássico usando o
    scaffold split estratificado do projeto.

Como rodar:
    python src/baseline/classical.py

Saída:
    Métricas impressas no terminal + salvas em results/metrics/baseline.json
"""

import sys, os, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
warnings.filterwarnings("ignore")

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.data.dataset import BBBPDataset
from src.utils.splits import scaffold_split


def smiles_to_ecfp4(smiles: str) -> list | None:
    """Converte SMILES em vetor ECFP4 (2048 bits)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    return list(generator.GetFingerprint(mol))


def build_feature_matrix(smiles_list: list[str]) -> np.ndarray:
    """Converte lista de SMILES em matriz de features ECFP4."""
    fps = [smiles_to_ecfp4(smi) for smi in smiles_list]
    return np.array([fp for fp in fps if fp is not None])


def evaluate(model, X_test, y_test, nome: str) -> dict:
    """Avalia um modelo e imprime as métricas."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    metrics = {
        "modelo":   nome,
        "auc_roc":  round(roc_auc_score(y_test, probs), 4),
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1":       round(f1_score(y_test, preds), 4),
    }
    print(f"\n{nome}:")
    print(f"  AUC-ROC:  {metrics['auc_roc']}")
    print(f"  Acurácia: {metrics['accuracy']}")
    print(f"  F1:       {metrics['f1']}")
    return metrics


def main():
    # ── 1. Carregar dataset e fazer o split ──────────────────────────────────
    dataset = BBBPDataset()
    train_data, val_data, test_data = scaffold_split(
        dataset, dataset.smiles_list, dataset.labels
    )

    # reconstruir índices para extrair SMILES e labels por split
    # (o scaffold_split retorna listas de objetos Data, não índices)
    # vamos extrair labels e smiles dos objetos Data e da lista original
    def get_smiles_labels(data_list, smiles_list_full, labels_full):
        # mapear objeto Data de volta ao índice via id()
        id_to_idx = {id(d): i for i, d in enumerate(dataset.data_list)}
        indices = [id_to_idx[id(d)] for d in data_list]
        smiles = [smiles_list_full[i] for i in indices]
        labels = [labels_full[i] for i in indices]
        return smiles, labels

    train_smiles, train_labels = get_smiles_labels(train_data, dataset.smiles_list, dataset.labels)
    test_smiles,  test_labels  = get_smiles_labels(test_data,  dataset.smiles_list, dataset.labels)

    # ── 2. Construir features ECFP4 ──────────────────────────────────────────
    print("\nCalculando fingerprints ECFP4...")
    X_train = build_feature_matrix(train_smiles)
    X_test  = build_feature_matrix(test_smiles)
    y_train = np.array(train_labels)
    y_test  = np.array(test_labels)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    # ── 3. Treinar e avaliar modelos ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("BASELINE CLÁSSICO — ECFP4 + Scaffold Split")
    print("=" * 50)

    rf = RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test, "Random Forest + ECFP4")

    svm = SVC(
        kernel='rbf', class_weight='balanced', probability=True, random_state=42
    )
    svm.fit(X_train, y_train)
    svm_metrics = evaluate(svm, X_test, y_test, "SVM + ECFP4")

    # ── 4. Salvar métricas ───────────────────────────────────────────────────
    results = {"baseline": [rf_metrics, svm_metrics]}
    os.makedirs("results/metrics", exist_ok=True)
    with open("results/metrics/baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nMétricas salvas em results/metrics/baseline.json")


if __name__ == "__main__":
    main()
